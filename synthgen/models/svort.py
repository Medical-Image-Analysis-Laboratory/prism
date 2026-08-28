import math
import torch
import torch.nn as nn
from lightning import LightningModule
import random
from typing import Dict, Any
from synthgen.generator.slice_acquisition.slice_acq import slice_acquisition
from synthgen.utils.scanutils import (
    remove_parameterized_bias_field,
    validateafterscandims,
)
from synthgen.generator.scanner import Scanner
from synthgen.models.transformer import (
    SVRtransformerV2,
    SVRtransformerV4,
    SVRtransformerV5,
)
from lightning_fabric.utilities.apply_func import apply_to_collection
from synthgen.utils.trainutils import (
    TVLoss3D,
    make_volume_comparison_image,
    make_stack_comparison_image,
    make_qa_comparison_image,
)

from monai.metrics import SSIMMetric

from synthgen.models.reconstructionorig import PSFreconstruction, SRR
from synthgen.generator.transform import (
    RigidTransform,
    mat_update_resolution,
    point2mat,
    mat2point,
)
import torch.nn.functional as F
from synthgen.utils.svortutils import (
    WarmupConstantSchedule,
    img_loss_l1,
    trans_loss,
    point_loss,
    load_svort_weights,
)

from lightning.pytorch.utilities import grad_norm


class SRRLightning(LightningModule):
    """
    PyTorch Lightning implementation of SRR, accepting the dictionary output
    directly from the Scanner.scan() method.
    """

    def __init__(
        self,
        n_iter=4,
        pe=True,
        cg_iter=2,
        freezesvr=False,
        freezerreg=False,
        freezesqm=False,
        regularizer=None,
        perceptual_loss=None,
        tvloss=None,
        gradloss=None,
        lr=0.0002,
        weight_decay=0.01,
        warmup_steps=100,
        weight_point=0.1,
        weight_img=1.0,
        weight_tv=10.0,
        weight_T=0.01,
        weight_R=10.0,
        weight_gradloss=0.0,
        weight_perceptual=0.0,
        n_train=200000,
        batch_size=8,
        sqm=None,
        bnet=None,
        dcregbalancer=None,
        weight_dc=0.0,
        weight_qa_reg=0.0,
        qa_target=0.85,
        active_tasks=("motion", "bias", "intensity", "quality"),
        weight_intshift=0.0,
        weight_bias=0.0,
        weight_qa=0.0,
        reg_lambda_init=0.05,
        n_reg_lambda=None,
        inference_mode=False,
        slice_reject_ncc_thresh=0.5,
        slice_reject_min_area=400,
    ):
        """
        Initializes the SRR model
        """
        super().__init__()

        # 1. Extract Model Configuration
        self.n_iter = n_iter
        self.pe = pe
        self.cg_iter = cg_iter
        self.freezesvr = freezesvr
        self.freezesqm = freezesqm
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.weight_point = weight_point
        self.weight_img = weight_img
        self.weight_tv = weight_tv
        self.weight_T = weight_T
        self.freezerreg = freezerreg
        self.weight_gradloss = weight_gradloss
        self.weight_R = weight_R
        self.weight_perceptual = weight_perceptual
        self.total_steps = int(n_train / batch_size)
        self.attn = None
        # 2. Init model components
        self.sqm = sqm
        self.bnet = bnet
        self.dcregbalancer = dcregbalancer
        # Step-by-step training: which task heads are active (and supervised).
        self.active_tasks = tuple(active_tasks)
        # Direct-supervision weights for the per-slice intensity scale, the
        # brain-masked bias field, and the brain-masked dense QA map.
        self.weight_intshift = weight_intshift
        self.weight_bias = weight_bias
        self.weight_qa = weight_qa

        self.svrnet1 = SVRtransformerV2(
            n_layers=4,
            n_head=4 * 2,
            d_in=9 + 2,
            d_out=9,
            d_model=256 * 2,
            d_inner=512 * 2,
            dropout=0.0,
            n_channels=1,
        )

        self.svrnet2 = SVRtransformerV2(
            n_layers=4 * 2,
            n_head=4 * 2,
            d_in=9 + 2,
            d_out=9,
            d_model=256 * 2,
            d_inner=512 * 2,
            dropout=0.0,
            n_channels=2,
        )

        self.ssim_slice = SSIMMetric(spatial_dims=2, reduction="sum")

        self.CG = SRR(
            n_iter=self.cg_iter,
            use_CG=True,
        )

        self.regularizer = regularizer
        # MoDL proximal weight λ in the data-consistency block
        # (AᵀPA + λI)x = AᵀPy + λz. Learned PER ITERATION (a vector); softplus keeps
        # each entry strictly positive. The loop indexes reg_lambda[i], clamping to
        # the last entry when there are fewer λ's than registration iterations
        # (so a single λ behaves as shared, and extra iterations reuse the last).
        # Only meaningful when a denoiser/regularizer prior is present.
        if self.regularizer is not None:
            n_lambda = int(n_reg_lambda) if n_reg_lambda is not None else int(n_iter)
            n_lambda = max(1, n_lambda)
            init = float(math.log(math.expm1(reg_lambda_init)))
            self.reg_lambda = nn.Parameter(torch.full((n_lambda,), init))
        else:
            self.reg_lambda = None
        self.perceptual_loss = perceptual_loss
        self.tvloss = tvloss
        self.gradloss = gradloss
        # Unsupervised loss weights: slice data-consistency and QA->1 prior.
        self.weight_dc = weight_dc
        self.weight_qa_reg = weight_qa_reg
        # Slice-rejection (NCC data consistency): before the regularized CG,
        # re-project the regularizer-free reconstruction and reject EVERY slice
        # whose slice-to-slice NCC falls below `ncc_thresh`, plus any slice with
        # in-brain area below `min_area` pixels, by zeroing their per-slice CG
        # weight. Toggled by `slice_reject`.
        self.inference_mode = inference_mode

        self.slice_reject_ncc_thresh = float(slice_reject_ncc_thresh)
        self.slice_reject_min_area = int(slice_reject_min_area)
        # Target mean QA for the budget prior: the penalty only acts when the
        # average in-brain QA falls below this, so the model can reject up to a
        # ~(1 - qa_target)/(1 - qa_floor) fraction of pixels for free.
        self.qa_target = qa_target

        if self.freezesvr:
            for p in self.svrnet1.parameters():
                p.requires_grad = False
            for p in self.svrnet2.parameters():
                p.requires_grad = False
        if self.freezerreg:
            if isinstance(self.regularizer, torch.nn.ModuleList):
                for reg in self.regularizer:
                    for p in reg.parameters():
                        p.requires_grad = False
            elif self.regularizer is not None:
                for p in self.regularizer.parameters():
                    p.requires_grad = False
        if self.freezesqm:
            if self.sqm is not None:
                for p in self.sqm.parameters():
                    p.requires_grad = False

        self.save_hyperparameters()

    def forward_svort(self, data, n_iter=None):

        if n_iter is None:
            n_iter = self.n_iter

        params = {
            "psf": data["psf_rec"],
            "slice_shape": data["slice_shape"],
            "interp_psf": False,
            "res_s": data["resolution_slice"],
            "res_r": data["resolution_recon"],
            "s_thick": data["slice_thickness"],
            "volume_shape": data["volume_shape"],
        }

        transforms = RigidTransform(data["transforms"])
        stacks = data["stacks"]
        positions = data["positions"]
        attn_mask = None

        thetas = []
        volumes = []
        trans = []
        int_shifts_all = []
        slice_qa_all = []
        bias_field_all = []
        conf_all = []
        dc_loss_all = []
        # Monitoring only: final data-consistency residual ‖Ax−y‖_P (set at the
        # last iteration below).
        dc_resid = torch.tensor(0.0, device=stacks.device)

        # When the SVR transformers expose aux heads, the bias-field and pixel-QA
        # estimates are produced per iteration inside the loop below.
        slice_qa = torch.ones_like(stacks)

        # calculate quality of input slices using qanet if available, and use as attn_mask
        if self.sqm is not None:
            slice_qa = self.sqm(stacks, slice_mask=(stacks > 0).float())

        slice_qa_all.append(slice_qa)  # slice_qa_i)
        bias_field_all.append(torch.ones_like(stacks))  # bias_field)

        if not self.pe:
            transforms = RigidTransform(transforms.axisangle() * 0)
            positions = positions * 0 + data["slice_thickness"]

        theta = mat2point(
            transforms.matrix(), stacks.shape[-1], stacks.shape[-2], params["res_s"]
        )
        volume = None
        mask_stacks = None
        # outer loop (joint motion reconstruction estimation)
        for i in range(n_iter):
            svrnet = self.svrnet2 if i else self.svrnet1

            theta, _, attn = svrnet(
                theta,
                stacks,
                positions,
                None if ((volume is None)) else volume.detach(),
                params,
                attn_mask,
            )
            # ignore svor confidence for now, just use the QA and ncc gate
            # conf_i = conf_i.view(-1, 1, 1, 1)  # (N,1,H,W)

            thetas.append(theta)

            _trans = RigidTransform(point2mat(theta))

            with torch.no_grad():

                if "transforms_gt" in data:
                    # adjust for GT
                    diff = (
                        RigidTransform(data["transforms_gt"])
                        .compose(_trans.inv())
                        .axisangle()
                        .detach()
                    )
                    err_R = torch.sum(diff[:, :3] ** 2, -1)
                    err_T = torch.sum(diff[:, 3:] ** 2, -1)
                    err = 1000.0 * err_R + err_T
                    _, min_idx = torch.topk(err, 5, largest=False, sorted=False)
                    diff = diff[min_idx, :].mean(0, keepdim=True)
                else:
                    diff = None

            if diff is not None:
                _trans = RigidTransform(diff).compose(_trans)
            trans.append(_trans)

            with torch.no_grad():
                mat = mat_update_resolution(
                    _trans.matrix().detach(), 1, params["res_r"]
                )
                volume_psf = PSFreconstruction(mat, stacks, None, None, params)

            # ---- MoDL iteration: denoise (prior) -> data-consistency (proximal CG) ----
            # The reconstruction is weighted per-pixel by the dense QA AND per-slice
            # by the SVoRT-style reliability weight, so slices the model finds
            # unreliable are down-weighted in the CG. The reliability is NOT detached:
            # it is learned purely through the image loss (down-weighting an
            # unreliable slice lowers the reconstruction error).

            volume_dcpsf = self.CG(
                mat,
                stacks,
                volume_psf,
                params,
                slice_qa=slice_qa,
            )

            with torch.no_grad():
                sim = self.CG.A(mat, volume_dcpsf.detach(), None, None, params)
                ncc = self.slice_ncc(sim, stacks, data.get("stacks_masks"))
                ncc_gate = ncc.clamp(0.0, 1.0).view(-1, 1, 1, 1)  # (N,1,1,1)

            cg_weight = slice_qa * ncc_gate

            conf_all.append(ncc_gate)  # conf_i)
            # MoDL prior z_k = D(x_{k-1})

            z = None
            mu = 0.0
            if self.regularizer is not None:  # and volume is not None:

                # ---- Slice-level data-consistency filtering (NCC) ----
                # Re-project the regularizer-free DC reconstruction at the estimated
                # poses and reject slices BEFORE the regularized CG -- every slice
                # whose slice-to-slice NCC vs the measured stacks is below
                # `ncc_thresh`, plus any slice too small to be reliable (in-brain
                # area < `min_area`) -- by zeroing their per-slice CG weight. Hard
                # rejection: the keep mask is computed under no_grad and applied as a
                # detached {0,1} factor, so it removes the corrupted slices from the
                # final solve while the learned reliability gradient still flows
                # through the kept entries of cg_weight. Skipped at iteration 0 (the
                # poses / reconstruction are still too crude to trust the NCC).
                if self.inference_mode:
                    slice_keep, _slice_ncc = self.slice_reject_mask(
                        mat,
                        volume_dcpsf,
                        stacks,
                        params,
                        slice_mask=data.get("stacks_masks"),
                        ncc_thresh=self.slice_reject_ncc_thresh,
                        min_area=self.slice_reject_min_area,
                    )
                    cg_weight = cg_weight * slice_keep

                if not isinstance(self.regularizer, torch.nn.ModuleList):
                    regularizer = self.regularizer
                elif len(self.regularizer) == 1:
                    # A single regularizer SHARED across every iteration: the
                    # canonical MoDL weight-tied denoiser prior. A list with
                    # one-per-iteration entries is indexed by iteration instead.
                    regularizer = self.regularizer[0]
                else:
                    regularizer = self.regularizer[min(i, len(self.regularizer) - 1)]

                z = (volume_dcpsf + regularizer(volume_dcpsf)).clamp(min=0.0)
                # Per-iteration λ; reuse the last entry when there are fewer λ's
                # than iterations.
                lam_idx = min(i, self.reg_lambda.numel() - 1)
                mu = F.softplus(self.reg_lambda[lam_idx])

                # Data-consistency block: (AᵀPA + μI)x = AᵀPy + μz, solved by CG. With a
                # prior, the DC step is pulled toward the denoised z where the data is
                # weak and toward the measured slices where the data is reliable, so the
                # OUTPUT stays data-consistent (not the oversmoothed raw denoiser output).
                # Warm-start from the (detached) prior z when present -- better
                # conditioned -- else the PSF adjoint; the prior's gradient flows through
                # the μz term in b, not the warm start.
                volume = self.CG(
                    mat,
                    stacks,
                    z if z is not None else volume_dcpsf,
                    params,
                    mu=mu,
                    z=z,
                    slice_qa=cg_weight,
                )
            else:
                volume = volume_dcpsf
            # The DC output is BOTH this iteration's reconstruction and the next
            # iteration's registration reference (J-MoDL: the poses are estimated
            # against the data-consistent volume, never the denoised one). The
            # denoiser only ever shapes the prior, never the output.
            volumes.append(volume)

            # Monitoring only (no gradient): final data-consistency residual
            # ‖Ax−y‖_P -- how well the reconstruction re-projects to the measured
            # slices under the CG weighting P (= cg_weight), restricted to in-brain
            # pixels. Lets us watch the data-vs-prior balance: a λ that over-trusts
            # the denoiser prior shows up as a rising DC residual.
            if i == n_iter - 1:
                with torch.no_grad():
                    sim_dc = self.CG.A(mat, volume.detach(), None, None, params)
                    # Use the post-rejection weight so the residual reflects the
                    # slices that actually drove the final solve.
                    p_w = cg_weight
                    if mask_stacks is not None:
                        p_w = p_w * mask_stacks
                    dc_resid = (p_w * (sim_dc - stacks).abs()).sum() / (
                        p_w.sum() + 1e-6
                    )

            # Self-supervised slice data-consistency: QA-weighted L1 between the
            # simulated slices (reconstruction re-projected at the estimated pose,
            # with gradient flowing to the motion params) and the bias+intensity-
            # corrected real slices. The volume is detached so this term refines
            # motion / bias / intensity, not the volume (trained by img loss). QA is
            # detached in the weighting below, so dc does not train the QA head.
            # Normalized by the QA weight and restricted to in-brain pixels.
            # NOTE: OFF by default (weight_dc=0); kept for self-supervised training.
            if self.weight_dc > 0.0:
                # Re-slice in the reconstruction frame (volume_dc lives in the
                # GT-aligned frame); gradient still flows to the pose through the raw
                # `_trans` factor inside the aligned `_trans`.
                # Re-slice the data-consistency output (volume_dc = the proximal CG
                # solution; with MoDL it equals `volume`). Where a prior is used this
                # volume is pulled toward the denoised z in weakly-sampled regions,
                # but the residual below is weighted by the GT quality (qa_w), which
                # down-weights exactly those corrupted regions; where the data is
                # reliable the CG fit dominates, so the re-projection is faithful for
                # the pose signal. The volume is detached, so dc refines
                # motion / bias / intensity, not the volume.
                mat_grad = mat_update_resolution(_trans.matrix(), 1, params["res_r"])
                sim_slices = self.CG.A(mat_grad, volume.detach(), None, None, params)
                # Weight the registration residual by the GROUND-TRUTH quality
                # (stacks_error_l), not the model's QA estimate: dc is a
                # training-only loss, and a wrong QA estimate would mis-weight the
                # pose signal (down-weighting good pixels / trusting corrupted ones).
                # GT quality cleanly excludes the genuinely un-registerable pixels
                # (void / motion). Falls back to the (detached) estimate when no GT
                # is available (e.g. inference). QA stays detached either way, so dc
                # never trains the quality head.
                qa_w = data["stacks_error_l"] if "stacks_error_l" in data else slice_qa
                w = qa_w * data["stacks_masks"]
                resid = (sim_slices - stacks).abs()
                dc_loss_all.append((w * resid).sum() / (w.sum() + 1e-6))
            # torch.set_grad_enabled(default_grad)

        self.attn = attn.detach()

        return (
            trans,
            volumes,
            thetas,
            volume_dcpsf.detach(),
            slice_qa_all,
            bias_field_all,
            int_shifts_all,
            dc_loss_all,
            conf_all,
            dc_resid.detach(),
        )

    @staticmethod
    def slice_ncc(a, b, mask=None, eps=1e-5):
        """Per-slice normalized cross-correlation between two slice stacks.

        a, b : (N, 1, H, W) -- e.g. the simulated (re-projected) slices and the
               measured stacks. N = number of slices.
        mask : (N, 1, H, W) or None -- in-brain pixels the NCC is restricted to.
               NCC of motion/intensity-corrupted slices drops well below 1.

        Returns (N,) NCC in [-1, 1], one scalar per slice.
        """
        N = a.shape[0]
        a = a.reshape(N, -1)
        b = b.reshape(N, -1)
        if mask is not None:
            m = (mask.reshape(N, -1) > 0).to(a.dtype)
        else:
            m = torch.ones_like(a)

        cnt = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        a_c = (a - (a * m).sum(1, keepdim=True) / cnt) * m
        b_c = (b - (b * m).sum(1, keepdim=True) / cnt) * m
        num = (a_c * b_c).sum(1)
        den = torch.sqrt(a_c.pow(2).sum(1) * b_c.pow(2).sum(1) + eps)
        return num / (den + eps)

    def slice_reject_mask(
        self,
        mat,
        volume,
        stacks,
        params,
        slice_mask=None,
        ncc_thresh=0.0,
        min_area=100,
    ):
        """Binary per-slice keep mask from slice-to-slice data consistency.

        Re-projects `volume` (the regularizer-free DC reconstruction) at the
        estimated poses and scores each slice by NCC against the measured
        `stacks`. A slice is REJECTED (keep = 0) when EITHER:
          * its in-brain area is below `min_area` pixels (too small to be
            reliable -- void / brain-edge slices), OR
          * its slice-to-slice NCC is below `ncc_thresh`.
        Every slice below the NCC threshold is removed -- there is no top-k cap.

        Hard rejection, so computed entirely under no_grad and returned as a
        DETACHED {0,1} mask; the caller multiplies it into the (grad-carrying)
        CG weight, which zeros the rejected slices while leaving the learned
        reliability gradient on the kept slices intact.

        Returns (keep, ncc):
          keep : (N, 1, 1, 1) float in {0, 1}, 1 = keep, 0 = reject (broadcasts
                 over the per-slice CG weight).
          ncc  : (N,) per-slice NCC, for logging/monitoring.
        """
        with torch.no_grad():
            sim = self.CG.A(mat, volume.detach(), None, None, params)
            ncc = self.slice_ncc(sim, stacks, slice_mask)

            N = ncc.shape[0]

            # In-brain area (number of mask pixels) per slice. Without a mask,
            # treat every slice as full-size so only the NCC criterion applies.
            if slice_mask is not None:
                area = (slice_mask.reshape(N, -1) > 0.01).sum(1)
            else:
                area = torch.full(
                    (N,), stacks[0].numel(), device=ncc.device, dtype=torch.long
                )

            # Reject too-small slices outright, plus every slice whose NCC falls
            # below the threshold.
            reject = (area < min_area) | (ncc < ncc_thresh)
            keep = (~reject).to(stacks.dtype).view(N, 1, 1, 1)

        return keep, ncc

    def forward(self, data: Dict[str, torch.Tensor]):
        """
        Runs the iterative SVoRT model forward pass.
        Data is the output of the scan() method.
        """
        validateafterscandims(data)
        (
            trans,
            volumes,
            points,
            volume_dc,
            slice_qa,
            bias_field,
            int_shifts,
            dc_loss_all,
            conf_all,
            dc_resid,
        ) = self.forward_svort(data)
        return (
            trans,
            volumes,
            points,
            volume_dc,
            slice_qa,
            bias_field,
            int_shifts,
            dc_loss_all,
            conf_all,
            dc_resid,
        )

    @staticmethod
    def batchitemselect(batch: Dict[str, torch.Tensor], batchidx: int):
        """
        Selects a single item from a batch dictionary.
        Handles nested dictionaries and Tensors.

        batch: Dict[str, torch.Tensor]
            The input batch dictionary.
        batchidx: int
            The index of the item to select.

        Returns:
            A dictionary containing only the selected item.
        """

        scalkeys = {
            "gap",
            "slice_thickness",
            "resolution_slice",
            "resolution_recon",
            "resolution",
        }
        tuplekeys = {"volume_shape", "slice_shape"}
        sync_keys = scalkeys | tuplekeys

        # All scalars/shape keys need .item() / float() which forces a GPU→CPU
        # round trip per call (~6ms latency each). Fuse into ONE D2H transfer.
        small: list[tuple[str, int]] = []  # (key, numel)
        parts: list[torch.Tensor] = []
        for k, v in batch.items():
            if k in sync_keys and isinstance(v, torch.Tensor) and v.is_cuda:
                parts.append(v.flatten())
                small.append((k, v.numel()))
        if parts:
            flat_cpu = torch.cat(parts).cpu()  # single D2H transfer + sync
            offset = 0
            cpu_vals: dict[str, torch.Tensor] = {}
            for k, n in small:
                cpu_vals[k] = flat_cpu[offset : offset + n]
                offset += n
        else:
            cpu_vals = {}

        selected = {}
        for k, v in batch.items():
            if k in tuplekeys:
                cv = cpu_vals.get(k, v)
                selected[k] = torch.Size([int(x) for x in cv])
            elif k in scalkeys:
                cv = cpu_vals.get(k, v)
                selected[k] = float(cv[batchidx])
            else:
                selected[k] = v[batchidx]  # GPU tensor view, no copy

        return selected

    def _shared_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        """
        Shared logic for both training, validation, and prediction:
        1. Data prep
        2. Forward pass
        3. Loss calculation
        4. Aggregation
        """
        # Batch size check
        batchsize = batch["volume_gt"].shape[0] if "volume_gt" in batch else 1
        assert batchsize == 1, "Batch size must be 1."

        # ---- Prepare single-item input ----
        # Use no_grad for data wrangling to ensure graph is clean
        with torch.no_grad():
            scanned_data = self.batchitemselect(batch, 0)

        # 1. Forward
        (
            transforms,
            volumes,
            points,
            volume_dc,
            slice_qa_list,
            bias_field_list,
            int_shifts,
            dc_loss_all,
            conf_all,
            dc_resid,
        ) = self(scanned_data)

        # 2. GT + constants
        transforms_gt_mat = (
            scanned_data["transforms_gt"] if "transforms_gt" in scanned_data else None
        )
        volume_gt = scanned_data["volume_gt"] if "volume_gt" in scanned_data else None
        rs = (
            scanned_data["resolution_slice"]
            if "resolution_slice" in scanned_data
            else None
        )
        sx, sy = (
            scanned_data["slice_shape"]
            if "slice_shape" in scanned_data
            else (None, None)
        )
        transforms_gt_rigid = (
            RigidTransform(transforms_gt_mat) if transforms_gt_mat is not None else None
        )

        # 3. Loss Calculation
        # Point Loss
        point_l = (
            point_loss(points, transforms_gt_mat, sx, sy, rs)
            if transforms_gt_mat is not None
            else [torch.tensor(0.0, device=self.device)]
        )

        # Rotation/Translation Loss
        loss_R, loss_T = (
            trans_loss(transforms, transforms_gt_rigid)
            if transforms_gt_rigid is not None
            else (
                [torch.tensor(0.0, device=self.device)] * len(transforms),
                [torch.tensor(0.0, device=self.device)] * len(transforms),
            )
        )

        # Image/Volume Loss
        # Image/Volume Loss
        img_l = (
            img_loss_l1(
                volumes,
                volume_gt,
                mask=scanned_data.get("mask_gt", volume_gt > 0),
            )
            if volume_gt is not None
            else [torch.tensor(0.0, device=self.device)]
        )

        # Gradient Loss
        if self.gradloss is not None and volume_gt is not None:
            grad_l = self.gradloss.forward(volumes[-1], volume_gt)
        else:
            grad_l = torch.tensor(0.0, device=self.device)

        # TV & Perceptual Loss
        if self.tvloss is not None:
            tv_loss = self.tvloss.forward(volumes[-1])
        else:
            tv_loss = torch.tensor(0.0, device=self.device)

        if self.perceptual_loss is not None and volume_gt is not None:
            perc_loss = self.perceptual_loss.forward(volumes[-1], volume_gt)
        else:
            perc_loss = torch.tensor(0.0, device=self.device)

        # Use scanned_data (the single-item view) so the per-slice mask matches the
        # per-slice slice_qa dims; equivalent to batch["stacks_masks"][0].
        stacks_mask = scanned_data["stacks_masks"]

        # --- Unsupervised: slice data-consistency (averaged over iterations) ---
        # The per-iteration QA-weighted residuals are computed inside
        # forward_svort (they need the per-iteration pose / corrected stacks / QA);
        # here we simply average them.
        dc_l = (
            sum(dc_loss_all) / len(dc_loss_all)
            if len(dc_loss_all) > 0
            else torch.tensor(0.0, device=self.device)
        )

        # --- Unsupervised: QA budget prior (averaged over iterations) ---
        # Instead of a per-pixel pull to 1 (a brittle global threshold against the
        # image-loss "reject" signal), penalize only the *mean* in-brain QA falling
        # below qa_target. This gives the model a rejection budget: QA stays ~1
        # where the image loss is indifferent, but a fraction of pixels can be
        # driven down on genuine artifacts for free, as long as the average holds.
        # Robust to weight_qa_reg (it just enforces the mean floor) and to changes
        # in weight_img. Background QA is already forced to 1.
        if (
            self.weight_qa_reg > 0.0
            and len(slice_qa_list) > 0
            and slice_qa_list[0] is not None
        ):
            qa_reg = torch.tensor(0.0, device=self.device)
            for qa in slice_qa_list:
                qa_reg = qa_reg + F.relu(self.qa_target - qa[stacks_mask].mean())
            qa_reg = qa_reg / len(slice_qa_list)
        else:
            qa_reg = torch.tensor(0.0, device=self.device)

        # --- Supervised: per-slice intensity scale L1 (averaged over iterations) ---
        # int_shifts[i] is (N,1); GT int_shifts is (N,) and is the multiplicative
        # factor applied at scan time (predicted scale should match it directly).
        int_gt = scanned_data.get("int_shifts")
        if (
            self.weight_intshift > 0.0
            and "intensity" in self.active_tasks
            and int_gt is not None
            and len(int_shifts) > 0
        ):
            int_tgt = int_gt.view(-1, 1)
            int_l = sum((s - int_tgt).abs().mean() for s in int_shifts) / len(
                int_shifts
            )
        else:
            int_l = torch.tensor(0.0, device=self.device)

        # --- Supervised: brain-masked bias-field L1 (averaged over iterations) ---
        # Bias outside the brain is unobservable/ambiguous, so the L1 is restricted
        # to in-brain pixels. bias_field_list[i] and gt_bias_field are both (N,H,W)
        # and come from the SAME remove_parameterized_bias_field, so are directly
        # comparable.
        bias_gt = scanned_data.get("gt_bias_field")
        if (
            self.weight_bias > 0.0
            and "bias" in self.active_tasks
            and bias_gt is not None
            and len(bias_field_list) > 0
        ):
            bm = stacks_mask.squeeze(1) > 0  # (N,H,W) bool
            bdenom = bm.sum().clamp_min(1.0)
            bias_l = sum(
                ((bf - bias_gt).abs() * bm).sum() / bdenom for bf in bias_field_list
            ) / len(bias_field_list)
        else:
            bias_l = torch.tensor(0.0, device=self.device)

        # --- Supervised: brain-masked dense QA L1 (averaged over iterations) ---
        # slice_qa_list[i] and stacks_error_l are both (N,1,H,W) in [0,1]; restrict
        # to in-brain pixels (background QA is already forced to 1).
        qa_gt = scanned_data.get("stacks_error_l")
        if (
            self.weight_qa > 0.0
            and "quality" in self.active_tasks
            and qa_gt is not None
            and len(slice_qa_list) > 0
            and slice_qa_list[0] is not None
        ):
            qdenom = stacks_mask.sum().clamp_min(1.0)
            qa_sup_l = sum(
                ((qa - qa_gt).abs() * stacks_mask).sum() / qdenom
                for qa in slice_qa_list
            ) / len(slice_qa_list)
        else:
            qa_sup_l = torch.tensor(0.0, device=self.device)

        # 4. Total Loss Aggregation
        #   Supervised:   reconstructed-volume L1  +  pose (point + R/T)
        #   Unsupervised: QA-weighted slice data-consistency  +  QA->1 prior
        # TV / perceptual / grad remain as weight-gated auxiliary terms (bias,
        # intensity and QA are supervised implicitly through the differentiable CG
        # by the volume L1). Pose is supervised by TWO COMPLEMENTARY terms (not
        # redundant): the point loss on the RAW absolute pose, and the R/T axis-angle
        # loss on the GT-gauge-ALIGNED pose -- i.e. the resolvable RELATIVE
        # registration error with the unobservable global gauge removed (see the
        # gauge-fix in forward_svort). The gauge-aligned R/T is the primary
        # registration signal; dropping it lets relative registration drift.
        #
        # SVoRT-faithful aggregation: EVERY per-iteration loss is SUMMED over the
        # registration iterations (deep supervision), exactly as the reference
        # (`weight_* * sum(loss_*)`) -- including the image loss. Averaging the image
        # loss instead would divide its effective weight by n_iter relative to SVoRT.
        vol_l1 = sum(img_l)
        motion_l2 = sum(point_l)
        rot_l = sum(loss_R)
        trans_l = sum(loss_T)

        total_loss = (
            # supervised
            self.weight_img * vol_l1
            + self.weight_point * motion_l2
            + self.weight_R * rot_l
            + self.weight_T * trans_l
            + self.weight_intshift * int_l
            + self.weight_bias * bias_l
            + self.weight_qa * qa_sup_l
            # unsupervised
            + self.weight_dc * dc_l
            + self.weight_qa_reg * qa_reg
            # auxiliary (weight-gated)
            + self.weight_tv * tv_loss
            + self.weight_perceptual * perc_loss
            + self.weight_gradloss * grad_l
        )

        outputs = {
            "volumes": volumes,
            "volume_gt": volume_gt,
            "volume_dc": volume_dc,
            # Added these to support analysis in predict_step
            "transforms": transforms,
            "transforms_gt": transforms_gt_mat,
            "stacks": scanned_data["stacks"].detach().float().cpu(),
            "transforms_pred": torch.stack([t.matrix() for t in transforms])
            .detach()
            .cpu(),
            "positions": scanned_data["positions"].detach().float().cpu(),
            "resolution_slice": scanned_data["resolution_slice"],
            "slice_thickness": scanned_data["slice_thickness"],
            "resolution_recon": scanned_data["resolution_recon"],
            "psf_acq": scanned_data["psf_rec"].detach().float().cpu(),
            "slice_shape": scanned_data["slice_shape"],
        }

        # Per-slice tensors for the stack-level validation visualization. Packaged
        # only outside training to avoid the extra forward projection and D2H copies
        # on every training step. `sim_stacks` re-projects the final reconstruction
        # at the estimated (GT-aligned) poses, so it is directly comparable, slice
        # for slice, with the real (corrupted, brain-masked) stacks.
        if (not self.training) and batch_idx == 1:
            viz_params = {
                "psf": scanned_data["psf_rec"],
                "slice_shape": scanned_data["slice_shape"],
                "interp_psf": False,
                "res_s": scanned_data["resolution_slice"],
                "res_r": scanned_data["resolution_recon"],
                "s_thick": scanned_data["slice_thickness"],
                "volume_shape": scanned_data["volume_shape"],
            }
            mat_final = mat_update_resolution(
                transforms[-1].matrix(), 1, scanned_data["resolution_recon"]
            )
            sim_stacks = self.CG.A(
                mat_final, volumes[-1].detach(), None, None, viz_params
            )

            def _cpu(x):
                return x.detach().float().cpu() if x is not None else None

            outputs["viz"] = {
                "real_stacks": _cpu(scanned_data["stacks"]),
                "sim_stacks": _cpu(sim_stacks),
                "slice_qa": _cpu(slice_qa_list[-1]) if slice_qa_list else None,
                "qa_gt": _cpu(scanned_data.get("stacks_error_l")),
                "bias_est": _cpu(bias_field_list[-1]) if bias_field_list else None,
                "bias_gt": _cpu(scanned_data.get("gt_bias_field")),
                "int_est": _cpu(int_shifts[-1]) if int_shifts else None,
                "int_gt": _cpu(scanned_data.get("int_shifts")),
                "conf_est": _cpu(conf_all[-1]) if conf_all else None,
                "positions": _cpu(scanned_data["positions"]),
            }

        return {
            "total_loss": total_loss,
            # SVoRT-style losses: raw, UNWEIGHTED, LAST-ITERATION values
            # (loss_*[-1]) -- exactly what the reference logs as point/rot/tran/img.
            # The optimizer sees the weighted SUM over iterations (total_loss).
            "losses": {
                "point": point_l[-1].detach(),
                "rot": loss_R[-1].detach(),
                "trans": loss_T[-1].detach(),
                "img": img_l[-1].detach() * self.weight_img,
            },
            # Non-motion / non-image auxiliary terms (weighted contributions),
            # unchanged.
            "components": {
                "int_sup": int_l * self.weight_intshift,
                "bias_sup": bias_l * self.weight_bias,
                "qa_sup": qa_sup_l * self.weight_qa,
                "dc": dc_l * self.weight_dc,
                "qa_reg": qa_reg * self.weight_qa_reg,
                "tv": tv_loss * self.weight_tv,
                "perc": perc_loss * self.weight_perceptual,
                "grad": grad_l * self.weight_gradloss,
            },
            # Unweighted monitoring metrics (not part of the optimized loss).
            "metrics": {
                "dc_resid": dc_resid.detach(),
            },
            "outputs": outputs,
        }

    def _log_modl_metrics(self, results: Dict, stage: str) -> None:
        """Log MoDL monitoring signals: the final data-consistency residual
        ‖Ax−y‖_P and the learned per-iteration proximal weights λ_k =
        softplus(reg_lambda_k). Both are unweighted diagnostics for watching the
        data-vs-prior balance, not part of the optimized loss."""
        log = {f"dc_resid/{stage}": results["metrics"]["dc_resid"]}
        if self.reg_lambda is not None:
            lam = F.softplus(self.reg_lambda.detach())
            for k in range(lam.numel()):
                log[f"lambda_{k}/{stage}"] = lam[k]
        self.log_dict(log, on_step=False, on_epoch=True, prog_bar=False)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        # Run shared processing
        results = self._shared_step(batch, batch_idx)

        total_loss = results["total_loss"]
        comps = results["components"]
        losses = results["losses"]

        # Logging for Training
        self.log_dict(
            {
                "loss_total/train": total_loss,
                # SVoRT-style: raw, UNWEIGHTED, LAST-ITERATION values (loss_*[-1]).
                "loss_point/train": losses["point"],
                "loss_rot/train": losses["rot"],
                "loss_trans/train": losses["trans"],
                "loss_imgl1/train": losses["img"],
                "loss_tv/train": comps["tv"],
                "loss_percept/train": comps["perc"],
                "loss_grad/train": comps["grad"],
                "loss_intshift/train": comps["int_sup"],
                "loss_bias/train": comps["bias_sup"],
                "loss_qa/train": comps["qa_sup"],
                "loss_dc/train": comps["dc"],
                "loss_qa_reg/train": comps["qa_reg"],
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self._log_modl_metrics(results, "train")

        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        # Run shared processing
        results = self._shared_step(batch, batch_idx)

        total_loss = results["total_loss"]
        comps = results["components"]
        losses = results["losses"]
        out = results["outputs"]

        # Logging for Validation
        self.log_dict(
            {
                "loss_total/val": total_loss,
                # SVoRT-style: raw, UNWEIGHTED, LAST-ITERATION values (loss_*[-1]).
                "loss_point/val": losses["point"],
                "loss_rot/val": losses["rot"],
                "loss_trans/val": losses["trans"],
                "loss_imgl1/val": losses["img"],
                "loss_tv/val": comps["tv"],
                "loss_intshift/val": comps["int_sup"],
                "loss_bias/val": comps["bias_sup"],
                "loss_qa/val": comps["qa_sup"],
                "loss_percept/val": comps["perc"],
                "loss_grad/val": comps["grad"],
                "loss_dc/val": comps["dc"],
                "loss_qa_reg/val": comps["qa_reg"],
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self._log_modl_metrics(results, "val")

        # Visualization (First 3 batches only)
        if batch_idx == 1:
            gt_one = out["volume_gt"][0].detach().float().cpu()
            recon_one = out["volumes"][-1][0].detach().float().cpu()
            dc_one = out["volume_dc"][0].detach().float().cpu()

            best_psf_volume = self.recon_gt_psf_volume(batch, batch_idx)

            fig = make_volume_comparison_image(
                gt_one,
                recon_one,
                dc_one,
                best_psf_volume,
                f"epoch={self.current_epoch}_step={self.global_step}_example={batch_idx}",
            )
            self.val_example_figures.append(fig)

            # Stack-level visualization: pick the stack with the most slices and
            # plot real vs simulated stacks and the per-slice estimates.
            viz = out.get("viz")
            if viz is not None and viz.get("real_stacks") is not None:
                stack_ids = viz["positions"][:, 1]
                uniq, counts = torch.unique(stack_ids, return_counts=True)
                chosen = uniq[counts.argmax()]
                sel = stack_ids == chosen

                def _sel(x):
                    return x[sel] if x is not None else None

                stack_fig = make_stack_comparison_image(
                    _sel(viz["real_stacks"]),
                    _sel(viz["sim_stacks"]),
                    _sel(viz["slice_qa"]),
                    _sel(viz.get("qa_gt")),
                    conf_est=_sel(viz.get("conf_est")),
                    caption=f"epoch={self.current_epoch}_step={self.global_step}_stack",
                )
                if stack_fig is not None:
                    self.val_stack_figures.append(stack_fig)

        return total_loss

    def recon_gt_psf_volume(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Reconstructs the ground truth volume using the best PSF for comparison.
        The 'batch' is the dictionary output from scanner.scan().

        batch: Dict[str, torch.Tensor]
            Expected keys (already on device):
                - 'image': Ground truth volume [B, C, D, H, W]
                - 'label': (optional) segmentation map [B, C, D, H, W]
        """

        batchsize = batch["volume_gt"].shape[0]
        assert batchsize == 1, "Predict step currently supports batch size of 1."
        # 1. Ensure no gradients are tracked during validation/testing
        scanned_data = self.batchitemselect(batch, 0)

        params = {
            "psf": scanned_data["psf_acq"],
            "slice_shape": scanned_data["slice_shape"],
            "interp_psf": False,
            "res_s": scanned_data["resolution_slice"],
            "res_r": scanned_data["resolution_recon"],
            "s_thick": scanned_data["slice_thickness"],
            "volume_shape": scanned_data["volume_shape"],
        }

        transforms_gt_mat = scanned_data["transforms_gt"]
        stacks = scanned_data["stacks"]

        mat = mat_update_resolution(transforms_gt_mat.detach(), 1, params["res_r"])
        best_psf_volume = PSFreconstruction(mat, stacks, None, None, params)

        return best_psf_volume.detach().float().cpu()

    def predict_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Calculates and logs the combined loss and key metrics using the
        exact same pipeline as training/validation.
        """

        # 1. Run shared processing to guarantee identical logic
        results = self._shared_step(batch, batch_idx)

        # 2. Extract results
        total_loss = results["total_loss"]
        comps = results["components"]
        out = results["outputs"]

        # 3. Unpack artifacts for saving/logging
        transforms = out["transforms"]
        volumes = out["volumes"]
        volume_gt = out["volume_gt"]

        # Extract predicted matrices (convert RigidTransform objects to tensor)
        transforms_pred = torch.stack([t.matrix() for t in transforms])

        # 4. Construct Results Dictionary
        # We move tensors to CPU/float here to save memory during inference aggregation
        results_dict = {
            "loss_total/predict": total_loss.item(),
            # SVoRT-style: raw, unweighted, last-iteration values (loss_*[-1]).
            "loss_point/predict": results["losses"]["point"].item(),
            "loss_rot/predict": results["losses"]["rot"].item(),
            "loss_trans/predict": results["losses"]["trans"].item(),
            "loss_img/predict": results["losses"]["img"].item(),
            "loss_tv/predict": comps["tv"].item(),
            "loss_grad/predict": comps["grad"].item(),
            "loss_perc/predict": comps["perc"].item(),
            # Artifacts (detached and moved to CPU)
            "gt_volume": (
                volume_gt[0].detach().float().cpu() if volume_gt is not None else None
            ),
            "recon_volume": volumes[-1][0].detach().float().cpu(),
            "stacks": results["outputs"]["stacks"].detach().float().cpu(),
            "transforms_gt": (
                results["outputs"]["transforms_gt"].detach().cpu()
                if "transforms_gt" in results["outputs"]
                and results["outputs"]["transforms_gt"] is not None
                else None
            ),
            "transforms_pred": transforms_pred.detach().cpu(),
            "psf_acq": results["outputs"]["psf_acq"].detach().float().to("cuda"),
            "slice_shape": results["outputs"]["slice_shape"],
            "resolution_slice": results["outputs"]["resolution_slice"],
            "slice_thickness": results["outputs"]["slice_thickness"],
            "resolution_recon": results["outputs"]["resolution_recon"],
            "positions": results["outputs"]["positions"].detach().float().cpu(),
        }

        # Merge with scanned_data metadata
        return results_dict | results["outputs"]

    def setup(self, stage: str):
        if self.perceptual_loss is not None:
            self.perceptual_loss = self.perceptual_loss.to(self.device)

    def configure_optimizers(self):
        """
        Sets up the AdamW optimizer and the WarmupConstantSchedule.
        """
        # Optimizer: AdamW [19, 25]
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Scheduler: linear warmup then CONSTANT LR. The previous linear
        # decay-to-zero schedule (a) stopped learning once the fixed horizon was
        # passed and (b) was ill-suited to resuming/fine-tuning from arbitrary
        # epochs. A warmup+constant schedule never decays to 0 and is horizon
        # independent, so continuing training from any checkpoint keeps a stable
        # LR. Lower `lr` in the config to anneal for a fine-tuning phase.
        scheduler = WarmupConstantSchedule(
            optimizer,
            warmup_steps=self.warmup_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def on_validation_epoch_start(self):
        # clear stored figures at the start of every val epoch
        self.val_example_figures = []
        self.val_stack_figures = []

    def on_validation_epoch_end(self):
        if self.logger is not None:
            run = self.logger.experiment
            log_dict = {}
            if self.val_example_figures:
                log_dict["Images"] = self.val_example_figures.pop(0)
            if self.val_stack_figures:
                log_dict["Stacks"] = self.val_stack_figures.pop(0)
            if log_dict:
                run.log(log_dict, step=self.global_step)

    def batch_to_device(self, batch, device, dataloader_idx=0):
        """
        Move batch to device using non-blocking transfers when possible.
        Works with nested structures and TorchIO batches.
        """

        def move_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.to(device, non_blocking=True)
            return x

        return apply_to_collection(batch, torch.Tensor, move_tensor)

    @staticmethod
    def total_variation_loss(img):
        """
        Calculates the Total Variation loss for an image tensor.

        Args:
            img: Tensor of shape (n_slices, 1, X, Y)
                where X is height and Y is width.
        Returns:
            A scalar tensor representing the TV loss.
        """
        # 1. Calculate TV along the X axis (Height)
        # We compare pixels shifted by 1 row.
        # img[:, :, 1:, :] drops the first row.
        # img[:, :, :-1, :] drops the last row.
        tv_x = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])

        # 2. Calculate TV along the Y axis (Width)
        # We compare pixels shifted by 1 column.
        # img[:, :, :, 1:] drops the first column.
        # img[:, :, :, :-1] drops the last column.
        tv_y = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])

        # 3. Sum the means
        # Taking the mean ensures your loss value doesn't explode
        # if you change your X, Y resolution or batch size (n_slices) later.
        return torch.mean(tv_x) + torch.mean(tv_y)

    # def on_after_backward(self):
    #     norms = grad_norm(self, norm_type=2)
    #     self.log("grad/global_norm", norms["grad_2.0_norm_total"], on_step=True)


class SRRLightningSQM(SRRLightning):
    """
    Lightning module for SVoRT-based Super-Resolution Reconstruction (SRR) with
    registration. Inherits from SRRLightning and adds registration-specific
    functionality.
    """

    def forward_svort(self, data, n_iter=None):
        """QA-only forward: run *only* the slice-quality model and skip the whole
        SVoRT registration / MoDL reconstruction stack. Every non-QA output is a
        cheap placeholder so the inherited 10-tuple contract is preserved, but no
        volume is reconstructed -- keeping QA-only training fast and light."""

        stacks = data["stacks"]

        # Dense per-slice quality map (N,1,H,W); background pixels forced to 1.0.
        if self.sqm is not None:
            slice_qa = self.sqm(stacks, slice_mask=(stacks > 0).float())
        else:
            slice_qa = torch.ones_like(stacks)

        # Placeholders for the unused registration / reconstruction outputs. The
        # dummy volume is a 1-element tensor (never indexed in QA-only steps); the
        # dc residual is a real 0 tensor so downstream `.detach()` calls are safe.
        dummy_vol = torch.zeros((1, 1, 1, 1, 1), device=stacks.device)
        dc_resid = torch.zeros((), device=stacks.device)
        return (
            [],  # trans
            [dummy_vol],  # volumes
            [],  # thetas (points)
            dummy_vol.detach(),  # volume_dc
            [slice_qa],  # slice_qa_all
            [],  # bias_field_all
            [],  # int_shifts_all
            [],  # dc_loss_all
            [],  # conf_all
            dc_resid,
        )

    def _shared_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        """QA-only shared step: forward the slice-quality model and supervise its
        dense QA map against the GT artifact map (`stacks_error_l`). Every other
        loss term (motion / image / bias / intensity / dc) is disabled -- only the
        QA head is trained, so no reconstruction is run."""
        # Batch size check
        batchsize = batch["volume_gt"].shape[0] if "volume_gt" in batch else 1
        assert batchsize == 1, "Batch size must be 1."

        # ---- Prepare single-item input (data wrangling under no_grad) ----
        with torch.no_grad():
            scanned_data = self.batchitemselect(batch, 0)

        # Forward: only the dense QA map(s) are meaningful here.
        _, _, _, _, slice_qa_list, _, _, _, _, _ = self(scanned_data)

        # Use scanned_data (the single-item view) so the per-slice mask matches the
        # per-slice slice_qa dims; equivalent to batch["stacks_masks"][0].
        stacks_mask = scanned_data["stacks_masks"]

        # --- Supervised: brain-masked dense QA L1 (averaged over QA maps) ---
        # slice_qa_list[i] and stacks_error_l are both (N,1,H,W) in [0,1]; restrict
        # to in-brain pixels (background QA is already forced to 1).
        qa_gt = scanned_data.get("stacks_error_l")
        if (
            self.weight_qa > 0.0
            and "quality" in self.active_tasks
            and qa_gt is not None
            and len(slice_qa_list) > 0
            and slice_qa_list[0] is not None
        ):
            qdenom = stacks_mask.sum().clamp_min(1.0)
            qa_sup_l = sum(
                ((qa - qa_gt).abs() * stacks_mask).sum() / qdenom
                for qa in slice_qa_list
            ) / len(slice_qa_list)
        else:
            qa_sup_l = torch.tensor(0.0, device=self.device)

        total_loss = self.weight_qa * qa_sup_l

        # Per-slice tensors for the QA visualization. Only materialized (D2H copies)
        # outside training, on the fixed viz batch, to keep training steps light.
        outputs = {}
        if (not self.training) and batch_idx == 1:

            def _cpu(x):
                return x.detach().float().cpu() if x is not None else None

            outputs = {
                "stacks": _cpu(scanned_data["stacks"]),
                "positions": _cpu(scanned_data["positions"]),
                "slice_qa": _cpu(slice_qa_list[-1]) if slice_qa_list else None,
                "qa_gt": _cpu(qa_gt),
            }

        return {
            "total_loss": total_loss,
            "qa_sup": qa_sup_l.detach(),
            "outputs": outputs,
        }

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        results = self._shared_step(batch, batch_idx)
        total_loss = results["total_loss"]
        self.log_dict(
            {
                "loss_total/train": total_loss,
                "loss_qa/train": results["qa_sup"],
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        results = self._shared_step(batch, batch_idx)
        total_loss = results["total_loss"]
        out = results["outputs"]

        self.log_dict(
            {
                "loss_total/val": total_loss,
                "loss_qa/val": results["qa_sup"],
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # QA stack visualization: pick the stack with the most slices and plot the
        # real stack, the estimated quality map, and the GT artifact map.
        if batch_idx == 1 and out.get("slice_qa") is not None:
            stack_ids = out["positions"][:, 1]
            uniq, counts = torch.unique(stack_ids, return_counts=True)
            chosen = uniq[counts.argmax()]
            sel = stack_ids == chosen

            def _sel(x):
                return x[sel] if x is not None else None

            qa_fig = make_qa_comparison_image(
                _sel(out["stacks"]),
                _sel(out["slice_qa"]),
                _sel(out["qa_gt"]),
                caption=f"epoch={self.current_epoch}_step={self.global_step}_qa",
            )
            if qa_fig is not None:
                self.val_stack_figures.append(qa_fig)

        return total_loss


class SRRLightningReg(SRRLightning):
    """Regularizer-only training: learn the MoDL volume denoiser in isolation.

    The forward reconstructs from the *ground-truth* per-slice poses (perturbed by
    a controllable amount of pose noise to mimic imperfect motion estimation):

        PSF init -> CG (data consistency) -> reg prior z=x+reg(x) -> CG (MoDL DC).

    The SVR registration transformers are never run (and should be frozen via
    `freezesvr`). Only the regularizer (+ its MoDL proximal weight λ) is trained,
    supervised by an UNMASKED L1 + perceptual loss on the final volume.
    """

    # Empirical gain relating the anchor-point noise std to the induced point
    # loss: after point2mat re-orthonormalization, point_loss ≈ GAIN * sigma^2
    # (measured ~1.2 across resolutions/poses in this pipeline). Used to size the
    # noise so its top-of-range point loss matches `max_motion_point_loss`.
    _POINT_NOISE_GAIN = 1.2

    def __init__(self, max_motion_point_loss: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        # Pose-noise severity is sampled uniformly in [0, sigma_max] per forward;
        # sigma_max is calibrated so the induced point loss reaches
        # ~max_motion_point_loss at the top of the range.
        self.max_motion_point_loss = float(max_motion_point_loss)
        self._point_noise_sigma_max = math.sqrt(
            max(self.max_motion_point_loss, 0.0) / self._POINT_NOISE_GAIN
        )
        # Parent already ran save_hyperparameters(); record the subclass arg too so
        # it round-trips through checkpoints.
        self.hparams["max_motion_point_loss"] = self.max_motion_point_loss

        # Learnable MoDL regularization weight λ (single scalar), owned by the reg
        # module. Stored in log-space and passed through softplus in the forward, so
        # the value used in the CG is ALWAYS non-negative -- a negative λ makes the
        # normal operator (AᵀPA + λI) indefinite and breaks CG. Initialized HIGH
        # (softplus(raw) == reg_lambda_init) so the data-consistency solve leans on
        # the regularizer prior from the start; the optimizer is then free to learn
        # the data-vs-prior balance. Replaces the parent's per-iteration reg_lambda
        # vector with one clearly-scoped scalar for this single reg/CG step.
        reg_lambda_init = float(self.hparams.get("reg_lambda_init", 0.5))
        raw = math.log(math.expm1(reg_lambda_init))  # softplus^{-1}(reg_lambda_init)
        self.reg_lambda = nn.Parameter(torch.full((1,), raw))

    def forward_svort(self, data, n_iter=None):
        """Reconstruct from noised GT poses: PSF -> CG -> reg -> CG."""
        params = {
            "psf": data["psf_rec"],
            "slice_shape": data["slice_shape"],
            "interp_psf": False,
            "res_s": data["resolution_slice"],
            "res_r": data["resolution_recon"],
            "s_thick": data["slice_thickness"],
            "volume_shape": data["volume_shape"],
        }
        stacks = data["stacks"]
        # Anchor-point conversion uses the same (sx, sy, rs) convention as the SVR
        # path / point_loss: sx = last dim, sy = second-to-last, rs = slice res.
        sx, sy, rs = stacks.shape[-1], stacks.shape[-2], params["res_s"]

        # --- Imperfect motion estimation: perturb the GT poses --------------- #
        # Perturb in anchor-point space (mm), so the induced error is directly the
        # point loss. Severity sampled uniformly per forward in [0, sigma_max].
        gt_mat = RigidTransform(data["transforms_gt"]).matrix()
        p_gt = mat2point(gt_mat, sx, sy, rs)
        scale = torch.rand((), device=stacks.device)
        sigma = scale * self._point_noise_sigma_max
        noisy_mat = point2mat(p_gt + torch.randn_like(p_gt) * sigma)

        # Induced point loss (monitoring only): the actual error the reg must fix.
        with torch.no_grad():
            induced_pl = F.mse_loss(mat2point(noisy_mat, sx, sy, rs), p_gt)

        mat = mat_update_resolution(noisy_mat, 1, params["res_r"])

        # GT per-pixel slice quality (artifact corruption): the base CG
        # data-consistency weight P (1 = trust, 0 = reject). Falls back to uniform
        # weighting if no GT quality map is available.
        cg_weight = data.get("stacks_error_l")

        # --- PSF init -> CG (data consistency, GT-quality weighted) ---------- #
        # First solve weighted by ARTIFACT quality only (the NCC gate below needs a
        # reconstruction to score against, so it cannot enter this pass).
        volume_psf = PSFreconstruction(mat, stacks, None, None, params)
        volume_dc = self.CG(mat, stacks, volume_psf, params, slice_qa=cg_weight)

        # --- Per-slice registration gate (NCC) ------------------------------- #
        # Re-project volume_dc at the (noised) poses and score each slice by NCC vs
        # the measured stacks: a mis-registered slice re-projects poorly -> low NCC.
        # Clamped to [0, 1] so it is a pure DOWN-WEIGHTING gate -- raw NCC in [-1, 1]
        # would otherwise flip the sign of a slice's data-term contribution. Computed
        # under no_grad (in reg-only training the registration quality is a given;
        # the gate just tells the final solve which slices to trust). Combined
        # MULTIPLICATIVELY with the GT artifact quality, so a slice must be BOTH
        # well-registered AND artifact-free to drive the data-consistency solve.
        # This keeps the per-slice reliability in MEASUREMENT (slice) space -- the
        # natural home of the weighted-least-squares weight P -- instead of splatting
        # it into the volume.
        with torch.no_grad():
            sim = self.CG.A(mat, volume_dc.detach(), None, None, params)
            ncc = self.slice_ncc(sim, stacks, data.get("stacks_masks"))
            ncc_gate = ncc.clamp(0.0, 1.0).view(-1, 1, 1, 1)  # (N,1,1,1)
        cg_weight_final = ncc_gate if cg_weight is None else cg_weight * ncc_gate

        # --- Regularizer prior -> CG (MoDL data-consistency solve) ----------- #
        if isinstance(self.regularizer, torch.nn.ModuleList):
            regularizer = self.regularizer[0]
        else:
            regularizer = self.regularizer
        if regularizer is not None:
            # Single-channel residual denoiser on the DC reconstruction.
            z = (volume_dc + regularizer(volume_dc)).clamp(min=0.0)
            # Constant scalar prior weight μ = softplus(λ) >= 0 keeps the normal
            # operator (AᵀPA + μI) positive-definite (a negative/indefinite μ breaks
            # CG). "Rely on the prior where the data is unreliable" comes for FREE
            # from P: in voxels seen only by gated-out slices BOTH AᵀPA·x and the RHS
            # AᵀPy vanish, so the solve (AᵀPA + μI)x = AᵀPy + μz collapses to x ≈ z;
            # where P is high the data term dominates and the CG keeps the final say.
            mu = F.softplus(self.reg_lambda[0]) if self.reg_lambda is not None else 0.0
            volume = self.CG(
                mat, stacks, z, params, mu=mu, z=z, slice_qa=cg_weight_final
            )
        else:
            volume = volume_dc

        trans = [RigidTransform(noisy_mat)]
        return (
            trans,  # noised poses used for PSF/CG
            [volume],  # final reg output
            [p_gt],  # GT anchor points
            volume_dc.detach(),  # pre-reg DC reconstruction (for viz)
            [torch.ones_like(stacks)],  # slice_qa (unused)
            [],  # bias_field_all
            [],  # int_shifts_all
            [],  # dc_loss_all
            [],  # conf_all
            induced_pl.detach(),  # induced point loss (monitoring), in dc_resid slot
        )

    def _shared_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        """Reg-only shared step: UNMASKED L1 + perceptual on the final volume."""
        batchsize = batch["volume_gt"].shape[0] if "volume_gt" in batch else 1
        assert batchsize == 1, "Batch size must be 1."

        with torch.no_grad():
            scanned_data = self.batchitemselect(batch, 0)

        _, volumes, _, volume_dc, _, _, _, _, _, induced_pl = self(scanned_data)
        volume_gt = scanned_data["volume_gt"]

        # UNMASKED L1 (smooth-L1 over the whole volume, not restricted to brain).
        img_l = img_loss_l1(volumes, volume_gt, mask=None)
        vol_l1 = sum(img_l)

        # Perceptual loss on the final reg output.
        if self.perceptual_loss is not None:
            perc_loss = self.perceptual_loss.forward(volumes[-1], volume_gt)
        else:
            perc_loss = torch.tensor(0.0, device=self.device)

        total_loss = self.weight_img * vol_l1 + self.weight_perceptual * perc_loss

        # Volumes for the comparison viz: only materialized outside training.
        outputs = {}
        if (not self.training) and batch_idx == 1:
            outputs = {
                "volume_gt": volume_gt,
                "volumes": volumes,
                "volume_dc": volume_dc,
            }

        return {
            "total_loss": total_loss,
            "l1_w": (self.weight_img * vol_l1).detach(),
            "perc_w": (self.weight_perceptual * perc_loss).detach(),
            "point_induced": induced_pl,
            "outputs": outputs,
        }

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        results = self._shared_step(batch, batch_idx)
        total_loss = results["total_loss"]
        self.log_dict(
            {
                "loss_total/train": total_loss,
                "loss_imgl1/train": results["l1_w"],
                "loss_percept/train": results["perc_w"],
                "point_induced/train": results["point_induced"],
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        results = self._shared_step(batch, batch_idx)
        total_loss = results["total_loss"]
        out = results["outputs"]

        self.log_dict(
            {
                "loss_total/val": total_loss,
                "loss_imgl1/val": results["l1_w"],
                "loss_percept/val": results["perc_w"],
                "point_induced/val": results["point_induced"],
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # Volume comparison: GT vs reg output vs pre-reg DC vs best-PSF reference.
        if batch_idx == 1 and out.get("volume_gt") is not None:
            gt_one = out["volume_gt"][0].detach().float().cpu()
            recon_one = out["volumes"][-1][0].detach().float().cpu()
            dc_one = out["volume_dc"][0].detach().float().cpu()
            best_psf_volume = self.recon_gt_psf_volume(batch, batch_idx)
            fig = make_volume_comparison_image(
                gt_one,
                recon_one,
                dc_one,
                best_psf_volume,
                f"epoch={self.current_epoch}_step={self.global_step}_example={batch_idx}",
            )
            self.val_example_figures.append(fig)

        return total_loss
