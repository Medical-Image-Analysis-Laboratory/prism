# from time import time

import os
import pathlib

import torch
import numpy as np
import torch.nn.functional as F

# from line_profiler import profile

from synthgen.generator.transform import (
    RigidTransform,
    mat_update_resolution,
)
from synthgen.generator.slice_acquisition import slice_acquisition
from synthgen.generator import artifacts
from synthgen.utils.scanutils import (
    get_PSF,
    init_stack_transforms,
    interleave_index,
    reset_transform,
)
from synthgen.generator.slice_acquisition.fetal_motion import sample_motion
from torchio.transforms import RandomMotion
from monai.metrics import SSIMMetric

from synthgen.utils.scanutils import (
    apply_parameterized_bias_field,
    remove_parameterized_bias_field,
    get_2d_polynomial_basis,
    percentile_bounds,
    percentile_normalize,
)


class Scanner:

    def __init__(self, cfg_scanner: dict):
        self.resolution_slice_fac = cfg_scanner["resolution_slice_fac"]
        self.resolution_slice_max = cfg_scanner["resolution_slice_max"]
        self.slice_thickness = cfg_scanner["slice_thickness"]
        self.gap = cfg_scanner["gap"]
        self.scan_segmentations = cfg_scanner.get("scan_segmentations", False)
        self.min_num_stack = cfg_scanner["min_num_stack"]
        self.max_num_stack = cfg_scanner["max_num_stack"]
        self.max_num_slices = cfg_scanner.get("max_num_slices", None)
        self.noise_sigma = cfg_scanner["noise_sigma"]
        self.TR = cfg_scanner["TR"]
        self.prob_void = cfg_scanner["prob_void"]
        self.slice_size = cfg_scanner.get("slice_size", None)
        self.resolution_recon = cfg_scanner.get("resolution_recon", None)
        self.restrict_transform = cfg_scanner.get("restrict_transform", False)
        self.bias_strength_max = cfg_scanner.get("bias_strength_max", 1.5)
        self.bias_strength_min = cfg_scanner.get("bias_strength_min", 0.1)
        self.bias_percentage_max = cfg_scanner.get("bias_percentage_max", 0.9)
        self.intensity_slice_std = cfg_scanner.get("intensity_slice_std", None)
        self.skull_strip = cfg_scanner.get("skull_strip", 0.5)
        self.intensity_slice_prb = cfg_scanner.get("intensity_slice_prb", 0.0)
        self.txy = cfg_scanner.get("txy", 0)
        self.device = "cuda:0"  # cfg_scanner.get("device", torch.device("cpu"))
        self.prob_motion = cfg_scanner.get("prob_motion", 0.0)
        # New structure-destroying artifacts (see synthgen.generator.artifacts).
        # Each is a per-subject MAX probability sampled in [0, max] in scan().
        self.prob_gray_void = cfg_scanner.get("prob_gray_void", 0.0)
        self.prob_local_blur = cfg_scanner.get("prob_local_blur", 0.0)
        self.prob_local_noise = cfg_scanner.get("prob_local_noise", 0.0)
        self.prob_slice_blur = cfg_scanner.get("prob_slice_blur", 0.0)
        self.prob_heavy_noise = cfg_scanner.get("prob_heavy_noise", 0.0)
        self.max_missing_slices = cfg_scanner.get("max_missing_slices", 0)
        # The actual missing-slice fraction is sampled per subject in scan()
        # (uniform in [0, max_missing_slices]); only validate the configured max.
        if self.max_missing_slices > 0:
            assert (
                0 <= self.max_missing_slices < 1
            ), "max_missing_slices should be in the range [0, 1)"
        self.ssim_metric = SSIMMetric(2)

    def get_resolution(self, data):
        resolution = data["resolution"]
        if hasattr(self.resolution_slice_fac, "__len__"):
            resolution_slice = np.random.uniform(
                self.resolution_slice_fac[0] * resolution,
                min(
                    self.resolution_slice_fac[-1] * resolution,
                    self.resolution_slice_max,
                ),
            )
        else:
            resolution_slice = self.resolution_slice_fac * resolution
        if self.resolution_recon is not None:
            data["resolution_recon"] = self.resolution_recon
        else:
            data["resolution_recon"] = np.random.uniform(resolution, resolution_slice)
        data["resolution_slice"] = resolution_slice

        if hasattr(self.slice_thickness, "__len__"):
            data["slice_thickness"] = np.random.uniform(
                self.slice_thickness[0], self.slice_thickness[-1]
            )
        else:
            data["slice_thickness"] = self.slice_thickness
        if self.gap is None:
            data["gap"] = data["slice_thickness"]
        elif hasattr(self.gap, "__len__"):
            data["gap"] = np.random.uniform(
                self.gap[0], min(self.gap[-1], data["slice_thickness"])
            )
        else:
            data["gap"] = self.gap
        return data

    def add_noise(self, slices, threshold, sigma=None):
        # `sigma` is the per-subject noise level (sampled in [0, max] by scan()).
        # Fall back to the legacy per-call uniform sample only when not provided.
        if sigma is None:
            if (not hasattr(self.noise_sigma, "__len__")) and self.noise_sigma == 0:
                return slices
            sigma = np.random.uniform(self.noise_sigma[0], self.noise_sigma[-1])
        if sigma <= 0:
            return slices
        mask = slices > threshold
        masked = slices[mask]
        noise1 = torch.randn_like(masked) * sigma
        noise2 = torch.randn_like(masked) * sigma
        slices[mask] = torch.sqrt((masked + noise1) ** 2 + noise2**2)
        return slices

    def signal_void(self, slices, prob=None, return_corrupted=False):
        # `prob` is the per-subject void probability (sampled in [0, prob_void]).
        # `idx` (which slices got a void) is the per-slice corruption mask returned
        # when `return_corrupted`, so the scanner can keep clean slices at quality 1.
        if prob is None:
            prob = self.prob_void
        idx = torch.rand(slices.shape[0], device=slices.device) < prob
        n = idx.sum()
        if n > 0:
            h, w = slices.shape[-2:]
            y = torch.linspace(-(h - 1) / 2, (h - 1) / 2, h, device=slices.device)
            x = torch.linspace(-(w - 1) / 2, (w - 1) / 2, w, device=slices.device)
            yc = (torch.rand(n, device=slices.device) - 0.5) * (h - 1)
            xc = (torch.rand(n, device=slices.device) - 0.5) * (w - 1)

            y = y.view(1, -1, 1) - yc.view(-1, 1, 1)
            x = x.view(1, 1, -1) - xc.view(-1, 1, 1)

            theta = 2 * np.pi * torch.rand((n, 1, 1), device=slices.device)
            c = torch.cos(theta)
            s = torch.sin(theta)
            x, y = c * x - s * y, s * x + c * y

            a = 10 + torch.rand_like(theta) * 20  # ellipse area, physical size
            A = torch.rand_like(theta) * 0.5 + 0.5  # max signal drop
            sx = torch.rand_like(theta) * 15 + 1
            sy = a**2 / sx

            sx = -0.5 / sx**2
            sy = -0.5 / sy**2

            mask = 1 - A * torch.exp(sx * x**2 + sy * y**2)
            slices[idx, 0] *= mask
        return (slices, idx) if return_corrupted else slices

    def sample_time(self, n_slice):
        TR = np.random.uniform(self.TR[0], self.TR[1])
        return np.arange(n_slice) * TR

    def add_motion(self, slices, prob=None, return_corrupted=False):
        # `prob` is the per-subject motion probability (sampled in [0, prob_motion]).
        # We sample the per-slice selection (`sel`) OURSELVES and apply RandomMotion
        # with p=1 only to the selected slices, so we know exactly which slices were
        # corrupted (returned when `return_corrupted`) instead of relying on
        # torchio's hidden per-call coin flip.
        if prob is None:
            prob = self.prob_motion
        device = slices.device
        n = slices.shape[0]
        sel = torch.rand(n) < prob
        random_motion = RandomMotion(
            degrees=30,
            translation=30,
            num_transforms=3,
            image_interpolation="linear",
            p=1.0,
        )
        slices = slices.cpu()
        motion_slice = []
        for i, slc in enumerate(slices):
            slc = slc.unsqueeze(0)  # add batch and channel dims
            if bool(sel[i]):
                slc = random_motion(slc)
            motion_slice.append(slc.squeeze(0).squeeze(0))
        motion_slice = torch.stack(motion_slice, 0).to("cuda:0").unsqueeze(1)
        return (motion_slice, sel.to(device)) if return_corrupted else motion_slice

    def add_slice_bias(self, slices, percentage=None):
        """_summary_

        Args:
            slices (torch.Tensor): Tensor of shape (N_Slices, H, W)
            percentage (float | None): per-subject fraction of slices to bias
                (sampled in [0, bias_percentage_max] by scan()). If None, the
                fraction is sampled per-call inside apply_parameterized_bias_field.
        """
        slices = slices.squeeze(1)  # add channel dim
        assert (
            slices.ndim == 3
        ), f"slices must be a 3D tensor (N_Slices, H, W). Found {slices.ndim}D: {slices.shape}."

        N, H, W = slices.shape
        # 2. Setup the basis
        # Order 2 is standard for slice-by-slice operations
        basis = get_2d_polynomial_basis(H, W, order=2, device=slices.device)

        # 3. Apply the bias field (Simulation / Training Forward Pass)
        corrupted_stack, gt_coeffs = apply_parameterized_bias_field(
            slices,
            basis,
            strength_max=self.bias_strength_max,
            strength_min=self.bias_strength_min,
            percentage_max=self.bias_percentage_max,
            percentage=percentage,
        )
        corrupted_stack = corrupted_stack.unsqueeze(
            1
        )  # add channel dim back for consistency
        return corrupted_stack, gt_coeffs, basis

    def add_slice_intensity_shift(self, slices, prb=None):
        """Apply random intensity shift to a subset of slices.
        Args:
            slices (torch.Tensor): Tensor of shape (N_Slices, 1, H, W) with values in [0, 1]
        Returns:
            transformed_slices (torch.Tensor): Tensor of same shape as input with intensity shift applied to a subset of slices.
            intshifts (torch.Tensor): Tensor of shape (N_Slices,) containing the intensity shift values applied to each slice (0.0 for unchanged slices).
        """
        # `prb` is the per-subject shift probability (sampled in [0, intensity_slice_prb]).
        if prb is None:
            prb = self.intensity_slice_prb
        if self.intensity_slice_std is None or prb == 0:
            return slices, torch.ones(
                slices.shape[0], device=slices.device
            )  # no intensity shift applied
        idx = torch.rand(slices.shape[0], device=slices.device) < prb
        intshifts = torch.ones(slices.shape[0], device=slices.device)

        # sample intensity shifts for the selected slices 1 ± intensity_slice_std
        if idx.sum() > 0:
            intshifts[idx] = (
                1
                + (2 * torch.rand(idx.sum(), device=slices.device) - 1)
                * self.intensity_slice_std
            )

            slices[idx] = slices[idx] * intshifts[idx].view(-1, 1, 1, 1)
        return slices, intshifts

    @staticmethod
    def dilate_mask(mask, kernel_size=3):
        """
        Dilate a binary mask tensor.

        Args:
            mask: (H, W) or (B, H, W) or (B, 1, H, W) bool/float tensor
            kernel_size: dilation amount (must be odd)

        Returns:
            Dilated mask of same shape
        """
        original_shape = mask.shape

        # Ensure float and 4D: (B, 1, H, W)
        x = mask.float()
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)

        padding = kernel_size // 2
        dilated = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)

        # Restore original shape and dtype
        dilated = dilated.view(original_shape)

        return dilated.bool() if mask.dtype == torch.bool else dilated

    @staticmethod
    def dilate_mask3d(mask, kernel_size=3):
        """
        Dilate a 3D volume mask with a cubic structuring element.

        Args:
            mask: (..., D, H, W) bool/float tensor (e.g. 5D (B, 1, D, H, W))
            kernel_size: dilation amount (must be odd)

        Returns:
            Dilated mask of the same shape and dtype as the input.
        """
        original_dtype = mask.dtype
        x = mask.float()
        n_unsqueeze = 0
        while x.dim() < 5:
            x = x.unsqueeze(0)
            n_unsqueeze += 1
        padding = kernel_size // 2
        dilated = F.max_pool3d(x, kernel_size=kernel_size, stride=1, padding=padding)
        for _ in range(n_unsqueeze):
            dilated = dilated.squeeze(0)
        return dilated.bool() if original_dtype == torch.bool else dilated

    def scan(self, data):
        """
        Simulate the acquisition of slices from the volume.

        Args:
            data: Dictionary containing the data. Expects keys:
                - "volume": The high-resolution 3D volume (torch.Tensor).
                - "resolution": The isotropic resolution of the volume (float).
                - "mask": The mask of the volume (torch.Tensor).
                - "threshold": Threshold value for selecting voxels for noise addition (background) (float).

        Returns:
            dict: The updated data dictionary with simulated acquired slices.
        """
        data = self.get_resolution(data)
        res = data["resolution"]
        res_r = data["resolution_recon"]
        res_s = data["resolution_slice"]
        s_thick = data["slice_thickness"]
        gap = data["gap"]
        device = data["volume"].device

        # flag to check if segmentation is provided
        segmcompute = True if "label" in data.keys() else False

        # compute stack affine matrix
        stack_affine = torch.eye(4, device=device)
        stack_affine[0, 0] = res_s
        stack_affine[1, 1] = res_s
        stack_affine[2, 2] = s_thick
        stack_affine[0, 3] = -res_s / 2
        stack_affine[1, 3] = -res_s / 2
        stack_affine[2, 3] = -s_thick / 2
        data["stack_affine"] = stack_affine

        if res_r != res:
            grids = []
            for i in range(3):
                size_new = int(data["volume"].shape[i + 2] * res / res_r)
                grid_max = (
                    (size_new - 1) * res_r / (data["volume"].shape[i + 2] - 1) / res
                )
                grids.append(
                    torch.linspace(-grid_max, grid_max, size_new, device=device)
                )
            grid = torch.stack(
                torch.meshgrid(*grids, indexing="ij")[::-1], -1
            ).unsqueeze_(0)
            volume_gt = F.grid_sample(data["volume"], grid, align_corners=True)
            # compute segmentation as well if provided
            if segmcompute:
                seg_gt = F.grid_sample(
                    data["label"], grid, mode="nearest", align_corners=True
                )
        else:
            # compute segmentation as well if provided
            volume_gt = data["volume"].clone()
            if segmcompute:
                seg_gt = data["label"].clone()

        # store segmentation as well if provided
        if segmcompute:
            data["seg_gt"] = seg_gt

        # --- Up-front skull stripping shared by the GT volume and the stacks ---
        # Decide ONCE whether to skull-strip and the dilation kernel, build the
        # brain mask from the segmentation (random dilation, same policy/params as
        # the old per-stack skull strip), and apply it to BOTH the high-res volume
        # that gets sliced AND volume_gt. The stacks are then sliced from this
        # masked volume and their per-slice masks come from slicing the same brain
        # mask, so GT and stacks cover an identical region. When skull stripping
        # does not trigger we fall back to the intensity support of the volume.
        do_strip = (
            segmcompute
            and self.skull_strip is not None
            and torch.rand(1).item() < self.skull_strip
        )

        if do_strip:
            # Skull-strip: brain mask = randomly dilated segmentation support. The
            # random kernel varies how tightly the brain is cropped between subjects.
            kernel_size = int(np.random.choice([3, 5, 7, 9]))
            brain_mask_hr = self.dilate_mask3d(data["label"] > 0, kernel_size).float()
        else:
            # No skull strip: keep the whole head, i.e. the volume's intensity
            # support. The per-slice masks (sliced from brain_mask_hr) and mask_gt
            # then cover this same region, so stacks and volume_gt stay consistent.
            # blur the volume and then used it to get a mask
            kernel_size = 9
            brain_mask_dilated = self.dilate_mask3d(
                data["label"] > 0, kernel_size
            ).float()
            brain_mask_hr = (
                (data["volume"] > data["threshold"]) | brain_mask_dilated.bool()
            ).float()

        # skull-strip the volume that gets sliced and the GT volume
        data["volume"] = data["volume"] * brain_mask_hr
        volume_gt = volume_gt * brain_mask_hr

        data["mask_gt"] = brain_mask_hr > 0
        data["volume_gt"] = volume_gt

        psf_acq = get_PSF(
            res_ratio=(res_s / res, res_s / res, s_thick / res), device=device
        )
        psf_rec = get_PSF(
            res_ratio=(res_s / res_r, res_s / res_r, s_thick / res_r), device=device
        )
        data["psf_rec"] = psf_rec
        data["psf_acq"] = psf_acq

        vs = data["volume"].shape
        if self.slice_size is None:
            ss = int(
                np.sqrt((vs[-1] ** 2 + vs[-2] ** 2 + vs[-3] ** 2) / 2.0) * res / res_s
            )
            ss = int(np.ceil(ss / 32.0) * 32)
        else:
            ss = self.slice_size
        ns = int(max(vs) * res / gap) + 2
        bias_basis = None
        stacks = []
        stack_masks = []
        stacks_clean = []
        stacksegms = []
        transforms = []
        transforms_gt = []
        positions = []
        bias_coefs = []
        gt_stacks_bias_field = []
        intshifts = []
        # per-slice bool: True if any per-slice artifact (void/blur/gray/heavy-noise/
        # motion) touched the slice. Slices with only the mild global acquisition
        # noise stay False -> their quality GT is forced to 1 (see end of scan()).
        slices_corrupted = []

        # --- Per-subject artifact severities --------------------------------
        # Each artifact's controlling probability / strength is sampled uniformly
        # in [0, configured_max] ONCE per subject (the config value is the MAX, not
        # a fixed level). This lets a fraction of subjects come out (near) clean, so
        # the model also sees good-quality data instead of every subject getting
        # the full configured corruption. Applied to ALL stacks of this subject.
        def _u(maxv):
            m = 0.0 if maxv is None else float(maxv)
            return float(torch.rand(1).item()) * m

        noise_max = (
            self.noise_sigma[-1]
            if hasattr(self.noise_sigma, "__len__")
            else self.noise_sigma
        )
        subj_noise_sigma = _u(noise_max)
        subj_prob_void = _u(self.prob_void)
        subj_prob_motion = _u(self.prob_motion)
        subj_prob_gray_void = _u(self.prob_gray_void)
        subj_prob_local_blur = _u(self.prob_local_blur)
        subj_prob_local_noise = _u(self.prob_local_noise)
        subj_prob_slice_blur = _u(self.prob_slice_blur)
        subj_prob_heavy_noise = _u(self.prob_heavy_noise)
        subj_bias_pct = _u(self.bias_percentage_max)
        subj_int_prb = _u(self.intensity_slice_prb)
        subj_missing = _u(self.max_missing_slices)

        max_num_stack = np.random.randint(self.min_num_stack, self.max_num_stack + 1)
        while True:
            # stack transformation
            transform_init = init_stack_transforms(
                ns, gap, self.restrict_transform, self.txy, device
            )
            ts = self.sample_time(ns)
            transform_motion = sample_motion(ts, device)
            # interleaved acquisition
            interleave_idx = interleave_index(
                ns, np.random.randint(2, int(np.sqrt(ns)) + 1)
            )
            transform_motion = transform_motion[interleave_idx]
            # apply motion
            transform_target = transform_motion.compose(transform_init)
            # sample slices
            mat = mat_update_resolution(transform_target.matrix(), 1, res)
            slices = slice_acquisition(
                mat,
                data["volume"],
                None,
                None,
                psf_acq,
                (ss, ss),
                res_s / res,
                False,
                False,
            )
            # slice the GT brain mask (sharp PSF) so the per-slice mask matches the
            # volume's skull strip exactly -> stacks and volume_gt share one region
            slices_brain = slice_acquisition(
                mat,
                brain_mask_hr,
                None,
                None,
                get_PSF(0, device=device),
                (ss, ss),
                res_s / res,
                False,
                False,
            )

            # segmentation transformation if provided
            if segmcompute and self.scan_segmentations:
                labs = torch.unique(data["label"])
                labs2indx = {l: i for i, l in enumerate(labs)}
                seg1h = torch.stack([data["label"] == l for l in labs], 0).float()
                stacks_segm = []
                for lab1h in range(len(labs)):
                    seg = slice_acquisition(
                        mat,
                        seg1h[lab1h],
                        None,
                        None,
                        get_PSF(0, device=device),
                        (ss, ss),
                        res_s / res,
                        False,
                        False,
                    )
                    stacks_segm.append(seg)
                stacks_segm = torch.stack(tensors=stacks_segm, dim=0)
                stacks_segm = torch.argmax(stacks_segm, 0)

                # remap idx back to original labels
                stacks_segm_corr = torch.zeros_like(stacks_segm, device=device)
                for lab, idx in labs2indx.items():
                    stacks_segm_corr[stacks_segm == idx] = lab.long()
                stacks_segm = stacks_segm_corr
            # segmentation transformation end

            idx = torch.ones(slices.shape[0], dtype=torch.bool, device=device)
            # keep only slices with some signal to make the task more realistic
            # (in real data empty slices are not included in the stacks)

            max_vals = slices.view(slices.shape[0], -1).max(dim=1)[0]
            idx = max_vals >= 0.01

            nz = torch.nonzero(idx)
            idx[nz[0, 0] : nz[-1, 0]] = True
            slices = slices[idx]
            slices_brain = slices_brain[idx]
            if segmcompute and self.scan_segmentations:
                stacks_segm = stacks_segm[idx]
            transform_init = transform_init[idx]
            transform_init = reset_transform(transform_init)
            transform_target = transform_target[idx]

            # per-slice brain mask = sliced GT brain mask (consistent with volume_gt)
            slices_mask = slices_brain > 0.5
            # add bias field and intensity shifts before clean stack saving
            # as not considered an artifact per say
            slices, stack_int_shifts = self.add_slice_intensity_shift(
                slices, prb=subj_int_prb
            )
            slices, stack_bias_coefs, bias_basis = self.add_slice_bias(
                slices, percentage=subj_bias_pct
            )
            _, slices_gt_bias_field = remove_parameterized_bias_field(
                slices, stack_bias_coefs, bias_basis
            )
            # Mild global acquisition noise: applied to every slice, NOT counted as
            # a per-slice corruption (so clean-but-noisy slices keep quality 1).
            slices = self.add_noise(slices, data["threshold"], sigma=subj_noise_sigma)

            # add artifacts
            slices_clean = slices.clone()  # save clean version for supervision of sqm

            # Track which slices each per-slice artifact actually corrupts, so clean
            # slices can be set to quality 1 regardless of metric noise. Every
            # artifact returns a (N,) bool of the slices it touched; we OR them.
            stack_corrupted = torch.zeros(
                slices.shape[0], dtype=torch.bool, device=device
            )
            slices, c = self.signal_void(
                slices, prob=subj_prob_void, return_corrupted=True
            )
            stack_corrupted |= c
            if subj_prob_motion > 0:
                slices, c = self.add_motion(
                    slices, prob=subj_prob_motion, return_corrupted=True
                )
                stack_corrupted |= c
            # New structure-destroying artifacts. Applied AFTER the slices_clean
            # snapshot so they show up in the SQM quality GT below. slices_mask is
            # the per-slice brain mask; it confines the void/blur to tissue and
            # provides the local-mean support for the gray void.
            # if subj_prob_gray_void > 0:
            #     slices, c = artifacts.local_gray_void(
            #         slices,
            #         prob=subj_prob_gray_void,
            #         brain_mask=slices_mask,
            #         return_corrupted=True,
            #     )
            #     stack_corrupted |= c
            if subj_prob_local_blur > 0:
                slices, c = artifacts.local_blur(
                    slices,
                    prob=subj_prob_local_blur,
                    brain_mask=slices_mask,
                    return_corrupted=True,
                )
                stack_corrupted |= c
            # Spatially-varying noise: a blob region gets strong (optionally
            # correlated) Rician noise, the rest of the slice stays clean.
            if subj_prob_local_noise > 0:
                slices, c = artifacts.local_noise(
                    slices,
                    prob=subj_prob_local_noise,
                    threshold=data["threshold"],
                    brain_mask=slices_mask,
                    return_corrupted=True,
                )
                stack_corrupted |= c
            # if subj_prob_slice_blur > 0:
            #     slices, c = artifacts.slice_blur(
            #         slices, prob=subj_prob_slice_blur, return_corrupted=True
            #     )
            #     stack_corrupted |= c
            # if subj_prob_heavy_noise > 0:
            #     slices, c = artifacts.heavy_noise(
            #         slices,
            #         prob=subj_prob_heavy_noise,
            #         threshold=data["threshold"],
            #         return_corrupted=True,
            #     )
            #     stack_corrupted |= c

            # re-apply the brain mask to keep outside-brain at 0 after artifacts.
            # The mask is the sliced GT brain mask, so stacks and volume_gt cover the
            # same region (skull stripping was applied up front to the volume).
            slices = torch.where(slices_mask, slices, torch.zeros_like(slices))
            slices_clean = torch.where(
                slices_mask, slices_clean, torch.zeros_like(slices_clean)
            )

            # quit stack generation if we already have enough slices for this scan
            if (
                self.max_num_slices is not None
                and sum(st.shape[0] for st in stacks) + slices.shape[0]
                >= self.max_num_slices
            ):
                break

            stacks.append(slices)
            stack_masks.append(slices_mask)
            bias_coefs.append(stack_bias_coefs)
            intshifts.append(stack_int_shifts)
            gt_stacks_bias_field.append(slices_gt_bias_field)
            stacks_clean.append(slices_clean)
            slices_corrupted.append(stack_corrupted)
            transforms.append(transform_init)
            if segmcompute and self.scan_segmentations:
                stacksegms.append(stacks_segm)
            transforms_gt.append(transform_target)
            positions.append(
                torch.arange(slices.shape[0], dtype=slices.dtype, device=device)
                - slices.shape[0] // 2
            )
            if len(stacks) >= max_num_stack:
                break
        # End of stack generation loop

        # add stack index
        stacks_ids = np.random.choice(20, len(stacks), replace=False)
        positions = torch.cat(
            [
                torch.stack((positions[i], torch.full_like(positions[i], s_i)), -1)
                for i, s_i in enumerate(stacks_ids)
            ],
            0,
        )
        stacks = torch.cat(stacks, 0)
        stack_masks = torch.cat(stack_masks, 0)
        gt_stacks_bias_field = torch.cat(gt_stacks_bias_field, 0)
        bias_coefs = torch.cat(bias_coefs, 0)
        intshifts = torch.cat(intshifts, 0)
        stacks_clean = torch.cat(stacks_clean, 0)
        stacks_corrupted = torch.cat(slices_corrupted, 0)  # (N,) bool
        transforms = RigidTransform.cat(transforms)
        transforms_gt = RigidTransform.cat(transforms_gt)
        if segmcompute and self.scan_segmentations:
            stacks_segm = torch.cat(stacksegms, 0)

        # Structure-aware SQM quality GT: min(local SSIM, gradient detail-retention)
        # between the clean and corrupted slices, hardened by finalize_quality's
        # sigmoid (thr=0.8, sharp=20). Unlike 1 - |corrupted - clean|, this flags blur /
        # gray voids that barely move intensity but destroy local structure. The
        # bias/intensity shifts live in BOTH stacks_clean and stacks, so they cancel
        # in the comparison and are (correctly) NOT flagged here.
        qa_soft = artifacts.quality_composite(stacks_clean, stacks)
        stacks_error_l = artifacts.finalize_quality(qa_soft)
        # Slices that NO per-slice artifact touched carry only the mild global
        # acquisition noise -> force their whole quality map to 1, so the SQM is not
        # supervised on metric false-positives over genuinely clean slices. Slices
        # that WERE corrupted keep the per-pixel map (which flags the bad region).
        stacks_error_l[~stacks_corrupted] = 1.0

        # Whole-slice rejection: a slice whose MEAN in-brain quality falls below 0.5
        # is too corrupted to be partially usable -> zero its entire quality map
        # (hard reject) instead of keeping a mostly-bad per-pixel map. The mean is
        # taken over in-brain pixels only (background is ~1 and would otherwise
        # inflate it); clean slices were just forced to 1.0, so they never trip this.
        mask_f = stack_masks.float()
        slice_area = mask_f.flatten(1).sum(1).clamp_min(1.0)  # (N,)
        mean_quality = (stacks_error_l * mask_f).flatten(1).sum(1) / slice_area
        stacks_error_l[mean_quality < 0.3] = 0.0

        # add missing slices simulation
        # subj_missing is the per-subject fraction, already sampled in
        # [0, max_missing_slices] above (so some subjects drop nothing). Build ONE
        # boolean keep-mask and apply it to every per-slice tensor, so adding a new
        # per-slice field only needs one line below (and can't silently desync).
        if subj_missing > 0:
            n_slices = stacks.shape[0]
            missing_slices_num = int(n_slices * subj_missing)
            if missing_slices_num > 0:
                keep = torch.ones(n_slices, dtype=torch.bool, device=stacks.device)
                missing_idx = torch.randperm(n_slices, device=stacks.device)[
                    :missing_slices_num
                ]
                keep[missing_idx] = False

                stacks = stacks[keep]
                stacks_clean = stacks_clean[keep]
                positions = positions[keep]
                bias_coefs = bias_coefs[keep]
                intshifts = intshifts[keep]
                gt_stacks_bias_field = gt_stacks_bias_field[keep]
                stack_masks = stack_masks[keep]
                stacks_error_l = stacks_error_l[keep]
                transforms = RigidTransform(transforms.matrix()[keep])
                transforms_gt = RigidTransform(transforms_gt.matrix()[keep])
                if segmcompute and self.scan_segmentations:
                    stacks_segm = stacks_segm[keep]

        # normalize stacks to ~[0, 1]
        stacks = F.relu(stacks)

        # mask out background in stacks to zero
        stacks = torch.where(stack_masks, stacks, torch.zeros_like(stacks))
        stacks_clean = torch.where(
            stack_masks, stacks_clean, torch.zeros_like(stacks_clean)
        )

        # # Robust 99th-percentile scaling (x / hi) over the in-brain pixels, instead of
        # # a fragile global min-max keyed on a single bright artifact voxel. This is a
        # # PURE SCALE (multiplicative=True forces lo=0): the per-slice intensity shifts
        # # and bias fields are multiplicative, so an affine (x - lo)/(hi - lo) map would
        # # turn them into per-slice affines that the model's multiplicative correction
        # # (/ int_shift, / bias_field) cannot undo -- leaving slice-to-slice streaks.
        # # The clean stacks reuse the SAME bounds so their scale stays comparable. The
        # # GT volume is scaled by ITS OWN in-brain 99th percentile, landing on the same
        # # kind of /hi scale as the stacks (matches InferenceStackDataset).
        # stack_bounds = percentile_bounds(stacks, stack_masks, multiplicative=True)
        # stacks = percentile_normalize(stacks, stack_masks, bounds=stack_bounds)
        # stacks_clean = percentile_normalize(
        #     stacks_clean, stack_masks, bounds=stack_bounds
        # )
        # volume_gt = percentile_normalize(
        #     volume_gt, data["mask_gt"], multiplicative=True
        # )

        data["volume_gt"] = volume_gt

        data["slice_shape"] = (ss, ss)
        data["volume_shape"] = volume_gt.shape[-3:]
        data["stacks"] = stacks
        data["stacks_clean"] = stacks_clean
        data["stacks_masks"] = stack_masks
        if segmcompute and self.scan_segmentations:
            # assign here (not before the missing-slices block) so segmentation
            # slices stay aligned with stacks/positions after slice dropping
            data["stacks_segm"] = stacks_segm
        data["positions"] = positions
        data["transforms"] = transforms.matrix()
        data["transforms_gt"] = transforms_gt.matrix()
        data["bias_coefs"] = bias_coefs
        data["gt_bias_field"] = gt_stacks_bias_field
        data["int_shifts"] = intshifts
        data["bias_basis"] = bias_basis
        data["stacks_error_l"] = stacks_error_l
        data.pop("volume")

        return data


def main():
    # test scanner on a generated batch
    import torchio as tio
    from torchio.transforms import Compose
    from synthgen.data.datamodules import DataModule
    from synthgen.generator.generator import SynthGen
    from synthgen.generator.intensity import ImageFromSeeds
    import nibabel as nib
    from torchio.transforms import (
        RandomAffineElasticDeformation,
        RandomAnisotropy,
        RandomBiasField,
        RandomGamma,
        RandomNoise,
        RescaleIntensity,
        RandomBlur,
    )

    def print_sample(sample):
        print(f"Sample type: {type(sample)}")
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                print(
                    f"  {k}: shape: {v.shape}: dtype {v.dtype} min: {v.min()} max: {v.max()} device: {v.device}"
                )
            elif isinstance(v, tio.data.image.ScalarImage) or isinstance(
                v, tio.data.image.LabelMap
            ):
                print(
                    f"  {k}: type:{type(v)} shape: {v['data'].shape}: dtype {v['data'].dtype} min: {v['data'].min()} max: {v['data'].max()} device: {v['data'].device}"
                )
            else:
                print(f"  {k}: {type(v)} value: {v}")

    blurtransf = RandomBlur(
        std=(0.4, 0.8),
        p=1,
    )
    spatial_deform = RandomAffineElasticDeformation(
        p=0.9,
        affine_first=False,
        affine_kwargs={
            "scales": 0.1,
            "degrees": 20,
            "translation": 5,
            "isotropic": False,
        },
        elastic_kwargs={
            "num_control_points": 7,
            "max_displacement": 8,
        },
    )
    resampletransf = RandomAnisotropy(
        axes=(0, 1, 2),
        downsampling=(1.5, 5),
        scalars_only=True,
        p=1.0,
    )

    gammatransf = RandomGamma(
        log_gamma=(-0.5, 0.5),
        p=0.8,
    )

    biasfield = RandomBiasField(coefficients=0.8, order=1, p=1)

    noistransf = RandomNoise(
        mean=1.0,
        std=(0, 0.25),
        p=0.2,
    )

    generator = SynthGen(
        shape=(128, 128, 128),
        resolution=(1, 1, 1),
        intensity_generator=ImageFromSeeds(
            min_subclusters=1,
            max_subclusters=6,
            seed_labels=list(range(0, 90)),
            generation_classes=list(range(0, 90)),
            meta_labels=list(range(1, 5)),
            empty_background_prob=1,  # 50% chance to have empty background
            skullstripprob=0.0,  # 90% chance to have skull
        ),
        # image_transforms=None,
        image_transforms=Compose(
            [
                # blurtransf,
                # spatial_deform,
                # resampletransf,
                gammatransf,
                biasfield,
                noistransf,
                RescaleIntensity((0, 1)),
            ]
        ),
    )
    scanner = Scanner(
        {
            "slice_size": 128,
            "resolution_slice_fac": 1,
            "resolution_slice_max": 2.0,
            "slice_thickness": [2.0, 4.0],
            "gap": [2.0, 4.0],
            "min_num_stack": 3,
            "max_num_stack": 5,
            "noise_sigma": [0.0, 0.05],
            "TR": [1.5, 2.5],
            "prob_void": 0.3,
            "device": "cuda:0",
        }
    )

    # Smoke test on the small demo BIDS folder shipped with the repo
    # (override with PRISM_DEMO_BIDS=/path/to/bids).
    demo_bids = os.environ.get("PRISM_DEMO_BIDS", "data")
    dm = DataModule(
        split_file=os.path.join(demo_bids, "participants.csv"),
        bids_path=demo_bids,
        seed_path=None,
        train_type="real",
        train_split="train",
        val_type="real",
        val_split="val",
        test_split="test",
        generator=generator,
        transforms=None,
        num_workers=0,
        batch_size=1,
        scanner=scanner,
    )

    # The dataset simulates the stacks: it unwraps the torchio Subject and runs
    # `scanner.scan` on it (see SynthDataset.sample).
    traindl = dm.train_dataloader()
    trainbatch = next(iter(traindl))
    procesed_data = {
        k: (v[0] if isinstance(v, torch.Tensor) and v.ndim > 0 else v)
        for k, v in trainbatch.items()
    }
    print_sample(procesed_data)

    from synthgen.utils.io import save_stack

    out_dir = pathlib.Path(os.environ.get("PRISM_DEMO_OUT", "simulated_stacks"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stacks_ids = procesed_data["positions"].cpu().numpy()[:, 1]
    for i, s_id in enumerate(np.unique(stacks_ids)):
        idx = stacks_ids == s_id
        save_stack(
            procesed_data["stacks"][idx],
            RigidTransform(procesed_data["transforms"][idx]),
            procesed_data["resolution_slice"],
            procesed_data["slice_thickness"],
            scale=1.0,
            dtype=np.float32,
            fname=str(out_dir / f"simulated_stack_{int(s_id)}.nii.gz"),
        )


if __name__ == "__main__":
    main()
