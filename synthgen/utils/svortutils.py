import torch
import torch.nn.functional as F
from typing import List, Dict
import os

from synthgen.generator.transform import (
    RigidTransform,
    mat2point,
)

from synthgen.utils.io import save_volume
from synthgen.generator.slice_acquisition import slice_acquisition
from monai.losses import LocalNormalizedCrossCorrelationLoss


# --- D. Scheduler (Derived from src/train.py) ---
class WarmupLinearSchedule(torch.optim.lr_scheduler.LambdaLR):
    """Linear warmup followed by linear decay."""

    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super().__init__(optimizer, self._lr_lambda, last_epoch=last_epoch)  # [12]

    def _lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))  # [12]
        return max(
            0.0,
            float(self.t_total - step)
            / float(max(1.0, self.t_total - self.warmup_steps)),
        )  # [12]


class WarmupConstantSchedule(torch.optim.lr_scheduler.LambdaLR):
    """Linear warmup over ``warmup_steps``, then a constant learning rate.

    Chosen for a continual fine-tuning workflow (resuming from arbitrary epochs,
    adding components such as a regularizer mid-training): unlike a linear/cosine
    decay, the LR never falls to 0 and does not depend on a fixed training
    horizon, so continuing from any checkpoint keeps a stable, sensible LR.
    Resuming past ``warmup_steps`` lands directly on the constant LR with no
    spurious re-warmup (Lightning restores ``last_epoch`` from the checkpoint).
    To drop the LR for a fine-tuning phase, just lower ``lr`` in the config.
    """

    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, self._lr_lambda, last_epoch=last_epoch)

    def _lr_lambda(self, step):
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return 1.0


# def img_loss(volumes, volume_gt, beta=0.01):
#     """L1 Loss on estimated volume x̂k against Ground Truth volume x."""
#     loss_img = [
#         F.smooth_l1_loss(v, volume_gt, beta=beta, reduction="mean") for v in volumes
#     ]  # [9]
#     return loss_img


def img_loss_masked(
    volumes,
    volume_gt,
    mask=None,
    beta=0.01,
):
    """
    Computes Smooth L1 loss between predicted volumes and GT,
    optionally applying a voxel-wise mask.

    Args:
        volumes (list[Tensor]): list of predicted 3D volumes.
        volume_gt (Tensor): ground-truth volume.
        beta (float): Smooth L1 loss beta parameter.
        mask (Tensor or list[Tensor] or None): voxel-wise mask(s) with 0/1 values.

    Returns:
        list[Tensor]: list of masked losses.
    """

    # If no mask given → same behaviour as before
    if mask is None:
        return [
            F.smooth_l1_loss(v, volume_gt, beta=beta, reduction="mean") for v in volumes
        ]

    losses = []

    # If one mask is provided, use it for all volumes
    if isinstance(mask, torch.Tensor):
        mask_list = [mask] * len(volumes)
    else:
        mask_list = mask  # assume list of masks

    for v, m in zip(volumes, mask_list):
        # Ensure mask is float
        m = m.float()

        # Compute elementwise smooth L1 loss (without reduction)
        elem_loss = F.smooth_l1_loss(v, volume_gt, beta=beta, reduction="none")

        # Apply mask
        masked_loss = elem_loss * m

        # Normalize only over masked elements
        loss = masked_loss.sum() / (m.sum() + 1e-8)
        losses.append(loss)

    return losses


def img_loss_l1(volumes, volume_gt, mask=None, beta=0.1, bg_weight=0.1):
    """Masked smooth-L1 loss on estimated volume x̂k against Ground Truth x.

    When a mask is given the loss is a proper masked mean: the error is averaged
    over the in-brain voxels (normalized by the in-brain voxel count) so the
    magnitude is independent of brain size and volume size. A small background
    term (averaged over its own voxel count, scaled by ``bg_weight``) keeps the
    reconstruction near zero outside the brain.
    """
    if mask is not None:
        brain = (mask > 0).float()
        bg = 1.0 - brain
        brain_n = brain.sum() + 1e-6
        bg_n = bg.sum() + 1e-6
        loss_img = []
        for v in volumes:
            elem = F.smooth_l1_loss(v, volume_gt, beta=beta, reduction="none")
            loss = (elem * brain).sum() / brain_n
            loss = loss + bg_weight * (elem * bg).sum() / bg_n
            loss_img.append(loss)
    else:
        loss_img = [
            F.smooth_l1_loss(
                v,
                volume_gt,
                beta=beta,
            )
            for v in volumes
        ]
    return loss_img


def img_loss_l2(volumes, volume_gt, beta=0.01):
    """L2 Loss on estimated volume x̂k against Ground Truth volume x."""
    loss_img = [F.mse_loss(v, volume_gt, reduction="mean") for v in volumes]
    return loss_img


def trans_loss(transforms: List[RigidTransform], transforms_gt: RigidTransform):
    """L2 Loss on residual error in axis-angle space (R and T components)."""
    # Calculate T_err = T_gt * T_pred^-1 (transforms_gt.inv().compose(t)) [9]
    transfroms_err = [transforms_gt.inv().compose(t) for t in transforms]
    err = [
        t_err.axisangle() for t_err in transfroms_err
    ]  # Get 6D axis-angle/translation error
    loss_R = [torch.mean(e[:, :3] ** 2) for e in err]  # Rotation error squared mean [9]
    loss_T = [
        torch.mean(e[:, 3:] ** 2) for e in err
    ]  # Translation error squared mean [9]
    return loss_R, loss_T


def point_loss(ps, transforms_gt_mat, sx, sy, rs):
    """L2 Loss on predicted anchor points P̂k against Ground Truth points P."""
    # ps is list of predicted anchor points (N_slices, 9)
    # P_gt is calculated from transforms_gt (3 anchor points x 3 coords = 9)
    p_gt = mat2point(transforms_gt_mat, sx, sy, rs)  # [10]
    loss_p = [F.mse_loss(p, p_gt) for p in ps]  # L2 loss [11]
    return loss_p


def save_reconstructed_volumes(
    data: Dict[str, torch.Tensor],
    volumes: List[torch.Tensor],
    output_dir: str,
    prefix: str,
):
    """
    Saves the final reconstructed volume (estimated volume x̂K) and the
    Ground Truth volume (x) using the resolution metadata for correct affines.

    Args:
        data: The dictionary output from scanner.scan(), containing 'volume_gt' and 'resolution_recon'.
        volumes: List of estimated volume tensors (x̂1, ..., x̂K), where x̂K is the best result.
        output_dir: Directory where volumes should be saved.
        prefix: Unique string prefix for filenames (e.g., 'epoch_05', 'step_1000').
    """

    # 1. Extract necessary data

    # The estimated volume x̂K is the last element in the list [4]
    reconstructed_volume = volumes[-1]
    # undo the permutation
    reconstructed_volume = reconstructed_volume.permute(0, 1, 4, 3, 2)

    # The Ground Truth volume x is generated by the scanner [3]
    gt_volume = data["volume_gt"]
    gt_volume = gt_volume.permute(0, 1, 4, 3, 2)
    resolution_recon = data["resolution_recon"]

    # Ensure tensors are detached and on CPU for saving (save_volume handles final CPU transfer)
    reconstructed_volume = reconstructed_volume.detach()
    gt_volume = gt_volume.detach()

    # 2. Save Ground Truth Volume
    gt_fname = os.path.join(output_dir, f"{prefix}_gt.nii.gz")
    save_volume(
        fname=gt_fname,
        data=gt_volume,
        res=resolution_recon,
        affine=data["input_affine"],
    )

    # 3. Save Reconstructed Volume (Best Estimate)
    rec_fname = os.path.join(output_dir, f"{prefix}_out.nii.gz")
    save_volume(
        fname=rec_fname,
        data=reconstructed_volume,
        res=resolution_recon,
        affine=data["input_affine"],
    )

    # The function save_volume ensures the affines are correctly calculated
    # based on the reconstruction resolution (res_r) [1].


def load_svort_weights(model, ckpt_path):
    model.regularizer = None
    ckpt = torch.load(ckpt_path, map_location="cpu")

    state_dict = ckpt["model"]
    new_state = {}
    for k, v in state_dict.items():
        # Add prefix ONLY if needed
        if (
            k.startswith("svrnet1.")
            or k.startswith("svrnet2.")
            or k.startswith("svrnet.")
        ):
            # new_state["model." + k] = v
            new_state[k] = v

        else:
            new_state[k] = v

    model.load_state_dict(new_state, strict=True)
    return model
