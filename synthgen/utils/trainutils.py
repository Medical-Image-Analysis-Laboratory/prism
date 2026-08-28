import io
import wandb
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
import torch.nn as nn


class MaskedSobelGradientLoss3D(nn.Module):
    """
    Computes the L1 difference between the 3D Sobel gradients.
    Supports an optional binary or soft mask to ignore background regions.
    """

    def __init__(
        self,
        blur: bool = True,
        weights: tuple[int] = (1.0, 1.0, 1.0),
        sigma=1.0,
        kernel_size=5,
    ):
        super(MaskedSobelGradientLoss3D, self).__init__()
        self.weights = weights  # (Z_weight, Y_weight, X_weight)
        self.blur = blur
        # --- Kernel Construction (Same as before) ---
        smooth = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float32)
        diff = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)

        k_z = diff.view(3, 1, 1) * smooth.view(1, 3, 1) * smooth.view(1, 1, 3)
        k_y = smooth.view(3, 1, 1) * diff.view(1, 3, 1) * smooth.view(1, 1, 3)
        k_x = smooth.view(3, 1, 1) * smooth.view(1, 3, 1) * diff.view(1, 1, 3)

        self.register_buffer("k_z", k_z.view(1, 1, 3, 3, 3))
        self.register_buffer("k_y", k_y.view(1, 1, 3, 3, 3))
        self.register_buffer("k_x", k_x.view(1, 1, 3, 3, 3))

        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        self.gaussian_kernel = self._create_gaussian_kernel(sigma, kernel_size)

    def _create_gaussian_kernel(self, sigma, kernel_size):
        # 1. Create a 1D coordinate vector: [-2, -1, 0, 1, 2]
        coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2

        # 2. Compute 1D Gaussian values
        k_1d = torch.exp(-(coords**2) / (2 * sigma**2))
        k_1d = k_1d / k_1d.sum()  # Normalize sum to 1.0

        # 3. Create 3D kernel using outer products (einsum is cleanest for this)
        # 'i' is Z, 'j' is Y, 'k' is X
        k_3d = torch.einsum("i,j,k->ijk", k_1d, k_1d, k_1d)

        # 4. Reshape for Conv3d: (Out_Channels, In_Channels, D, H, W)
        return k_3d.view(1, 1, kernel_size, kernel_size, kernel_size)

    def forward(self, pred, target, mask=None):
        if self.blur:
            # Ensure kernel is on the correct device
            if self.gaussian_kernel.device != pred.device:
                self.gaussian_kernel = self.gaussian_kernel.to(pred.device).type_as(
                    pred
                )

            # 1. Pre-smooth both prediction and target
            # We use padding = kernel_size // 2 to maintain the same spatial dimensions
            pad = self.gaussian_kernel.shape[-1] // 2

            pred = F.conv3d(pred, self.gaussian_kernel, padding=pad)
            target = F.conv3d(target, self.gaussian_kernel, padding=pad)

        # 1. Calculate Sobel Gradients (Directional/Signed)
        pred_gz = F.conv3d(pred, self.k_z, padding=1)
        targ_gz = F.conv3d(target, self.k_z, padding=1)

        pred_gy = F.conv3d(pred, self.k_y, padding=1)
        targ_gy = F.conv3d(target, self.k_y, padding=1)

        pred_gx = F.conv3d(pred, self.k_x, padding=1)
        targ_gx = F.conv3d(target, self.k_x, padding=1)

        # 2. Calculate Element-wise L1 Error (Absolute Difference)
        #    We do NOT reduce to a single number yet.
        diff_z = torch.abs(pred_gz - targ_gz)
        diff_y = torch.abs(pred_gy - targ_gy)
        diff_x = torch.abs(pred_gx - targ_gx)

        # 3. Apply Masking
        if mask is not None:
            # Check mask shape
            if mask.shape != pred.shape:
                # Try to broadcast if mask is (B, 1, D, H, W) and pred is (B, C, D, H, W)
                mask = mask.expand_as(pred)

            # Weight the errors by the mask
            diff_z = diff_z * mask
            diff_y = diff_y * mask
            diff_x = diff_x * mask

            # Normalize by the number of valid pixels in the mask
            # Add epsilon to avoid division by zero
            normalization = mask.sum() + 1e-8

            loss_z = diff_z.sum() / normalization
            loss_y = diff_y.sum() / normalization
            loss_x = diff_x.sum() / normalization
        else:
            # Standard Mean Reduction
            loss_z = diff_z.mean()
            loss_y = diff_y.mean()
            loss_x = diff_x.mean()

        # 4. Weighted Combination
        total_loss = (
            (self.weights[0] * loss_z)
            + (self.weights[1] * loss_y)
            + (self.weights[2] * loss_x)
        )

        return total_loss


class TVLoss3D:
    r"""
    Total variation loss (:math:`\ell_2` norm) for 3D images (volumes).

    It computes the loss :math:`\|D\hat{x}\|_2^2`, where :math:`D` is
    a normalized linear operator that computes the depth, vertical,
    and horizontal first order differences of the reconstructed volume
    :math:`\hat{x}`.

    Assumes input shape: (Batch, Channel, Depth, Height, Width).

    :param float weight: scalar weight for the TV loss.
    """

    def __init__(self, weight: float = 1.0):
        self.tv_loss_weight = weight
        self.name = "tv3d"

    def forward(self, x_net: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        Computes the TV loss for 3D volumes.

        :param torch.Tensor x_net: reconstructed image of size (N, C, D, H, W).
        :return: torch.Tensor loss of size (batch_size,)
        """
        # Ensure input is 5D (N, C, D, H, W)
        if x_net.dim() != 5:
            raise ValueError(f"Expected 5D input (N, C, D, H, W),\
                got {x_net.dim()}D")

        batch_size = x_net.size()[0]
        d_x = x_net.size()[2]
        h_x = x_net.size()[3]
        w_x = x_net.size()[4]

        # Calculate number of elements for normalization in each direction
        count_d = self.tensor_size(x_net[:, :, 1:, :, :])
        count_h = self.tensor_size(x_net[:, :, :, 1:, :])
        count_w = self.tensor_size(x_net[:, :, :, :, 1:])

        # Depth (D) differences
        d_tv = (
            torch.pow((x_net[:, :, 1:, :, :] - x_net[:, :, : d_x - 1, :, :]), 2)
            .reshape(batch_size, -1)
            .sum(1)
        )

        # Height (H) differences
        h_tv = (
            torch.pow((x_net[:, :, :, 1:, :] - x_net[:, :, :, : h_x - 1, :]), 2)
            .reshape(batch_size, -1)
            .sum(1)
        )

        # Width (W) differences
        w_tv = (
            torch.pow((x_net[:, :, :, :, 1:] - x_net[:, :, :, :, : w_x - 1]), 2)
            .reshape(batch_size, -1)
            .sum(1)
        )

        # Sum component losses and apply weight
        return (
            self.tv_loss_weight * 2 * (d_tv / count_d + h_tv / count_h + w_tv / count_w)
        )

    @staticmethod
    def tensor_size(t: torch.Tensor) -> int:
        # Computes size excluding the batch dimension (C * D * H * W)
        return t.size()[1] * t.size()[2] * t.size()[3] * t.size()[4]


def make_volume_comparison_image(
    gt, recon_final, recon_dc, recon_best, caption: str = ""
) -> "wandb.Image":

    # print(
    #     f"Shapes and types of inputs: GT {gt.shape} {type(gt)}, Recon Final {recon_final.shape} {type(recon_final)}, Recon DC {recon_dc.shape} {type(recon_dc)}, Recon Best {recon_best.shape} {type(recon_best)}"
    # )

    gt = gt.detach().cpu()
    recon_final = recon_final.detach().cpu()
    recon_dc = recon_dc.detach().cpu()
    recon_best = recon_best.detach().cpu()[0]

    if gt.ndim == 4:
        gt = gt[0]
    if recon_final.ndim == 4:
        recon_final = recon_final[0]
    if recon_dc.ndim == 4:
        recon_dc = recon_dc[0]
    if recon_best.ndim == 4:
        recon_best = recon_best[0]

    D, H, W = gt.shape
    axial_idx = D // 2
    coronal_idx = H // 2
    sagittal_idx = W // 2

    # Convert once (avoid repeated .numpy() calls)
    gt_np = gt.numpy()
    rec_final_np = recon_final.numpy()
    rec_dc_np = recon_dc.numpy()
    rec_best_np = recon_best.numpy()

    fig, axes = plt.subplots(4, 3, figsize=(9, 8))

    # Note: your titles are swapped vs the slice variables; fix labels for clarity
    axes[0, 0].imshow(gt_np[axial_idx].T, origin="lower", cmap="gray")
    axes[0, 0].set_title("GT - Axial")
    axes[0, 1].imshow(gt_np[:, coronal_idx, :].T, origin="lower", cmap="gray")
    axes[0, 1].set_title("GT - Coronal")
    axes[0, 2].imshow(gt_np[:, :, sagittal_idx].T, origin="lower", cmap="gray")
    axes[0, 2].set_title("GT - Sagittal")

    axes[1, 0].imshow(rec_dc_np[axial_idx].T, origin="lower", cmap="gray")
    axes[1, 0].set_title("DC - Axial")
    axes[1, 1].imshow(rec_dc_np[:, coronal_idx, :].T, origin="lower", cmap="gray")
    axes[1, 1].set_title("DC - Coronal")
    axes[1, 2].imshow(rec_dc_np[:, :, sagittal_idx].T, origin="lower", cmap="gray")
    axes[1, 2].set_title("DC - Sagittal")
    axes[2, 0].imshow(rec_best_np[axial_idx].T, origin="lower", cmap="gray")
    axes[2, 0].set_title("Best PSF- Axial")
    axes[2, 1].imshow(rec_best_np[:, coronal_idx, :].T, origin="lower", cmap="gray")
    axes[2, 1].set_title("Best PSF- Coronal")
    axes[2, 2].imshow(rec_best_np[:, :, sagittal_idx].T, origin="lower", cmap="gray")
    axes[2, 2].set_title("Best PSF- Sagittal")
    axes[3, 0].imshow(rec_final_np[axial_idx].T, origin="lower", cmap="gray")
    axes[3, 0].set_title("Final - Axial")
    axes[3, 1].imshow(rec_final_np[:, coronal_idx, :].T, origin="lower", cmap="gray")
    axes[3, 1].set_title("Final Estimated - Coronal")
    axes[3, 2].imshow(rec_final_np[:, :, sagittal_idx].T, origin="lower", cmap="gray")
    axes[3, 2].set_title("Final Estimated - Sagittal")
    for ax in axes.ravel():
        ax.axis("off")

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    pil_img = Image.open(buf).convert("RGB")

    # Important: close buffer after PIL loads (PIL keeps file handle otherwise)
    buf.close()
    plt.close(fig)
    plt.close("all")
    plt.clf()
    plt.cla()
    return wandb.Image(pil_img, caption=caption)


def make_stack_comparison_image(
    real_stacks,
    sim_stacks,
    slice_qa,
    qa_gt,
    conf_est=None,
    caption: str = "",
):
    """Visualize a single acquired stack and the model's per-slice estimates.

    Image inputs are (N, H, W) (a leading channel dim is squeezed if present);
    the per-slice scalar input (reliability) is (N,). Rows:
        real_stacks, simulated_stacks, slice_quality_estimated,
        slice_quality_gt, registration_reliability_estimated (big pixels)
    For the image rows the three columns are orientations of the (N, H, W) stack
    volume: in-plane (a random slice) and two through-plane resamples. The
    reliability row is skipped (left blank) when not provided.
    """
    import numpy as np
    import matplotlib.gridspec as gridspec

    def _img(x):
        if x is None:
            return None
        x = x.detach().float().cpu()
        if x.dim() == 4:  # (N, 1, H, W) -> (N, H, W)
            x = x[:, 0]
        return x.numpy()

    def _vec(x):
        return None if x is None else x.detach().float().cpu().reshape(-1).numpy()

    real, sim, qa = _img(real_stacks), _img(sim_stacks), _img(slice_qa)
    qa_g = _img(qa_gt)
    c_est = _vec(conf_est)

    ref = next((v for v in (real, sim, qa, qa_g) if v is not None), None)
    if ref is None:
        return None
    N, H, W = ref.shape

    # In-plane slice = the slice with the largest non-zero (brain) area, so the
    # panel is never an empty / edge slice. The two through-plane cuts pass through
    # that slice's brain centroid so they also show content.
    base = real if real is not None else ref
    nz_per_slice = (np.abs(base) > 1e-6).reshape(N, -1).sum(axis=1)
    n_in = int(nz_per_slice.argmax()) if nz_per_slice.max() > 0 else N // 2
    ys, xs = np.nonzero(np.abs(base[n_in]) > 1e-6)
    h_mid = int(round(ys.mean())) if ys.size else H // 2
    w_mid = int(round(xs.mean())) if xs.size else W // 2

    def _orient(vol):
        if vol is None:
            return (None, None, None)
        # in-plane (largest-brain slice), through-plane x2
        return (vol[n_in], vol[:, h_mid, :], vol[:, :, w_mid])

    def _range(*arrs, lo=1, hi=99):
        vals = [a.ravel() for a in arrs if a is not None and a.size > 0]
        if not vals:
            return 0.0, 1.0
        v = np.concatenate(vals)
        return float(np.percentile(v, lo)), float(np.percentile(v, hi))

    stack_vmax = _range(real, sim, lo=1, hi=99)[1]

    image_rows = [
        ("real_stacks", real, "gray", (0.0, stack_vmax)),
        ("simulated_stacks", sim, "gray", (0.0, stack_vmax)),
        ("slice_quality_estimated", qa, "viridis", (0.0, 1.0)),
        ("slice_quality_gt", qa_g, "viridis", (0.0, 1.0)),
    ]
    orient_names = ("in-plane", "through-plane-1", "through-plane-2")

    n_rows = len(image_rows) + 1  # + registration-reliability big-pixel strip
    fig = plt.figure(figsize=(10, 3 * n_rows), constrained_layout=True)
    gs = gridspec.GridSpec(n_rows, 3, figure=fig)

    qa_im, qa_axes = None, []
    for r, (name, vol, cmap, (vmin, vmax)) in enumerate(image_rows):
        views = _orient(vol)
        for c in range(3):
            ax = fig.add_subplot(gs[r, c])
            if views[c] is not None:
                im = ax.imshow(
                    views[c].T,
                    origin="lower",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    aspect="auto",
                )
                if cmap == "viridis":
                    qa_im = im
                    qa_axes.append(ax)
            ax.set_title(f"{name} - {orient_names[c]}", fontsize=8)
            ax.axis("off")

    # legend (colorbar) shared by the slice-quality rows: 0 = reject, 1 = trust
    if qa_im is not None:
        cbar = fig.colorbar(
            qa_im, ax=qa_axes, fraction=0.02, pad=0.01, label="slice quality"
        )
        cbar.set_ticks([0.0, 0.5, 1.0])

    # Per-slice registration reliability rendered as a (1, N) big-pixel strip: the
    # SVoRT-style softmax weight in [0, 1] (higher = the slice is trusted more).
    ax = fig.add_subplot(gs[len(image_rows), :])
    if c_est is not None:
        im = ax.imshow(
            c_est[None, :], cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto"
        )
        ax.set_yticks([])
        ax.set_xlabel("slice index", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    ax.set_title("registration_reliability_estimated", fontsize=8)

    fig.suptitle(caption, fontsize=10)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    pil_img = Image.open(buf).convert("RGB")
    buf.close()
    plt.close(fig)
    plt.close("all")
    plt.clf()
    plt.cla()
    return wandb.Image(pil_img, caption=caption)


def make_qa_comparison_image(real_stacks, qa_est, qa_gt, caption: str = ""):
    """Visualize a single acquired stack and its slice-quality maps.

    For QA-only training (no reconstruction): three rows -- the real stack, the
    estimated dense quality map, and the GT artifact map -- each shown in three
    orientations of the (N, H, W) stack volume (an in-plane slice + two
    through-plane resamples). Inputs are (N, H, W) or (N, 1, H, W) (a leading
    channel dim is squeezed). Any input may be None (its row is left blank).
    """
    import numpy as np
    import matplotlib.gridspec as gridspec

    def _img(x):
        if x is None:
            return None
        x = x.detach().float().cpu()
        if x.dim() == 4:  # (N, 1, H, W) -> (N, H, W)
            x = x[:, 0]
        return x.numpy()

    real, est, gt = _img(real_stacks), _img(qa_est), _img(qa_gt)
    ref = next((v for v in (real, est, gt) if v is not None), None)
    if ref is None:
        return None
    N, H, W = ref.shape

    # In-plane slice = the slice with the largest brain area (never an empty edge
    # slice); the two through-plane cuts pass through that slice's brain centroid.
    base = real if real is not None else ref
    nz_per_slice = (np.abs(base) > 1e-6).reshape(N, -1).sum(axis=1)
    n_in = int(nz_per_slice.argmax()) if nz_per_slice.max() > 0 else N // 2
    ys, xs = np.nonzero(np.abs(base[n_in]) > 1e-6)
    h_mid = int(round(ys.mean())) if ys.size else H // 2
    w_mid = int(round(xs.mean())) if xs.size else W // 2

    def _orient(vol):
        if vol is None:
            return (None, None, None)
        return (vol[n_in], vol[:, h_mid, :], vol[:, :, w_mid])

    stack_vmax = float(np.percentile(real, 99)) if real is not None else 1.0
    rows = [
        ("real_stacks", real, "gray", (0.0, stack_vmax)),
        ("slice_quality_estimated", est, "viridis", (0.0, 1.0)),
        ("slice_quality_gt", gt, "viridis", (0.0, 1.0)),
    ]
    orient_names = ("in-plane", "through-plane-1", "through-plane-2")

    fig = plt.figure(figsize=(10, 9), constrained_layout=True)
    gs = gridspec.GridSpec(3, 3, figure=fig)

    qa_im, qa_axes = None, []
    for r, (name, vol, cmap, (vmin, vmax)) in enumerate(rows):
        views = _orient(vol)
        for c in range(3):
            ax = fig.add_subplot(gs[r, c])
            if views[c] is not None:
                im = ax.imshow(
                    views[c].T,
                    origin="lower",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    aspect="auto",
                )
                if cmap == "viridis":
                    qa_im = im
                    qa_axes.append(ax)
            ax.set_title(f"{name} - {orient_names[c]}", fontsize=8)
            ax.axis("off")

    # Shared colorbar for the quality rows: 0 = reject, 1 = trust.
    if qa_im is not None:
        cbar = fig.colorbar(
            qa_im, ax=qa_axes, fraction=0.02, pad=0.01, label="slice quality"
        )
        cbar.set_ticks([0.0, 0.5, 1.0])

    fig.suptitle(caption, fontsize=10)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    pil_img = Image.open(buf).convert("RGB")
    buf.close()
    plt.close(fig)
    plt.close("all")
    plt.clf()
    plt.cla()
    return wandb.Image(pil_img, caption=caption)
