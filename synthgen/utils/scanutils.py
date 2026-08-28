import torch
import numpy as np
import collections
import torch.nn.functional as F
from synthgen.generator.transform import (
    random_angle,
    RigidTransform,
)
from typing import Sequence, Tuple, Union

import collections

RangeLike = Union[float, int, Sequence[float], Tuple[float, float]]


def validatebeforescandims(scan_data):
    assert isinstance(
        scan_data["resolution"], float
    ), f"Resolution must be a float. Found {type(scan_data['resolution'])} {scan_data['resolution']}."

    for k in ["mask", "volume", "label"]:
        assert scan_data[k].ndim == 5 and isinstance(
            scan_data[k], torch.Tensor
        ), f"{k} must be torch.Tensor 5D (B, C, H, W, D). Found {scan_data[k].ndim}D with shape {scan_data[k].shape} and type {type(scan_data[k])}."

    for k in [
        "threshold",
    ]:  # not checking scale as we apply gamma with torchio
        assert (
            isinstance(scan_data[k], torch.Tensor) and scan_data[k].ndim == 0
        ), f"{k} must be a scalar torch.Tensor. Found {type(scan_data[k])} {scan_data[k]}."


def validateafterscandims(scan_data):

    for k in [
        "resolution",
        "resolution_recon",
        "resolution_slice",
        "slice_thickness",
        "gap",
    ]:
        assert isinstance(
            scan_data[k], float
        ), f"{k} must be a float. Found {type(scan_data[k])} {scan_data[k]}."

    for k in ["mask", "volume_gt", "label", "mask_gt", "seg_gt"]:
        if k in scan_data:
            assert scan_data[k].ndim == 5 and isinstance(
                scan_data[k], torch.Tensor
            ), f"{k} must be torch.Tensor 5D (B, C, H, W, D). Found {scan_data[k].ndim}D with shape {scan_data[k].shape} and type {type(scan_data[k])}."

    for k in [
        "threshold",
    ]:
        assert (
            isinstance(scan_data[k], torch.Tensor) and scan_data[k].ndim == 0
        ), f"{k} must be a scalar torch.Tensor. Found {type(scan_data[k])} {scan_data[k]}."

    assert (
        isinstance(scan_data["volume_shape"], torch.Size)
        and len(scan_data["volume_shape"]) == 3
    ), f"volume_shape must be torch.Size of length 3. Found {type(scan_data['volume_shape'])} {scan_data['volume_shape']}."

    assert (
        isinstance(scan_data["positions"], torch.Tensor)
        and scan_data["positions"].ndim == 2
    ), f"positions must be torch.Tensor 2D (N_slices, 2). Found {type(scan_data['positions'])} {scan_data['positions']}."

    for k in ["transforms", "transforms_gt"]:
        if k in scan_data:
            assert (
                isinstance(scan_data[k], torch.Tensor) and scan_data[k].ndim == 3
            ), f"{k} must be torch.Tensor 3D (N_slices, 3, 4). Found {type(scan_data[k])} {scan_data[k]}."

    assert (
        isinstance(scan_data["slice_shape"], tuple)
        and len(scan_data["slice_shape"]) == 2
    ), f"slice_shape must be a tuple of length 2. Found {type(scan_data['slice_shape'])} {scan_data['slice_shape']}."

    assert (
        isinstance(scan_data["psf_rec"], torch.Tensor)
        and scan_data["psf_rec"].ndim == 3
    ), f"psf_rec must be torch.Tensor 3D (H, W, D). Found {type(scan_data['psf_rec'])} {scan_data['psf_rec']}."
    assert (
        isinstance(scan_data["psf_acq"], torch.Tensor)
        and scan_data["psf_acq"].ndim == 3
    ), f"psf_acq must be torch.Tensor 3D (H, W, D). Found {type(scan_data['psf_acq'])} {scan_data['psf_acq']}."


def interleave_index(N, n_i):
    idx = [None] * N
    t = 0
    for i in range(n_i):
        j = i
        while j < N:
            idx[j] = t
            t += 1
            j += n_i
    return idx


def get_PSF(
    r_max=None, res_ratio=(1, 1, 3), threshold=1e-3, device=torch.device("cpu")
):
    sigma_x = 1.2 * res_ratio[0] / 2.3548
    sigma_y = 1.2 * res_ratio[1] / 2.3548
    sigma_z = res_ratio[2] / 2.3548
    if r_max is None:
        r_max = max(int(2 * r + 1) for r in (sigma_x, sigma_y, sigma_z))
        r_max = max(r_max, 4)
    x = torch.linspace(-r_max, r_max, 2 * r_max + 1, dtype=torch.float32, device=device)
    grid_z, grid_y, grid_x = torch.meshgrid(x, x, x, indexing="ij")
    psf = torch.exp(
        -0.5
        * (grid_x**2 / sigma_x**2 + grid_y**2 / sigma_y**2 + grid_z**2 / sigma_z**2)
    )
    psf[psf.abs() < threshold] = 0
    rx = torch.nonzero(psf.sum((0, 1)) > 0)[0, 0].item()
    ry = torch.nonzero(psf.sum((0, 2)) > 0)[0, 0].item()
    rz = torch.nonzero(psf.sum((1, 2)) > 0)[0, 0].item()
    psf = psf[
        rz : 2 * r_max + 1 - rz, ry : 2 * r_max + 1 - ry, rx : 2 * r_max + 1 - rx
    ].contiguous()
    psf = psf / psf.sum()
    return psf


def init_stack_transforms(n_slice, gap, restricted, txy, device):
    angle = random_angle(1, restricted, device).expand(n_slice, -1)
    tz = (
        torch.arange(0, n_slice, device=device, dtype=torch.float32)
        - (n_slice - 1) / 2.0
    ) * gap
    if txy:
        tx = torch.ones_like(tz) * np.random.uniform(-txy, txy)
        ty = torch.ones_like(tz) * np.random.uniform(-txy, txy)
    else:
        tx = ty = torch.zeros_like(tz)
    t = torch.stack((tx, ty, tz), -1)
    return RigidTransform(torch.cat((angle, t), -1), trans_first=True)


def reset_transform(transform):
    transform = transform.axisangle()
    transform[:, :-1] = 0
    transform[:, -1] -= transform[:, -1].mean()
    return RigidTransform(transform)


def bias_field(volume, bias_size, bias_sigma):
    if isinstance(bias_size, collections.abc.Iterable):
        bias_size = np.random.randint(bias_size[0], bias_size[-1] + 1)
    if isinstance(bias_sigma, collections.abc.Iterable):
        bias_sigma = np.random.uniform(bias_sigma[0], bias_sigma[-1])
    bias = (
        torch.randn(
            (1, 1, bias_size, bias_size, bias_size),
            dtype=volume.dtype,
            device=volume.device,
        )
        * bias_sigma
    )
    bias = F.interpolate(
        bias, size=volume.shape[-3:], mode="trilinear", align_corners=False
    )
    volume *= torch.exp(bias)
    return volume


def gamma_transform(volume, g, q, threshold):
    if isinstance(g, collections.abc.Iterable):
        g = np.random.uniform(g[0], g[-1])
    if isinstance(q, collections.abc.Iterable):
        q = np.random.uniform(q[0], q[-1])
    vq = torch.quantile(volume[volume > threshold], q)
    volume = (volume / vq) ** g
    threshold = (threshold / vq) ** g
    return volume, threshold, vq


def get_2d_polynomial_basis(H, W, order=2, device="cpu"):
    """
    Generates a fixed 2D polynomial basis tensor.
    For order=2, returns 6 basis maps: [1, x, y, x^2, xy, y^2]
    Shape: (num_coeffs, H, W)
    """
    y_coords = torch.linspace(-1, 1, H, device=device)
    x_coords = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

    basis = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            basis.append((xx**j) * (yy**i))

    return torch.stack(basis, dim=0)


def apply_parameterized_bias_field(
    image_stack,
    basis,
    strength_max=1.5,
    strength_min=0.1,
    percentage_max=1.0,
    percentage=None,
):
    """
    Applies a random polynomial bias field slice-by-slice.

    Args:
        image_stack: Tensor of shape (D, H, W)
        basis: Precomputed basis tensor of shape (num_coeffs, H, W)
        strength_max: Maximum standard deviation for the random coefficients
        strength_min: Minimum standard deviation for the random coefficients
        percentage_max: Maximum probability (0 to 1.0) of applying the artifact to a given slice
        percentage: Explicit fraction of slices to bias. If None, sampled per-call
            as ``rand * percentage_max``; pass a value to fix it per subject.

    Returns:
        corrupted_stack: (D, H, W)
        gt_coefficients: (D, num_coeffs) - Target labels for the network
    """

    D, H, W = image_stack.shape
    num_coeffs = basis.shape[0]
    device = image_stack.device

    corrupted_stack = image_stack.clone()
    gt_coefficients = torch.zeros((D, num_coeffs), device=device)
    if percentage is None:
        percentage = (
            torch.rand(1).item() * percentage_max
        )  # Random percentage for this run
    for d in range(D):
        if torch.rand(1).item() < percentage:
            # Sample random coefficients
            strength = (
                torch.rand(1).item() * (strength_max - strength_min) + strength_min
            )
            coeffs = torch.randn(num_coeffs, device=device) * strength

            # Set the 0th coefficient (constant plane) to 0 to prevent global intensity shifts
            coeffs[0] = 0.0

            # Reconstruct log-bias field and exponentiate
            log_bias = torch.sum(coeffs.view(-1, 1, 1) * basis, dim=0)
            bias_field = torch.exp(log_bias)

            # Apply and store
            corrupted_stack[d] = image_stack[d] * bias_field
            gt_coefficients[d] = coeffs

    return corrupted_stack, gt_coefficients


def remove_parameterized_bias_field(corrupted_stack, coefficients, basis):
    """
    Removes the bias field from an image stack using vectorized operations.

    Args:
        corrupted_stack: Tensor of shape (D, H, W)
        coefficients: Predicted or ground truth coefficients, shape (D, num_coeffs)
        basis: Precomputed basis tensor, shape (num_coeffs, H, W)

    Returns:
        corrected_stack: Tensor of shape (D, H, W), bias_field: Tensor of shape (D, H, W)
    """

    corrupted_stack = corrupted_stack.squeeze(1)
    assert (
        corrupted_stack.ndim == 3
    ), "corrupted_stack must be (D, H, W), found shape {}".format(corrupted_stack.shape)

    epsilon = 1e-8

    # Reshape for broadcasting
    coeffs_reshaped = coefficients.view(-1, basis.shape[0], 1, 1)
    basis_reshaped = basis.unsqueeze(0)

    # Reconstruct the log bias field and exponentiate
    log_bias = torch.sum(coeffs_reshaped * basis_reshaped, dim=1)
    bias_field = torch.exp(log_bias)

    # Remove the field
    corrected_stack = corrupted_stack / (bias_field + epsilon)
    corrected_stack = corrected_stack.unsqueeze(1)  # Add back batch dimension if needed
    return corrected_stack, bias_field


def percentile_bounds(
    x, mask=None, lo_p=0.01, hi_p=0.99, max_samples=4_000_000, multiplicative=False
):
    """Lower/upper percentile bounds of the in-mask voxels of ``x``.

    Robust alternative to ``(min, max)``: defaults to the 1st/99th percentiles so
    a few bright artifact voxels (noise spikes, bias amplification) cannot dictate
    the scale. ``mask`` selects the voxels used to estimate the bounds (e.g. the
    brain); if None all voxels are used. Returns scalar tensors ``(lo, hi)``.

    With ``multiplicative=True`` the lower bound is forced to 0 so the normalization
    becomes a pure scale (``x / hi``) rather than an affine ``(x - lo) / (hi - lo)``.
    This keeps a per-slice MULTIPLICATIVE intensity shift / bias field removable by a
    matching multiplicative correction: subtracting ``lo`` turns a multiplicative
    shift into an affine one that no scalar division can undo, leaving a per-slice DC
    residual (slice-to-slice streaks).
    """
    vals = x[mask > 0] if mask is not None else x.reshape(-1)
    vals = vals.float()
    if vals.numel() == 0:
        return x.new_tensor(0.0), x.new_tensor(1.0)
    if vals.numel() > max_samples:
        # torch.quantile is exact but limited/expensive on huge inputs; a random
        # subsample gives a statistically equivalent percentile estimate.
        idx = torch.randperm(vals.numel(), device=vals.device)[:max_samples]
        vals = vals[idx]
    lo = x.new_tensor(0.0) if multiplicative else torch.quantile(vals, lo_p)
    return lo, torch.quantile(vals, hi_p)


def percentile_normalize(
    x, mask=None, lo_p=0.01, hi_p=0.99, bounds=None, eps=1e-6, multiplicative=False
):
    """Robust [0, 1]-style scaling using lo/hi percentiles of the in-mask voxels.

    Maps ``p_lo -> 0`` and ``p_hi -> 1`` (bright detail above ``p_hi`` is kept, not
    clipped), forces out-of-mask voxels to 0, and clamps negatives to 0. Pass a
    precomputed ``bounds=(lo, hi)`` to place a second tensor on an identical scale
    (e.g. the clean stacks on the corrupted stacks' bounds).

    With ``multiplicative=True`` (and no precomputed ``bounds``) the lower bound is
    forced to 0, so the map is the pure scale ``x / hi`` -- required when the inputs
    carry multiplicative per-slice intensity shifts / bias fields that the model
    corrects by division (see ``percentile_bounds``). When ``bounds`` is supplied it
    is used verbatim, so pass ``lo=0`` bounds for the multiplicative case.
    """
    if bounds is None:
        lo, hi = percentile_bounds(x, mask, lo_p, hi_p, multiplicative=multiplicative)
    else:
        lo, hi = bounds
    x = (x - lo) / (hi - lo + eps)
    if mask is not None:
        x = torch.where(mask > 0, x, torch.zeros_like(x))
    return x.clamp(min=0)
