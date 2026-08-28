"""Slice-level artifact simulation and slice-quality ground-truth options.

This module is a *prototype / proposal* companion to ``scanner.py``. It collects:

1. New per-slice artifact simulators aimed at two real-data failure modes that
   the existing :meth:`Scanner.signal_void` (a pure elliptical *darkening*) does
   not reproduce:
     - **local gray signal void** -- a blob where tissue loses contrast, is
       smeared and washes out to a flat mid-gray (the "blurred + grayed patch"
       look), and
     - **blur / heavy-noise corruption** -- partial (blob) or whole-slice blur,
       and strong (optionally spatially-correlated) noise.

2. Several options for the per-pixel **quality ground truth** used to supervise
   the slice-quality module (SQM). The scanner currently derives the GT from the
   raw intensity difference ``|corrupted - clean|``. That is blind to blur in
   low-contrast regions (intensity barely moves, yet structure is destroyed) and
   over-reacts to the bias/intensity-shift artifacts that are corrected by other
   heads. The structure-aware options (local NCC / local SSIM) fix both.

All functions operate on float tensors shaped ``(N, 1, H, W)`` (a stack of N
slices), values roughly in ``[0, 1]``, and are device-agnostic. They are written
to drop into ``Scanner.scan`` next to the existing ``signal_void`` / ``add_noise``
calls.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur


# --------------------------------------------------------------------------- #
#  Small helpers
# --------------------------------------------------------------------------- #
def _as_n1hw(slices: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Return slices as ``(N, 1, H, W)`` plus a flag whether a channel was added."""
    if slices.ndim == 3:  # (N, H, W)
        return slices.unsqueeze(1), True
    assert slices.ndim == 4, f"expected (N,H,W) or (N,1,H,W), got {slices.shape}"
    return slices, False


def _kernel_size(sigma: float) -> int:
    """Odd Gaussian kernel size covering +-3 sigma (torchvision needs odd k)."""
    k = int(2 * math.ceil(3 * float(sigma)) + 1)
    return max(k, 3)


def _gblur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian blur with an automatically sized odd kernel."""
    if sigma <= 0:
        return x
    return gaussian_blur(x, kernel_size=_kernel_size(sigma), sigma=float(sigma))


def _box_blur(x: torch.Tensor, k: int) -> torch.Tensor:
    """Mean filter (box blur) with reflect padding; k forced odd."""
    k = int(k) | 1
    pad = k // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(x, kernel_size=k, stride=1)


def masked_local_mean(x: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    """Local mean of ``x`` over a ``k x k`` window, ignoring ``mask==0`` pixels.

    Used so the "gray" target of a void is the regional *tissue* mean and is not
    dragged toward 0 by the masked-out background.
    """
    m = mask.float()
    num = _box_blur(x * m, k)
    den = _box_blur(m, k).clamp_min(1e-6)
    return num / den


def random_blob_mask(
    n: int,
    h: int,
    w: int,
    device,
    n_blobs=(1, 2),
    area=(10.0, 30.0),
    aspect_max: float = 15.0,
    feather: float = 1.0,
    seed_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generate smooth-edged soft blob masks in ``[0, 1]``, shape ``(n, 1, h, w)``.

    Each slice gets 1..``n_blobs`` anisotropic, randomly rotated Gaussian blobs
    (same parameterization as :meth:`Scanner.signal_void`, but returned as a
    *soft* falloff so the corruption fades gradually into clean tissue -- which is
    what real grayed-out patches look like). ``feather`` extra-blurs the mask edge.

    Args:
        n, h, w: number and spatial size of slices.
        n_blobs: ``(min, max)`` blobs per slice.
        area: ``(min, max)`` blob "radius" in pixels (physical-ish size).
        aspect_max: max anisotropy of a blob.
        feather: Gaussian sigma applied to the final mask to soften edges.
        seed_mask: optional ``(n,1,h,w)`` brain mask; blob centers are biased to
            lie inside it so voids land on tissue rather than background.
    """
    y = torch.linspace(-(h - 1) / 2, (h - 1) / 2, h, device=device)
    x = torch.linspace(-(w - 1) / 2, (w - 1) / 2, w, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")  # (h, w)

    out = torch.zeros(n, 1, h, w, device=device)
    nb_lo, nb_hi = n_blobs
    for i in range(n):
        nb = int(np.random.randint(nb_lo, nb_hi + 1))
        acc = torch.zeros(h, w, device=device)
        for _ in range(nb):
            # blob center: random, optionally pulled to a tissue pixel
            yc = (torch.rand(1, device=device) - 0.5).item() * (h - 1)
            xc = (torch.rand(1, device=device) - 0.5).item() * (w - 1)
            if seed_mask is not None:
                idxs = torch.nonzero(seed_mask[i, 0] > 0.5)
                if len(idxs) > 0:
                    py, px = idxs[np.random.randint(len(idxs))].tolist()
                    yc = py - (h - 1) / 2
                    xc = px - (w - 1) / 2
            yb = yy - yc
            xb = xx - xc
            theta = 2 * math.pi * float(torch.rand(1))
            c, s = math.cos(theta), math.sin(theta)
            xr, yr = c * xb - s * yb, s * xb + c * yb
            a = float(np.random.uniform(*area))
            sx = float(np.random.uniform(1.0, aspect_max))
            sy = a**2 / sx
            blob = torch.exp(-0.5 * (xr**2 / sx**2 + yr**2 / sy**2))
            acc = torch.maximum(acc, blob)
        out[i, 0] = acc
    if feather > 0:
        out = _gblur(out, feather)
    return out.clamp(0, 1)


# --------------------------------------------------------------------------- #
#  Artifact simulators
# --------------------------------------------------------------------------- #
def local_gray_void(
    slices: torch.Tensor,
    prob: float = 0.1,
    brain_mask: torch.Tensor | None = None,
    blur_sigma=(2.0, 5.0),
    contrast=(0.15, 0.5),
    lift=(0.0, 0.25),
    mean_window: int = 31,
    return_mask: bool = False,
    return_corrupted: bool = False,
    **blob_kwargs,
):
    """Local "gray signal void": a blob where tissue is smeared toward a flat gray.

    Within a soft random blob the slice is replaced by a target that is (a) heavily
    blurred (detail lost) and (b) contrast-compressed toward the local tissue mean
    and slightly brightened -- reproducing the washed-out gray patch seen in real
    slices. Outside the blob the slice is untouched; the soft mask gives a gradual
    transition.

        target = local_mean * (1 + lift) + contrast * (blur(slice) - local_mean)
        out    = (1 - m) * slice + m * target

    Args:
        slices: ``(N, 1, H, W)`` (or ``(N, H, W)``).
        prob: per-slice probability of receiving a gray void.
        brain_mask: optional ``(N,1,H,W)`` tissue mask for the local mean / centers.
        blur_sigma, contrast, lift: ``(min, max)`` ranges sampled per affected slice.
        mean_window: window (px) for the regional tissue mean.
        return_mask: also return the applied soft mask (0 where untouched).
        return_corrupted: also return a ``(N,)`` bool of which slices were touched.
        blob_kwargs: forwarded to :func:`random_blob_mask`.

    Returns:
        corrupted slices (same shape as input). If ``return_mask`` the soft mask is
        appended, and if ``return_corrupted`` the per-slice bool is appended (in
        that order), e.g. ``(slices, corrupted)`` with only ``return_corrupted``.
    """
    x, squeezed = _as_n1hw(slices)
    n, _, h, w = x.shape
    if brain_mask is None:
        bm = (x > 1e-4).float()
    else:
        bm, _ = _as_n1hw(brain_mask)
        bm = bm.float()

    sel = torch.rand(n, device=x.device) < prob
    applied = torch.zeros_like(x)
    if sel.any():
        idx = torch.nonzero(sel).flatten()
        m = random_blob_mask(
            len(idx), h, w, x.device, seed_mask=bm[idx], **blob_kwargs
        )
        m = m * bm[idx]  # confine void to tissue
        sub = x[idx]
        lm = masked_local_mean(sub, bm[idx], mean_window)
        for j in range(len(idx)):
            bs = float(np.random.uniform(*blur_sigma))
            cs = float(np.random.uniform(*contrast))
            lf = float(np.random.uniform(*lift))
            blurred = _gblur(sub[j : j + 1], bs)
            target = lm[j : j + 1] * (1 + lf) + cs * (blurred - lm[j : j + 1])
            sub[j : j + 1] = (1 - m[j : j + 1]) * sub[j : j + 1] + m[j : j + 1] * target
        x[idx] = sub
        applied[idx] = m
    out = x.squeeze(1) if squeezed else x
    extras = []
    if return_mask:
        extras.append(applied.squeeze(1) if squeezed else applied)
    if return_corrupted:
        extras.append(sel)
    return (out, *extras) if extras else out


def local_blur(
    slices: torch.Tensor,
    prob: float = 0.1,
    blur_sigma=(1.5, 4.0),
    brain_mask: torch.Tensor | None = None,
    return_mask: bool = False,
    return_corrupted: bool = False,
    **blob_kwargs,
):
    """Partial blur: a blob region is smeared while contrast is roughly preserved.

    Like :func:`local_gray_void` but only blurs (no graying) -- for slices that are
    locally out of focus rather than signal-dropped. ``return_mask`` /
    ``return_corrupted`` behave as in :func:`local_gray_void`.
    """
    x, squeezed = _as_n1hw(slices)
    n, _, h, w = x.shape
    bm = (x > 1e-4).float() if brain_mask is None else _as_n1hw(brain_mask)[0].float()
    sel = torch.rand(n, device=x.device) < prob
    applied = torch.zeros_like(x)
    if sel.any():
        idx = torch.nonzero(sel).flatten()
        m = random_blob_mask(len(idx), h, w, x.device, seed_mask=bm[idx], **blob_kwargs)
        m = m * bm[idx]
        sub = x[idx]
        for j in range(len(idx)):
            bs = float(np.random.uniform(*blur_sigma))
            blurred = _gblur(sub[j : j + 1], bs)
            sub[j : j + 1] = (1 - m[j : j + 1]) * sub[j : j + 1] + m[j : j + 1] * blurred
        x[idx] = sub
        applied[idx] = m
    out = x.squeeze(1) if squeezed else x
    extras = []
    if return_mask:
        extras.append(applied.squeeze(1) if squeezed else applied)
    if return_corrupted:
        extras.append(sel)
    return (out, *extras) if extras else out


def local_noise(
    slices: torch.Tensor,
    prob: float = 0.1,
    sigma=(0.05, 0.25),
    brain_mask: torch.Tensor | None = None,
    correlated_prob: float = 0.5,
    correlation_sigma=(0.6, 1.5),
    threshold: float = 1e-4,
    return_mask: bool = False,
    return_corrupted: bool = False,
    **blob_kwargs,
):
    """Local heavy noise: a blob region is corrupted by strong (optionally
    spatially-correlated) Rician noise while the rest of the slice is untouched.

    The *spatially-varying* analogue of :func:`heavy_noise` -- like
    :func:`local_blur` / :func:`local_gray_void`, the corruption is confined to a
    soft random blob, so only part of the slice is degraded and the noise fades
    gradually into clean tissue via the soft mask:

        noisy = sqrt((slice + n1)^2 + n2^2)        # Rician, as in heavy_noise
        out   = (1 - m) * slice + m * noisy

    ``correlated_prob`` / ``correlation_sigma`` optionally blur the noise field
    (variance-renormalized) to mimic the grainy/structured texture of real noise
    rather than flat white noise. ``return_mask`` / ``return_corrupted`` behave as
    in :func:`local_gray_void`.

    Args:
        slices: ``(N, 1, H, W)`` (or ``(N, H, W)``).
        prob: per-slice probability of receiving a local-noise blob.
        sigma: ``(min, max)`` Rician noise strength, sampled per affected slice.
        brain_mask: optional ``(N,1,H,W)`` tissue mask confining the blob / centers.
        correlated_prob: probability the noise field is spatially correlated.
        correlation_sigma: ``(min, max)`` blur sigma for the correlated noise field.
        threshold: tissue cutoff used when ``brain_mask`` is not given.
        blob_kwargs: forwarded to :func:`random_blob_mask`.
    """
    x, squeezed = _as_n1hw(slices)
    n, _, h, w = x.shape
    bm = (x > threshold).float() if brain_mask is None else _as_n1hw(brain_mask)[0].float()
    sel = torch.rand(n, device=x.device) < prob
    applied = torch.zeros_like(x)
    if sel.any():
        idx = torch.nonzero(sel).flatten()
        m = random_blob_mask(len(idx), h, w, x.device, seed_mask=bm[idx], **blob_kwargs)
        m = m * bm[idx]  # confine the noisy blob to tissue
        sub = x[idx]
        for j in range(len(idx)):
            sg = float(np.random.uniform(*sigma))
            n1 = torch.randn_like(sub[j : j + 1]) * sg
            n2 = torch.randn_like(sub[j : j + 1]) * sg
            if float(torch.rand(1)) < correlated_prob:
                cs = float(np.random.uniform(*correlation_sigma))
                # blur then renormalize variance so correlated noise keeps strength sg
                n1 = _gblur(n1, cs) * (cs * math.sqrt(4 * math.pi))
                n2 = _gblur(n2, cs) * (cs * math.sqrt(4 * math.pi))
            noisy = torch.sqrt((sub[j : j + 1] + n1) ** 2 + n2**2)
            mj = m[j : j + 1]
            sub[j : j + 1] = (1 - mj) * sub[j : j + 1] + mj * noisy
        x[idx] = sub
        applied[idx] = m
    out = x.squeeze(1) if squeezed else x
    extras = []
    if return_mask:
        extras.append(applied.squeeze(1) if squeezed else applied)
    if return_corrupted:
        extras.append(sel)
    return (out, *extras) if extras else out


def slice_blur(
    slices: torch.Tensor,
    prob: float = 0.1,
    blur_sigma=(0.8, 2.5),
    return_corrupted: bool = False,
):
    """Whole-slice Gaussian blur on a random subset of slices (out-of-focus slices).

    With ``return_corrupted`` also returns a ``(N,)`` bool of the blurred slices.
    """
    x, squeezed = _as_n1hw(slices)
    n = x.shape[0]
    sel = torch.rand(n, device=x.device) < prob
    for j in torch.nonzero(sel).flatten():
        bs = float(np.random.uniform(*blur_sigma))
        x[j : j + 1] = _gblur(x[j : j + 1], bs)
    out = x.squeeze(1) if squeezed else x
    return (out, sel) if return_corrupted else out


def heavy_noise(
    slices: torch.Tensor,
    prob: float = 0.1,
    sigma=(0.05, 0.2),
    threshold: float = 1e-4,
    correlated_prob: float = 0.5,
    correlation_sigma=(0.6, 1.5),
    return_corrupted: bool = False,
):
    """Strong Rician noise on a random subset of slices, optionally spatially
    correlated (blurred noise field) to mimic the grainy/structured texture of
    real noise-corrupted slices rather than flat white noise.

    With ``return_corrupted`` also returns a ``(N,)`` bool of the noised slices.
    """
    x, squeezed = _as_n1hw(slices)
    n = x.shape[0]
    sel = torch.rand(n, device=x.device) < prob
    for j in torch.nonzero(sel).flatten():
        sl = x[j : j + 1]
        mask = sl > threshold
        sg = float(np.random.uniform(*sigma))
        n1 = torch.randn_like(sl) * sg
        n2 = torch.randn_like(sl) * sg
        if float(torch.rand(1)) < correlated_prob:
            cs = float(np.random.uniform(*correlation_sigma))
            # blur then renormalize variance so correlated noise keeps strength sg
            n1 = _gblur(n1, cs) * (cs * math.sqrt(4 * math.pi))
            n2 = _gblur(n2, cs) * (cs * math.sqrt(4 * math.pi))
        noisy = torch.sqrt((sl + n1) ** 2 + n2**2)
        x[j : j + 1] = torch.where(mask, noisy, sl)
    out = x.squeeze(1) if squeezed else x
    return (out, sel) if return_corrupted else out


# --------------------------------------------------------------------------- #
#  Slice-quality ground-truth options
# --------------------------------------------------------------------------- #
# Each returns a per-pixel quality map in [0, 1] (1 = good, 0 = corrupted), same
# spatial shape as the inputs. ``clean`` is the artifact-free reference (the
# scanner's ``slices_clean``); ``corrupted`` is the final slice.


def finalize_quality(
    q: torch.Tensor, thr: float = 0.8, sharp: float = 20.0
) -> torch.Tensor:
    """Harden a soft ``[0,1]`` quality score into the actual SQM training target.

    This is the scanner's "diff -> quality map" step: a sigmoid that pushes pixels
    above ``thr`` toward 1 (good) and below toward 0 (corrupted). The scanner GT is
    ``finalize_quality(quality_composite(clean, corrupted))``.

    ``thr`` is the soft score below which a pixel counts as corrupted; ``sharp``
    sets how hard the soft->binary transition is. The defaults were softened from
    ``(0.9, 30)`` to ``(0.8, 20)`` to reduce over-flagging of mild changes: a soft
    score of ~0.85 (a subtle blur) now maps to ~0.73 (kept mostly good) instead of
    ~0.18, while genuinely corrupted regions (soft <~0.7, e.g. voids / heavy noise)
    still collapse to ~0. Raise ``thr``/``sharp`` for a stricter, more binary
    target, or pass the soft map straight through for a smooth one.
    """
    return torch.sigmoid(sharp * (q - thr))


def quality_intensity(
    clean: torch.Tensor,
    corrupted: torch.Tensor,
    smooth_sigma: float = 1.0,
) -> torch.Tensor:
    """**Option A (current scanner GT, soft form).** ``1 - blur(|corrupted - clean|)``.

    This is a pre-sigmoid map; wrap it in :func:`finalize_quality` to harden it into
    a SQM target. Simple but blind to blur in flat regions and reacts to any
    intensity change. Kept as the baseline to compare against."""
    err = (corrupted - clean).abs()
    return 1 - _gblur(err, smooth_sigma) if smooth_sigma > 0 else 1 - err


def quality_ncc(
    clean: torch.Tensor,
    corrupted: torch.Tensor,
    win: int = 9,
    c: float = 1e-3,
) -> torch.Tensor:
    """**Option D.** Local normalized cross-correlation map.

    Windowed zero-mean NCC between clean and corrupted. NCC is invariant to local
    *affine* intensity changes: useful if you compute the GT against a truly clean
    reference (no bias/intensity shift) and want to ignore those, but in this
    pipeline ``slices_clean`` already contains the bias/intensity shift so any
    metric cancels them. Empirically NCC is the *least* sensitive of the options to
    blur and graying (correlation tolerates contrast loss and mild blur), and only
    moderately sensitive to noise -- prefer SSIM/gradient/composite for flagging
    these artifacts. Mapped from ``[-1, 1]`` to ``[0, 1]``.

    A constant stabilizer ``c`` (the SSIM "structure" term, ``~ (0.03 * range)^2``)
    makes *flat, unchanged* windows -- background and low-contrast tissue -- read as
    1 instead of 0/0. Without it, NCC is undefined where there is no local variance
    and would wrongly flag clean flat regions as corrupted. Matches the repo's
    existing NCC slice-reject direction.
    """
    mu_c = _box_blur(clean, win)
    mu_d = _box_blur(corrupted, win)
    var_c = (_box_blur(clean * clean, win) - mu_c * mu_c).clamp_min(0)
    var_d = (_box_blur(corrupted * corrupted, win) - mu_d * mu_d).clamp_min(0)
    cov = _box_blur(clean * corrupted, win) - mu_c * mu_d
    ncc = (cov + c) / (torch.sqrt(var_c * var_d) + c)
    return ncc.clamp(0, 1)


def quality_ssim(
    clean: torch.Tensor,
    corrupted: torch.Tensor,
    win: int = 7,
    C1: float = 0.01**2,
    C2: float = 0.03**2,
) -> torch.Tensor:
    """**Option C (recommended single metric).** Local SSIM map (per-pixel).
    Jointly captures luminance shift (gray void), contrast loss (blur, void) and
    structure loss (noise) -- empirically the only single option sensitive to all
    three new artifact types at once. Its luminance/contrast terms are not
    affine-invariant, but that is harmless here because ``slices_clean`` already
    contains the bias/intensity shift, so those cancel in the comparison."""
    mu_c = _box_blur(clean, win)
    mu_d = _box_blur(corrupted, win)
    mu_c2, mu_d2, mu_cd = mu_c * mu_c, mu_d * mu_d, mu_c * mu_d
    var_c = _box_blur(clean * clean, win) - mu_c2
    var_d = _box_blur(corrupted * corrupted, win) - mu_d2
    cov = _box_blur(clean * corrupted, win) - mu_cd
    ssim = ((2 * mu_cd + C1) * (2 * cov + C2)) / (
        (mu_c2 + mu_d2 + C1) * (var_c + var_d + C2)
    )
    return ssim.clamp(0, 1)


def quality_gradient(
    clean: torch.Tensor,
    corrupted: torch.Tensor,
    smooth_sigma: float = 1.5,
    eps: float = 1e-4,
) -> torch.Tensor:
    """**Option B.** Local detail-retention ratio:
    ``blur(min(|grad corrupted|, |grad clean|)) / blur(|grad clean|)``.
    Directly measures how much edge detail survived -> good for blur/void. Note it
    is *blind to noise* (the ``min`` caps noise-inflated gradients), so it is best
    combined with an intensity/structural term rather than used alone."""

    def grad_mag(t):
        gx = torch.zeros_like(t)
        gy = torch.zeros_like(t)
        gx[..., :, 1:] = t[..., :, 1:] - t[..., :, :-1]
        gy[..., 1:, :] = t[..., 1:, :] - t[..., :-1, :]
        return torch.sqrt(gx * gx + gy * gy + 1e-12)

    gc = grad_mag(clean)
    gd = grad_mag(corrupted)
    # ``eps`` in BOTH numerator and denominator so flat, unchanged regions
    # (no edges in either image) read as 1 (eps/eps) rather than 0/eps.
    ratio = (_gblur(torch.minimum(gd, gc), smooth_sigma) + eps) / (
        _gblur(gc, smooth_sigma) + eps
    )
    return ratio.clamp(0, 1)


def quality_composite(
    clean: torch.Tensor,
    corrupted: torch.Tensor,
    ssim_win: int = 7,
    grad_sigma: float = 1.5,
) -> torch.Tensor:
    """**Option E (recommended overall).** ``min(SSIM, gradient-retention)``.

    A pixel is flagged bad if *either* term fires, combining each option's
    strength: SSIM catches noise and graying / luminance-contrast loss (gradient
    is blind to noise), and the gradient detail-retention term catches blur more
    sharply than SSIM. Empirically the most sensitive choice across all three new
    artifacts (blur, gray void, heavy noise) -- at the cost of being the strictest
    GT, so pair it with a slightly looser SQM weight if it over-penalizes."""
    struct = quality_ssim(clean, corrupted, win=ssim_win)
    detail = quality_gradient(clean, corrupted, smooth_sigma=grad_sigma)
    return torch.minimum(struct, detail)


QUALITY_GT_OPTIONS = {
    "intensity": quality_intensity,  # A: current
    "gradient": quality_gradient,  # B
    "ssim": quality_ssim,  # C
    "ncc": quality_ncc,  # D (recommended)
    "composite": quality_composite,  # E
}
