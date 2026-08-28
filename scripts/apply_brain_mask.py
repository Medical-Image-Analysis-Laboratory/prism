#!/usr/bin/env python
"""Prepare acquired 2D stacks for PRISM: mask, crop and normalise.

This is the last preprocessing step before ``scripts/predict_bids.py``, and the
one used for the clinical data of the paper. For every stack of an input BIDS
folder it

1. multiplies the image by its brain mask (background -> 0),
2. clips to the 1st/99th intensity percentile and rescales to ``[0, 1]``,
3. crops/pads to a ``--size`` (default 128) cube centred on the brain,

and writes the result to the same relative path under ``--output-dir`` with the
suffix ``--out-suffix`` (default ``T2w_maskout``, so
``sub-001_..._T2w.nii.gz`` -> ``sub-001_..._T2w_maskout.nii.gz``).

Masks are looked up anywhere under ``--masks`` by matching the BIDS entity stem
of the image, so both ``<stem>_T2w_mask.nii.gz`` (the layout
``scripts/make_bids_masks.py`` writes) and ``<stem>_mask.nii.gz`` (the layout
FetPype writes) are found. Any fetal brain extraction tool can produce them; the
paper used the masks of the FetPype pipeline (https://fetpype.github.io/fetpype/).

Only the in-plane geometry is touched; the affine is carried over unchanged,
which is harmless because the model re-derives the initial slice poses from the
stack direction and spacing alone (the origin is reset, see
``synthgen.utils.io.load_stack``).

Examples
--------
    python scripts/apply_brain_mask.py \\
        --bids-dir /data/mybids \\
        --masks    /data/mybids/derivatives/masks \\
        --output-dir /data/mybids/derivatives/masked
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


def find_mask(image: Path, masks_dir: Path, img_suffix: str) -> Path | None:
    """Locate the brain mask of ``image`` anywhere under ``masks_dir``.

    Matched on the BIDS entity stem, so both the mirrored layout of
    ``make_bids_masks.py`` (``<stem>_T2w_mask.nii.gz``) and the flat layout of
    FetPype (``<stem>_mask.nii.gz``) are found.
    """
    stem = image.name[: -len(f"_{img_suffix}.nii.gz")]
    for pattern in (
        f"{stem}_{img_suffix}_mask.nii.gz",
        f"{stem}_mask.nii.gz",
        f"{stem}*mask*.nii.gz",
    ):
        matches = sorted(masks_dir.rglob(pattern))
        if matches:
            return matches[0]
    return None


def crop_or_pad(data: np.ndarray, brain_idx: tuple, size: int) -> np.ndarray:
    """Return a ``size``-cubed volume centred on the brain bounding box.

    Axes longer than ``size`` are cropped, shorter ones zero-padded, so every
    stack ends up on the same grid whatever its field of view or slice count
    (the through-plane axis of a thick-slice stack is usually padded, which is
    why most slices of the result are empty).
    """
    out = np.zeros((size,) * data.ndim, dtype=data.dtype)
    src, dst = [], []
    for ax, dim in zip(brain_idx, data.shape):
        centre = (int(ax.min()) + int(ax.max())) // 2
        lo = centre - size // 2
        s_lo, s_hi = max(0, lo), min(dim, lo + size)
        src.append(slice(s_lo, s_hi))
        dst.append(slice(s_lo - lo, s_lo - lo + (s_hi - s_lo)))
    out[tuple(dst)] = data[tuple(src)]
    return out


def squeeze_4d(data: np.ndarray) -> np.ndarray:
    """Drop a trailing singleton 4th dimension, as some DICOM exports carry."""
    if data.ndim == 4 and data.shape[3] == 1:
        return data[..., 0]
    return data


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-i", "--bids-dir", type=Path, required=True,
                   help="input BIDS folder of acquired stacks")
    p.add_argument("-m", "--masks", type=Path, required=True,
                   help="BIDS-mirrored folder of brain masks")
    p.add_argument("-o", "--output-dir", type=Path, required=True,
                   help="output folder (same BIDS layout)")
    p.add_argument("--img-suffix", default="T2w",
                   help="BIDS suffix of the input stacks (default: T2w)")
    p.add_argument("--out-suffix", default="T2w_maskout",
                   help="BIDS suffix to write (default: T2w_maskout)")
    p.add_argument("--size", type=int, default=128,
                   help="side of the output cube, in voxels (default: 128, "
                        "the in-plane size the model expects)")
    p.add_argument("--percentiles", type=float, nargs=2, default=(1.0, 99.0),
                   metavar=("LO", "HI"),
                   help="intensity percentiles to clip to (default: 1 99)")
    p.add_argument("-f", "--overwrite", action="store_true",
                   help="overwrite existing outputs (default: skip them)")
    args = p.parse_args(argv)

    if not args.bids_dir.is_dir():
        print(f"ERROR: no such BIDS folder: {args.bids_dir}", file=sys.stderr)
        return 2
    if not args.masks.is_dir():
        print(f"ERROR: no such masks folder: {args.masks}", file=sys.stderr)
        return 2

    images = sorted(args.bids_dir.glob(f"sub-*/**/*_{args.img_suffix}.nii.gz"))
    if not images:
        print(f"ERROR: no sub-*/**/*_{args.img_suffix}.nii.gz under "
              f"{args.bids_dir}", file=sys.stderr)
        return 1
    print(f"Found {len(images)} stack(s) in {args.bids_dir}")

    lo_q, hi_q = args.percentiles[0] / 100.0, args.percentiles[1] / 100.0
    n_done = n_skipped = n_missing = n_empty = 0

    for image in tqdm(images, desc="Masking", unit="stack"):
        mask_file = find_mask(image, args.masks, args.img_suffix)
        if mask_file is None:
            tqdm.write(f"no mask found for {image.name} under {args.masks}")
            n_missing += 1
            continue

        rel = image.relative_to(args.bids_dir)
        out_file = args.output_dir / rel.parent / rel.name.replace(
            f"_{args.img_suffix}.nii.gz", f"_{args.out_suffix}.nii.gz"
        )
        if out_file.exists() and not args.overwrite:
            n_skipped += 1
            continue

        img = nib.load(str(image))
        data = squeeze_4d(np.asanyarray(img.dataobj).astype(np.float32))
        mask = squeeze_4d(np.asanyarray(nib.load(str(mask_file)).dataobj))

        masked = data * (mask > 0)
        idx = (mask > 0).nonzero()
        if len(idx[0]) == 0:
            tqdm.write(f"empty mask for {image.name}")
            n_empty += 1
            continue

        # Percentiles come from the masked volume, before padding, so the
        # amount of padding cannot influence the intensity mapping.
        lo = np.quantile(masked, lo_q)
        hi = np.quantile(masked, hi_q)
        masked = np.clip(masked, lo, hi)
        masked = (masked - lo) / (hi - lo) if hi > lo else np.zeros_like(masked)
        masked = crop_or_pad(masked, idx, args.size)

        out_file.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(masked, img.affine), str(out_file))
        n_done += 1

    print(f"\nWrote {n_done} stack(s) to {args.output_dir} "
          f"(skipped {n_skipped}, no mask {n_missing}, empty mask {n_empty})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
