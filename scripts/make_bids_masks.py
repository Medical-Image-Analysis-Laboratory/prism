#!/usr/bin/env python
"""Create a BIDS-mirrored mask dataset from an input BIDS folder.

Masks are obtained by thresholding every NIfTI at ``> threshold`` (default 0),
i.e. a binary brain/foreground mask. The output keeps the exact same directory
structure, subject/session naming and BIDS entities; only the suffix changes
from ``_T2w(.nii.gz)`` to ``_T2w_mask.nii.gz``.

Examples
--------
    ./make_bids_masks.py \
        /data/mybids/derivatives/masked \
        /data/mybids/derivatives/masks

    ./make_bids_masks.py --dry-run <input_bids> <output_bids>
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


def mask_filename(name: str) -> str:
    """Map an input NIfTI filename to its mask filename.

    ``..._T2w.nii.gz``         -> ``..._T2w_mask.nii.gz``
    ``..._T2w_maskout.nii.gz`` -> ``..._T2w_mask.nii.gz``
    anything else              -> insert ``_mask`` before the extension.
    """
    # Replace the BIDS suffix and any trailing desc (e.g. _maskout) with _mask.
    new = re.sub(r"_T2w[A-Za-z0-9_-]*\.nii(\.gz)?$",
                 lambda m: "_T2w_mask.nii" + (m.group(1) or ""),
                 name)
    if new != name:
        return new
    # Fallback for non-T2w files: ..._foo.nii.gz -> ..._foo_mask.nii.gz
    return re.sub(r"\.nii(\.gz)?$",
                  lambda m: "_mask.nii" + (m.group(1) or ""),
                  name)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_dir", type=Path,
                   help="input BIDS directory")
    p.add_argument("output_dir", type=Path,
                   help="output BIDS directory for masks")
    p.add_argument("-t", "--threshold", type=float, default=0.0,
                   help="mask = data > threshold (default: 0)")
    p.add_argument("-n", "--dry-run", action="store_true",
                   help="list what would be written without creating anything")
    p.add_argument("-f", "--overwrite", action="store_true",
                   help="overwrite existing mask files (default: skip them)")
    args = p.parse_args()

    in_root: Path = args.input_dir.resolve()
    out_root: Path = args.output_dir.resolve()

    if not in_root.is_dir():
        print(f"Error: input dir does not exist: {in_root}",
              file=sys.stderr)
        return 1

    # Only descend into top-level BIDS subject folders (sub-*); ignore any
    # other folders next to them (code/, derivatives/, sourcedata/, ...).
    sub_dirs = sorted(d for d in in_root.glob("sub-*") if d.is_dir())
    if not sub_dirs:
        print(f"No sub-* folders found directly under: {in_root}",
              file=sys.stderr)
        return 1

    niftis = sorted(f for d in sub_dirs for f in d.rglob("*.nii.gz"))
    # Don't re-mask files that are already masks.
    niftis = [f for f in niftis if not f.name.endswith("_mask.nii.gz")]
    if not niftis:
        print(f"No *.nii.gz files found under sub-* folders in: {in_root}",
              file=sys.stderr)
        return 1

    print(f"Input : {in_root}")
    print(f"Output: {out_root}")
    print(f"Found {len(niftis)} NIfTI file(s). "
          f"Threshold: data > {args.threshold}")
    if args.dry_run:
        print("(dry-run: nothing will be written)\n")

    n_done = n_skipped = n_failed = 0
    for src in tqdm(niftis, desc="Masking", unit="file"):
        rel = src.relative_to(in_root)
        dst = out_root / rel.parent / mask_filename(src.name)

        if args.dry_run:
            tqdm.write(f"[DRY-RUN] {rel}  ->  {dst.relative_to(out_root)}")
            n_done += 1
            continue

        if dst.exists() and not args.overwrite:
            n_skipped += 1
            n_done += 1
            continue

        try:
            img = nib.load(str(src))
            data = np.asanyarray(img.dataobj)
            mask = (data > args.threshold).astype(np.uint8)

            out_img = nib.Nifti1Image(mask, img.affine, img.header)
            out_img.set_data_dtype(np.uint8)
            # Keep header consistent with the new dtype / binary range.
            out_img.header.set_data_dtype(np.uint8)

            dst.parent.mkdir(parents=True, exist_ok=True)
            nib.save(out_img, str(dst))
        except Exception as exc:  # noqa: BLE001 - report and keep going
            tqdm.write(f"FAILED {rel}: {exc}")
            n_failed += 1
        n_done += 1

    print(f"\nFinished: {n_done}/{len(niftis)} processed | "
          f"skipped: {n_skipped} | failed: {n_failed}")
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
