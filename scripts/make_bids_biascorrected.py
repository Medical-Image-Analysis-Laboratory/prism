#!/usr/bin/env python
"""Run N4 bias field correction over a BIDS folder, in place.

Walks every ``sub-*/ses-*/anat/*T2w.nii.gz`` image, applies N4 bias field
correction (ANTsPy) and writes the result *alongside* the input with
``biascorrected`` inserted before the extension, e.g.
``..._T2w.nii.gz`` -> ``..._T2wbiascorrected.nii.gz``.

Examples
--------
    ./make_bids_biascorrected.py /path/to/bids

    ./make_bids_biascorrected.py --dry-run /path/to/bids
    ./make_bids_biascorrected.py -f /path/to/bids        # overwrite existing
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import ants
from tqdm import tqdm


def biascorrected_filename(name: str) -> str:
    """Insert ``biascorrected`` before the extension, keeping the T2/T2w token.

    ``..._T2w.nii.gz`` -> ``..._T2wbiascorrected.nii.gz``
    ``..._T2.nii.gz``  -> ``..._T2biascorrected.nii.gz``
    """
    return re.sub(r"(T2w?)\.nii(\.gz)?$",
                  lambda m: m.group(1) + "biascorrected.nii" + (m.group(2) or ""),
                  name)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bids_dir", type=Path,
                   help="BIDS directory to process (results saved alongside)")
    p.add_argument("-p", "--pattern", default="sub-*/ses-*/anat/*T2w.nii.gz",
                   help="glob (relative to bids_dir) for input images "
                        "(default: %(default)s)")
    p.add_argument("--no-mask", action="store_true",
                   help="do not estimate a foreground mask for N4 "
                        "(by default ants.get_mask is used to focus N4)")
    p.add_argument("-n", "--dry-run", action="store_true",
                   help="list what would be written without creating anything")
    p.add_argument("-f", "--overwrite", action="store_true",
                   help="overwrite existing outputs (default: skip them)")
    args = p.parse_args()

    root: Path = args.bids_dir.resolve()
    if not root.is_dir():
        print(f"Error: BIDS dir does not exist: {root}", file=sys.stderr)
        return 1

    images = sorted(root.glob(args.pattern))
    # Never re-process an already-corrected output.
    images = [f for f in images
              if not f.name.endswith("biascorrected.nii.gz")]
    if not images:
        print(f"No files matching {args.pattern!r} under: {root}",
              file=sys.stderr)
        return 1

    print(f"BIDS  : {root}")
    print(f"Found {len(images)} image(s) matching {args.pattern!r}")
    if args.dry_run:
        print("(dry-run: nothing will be written)\n")

    n_done = n_skipped = n_failed = 0
    for src in tqdm(images, desc="N4", unit="file"):
        rel = src.relative_to(root)
        dst = src.with_name(biascorrected_filename(src.name))

        if dst.name == src.name:
            tqdm.write(f"SKIP {rel}: filename does not end in 'T2.nii.gz'")
            n_skipped += 1
            n_done += 1
            continue

        if args.dry_run:
            tqdm.write(f"[DRY-RUN] {rel}  ->  {dst.relative_to(root)}")
            n_done += 1
            continue

        if dst.exists() and not args.overwrite:
            n_skipped += 1
            n_done += 1
            continue

        try:
            img = ants.image_read(str(src))
            mask = None if args.no_mask else ants.get_mask(img)
            corrected = ants.n4_bias_field_correction(img, mask=mask)
            ants.image_write(corrected, str(dst))
        except Exception as exc:  # noqa: BLE001 - report and keep going
            tqdm.write(f"FAILED {rel}: {exc}")
            n_failed += 1
        n_done += 1

    print(f"\nFinished: {n_done}/{len(images)} processed | "
          f"skipped: {n_skipped} | failed: {n_failed}")
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
