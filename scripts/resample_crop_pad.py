#!/usr/bin/env python3
"""
Resample, orient, center-crop and pad BIDS-structured images & labels.

Example:
    python resample_crop_pad.py \
        --bids-path /path/to/bids \
        --res 0.5 \
        --target-size 256 256 256 \
        --image-pattern "*_T2w.nii.gz" \
        --label-pattern "*drawem9_dseg.nii.gz"

Notes:
- Expects subjects as sub-*/ses-*/anat/ under --bids-path or sub-*/anat/.
- Writes to --out-dir (default: <bids-root>/derivatives/resampled{res_mm}).
- Image is resampled with bilinear; label with nearest.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple, List

from tqdm import tqdm
import monai


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Resample to isotropic spacing, orient to RAS, center-crop and pad images/labels."
    )
    p.add_argument(
        "--bids-path",
        type=Path,
        required=True,
        help="Path to the BIDS root or its derivatives folder containing sub-*/ses-*/anat/ or sub-*/anat/.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <bids-root>/derivatives/resampled{res_mm}).",
    )
    p.add_argument(
        "--res",
        type=float,
        default=0.5,
        help="Isotropic resampling resolution in mm (default: 0.5).",
    )
    p.add_argument(
        "--target-size",
        type=int,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(256, 256, 256),
        help="Center crop-pad to the ROI size in voxels (default: 256 256 256).",
    )
    p.add_argument(
        "--axcodes",
        type=str,
        default="RAS",
        help="Orientation codes for Orientationd (default: RAS).",
    )

    p.add_argument(
        "--image-patterns",
        type=str,
        nargs="+",
        default=("T2w",),
        help="Patterns to match image files (T1w, T2w) Will be used in a regex: {sub}*_{pattern}.nii.gz",
    )

    p.add_argument(
        "--label-patterns",
        type=str,
        nargs="+",
        default=("drawem9_dseg",),
        help="Patterns to match label files (drawem9_dseg, etc.) Will be used in a regex: {sub}*_{pattern}.nii.gz",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    img_keys = args.image_patterns
    label_keys = args.label_patterns
    datakeys = args.image_patterns + args.label_patterns
    bids_path = args.bids_path
    out_path = (
        args.out_dir
        if args.out_dir is not None
        else bids_path / "derivatives" / f"resampled{args.res:.2f}"
    )
    # defining transforms
    loader = monai.transforms.LoadImaged(keys=datakeys, allow_missing_keys=True)
    resample_transform = monai.transforms.Spacingd(
        keys=datakeys,
        pixdim=(args.res, args.res, args.res),
        mode=["bilinear"] * len(img_keys) + ["nearest"] * len(label_keys),
        allow_missing_keys=True,
    )

    cropper = monai.transforms.CenterSpatialCropd(
        keys=datakeys, allow_missing_keys=True, roi_size=tuple(args.target_size)
    )
    orientation = monai.transforms.Orientationd(
        keys=datakeys, axcodes=args.axcodes, allow_missing_keys=True
    )

    padder = monai.transforms.SpatialPadd(
        keys=datakeys,
        spatial_size=args.target_size,
        mode="constant",
        allow_missing_keys=True,
    )

    # Try finding sessions and if not try without
    subject_sessions = list(bids_path.glob("sub-*/ses-*"))
    if len(subject_sessions) == 0:
        subject_sessions = list(bids_path.glob("sub-*"))
        print(
            f"No sessions found. Found {len(subject_sessions)} subjects in {bids_path}"
        )
    else:
        print(f"Found {len(subject_sessions)} subject sessions in {bids_path}")

    for subses in tqdm(subject_sessions):
        if "ses-" in subses.name:
            sub = subses.parent.name
            ses = subses.name
            output_dir = out_path / f"{sub}/{ses}/anat/"
        else:
            sub = subses.name
            ses = ""
            output_dir = out_path / f"{sub}/anat/"

        saver = monai.transforms.SaveImaged(
            keys=datakeys,
            output_dir=output_dir,
            output_postfix="",
            resample=False,
            separate_folder=False,
            print_log=False,
            allow_missing_keys=True,
        )

        # gather all images and labels matching the patterns
        imgs = {}
        labels = {}

        for img_pattern in img_keys:
            imgs[img_pattern] = list(
                subses.glob(f"anat/{sub}{'_' + ses}*{img_pattern}.nii.gz")
            )
        for label_pattern in label_keys:
            labels[label_pattern] = list(
                subses.glob(f"anat/{sub}{'_' + ses}*{label_pattern}.nii.gz")
            )
        # check that for each pattern we have found at most one file
        data = {}
        for img_pattern in img_keys:
            if len(imgs[img_pattern]) > 1:
                raise ValueError(
                    f"More than one image found for pattern {img_pattern} in {subses}"
                )
            elif len(imgs[img_pattern]) == 0:
                # remove the key
                del imgs[img_pattern]
                print(f"No images found for pattern {img_pattern} in {subses}")
            elif len(imgs[img_pattern]) == 1:
                data[img_pattern] = str(imgs[img_pattern][0])

        for label_pattern in label_keys:
            if len(labels[label_pattern]) > 1:
                raise ValueError(
                    f"More than one label found for pattern {label_pattern} in {subses}"
                )
            elif len(labels[label_pattern]) == 0:
                # remove the key
                del labels[label_pattern]
                print(f"No labels found for pattern {label_pattern} in {subses}")
            elif len(labels[label_pattern]) == 1:
                data[label_pattern] = str(labels[label_pattern][0])

        data = loader(data)
        for imgkey in img_keys:
            if imgkey in data:
                data[imgkey] = data[imgkey].unsqueeze(0)

        for labkey in label_keys:
            if labkey in data:
                data[labkey] = data[labkey].unsqueeze(0)
                data[labkey] = data[labkey].squeeze(-1)

        data = resample_transform(data)
        data = orientation(data)
        data = cropper(data)
        data = padder(data)
        saver(data)
        # break
