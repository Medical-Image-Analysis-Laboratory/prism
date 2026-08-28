#!/usr/bin/env python
"""Run PRISM slice-to-volume registration over a BIDS folder of 2D stacks.

This is the standalone entry point: no Hydra config, no split CSV, no ground
truth. Every ``sub-*/[ses-*/]anat/*_<suffix>.nii.gz`` stack of a subject/session
is loaded together, the model estimates the rigid pose of every slice, and the
results are written in a BIDS-like layout under ``--output-dir``:

    <output>/<sub>/<ses>/anat/
        <sub>_<ses>_type-SRRest_T2w.nii.gz        reconstructed volume (1 mm iso)
        <sub>_<ses>_motion-pred.npz               per-slice rigid poses
        <sub>_<ses>_type-ESTslices_T2w.nii.gz     slices re-projected from it
        <sub>_<ses>_type-GTslices_T2w.nii.gz      the acquired slices, as fed in
        <sub>_slice_metrics.csv                   per-slice agreement metrics
    <output>/metrics.csv                          one row per subject/session

The per-slice metrics compare each acquired slice with the same slice
re-projected from the reconstruction at the estimated pose -- the slice-level
agreement reported for real data in the paper (no ground-truth pose needed).

Inputs must be **brain-masked** stacks in the same intensity range the model was
trained on; see the README for the preprocessing steps (brain masking + N4).

Requires a CUDA GPU: the forward model (slice acquisition and its adjoint) is a
CUDA extension.

Examples
--------
    python scripts/predict_bids.py \\
        --bids-dir  /data/mybids/derivatives/masked \\
        --output-dir /data/mybids/derivatives/prism \\
        --weights   /weights/prism_weights.pth

    # only two subjects, and keep the re-projected slices
    python scripts/predict_bids.py -i <bids> -o <out> -w <weights> \\
        --subjects sub-001 sub-002 --save-slices
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd
import rootutils
import torch
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from synthgen.data.datasets import InferenceStackDataset  # noqa: E402
from synthgen.generator.slice_acquisition import slice_acquisition  # noqa: E402
from synthgen.models.prism import VARIANTS, build_prism  # noqa: E402
from synthgen.utils.io import save_motion_params, save_volume  # noqa: E402

warnings.filterwarnings("ignore")

# Near-empty slices (fewer brain voxels than this) are still registered and
# still written to the motion file, but are not scored.
EMPTY_SLICE_MIN_VOXELS = 100


def ncc(a: torch.Tensor, b: torch.Tensor, mask=None, eps: float = 1e-8) -> float:
    """Normalized cross-correlation in [-1, 1] (over ``mask`` if given)."""
    a = a.flatten().float()
    b = b.flatten().float()
    if mask is not None:
        m = mask.flatten() > 0
        if m.sum() < 2:
            return float("nan")
        a, b = a[m], b[m]
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + eps))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    io = p.add_argument_group("input / output")
    io.add_argument("-i", "--bids-dir", type=Path, required=True,
                    help="BIDS folder of brain-masked 2D stacks")
    io.add_argument("-o", "--output-dir", type=Path, required=True,
                    help="where to write reconstructions, poses and metrics")
    io.add_argument("-w", "--weights", type=Path, required=True,
                    help="PRISM weights (released .pth or training .ckpt)")
    io.add_argument("--img-suffix", default="T2w",
                    help="BIDS suffix of the stacks to read (default: T2w)")
    io.add_argument("--subjects", nargs="+", default=None,
                    help="only these subjects (default: every sub-* found)")

    mdl = p.add_argument_group("model")
    mdl.add_argument("--variant", default="prism", choices=VARIANTS,
                    help="model variant matching the weights (default: prism)")
    mdl.add_argument("--n-iter", type=int, default=4,
                    help="registration iterations (default: 4)")
    mdl.add_argument("--cg-iter", type=int, default=4,
                    help="CG iterations per data-consistency solve (default: 4)")
    mdl.add_argument("--device", default="cuda:0", help="CUDA device")
    mdl.add_argument("--recon", default="dc", choices=("dc", "regularized"),
                    help="which reconstruction to save and score: 'dc' is the "
                         "plain data-consistency solve at the final poses (the "
                         "protocol reported in the paper, default); "
                         "'regularized' is the prior-regularized volume the SVR "
                         "loop uses internally")

    geo = p.add_argument_group("geometry")
    geo.add_argument("--resolution", type=float, default=1.0,
                    help="isotropic reconstruction resolution in mm (default: 1)")
    geo.add_argument("--volume-shape", type=int, nargs=3, default=(128, 128, 128),
                    metavar=("X", "Y", "Z"), help="reconstructed volume shape")
    geo.add_argument("--slice-size", type=int, default=128,
                    help="in-plane size the slices are padded to (default: 128)")
    geo.add_argument("--keep-empty-slices", dest="drop_empty_slices",
                    action="store_false", default=True,
                    help="feed near-empty slices (<=100 brain voxels) to the "
                         "model as well; they are dropped by default, as in the "
                         "evaluation reported in the paper")

    out = p.add_argument_group("what to save")
    out.add_argument("--save-slices", action="store_true",
                    help="also save the acquired and re-projected slice stacks")
    out.add_argument("--no-volume", action="store_true",
                    help="skip writing the reconstructed volume")
    out.add_argument("--no-motion", action="store_true",
                    help="skip writing the estimated per-slice poses")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not torch.cuda.is_available():
        print("ERROR: a CUDA GPU is required (the forward model is a CUDA "
              "extension).", file=sys.stderr)
        return 2
    if not args.bids_dir.is_dir():
        print(f"ERROR: no such BIDS folder: {args.bids_dir}", file=sys.stderr)
        return 2
    if not args.weights.is_file():
        print(f"ERROR: no such weights file: {args.weights}", file=sys.stderr)
        return 2

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    dataset = InferenceStackDataset(
        bids_path=str(args.bids_dir),
        sub_list=args.subjects,
        img_suffix=args.img_suffix,
        seg_suffix=None,
        target_resolution_recon=args.resolution,
        slice_size=args.slice_size,
        reconstructed_volume_shape=tuple(args.volume_shape),
        drop_empty_slices=args.drop_empty_slices,
    )
    if len(dataset) == 0:
        print(f"ERROR: no subject with a *_{args.img_suffix}.nii.gz stack found "
              f"under {args.bids_dir}", file=sys.stderr)
        return 1
    print(f"Found {len(dataset)} subject/session(s) in {args.bids_dir}")

    model = build_prism(
        variant=args.variant,
        weights=str(args.weights),
        n_iter=args.n_iter,
        cg_iter=args.cg_iter,
        device=device,
    )
    print(f"Running {args.variant} with n_iter={args.n_iter}, "
          f"cg_iter={args.cg_iter} on {device}")

    # monai metrics are imported late: they pull in a large dependency tree.
    from monai.metrics import MSEMetric, PSNRMetric, SSIMMetric

    mse, psnr = MSEMetric(), PSNRMetric(1.0)
    ssim2d = SSIMMetric(2)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scan_rows = []

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=0, shuffle=False
    )
    for batch in tqdm(loader, total=len(dataset), desc="PRISM"):
        name = batch["name"][0]
        sub, ses = (name.split("_") + [None])[:2]
        subdir = args.output_dir / sub / (ses or "") / "anat"
        subdir.mkdir(parents=True, exist_ok=True)
        stem = f"{sub}_{ses}" if ses else sub

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)

        with torch.no_grad():
            res = model.predict_step(batch, 0)

            transforms_pred = res["transforms_pred"][-1].to(device)
            # `volume_dc`: the plain data-consistency solve at the final poses
            # (Eq. 2), i.e. what the paper scores; `recon_volume`: the
            # prior-regularized volume the SVR loop refines the poses against.
            volume = (
                res["volume_dc"][0].detach().float().cpu()
                if args.recon == "dc"
                else res["recon_volume"]
            )

            # Re-project the reconstruction at the estimated poses: the
            # simulated counterpart of every acquired slice.
            est_slices = slice_acquisition(
                transforms=transforms_pred,
                vol=volume.to(device).unsqueeze(0),
                vol_mask=None,
                slices_mask=None,
                psf=res["psf_acq"].to(device),
                slice_shape=res["slice_shape"],
                res_slice=res["resolution_slice"] / res["resolution_recon"],
                need_weight=False,
                interp_psf=False,
            ).cpu()
            acq_slices = res["stacks"].cpu()

        slice_rows = []
        for i in range(est_slices.shape[0]):
            gt_slice = acq_slices[i].unsqueeze(0)
            n_brain = int((gt_slice > 0).sum().item())
            if n_brain <= EMPTY_SLICE_MIN_VOXELS:
                continue
            est_slice = est_slices[i].unsqueeze(0)
            slice_rows.append({
                "name": name,
                "stack_index": int(res["positions"][i, 1].item()),
                "slice_in_scan": int(res["positions"][i, 0].item()),
                "slice_index": i,
                "MSE": mse(y_pred=est_slice, y=gt_slice).item(),
                "PSNR": psnr(y_pred=est_slice, y=gt_slice).item(),
                "SSIM": ssim2d(y_pred=est_slice, y=gt_slice).item(),
                "NCC": ncc(est_slice, gt_slice),
                "NCC_brain": ncc(est_slice, gt_slice, mask=gt_slice > 0),
                "brain_voxels": n_brain,
            })

        row = {"name": name, "n_slices_total": int(est_slices.shape[0])}
        if slice_rows:
            slice_df = pd.DataFrame(slice_rows)
            slice_df.to_csv(subdir / f"{stem}_slice_metrics.csv", index=False)
            row.update({
                f"slice_{k}": float(slice_df[k].mean())
                for k in ("MSE", "PSNR", "SSIM", "NCC", "NCC_brain")
            })
            row["n_slices_scored"] = len(slice_rows)
        scan_rows.append(row)

        if not args.no_volume:
            save_volume(
                subdir / f"{stem}_type-SRRest_T2w.nii.gz",
                volume.unsqueeze(0),
                res=res["resolution_recon"],
            )
        if not args.no_motion:
            save_motion_params(
                subdir / f"{stem}_motion-pred.npz",
                res["transforms_pred"][-1],
                res["positions"],
                res["slice_shape"],
                res["resolution_slice"],
                slice_thickness=res["slice_thickness"],
                resolution_recon=res["resolution_recon"],
            )
        if args.save_slices:
            slice_res = [
                res["resolution_slice"],
                res["resolution_slice"],
                res["slice_thickness"],
            ]
            save_volume(
                subdir / f"{stem}_type-GTslices_T2w.nii.gz",
                acq_slices[:, 0].unsqueeze(0).unsqueeze(0),
                res=slice_res,
            )
            save_volume(
                subdir / f"{stem}_type-ESTslices_T2w.nii.gz",
                est_slices[:, 0].unsqueeze(0).unsqueeze(0),
                res=slice_res,
            )

    metrics = pd.DataFrame(scan_rows)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    print(f"\nWrote {len(metrics)} scan(s) to {args.output_dir}")
    cols = [c for c in ("slice_NCC", "slice_PSNR", "slice_SSIM") if c in metrics]
    if cols:
        means = ", ".join(f"{c[6:]}={metrics[c].mean():.4f}" for c in cols)
        print(f"Mean slice-level agreement over all scans: {means}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
