"""
Build a BIDS database of *simulated* stacks for a split (validation by default).

For every subject of the requested split this script reproduces the exact
forward simulation that feeds the model during training/eval (SynthDataset ->
generator.sample -> scanner.scan, see synthgen/inference.py) and writes the
resulting stacks to disk in a BIDS-compliant layout.

Output layout (``output_dir``)::

    output_dir/
    |-- dataset_description.json
    |-- sub-XXX/[ses-YYY/]anat/
    |       sub-XXX[_ses-YYY][_acq-scN]_run-K_T2w.nii.gz   <- simulated stack
    |       sub-XXX[_ses-YYY][_acq-scN]_run-K_T2w.json      <- per-stack params
    |-- derivatives/
        |-- config.yaml                 <- full resolved config used
        |-- stacks_metadata.csv         <- one row per simulated stack
        |-- gt_volume/
            |-- dataset_description.json
            |-- sub-XXX/[ses-YYY/]anat/
                    sub-XXX[_ses-YYY][_acq-scN]_T2w.nii.gz  <- GT reference volume
                    sub-XXX[_ses-YYY][_acq-scN]_dseg.nii.gz <- GT segmentation

Usage::

    python -m synthgen.generate_stack_database experiment=datagen/simrealbad

  The three `datagen/simreal{good,med,bad}` experiments are the test sets of the
  paper; `output_dir` and the source data can be overridden on the command line.

Notes
-----
* Requires a CUDA device: ``Scanner`` runs on ``cuda:0``.
* Within a single scan all stacks share the sampled geometry (slice thickness,
  in-plane resolution, gap) - that is how ``Scanner`` works. Noise, motion,
  signal void, bias field and slice intensity shifts vary per stack / per slice.
  Use ``n_scans_per_subject>1`` to draw several independent geometries per
  subject.
"""

import json
import warnings
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import pandas as pd
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

warnings.filterwarnings("ignore")

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from synthgen.data.datasets import SynthDataset
from synthgen.generator.scanner import Scanner
from synthgen.generator.transform.transform import RigidTransform
from synthgen.utils.io import save_stack, save_volume, save_motion_params
from synthgen.utils.lightning import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class RecordingScanner(Scanner):
    """``Scanner`` that records the per-stack noise sigma it samples.

    ``Scanner.add_noise`` draws a fresh sigma for every stack and discards it,
    so the only way to report the noise level that actually generated a stack is
    to capture it at sampling time. Everything else (slice thickness, in-plane
    resolution, gap, bias coefficients, intensity shifts) is already returned in
    the data dict, so we only override ``add_noise`` here. ``add_noise`` is
    called exactly once per stack, in stack-acquisition order.
    """

    def scan(self, data):
        self._noise_sigmas: list[float] = []
        return super().scan(data)

    def add_noise(self, slices, threshold, sigma=None):
        # Faithful copy of Scanner.add_noise that also logs the sampled sigma.

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
        sigma = float(np.random.uniform(self.noise_sigma[0], self.noise_sigma[-1]))
        self._noise_sigmas.append(sigma)
        noise1 = torch.randn_like(masked) * sigma
        noise2 = torch.randn_like(masked) * sigma
        slices[mask] = torch.sqrt((masked + noise1) ** 2 + noise2**2)
        return slices


def _bids_anat_dir(root: Path, sub: str, ses: str | None) -> Path:
    """Return the (created) ``anat`` directory for a subject/session."""
    d = root / sub / ses / "anat" if ses else root / sub / "anat"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _entity_base(
    sub: str,
    ses: str | None,
    suffix: str,
    n_scans: int,
    scan_idx: int,
    run: int | None = None,
    stack_id: int | None = None,
) -> str:
    """Build a BIDS filename stem like ``sub-X_ses-Y_acq-sc1_stackid-3_run-2_T2w``.

    ``stack_id`` is the scanner's actual stack id (``positions[:, 1]``), placed
    before ``run`` so the file name keeps both the run index and the stack id.
    """
    parts = [sub]
    if ses:
        parts.append(ses)
    if n_scans > 1:
        parts.append(f"acq-sc{scan_idx + 1}")
    if stack_id is not None:
        parts.append(f"stackid-{stack_id}")
    if run is not None:
        parts.append(f"run-{run}")
    parts.append(suffix)
    return "_".join(parts)


def _write_dataset_description(root: Path, name: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "dataset_description.json", "w") as f:
        json.dump(
            {"Name": name, "BIDSVersion": "1.8.0", "DatasetType": "raw"}, f, indent=2
        )


def _stack_indices(positions: torch.Tensor) -> list[int]:
    """Stack ids in acquisition order (matches RecordingScanner noise log)."""
    return [int(round(float(s))) for s in pd.unique(positions[:, 1].cpu().numpy())]


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="generate_stack_database.yaml",
)
def main(cfg: DictConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "generate_stack_database requires a CUDA device (Scanner runs on cuda:0)."
        )
    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)

    out_root = Path(cfg.output_dir)
    deriv = out_root / "derivatives"
    gt_root = deriv / "gt_volume"
    out_root.mkdir(parents=True, exist_ok=True)
    deriv.mkdir(parents=True, exist_ok=True)

    img_suffix = cfg.data.img_suffix
    seg_suffix = cfg.data.seg_suffix
    n_scans = int(cfg.n_scans_per_subject)
    split = cfg.split
    val_type = cfg.data.get("val_type", "real")

    # BIDS housekeeping
    _write_dataset_description(
        out_root, f"Simulated stacks ({cfg.get('task_name', 'synthgen')})"
    )
    _write_dataset_description(gt_root, "Ground-truth reference volumes")
    OmegaConf.save(cfg, deriv / "config.yaml")

    # ----- subjects of the requested split ---------------------------------
    split_df = pd.read_csv(cfg.data.split_file)
    subjects = split_df[split_df.splits == split].participant_id.tolist()
    if not subjects:
        raise RuntimeError(
            f"No subjects found for split='{split}' in {cfg.data.split_file}"
        )

    # ----- scanner (records per-stack noise) + generator -------------------
    scanner_cfg = OmegaConf.to_container(cfg.data.scanner.cfg_scanner, resolve=True)
    # acquire per-stack segmentations with the same geometry as the image slices
    scanner_cfg["scan_segmentations"] = bool(cfg.save_stack_segmentations)
    scanner = RecordingScanner(scanner_cfg)
    generator = (
        hydra.utils.instantiate(cfg.data.generator)
        if cfg.data.get("generator")
        else None
    )

    # ----- dataset (mirror DataModule's val-set construction) --------------
    if val_type == "real":
        dataset = SynthDataset(
            bids_path=cfg.data.bids_path,
            seed_path=None,
            sub_list=subjects,
            load_image=True,
            image_as_intensity=True,
            generator=generator,
            img_suffix=img_suffix,
            seg_suffix=seg_suffix,
            scanner=scanner,
        )
    elif val_type == "synth":
        dataset = SynthDataset(
            bids_path=cfg.data.bids_path,
            seed_path=cfg.data.seed_path,
            sub_list=subjects,
            load_image=False,
            image_as_intensity=False,
            generator=generator,
            img_suffix=img_suffix,
            seg_suffix=seg_suffix,
            scanner=scanner,
        )
    else:
        raise ValueError(
            f"val_type='{val_type}' cannot be simulated; use an experiment with "
            "val_type 'real' or 'synth'."
        )

    n_subjects = len(dataset)
    if n_subjects == 0:
        raise RuntimeError(
            f"Dataset is empty for split='{split}' (no subjects matched in BIDS)."
        )
    log.info(
        f"Simulating split='{split}' ({val_type}): {n_subjects} subjects, {n_scans} scan(s) each -> {out_root}"
    )

    rows: list[dict] = []
    n_stacks_total = 0
    with tqdm(total=n_subjects * n_scans, desc="Simulating", unit="scan") as pbar:
        for idx in range(n_subjects):
            sub, ses = dataset.sub_ses[idx]
            anat_dir = _bids_anat_dir(out_root, sub, ses)
            gt_anat_dir = _bids_anat_dir(gt_root, sub, ses)

            for scan_idx in range(n_scans):
                # try:
                data = dataset.sample(idx)
                # except Exception as exc:  # noqa: BLE001
                # tqdm.write(f"WARNING: {sub} {ses} scan {scan_idx} failed: {exc}")
                # pbar.update(1)
                # continue

                res_s = float(data["resolution_slice"])
                s_thick = float(data["slice_thickness"])
                positions = data["positions"]
                noise_log = getattr(scanner, "_noise_sigmas", [])

                # --- GT reference volume (and segmentation) ----------------
                gt_base = _entity_base(sub, ses, img_suffix, n_scans, scan_idx)
                gt_path = gt_anat_dir / f"{gt_base}.nii.gz"
                save_volume(
                    gt_path, data["volume_gt"], res=float(data["resolution_recon"])
                )
                if cfg.save_gt_segmentation and "seg_gt" in data:
                    seg_base = _entity_base(sub, ses, seg_suffix, n_scans, scan_idx)
                    save_volume(
                        gt_anat_dir / f"{seg_base}.nii.gz",
                        data["seg_gt"].float(),
                        res=float(data["resolution_recon"]),
                    )

                # --- GT motion parameters (one .npz/.csv per scan) ----------
                # Per-slice GT poses (all stacks of this scan; `positions[:, 1]`
                # is the stack id). Saved in the same units/format the model
                # outputs at inference, so the point-loss MSE between a model's
                # prediction and this GT can be computed offline. See
                # synthgen.utils.io.save_motion_params.
                motion_base = _entity_base(sub, ses, "motion", n_scans, scan_idx)
                gt_motion_path = anat_dir / f"{motion_base}.npz"
                save_motion_params(
                    gt_motion_path,
                    data["transforms_gt"],
                    positions,
                    data["slice_shape"],
                    data["resolution_slice"],
                    slice_thickness=s_thick,
                    resolution_recon=float(data["resolution_recon"]),
                )
                gt_motion_rel = str(gt_motion_path.relative_to(out_root))

                # --- one file (+ json) per simulated stack -----------------
                for run, sid in enumerate(_stack_indices(positions), start=1):
                    smask = positions[:, 1] == sid
                    n_slices = int(smask.sum())

                    base = _entity_base(
                        sub, ses, img_suffix, n_scans, scan_idx, run=run, stack_id=sid
                    )
                    stack_path = anat_dir / f"{base}.nii.gz"
                    save_stack(
                        data["stacks"][smask],
                        RigidTransform(data["transforms"][smask]),
                        res_s,
                        s_thick,
                        scale=1.0,
                        dtype=np.float32,
                        fname=stack_path,
                    )

                    # matching segmentation stack (same geometry, label dtype)
                    stack_seg_rel = None
                    if cfg.save_stack_segmentations and "stacks_segm" in data:
                        seg_base = _entity_base(
                            sub, ses, seg_suffix, n_scans, scan_idx, run=run, stack_id=sid
                        )
                        stack_seg_path = anat_dir / f"{seg_base}.nii.gz"
                        save_stack(
                            data["stacks_segm"][smask],
                            RigidTransform(data["transforms"][smask]),
                            res_s,
                            s_thick,
                            scale=1.0,
                            dtype=np.int16,
                            fname=stack_seg_path,
                        )
                        stack_seg_rel = str(stack_seg_path.relative_to(out_root))

                    noise_sigma = (
                        noise_log[run - 1]
                        if (run - 1) < len(noise_log)
                        else float("nan")
                    )

                    # per-stack bias / intensity-shift summary (per-slice arrays)
                    bias_strength = float("nan")
                    if "bias_coefs" in data and data["bias_coefs"].ndim == 2:
                        bc = data["bias_coefs"][smask]
                        bias_strength = (
                            float(bc.norm(dim=1).mean()) if bc.numel() else float("nan")
                        )
                    n_int_shift = 0
                    int_shift_mean = 1.0
                    if "int_shifts" in data:
                        ish = data["int_shifts"][smask]
                        shifted = (ish - 1.0).abs() > 1e-6
                        n_int_shift = int(shifted.sum())
                        int_shift_mean = float(ish.mean()) if ish.numel() else 1.0

                    record = {
                        "subject": sub,
                        "session": ses if ses else "",
                        "scan_index": scan_idx + 1,
                        "run": run,
                        "stack_id": sid,
                        "stack_name": stack_path.name,
                        "stack_path": str(stack_path.relative_to(out_root)),
                        "stack_seg_path": stack_seg_rel,
                        "gt_volume_path": str(gt_path.relative_to(out_root)),
                        "gt_motion_path": gt_motion_rel,
                        "n_slices": n_slices,
                        "slice_size_h": int(data["stacks"].shape[-2]),
                        "slice_size_w": int(data["stacks"].shape[-1]),
                        "resolution_slice": res_s,
                        "slice_thickness": s_thick,
                        "gap": float(data["gap"]),
                        "resolution_recon": float(data["resolution_recon"]),
                        "resolution_original": float(data["resolution"]),
                        "noise_sigma": noise_sigma,
                        "bias_strength": bias_strength,
                        "n_slices_intensity_shifted": n_int_shift,
                        "intensity_shift_mean": int_shift_mean,
                        # scanner sampling config (context for the artifacts above)
                        "prob_void": scanner.prob_void,
                        "prob_motion": scanner.prob_motion,
                        "max_missing_slices": scanner.max_missing_slices,
                        "seed": int(cfg.seed) if cfg.get("seed") is not None else None,
                    }
                    rows.append(record)
                    n_stacks_total += 1

                    # sidecar json (BIDS): same info as the CSV row
                    with open(
                        stack_path.with_suffix("").with_suffix(".json"), "w"
                    ) as f:
                        json.dump(record, f, indent=2)

                pbar.update(1)

    # ----- big metadata CSV ------------------------------------------------
    csv_path = deriv / "stacks_metadata.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    log.info(f"Done. {n_stacks_total} stacks for {n_subjects} subjects -> {out_root}")
    log.info(f"Metadata CSV: {csv_path}")


if __name__ == "__main__":
    main()
