from dotenv import load_dotenv

load_dotenv(".env", override=True)  # loads .env from current working directory

from typing import Any, Dict, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig
import warnings
from pathlib import Path
from tqdm import tqdm
from monai.metrics import MSEMetric, PSNRMetric, SSIMMetric
from synthgen.generator.slice_acquisition import slice_acquisition
from synthgen.utils.checkpoint import load_weights
import numpy as np
from synthgen.generator.transform.transform import RigidTransform
from synthgen.utils.io import save_stack, save_volume, save_motion_params

import pandas as pd

# suppress warnings
warnings.filterwarnings("ignore")

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from synthgen.utils.lightning import (
    RankedLogger,
    extras,
    get_metric_value,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def experiment_name(ckpt_path: str) -> str:
    """A short, stable name for a checkpoint's run, used in the output paths.

    Training runs are laid out as ``<task_name>/runs/<timestamp>/checkpoints/
    <file>.ckpt``, in which case the name is ``<task_name>/<timestamp>``; any
    other layout falls back to the file name.
    """
    parts = Path(ckpt_path).parts
    if len(parts) >= 5 and parts[-2] == "checkpoints" and parts[-4] == "runs":
        return f"{parts[-5]}/{parts[-3]}"
    return Path(ckpt_path).stem


# Near-empty slices (<= this many brain voxels in the GT/input slice) are kept in
# the model input and in the saved motion (so `positions` stays 1:1 with the GT
# motion), but are NOT scored -- mirrors the old InferenceStackDataset filter.
EMPTY_SLICE_MIN_VOXELS = 100


# --------------------------------------------------------------------------- #
#  Metric helpers. `scripts/predict_bids.py` computes the same quantities the
#  same way -- edit both together.
# --------------------------------------------------------------------------- #
def ncc(a, b, mask=None, eps: float = 1e-8) -> float:
    """Global normalized cross-correlation in [-1, 1] (over `mask` if given)."""
    a = a.flatten().float()
    b = b.flatten().float()
    if mask is not None:
        m = mask.flatten() > 0
        if m.sum() < 2:
            return float("nan")
        a, b = a[m], b[m]
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm() + eps
    return float((a * b).sum() / denom)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """

    mse = MSEMetric()
    psnr = PSNRMetric(1.0)
    ssim = SSIMMetric(3)
    ssim_slice = SSIMMetric(2)
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Model: <{cfg.model._target_}>")

    with torch.no_grad():
        ckpt_paths = cfg.get("ckpt_paths", [])
        if not ckpt_paths:
            raise ValueError(
                "No checkpoint to evaluate: pass `ckpt_paths=[/path/to.ckpt]`."
            )
        for ckpt_path in ckpt_paths:
            exp_name = experiment_name(ckpt_path)
            log.info(f"Loading model weights from {ckpt_path}")
            # Weights are copied into the config-built model rather than
            # restored with `load_from_checkpoint`: a training checkpoint also
            # pickles the (training-only) perceptual loss, whose MedicalNet
            # backbone would otherwise have to be installed just to evaluate.
            model = hydra.utils.instantiate(cfg.model)
            load_weights(model, ckpt_path)
            model.eval()
            # move model to device
            model.to(cfg.device)
            out_pred = (
                Path(cfg["save_path"])
                / f"{exp_name}/{cfg.data.test_split}_{cfg.data.img_suffix}"
            )
            out_pred.mkdir(parents=True, exist_ok=True)
            log.info(f"Testing {exp_name} on test split {cfg.data.test_split}")
            log.info(f"Saving predictions to {out_pred}")

            datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
            model.CG.n_iter = cfg.model.cg_iter
            model.n_iter = cfg.model.n_iter
            valdl = datamodule.test_dataloader()
            valdl_len = len(valdl.dataset)
            valsplit_df = []
            for valbactch in tqdm(
                valdl, total=valdl_len, desc=f"Test {exp_name} on {cfg.data.test_split}"
            ):
                name = valbactch["name"][0]
                ses = name.split("_")[1]
                name = name.split("_")[0]
                subfolder = out_pred / f"{name}/{ses}/anat"
                subfolder.mkdir(exist_ok=True, parents=True)
                with torch.no_grad():
                    for key in valbactch.keys():
                        if isinstance(valbactch[key], torch.Tensor):
                            valbactch[key] = valbactch[key].to(cfg.device)
                    # try:
                    res = model.predict_step(valbactch, 0)
                    # except RuntimeError as e:
                    #     log.error(f"RuntimeError for {name}: {e}")
                    #     continue
                    transforms_pred = res["transforms_pred"][-1].to("cuda")

                    # save validation step metrics
                    res_metrics = {
                        "name": valbactch["name"][0],
                        "experiment": exp_name,
                    }

                    # PART 1: SRR metrics. Real inference usually has no GT
                    # volume, so these run only when one is present -- computed
                    # on the raw volumes (NCC is itself
                    # scale/shift-invariant).
                    # `dc`: the plain data-consistency solve at the final
                    # poses (Eq. 2) -- the reconstruction the paper reports.
                    # `regularized`: the prior-regularized volume that the SVR
                    # loop refines the poses against.
                    if cfg.get("recon", "dc") == "dc":
                        pred_volume = res["volume_dc"].detach().float().cpu()
                    else:
                        pred_volume = res["recon_volume"].unsqueeze(0)
                    if res.get("gt_volume") is not None:
                        gt_volume = res["gt_volume"].unsqueeze(0)
                        res_metrics.update(
                            {
                                "MSE": mse(y_pred=pred_volume, y=gt_volume).item(),
                                "PSNR": psnr(y_pred=pred_volume, y=gt_volume).item(),
                                "SSIM": ssim(y_pred=pred_volume, y=gt_volume).item(),
                                "NCC": ncc(pred_volume, gt_volume),
                                "NCC_brain": ncc(
                                    pred_volume, gt_volume, mask=gt_volume > 0
                                ),
                            }
                        )

                    # PART 2: Slice-wise metrics
                    recon_slices = slice_acquisition(
                        transforms=transforms_pred,
                        vol=pred_volume.to(
                            "cuda"
                        ),  # The 3D volume you want to project from
                        vol_mask=None,
                        slices_mask=None,
                        psf=res["psf_acq"].to("cuda"),  # The blurring kernel
                        slice_shape=res["slice_shape"],
                        res_slice=res["resolution_slice"]
                        / res[
                            "resolution_recon"
                        ],  # Ratio of slice res to reconstruction res
                        need_weight=False,
                        interp_psf=False,
                    ).cpu()
                    # check if recon slices have nans

                    gt_slices = res["stacks"][:, :, :, ...].cpu()
                    # print(gt_slices.shape, recon_slices.shape)
                    slice_metrics = []
                    # calculate mse, psnr, ssim for each slice
                    for i in range(recon_slices.shape[0]):
                        pred_slice = recon_slices[i].unsqueeze(0)
                        stack_idx = res["positions"][i, 1].item()
                        gt_slice = gt_slices[i].unsqueeze(0)
                        gt_nonzero = int(torch.sum(gt_slice > 0).item())
                        # skip near-empty slices (not scored, but still part of the
                        # input/motion output so `positions` matches the GT motion)
                        if gt_nonzero <= EMPTY_SLICE_MIN_VOXELS:
                            continue
                        mse_metric = mse(y_pred=pred_slice, y=gt_slice).item()
                        psnr_metric = psnr(y_pred=pred_slice, y=gt_slice).item()
                        ssim_metric = ssim_slice(y_pred=pred_slice, y=gt_slice).item()
                        slice_metrics.append(
                            {
                                "stack_index": stack_idx,
                                # GT-comparable slice index within the stack
                                # (centered; matches the GT motion positions)
                                "slice_in_scan": int(res["positions"][i, 0].item()),
                                "name": valbactch["name"][0],
                                "exp_name": exp_name,
                                "slice_index": i,
                                "MSE": mse_metric,
                                "PSNR": psnr_metric,
                                "SSIM": ssim_metric,
                                "NCC": ncc(pred_slice, gt_slice),  # full slice
                                "NCC_brain": ncc(  # over GT-brain voxels only
                                    pred_slice, gt_slice, mask=gt_slice > 0
                                ),
                                "GT_Slice_nonzero_voxels": gt_nonzero,
                            }
                        )
                    # save as dataframes
                    slice_metrics_df = pd.DataFrame(slice_metrics)
                    slice_metrics_df.to_csv(
                        subfolder / f"{name}_slice_metrics.csv", index=False
                    )
                    # Per-scan/session aggregate (mean) of the slice metrics ->
                    # one row per scan in the root metrics.csv. For real data
                    # (no GT volume) these are the only metrics available.
                    if slice_metrics:
                        res_metrics.update(
                            {
                                f"slice_{k}": float(slice_metrics_df[k].mean())
                                for k in ("MSE", "PSNR", "SSIM", "NCC", "NCC_brain")
                            }
                        )
                        res_metrics["n_slices"] = len(slice_metrics)
                    valsplit_df.append(res_metrics)
                    # PART 2b: Save estimated motion (and the GT used) per scan.
                    # Same units/format as generate_stack_database's GT motion, so
                    # the point-loss MSE between model versions can be computed
                    # offline:  ((pred["points"] - gt["points"]) ** 2).mean().
                    # See synthgen.utils.io.save_motion_params.
                    if cfg.get("save_motion", True):
                        save_motion_params(
                            subfolder / f"{name}_{ses}_motion-pred.npz",
                            res["transforms_pred"][-1],
                            res["positions"],
                            res["slice_shape"],
                            res["resolution_slice"],
                            slice_thickness=res["slice_thickness"],
                            resolution_recon=res["resolution_recon"],
                        )
                        if res.get("transforms_gt") is not None:
                            save_motion_params(
                                subfolder / f"{name}_{ses}_motion-gt.npz",
                                res["transforms_gt"],
                                res["positions"],
                                res["slice_shape"],
                                res["resolution_slice"],
                                slice_thickness=res["slice_thickness"],
                                resolution_recon=res["resolution_recon"],
                            )
                    # PART 3: Save volumes and stacks
                    if cfg.get("save_volumes", False):
                        # save pred and gt volumes
                        save_volume(
                            subfolder / f"{name}_{ses}_type-SRRest_T2w.nii.gz",
                            pred_volume,
                            res=res["resolution_recon"],
                        )

                    if cfg.get("save_slices", False):
                        # save simulated slices as volume
                        save_volume(
                            subfolder / f"{name}_{ses}_type-GTslices_T2wslices.nii.gz",
                            gt_slices[:, 0, ...].unsqueeze(0).unsqueeze(0),
                            res=[
                                res["resolution_slice"],
                                res["resolution_slice"],
                                res["slice_thickness"],
                            ],
                        )

                        save_volume(
                            subfolder / f"{name}_{ses}_type-ESTslices_T2wslices.nii.gz",
                            recon_slices[:, 0, ...].unsqueeze(0).unsqueeze(0),
                            res=[
                                res["resolution_slice"],
                                res["resolution_slice"],
                                res["slice_thickness"],
                            ],
                        )
                    if cfg.get("save_stacks", False):
                        # save the stacks
                        stacks_ids = res["positions"].cpu().numpy()[:, 1]
                        for i, s_id in enumerate(np.unique(stacks_ids)):
                            idx = stacks_ids == s_id
                            save_stack(
                                res["stacks"][idx],
                                RigidTransform(transforms_pred[idx]),
                                res["resolution_slice"],
                                res["slice_thickness"],
                                scale=1.0,
                                dtype=np.float32,
                                fname=subfolder
                                / f"{name}_{ses}_type-stackINP_stackid-{int(s_id)}_run-{i}_T2wstack.nii.gz",
                            )

                        # save stacks based on the reconstructed slices
                        stacks_ids = res["positions"].cpu().numpy()[:, 1]
                        for i, s_id in enumerate(np.unique(stacks_ids)):
                            idx = stacks_ids == s_id
                            save_stack(
                                recon_slices[idx],
                                RigidTransform(transforms_pred[idx]),
                                res["resolution_slice"],
                                res["slice_thickness"],
                                scale=1.0,
                                dtype=np.float32,
                                fname=subfolder
                                / f"{name}_{ses}_type-stackEST_stackid-{int(s_id)}_run-{i}_T2w_stack.nii.gz",
                            )
            valsplit_df = pd.DataFrame(valsplit_df)
            valsplit_df.to_csv(
                out_pred / "metrics.csv",
                index=False,
            )
            # save cfg used for evaluation
            cfg_savepath = out_pred / "eval_config.yaml"
            with open(cfg_savepath, "w") as f:
                f.write(str(dict(cfg)))
    return {}, {}


@hydra.main(version_base="1.3", config_path="../configs", config_name="inferencechuv")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
