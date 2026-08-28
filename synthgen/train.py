from dotenv import load_dotenv
import sys
import os

# The 3D perceptual loss used during training is MONAI's `PerceptualLoss` with a
# MedicalNet ResNet-10 backbone, which MONAI fetches through `torch.hub` from
# https://github.com/Warvito/MedicalNet-models. Set MEDICALNET_REPO to a local
# clone to run offline.
_medicalnet_repo = os.environ.get("MEDICALNET_REPO")
if _medicalnet_repo:
    sys.path.append(os.path.abspath(_medicalnet_repo))

load_dotenv(".env", override=True)  # loads .env from current working directory

from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from typing import Sequence
from torch.nn import Module
from synthgen.utils.svortutils import (
    load_svort_weights,
)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from synthgen.utils.checkpoint import load_state_dict_only
from synthgen.utils.lightning import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # if cfg.get("ckpt_path"):
    #     log.info(f"Loading model weights (partial) from <{cfg.ckpt_path}>")
    #     load_checkpoint_weights(
    #         model=model,
    #         ckpt_path=cfg.ckpt_path,
    #         map_location="cuda" if torch.cuda.is_available() else "cpu",
    #         # anything you were already removing, e.g. your final conv layer:
    #         # ignore_layers=["net.model.2.0.conv"],
    #     )

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # --- Checkpoint loading: toggle full-resume vs. weights-only ---------------
    # ckpt_load_mode:
    #   "resume"  -> full checkpoint resume (model + optimizer + scheduler + step;
    #                strict). Use to continue an interrupted run of the SAME model.
    #   "weights" -> load only the model state_dict (non-strict) and start a FRESH
    #                fit. Use to finetune a stage that changed the architecture
    #                (e.g. added the regconf head) from an earlier checkpoint.
    ckpt_path = cfg.get("ckpt_path") or None
    load_mode = cfg.get("ckpt_load_mode", "resume")
    resume_ckpt = None
    if ckpt_path:
        if load_mode == "weights":
            log.info(f"Loading model WEIGHTS only (non-strict) from <{ckpt_path}>")
            load_state_dict_weights(model, ckpt_path)
        elif load_mode == "resume":
            log.info(f"Full RESUME from <{ckpt_path}> (model+optim+sched+step)")
            resume_ckpt = ckpt_path
        else:
            raise ValueError(
                f"Unknown ckpt_load_mode '{load_mode}' (expected 'resume' or 'weights')"
            )

    # Optional: Lightning learning-rate range test to pick a starting LR.
    tune_cfg = cfg.get("tune_lr")
    if tune_cfg and tune_cfg.get("enable"):
        from lightning.pytorch.tuner import Tuner

        log.info("Running learning-rate finder (Tuner.lr_find)...")
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(
            model,
            train_dataloaders=datamodule.train_dataloader(),
            val_dataloaders=datamodule.val_dataloader(),
            min_lr=float(tune_cfg.get("min_lr", 1e-6)),
            max_lr=float(tune_cfg.get("max_lr", 1e-1)),
            num_training=int(tune_cfg.get("num_training", 100)),
            mode=tune_cfg.get("mode", "exponential"),
            update_attr=False,  # we set model.lr explicitly below
        )
        suggested_lr = lr_finder.suggestion() if lr_finder is not None else None
        log.info(f"LR finder suggested lr={suggested_lr}")

        plot_path = tune_cfg.get("plot_path")
        if plot_path and lr_finder is not None:
            try:
                fig = lr_finder.plot(suggest=True)
                fig.savefig(plot_path)
                log.info(f"Saved LR finder plot to <{plot_path}>")
            except Exception as e:  # plotting is best-effort (headless backends etc.)
                log.warning(f"Could not save LR finder plot: {e}")

        if tune_cfg.get("update_attr", True) and suggested_lr is not None:
            model.lr = suggested_lr
            model.hparams.lr = suggested_lr
            log.info(f"Set model.lr = {suggested_lr} for training")

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(
            model=model,
            train_dataloaders=datamodule.train_dataloader(),
            val_dataloaders=datamodule.val_dataloader(),
            # None in "weights" mode (fresh optimizer/scheduler); the checkpoint
            # path only when doing a full resume.
            ckpt_path=resume_ckpt,
        )
    train_metrics = trainer.callback_metrics
    print(train_metrics)
    return train_metrics


def load_checkpoint_weights(
    model: Module,
    ckpt_path: str,
    ignore_layers: Sequence[str] = (),
    map_location: str = "cuda",
    fine_tune_mode: str = "full",
) -> None:
    """
    Load matching weights from ckpt_path into model, plus:
      • For any layer whose only shape mismatch is in dim0, slice the checkpoint
        and load the first model_shape[0] entries.
      • Skip & warn on any other mismatch.
    Then apply the requested fine_tune_mode ("head" or "full") by toggling requires_grad.
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)
    state_dict = ckpt.get("state_dict", ckpt)
    model_dict = model.state_dict()

    loaded, skipped = [], []

    for name, param in state_dict.items():
        # 1) skip user-ignored layers
        if any(ign in name for ign in ignore_layers):
            skipped.append((name, f"ignored by pattern {ignore_layers}"))
            continue

        # 2) not in model
        if name not in model_dict:
            skipped.append((name, "not found in model"))
            continue

        mparam = model_dict[name]
        # 3) exact match → copy
        if param.shape == mparam.shape:
            model_dict[name] = param
            loaded.append(name)
            continue

        # 4) special slice case: only dim0 differs, others match
        if (
            param.ndim >= 1
            and param.shape[1:] == mparam.shape[1:]
            and param.shape[0] > mparam.shape[0]
        ):
            # copy just the first mparam.shape[0] output‐channels
            sliced = param[: mparam.shape[0], ...].clone()
            model_dict[name] = sliced
            loaded.append(name + " (sliced dim0)")
            continue

        # 4.75) special slice case: only dim1 differs, others match
        if (
            param.ndim >= 1
            and param.shape[2:] == mparam.shape[2:]
            and param.shape[0] == mparam.shape[0]
            and param.shape[0] >= mparam.shape[0]
        ):
            # copy just the first mparam.shape[0] output‐channels
            sliced = param[:, : mparam.shape[1], ...].clone()
            model_dict[name] = sliced
            loaded.append(name + " (sliced dim1)")
            continue

        # 4.5) special slice case: only dim0 and dim1 differs, others match
        if (
            param.ndim >= 1
            and param.shape[2:] == mparam.shape[2:]
            and param.shape[0] > mparam.shape[0]
        ):
            # copy just the first mparam.shape[0] output‐channels
            sliced = param[: mparam.shape[0], : mparam.shape[1], ...].clone()
            model_dict[name] = sliced
            loaded.append(name + " (sliced dim1 dim2)")
            continue

        # 5) anything else → skip
        skipped.append(
            (
                name,
                f"shape mismatch ckpt {tuple(param.shape)} vs model {tuple(mparam.shape)}",
            )
        )

    # actually load into model
    model.load_state_dict(model_dict)

    # report
    print(f"✔️ Loaded {len(loaded)} parameters:")
    for n in loaded:
        print(f"   • {n}")
    print(f"⚠️  Skipped {len(skipped)} parameters:")
    for n, reason in skipped:
        print(f"   • {n}: {reason}")


def load_state_dict_weights(
    model: Module, ckpt_path: str, map_location: str = "cpu"
) -> Module:
    """Load ONLY the model weights from a checkpoint, non-strict.

    Use this to finetune a new stage that changed the architecture (e.g. added a
    head): matching parameters are restored, any new/changed parameters keep
    their fresh init, and training starts with a fresh optimizer/scheduler (no
    `ckpt_path` passed to `trainer.fit`). Logs missing/unexpected keys so the
    architecture delta is visible.
    """
    state_dict = load_state_dict_only(ckpt_path)
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing:
        log.info(
            f"weights-load: {len(missing)} missing (kept fresh-init) keys, "
            f"e.g. {missing[:6]}"
        )
    if unexpected:
        log.info(
            f"weights-load: {len(unexpected)} unexpected (ignored) keys, "
            f"e.g. {unexpected[:6]}"
        )
    if not missing and not unexpected:
        log.info("weights-load: all keys matched exactly.")
    return model


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
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
