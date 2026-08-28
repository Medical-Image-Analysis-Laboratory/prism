"""The PRISM model, built in plain Python (no Hydra, no training config).

``configs/experiment/prism/*.yaml`` describe the same architectures for
training; this module is the dependency-free equivalent used by
``scripts/predict_bids.py`` and by anyone who just wants the model:

    >>> from synthgen.models.prism import build_prism
    >>> model = build_prism(weights="prism_weights.pth", device="cuda")

The four variants correspond to the ablation of the paper:

===============  =====================================  ======================
variant          paper                                  components
===============  =====================================  ======================
``svort``        SVoRT (retrained backbone)             transformer only
``svort_q``      SVoRT (+Q)                             + dense SQM
``svort_r``      SVoRT (+R)                             + learned regularizer
``prism``        PRISM                                  + both, + NCC gate
===============  =====================================  ======================
"""

from __future__ import annotations

from typing import Literal, Optional

import torch

from synthgen.models.regularizers import DenoisingResUNet3D
from synthgen.models.sbqm import QualityNet
from synthgen.models.svort import SRRLightning
from synthgen.utils.checkpoint import load_weights

__all__ = ["build_prism", "VARIANTS"]

VARIANTS = ("prism", "svort", "svort_q", "svort_r")

Variant = Literal["prism", "svort", "svort_q", "svort_r"]

# Registration iterations the models were trained with, and the number of
# per-iteration denoisers instantiated for the regularized variants.
N_TRAIN_ITER = 3


def _regularizer(n_denoisers: int) -> torch.nn.ModuleList:
    """The learned prior: one residual 3D denoiser per SVR iteration."""
    return torch.nn.ModuleList(
        [
            DenoisingResUNet3D(
                in_channels=1,
                out_channels=1,
                channels=[16, 32, 64, 128],
                up_sample_mode="pixelshuffle",
                strides=[2, 2, 2],
            )
            for _ in range(n_denoisers)
        ]
    )


def build_prism(
    variant: Variant = "prism",
    weights: Optional[str] = None,
    n_iter: int = 4,
    cg_iter: int = 4,
    device: str | torch.device = "cuda",
    eval_mode: bool = True,
    strict: bool = False,
) -> SRRLightning:
    """Build a PRISM (or ablation) model, optionally loading trained weights.

    Parameters
    ----------
    variant:
        Which model of the ablation to build (see the table above).
    weights:
        Path to the released PRISM weights or to a training checkpoint. The
        perceptual loss stored in a training checkpoint is ignored, so no
        MedicalNet installation is required.
    n_iter:
        Registration iterations to run. The default (4) matches the evaluation
        configs used to produce the reported results; the models were *trained*
        with ``N_TRAIN_ITER`` (3) and the per-iteration denoiser of the last
        trained iteration is reused for any extra iteration.
    cg_iter:
        Conjugate-gradient iterations of each data-consistency solve.
    device:
        Device to move the model to. A CUDA device is required: the forward
        model (slice acquisition / PSF adjoint) is a CUDA extension.
    eval_mode:
        Put the model in ``eval()`` and disable gradients on its parameters.
    strict:
        Fail if ``weights`` does not exactly match the architecture.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")

    use_sqm = variant in ("prism", "svort_q")
    use_reg = variant in ("prism", "svort_r")

    model = SRRLightning(
        n_iter=n_iter,
        cg_iter=cg_iter,
        sqm=QualityNet() if use_sqm else None,
        regularizer=_regularizer(N_TRAIN_ITER) if use_reg else None,
        reg_lambda_init=0.5,
        # One learned lambda per trained iteration.
        n_reg_lambda=N_TRAIN_ITER if use_reg else None,
        # Training-only components: never built for inference.
        perceptual_loss=None,
        tvloss=None,
        gradloss=None,
        # Hard slice rejection is an inference-time extra, not used in the paper.
        inference_mode=False,
    )

    if weights is not None:
        load_weights(model, weights, strict=strict)

    model = model.to(device)
    if eval_mode:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return model
