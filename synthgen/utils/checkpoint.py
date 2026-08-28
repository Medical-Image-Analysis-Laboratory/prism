"""Checkpoint loading helpers that do not depend on the training environment.

A Lightning training checkpoint of :class:`synthgen.models.svort.SRRLightning`
stores ``hyper_parameters`` by value, which for our training configs includes a
*pickled* ``monai.losses.PerceptualLoss`` whose MedicalNet backbone lives in an
external repository (``medicalnet_models``, see
https://github.com/Warvito/MedicalNet-models). Unpickling it therefore fails on
any machine that does not have that repository on ``sys.path`` -- even though
the perceptual loss is only ever used to *train* and plays no role at inference.

:func:`load_weights` sidesteps this: it reads only the tensors, stubbing out any
class it cannot import, and copies them into a model that the caller has already
built (from a config, or from :func:`synthgen.models.prism.build_prism`). It
accepts both a full training checkpoint and a bare ``state_dict`` such as the
released PRISM weights.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable

import torch

__all__ = ["load_state_dict_only", "load_weights"]


# Prefixes that never need to be restored: the perceptual loss is a frozen,
# training-only criterion whose weights come from MedicalNet, not from us.
_SKIP_PREFIXES = ("perceptual_loss.",)

# Modules the checkpoint may reference but that are irrelevant to the weights.
_STUB_MODULES = ("medicalnet_models",)


class _Stub:
    """Placeholder standing in for a class we cannot (and need not) import."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        pass


def _stub_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)

    def __getattr__(attr: str) -> type:  # noqa: N807 - module protocol
        return type(attr, (_Stub,), {"__module__": name})

    mod.__getattr__ = __getattr__  # type: ignore[attr-defined]
    mod.__path__ = []  # type: ignore[attr-defined]  # it may have submodules
    mod.__all__ = []  # type: ignore[attr-defined]
    return mod


class _StubFinder:
    """Import hook fabricating any module under one of ``_STUB_MODULES``.

    Only active while a checkpoint is being read, and only for the listed
    top-level names, so it can never shadow a real installation.
    """

    def __init__(self, roots: Iterable[str]) -> None:
        self.roots = tuple(roots)

    def _matches(self, fullname: str) -> bool:
        return any(
            fullname == r or fullname.startswith(r + ".") for r in self.roots
        )

    def find_module(self, fullname: str, path=None):  # legacy API, py<3.12
        return self if self._matches(fullname) else None

    def find_spec(self, fullname: str, path=None, target=None):
        if not self._matches(fullname):
            return None
        from importlib.machinery import ModuleSpec

        return ModuleSpec(fullname, self, is_package=True)

    def create_module(self, spec):
        return _stub_module(spec.name)

    def exec_module(self, module) -> None:
        return None

    def load_module(self, fullname: str):  # legacy API, py<3.12
        mod = sys.modules.get(fullname) or _stub_module(fullname)
        sys.modules[fullname] = mod
        return mod


def load_state_dict_only(path: str | Path) -> Dict[str, torch.Tensor]:
    """Read the tensors of a checkpoint, ignoring everything else.

    Works for a Lightning checkpoint (``{"state_dict": ...,
    "hyper_parameters": ...}``), for a bare ``state_dict``, and for the original
    SVoRT checkpoints (``{"model": ...}``).
    """
    path = str(path)
    finder = _StubFinder(_STUB_MODULES)
    sys.meta_path.append(finder)
    try:
        ckpt = torch.load(path, map_location="cpu")
    finally:
        sys.meta_path.remove(finder)
        for name in list(sys.modules):
            if finder._matches(name):
                sys.modules.pop(name, None)

    if isinstance(ckpt, dict):
        for key in ("state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    if not isinstance(ckpt, dict):
        raise ValueError(f"{path} does not contain a state_dict")
    return {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}


def load_weights(
    model: torch.nn.Module,
    path: str | Path,
    skip_prefixes: Iterable[str] = _SKIP_PREFIXES,
    strict: bool = False,
    verbose: bool = True,
) -> torch.nn.Module:
    """Copy the weights of ``path`` into an already-built ``model``.

    Parameters
    ----------
    model:
        A model built with the architecture the checkpoint was trained with.
    path:
        Training checkpoint or released ``state_dict``.
    skip_prefixes:
        Parameter-name prefixes to ignore (default: the perceptual loss).
    strict:
        Raise if any weight of ``model`` is left uninitialised, or if the
        checkpoint carries weights the model does not have.
    """
    state = load_state_dict_only(path)
    skip = tuple(skip_prefixes)
    state = {k: v for k, v in state.items() if not k.startswith(skip)}

    own = model.state_dict()
    missing = [
        k for k in own if k not in state and not k.startswith(skip)
    ]
    unexpected = [k for k in state if k not in own]
    mismatched = [
        k for k in state if k in own and own[k].shape != state[k].shape
    ]

    if strict and (missing or unexpected or mismatched):
        raise RuntimeError(
            f"Checkpoint {path} does not match the model:\n"
            f"  missing:    {missing}\n"
            f"  unexpected: {unexpected}\n"
            f"  mismatched: {mismatched}"
        )
    for k in mismatched:
        state.pop(k)

    model.load_state_dict(state, strict=False)
    if verbose:
        print(
            f"Loaded {len(state)} tensors from {Path(path).name} "
            f"({len(missing)} missing, {len(unexpected)} unexpected, "
            f"{len(mismatched)} shape-mismatched)"
        )
        for name, keys in (
            ("missing", missing),
            ("unexpected", unexpected),
            ("mismatched", mismatched),
        ):
            if keys:
                head = ", ".join(keys[:6])
                more = "" if len(keys) <= 6 else f", ... (+{len(keys) - 6})"
                print(f"  {name}: {head}{more}")
    return model
