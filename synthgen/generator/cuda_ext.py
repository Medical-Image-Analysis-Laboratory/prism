"""Import the CUDA extensions of the forward model, JIT-compiling if needed.

``pip install -e .`` (or ``pip install .``) builds ``slice_acq_cuda`` and
``transform_convert_cuda`` ahead of time -- that is the fast path and the only
one that runs here. If the compiled module is not importable (the package was
copied rather than installed, or the extensions were built for another Python /
PyTorch), we fall back to compiling them on first use with
``torch.utils.cpp_extension.load``, which needs ``nvcc`` on the ``PATH`` and
takes ~1 min once; the result is cached in ``$SYNTHGEN_EXT_CACHE`` (default
``~/.cache/synthgen/cuda_ext``).
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import ModuleType
from typing import Sequence

__all__ = ["load_extension"]


def _cache_dir(name: str) -> Path:
    root = os.environ.get("SYNTHGEN_EXT_CACHE")
    if root is None:
        root = Path.home() / ".cache" / "synthgen" / "cuda_ext"
    path = Path(root) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_extension(name: str, sources: Sequence[str | Path]) -> ModuleType:
    """Return the extension ``name``, compiling ``sources`` if it is missing."""
    try:
        return importlib.import_module(name)
    except ImportError:
        pass

    from torch.utils.cpp_extension import load

    missing = [str(s) for s in sources if not Path(s).is_file()]
    if missing:
        raise ImportError(
            f"CUDA extension '{name}' is not installed and its sources are "
            f"missing: {missing}. Install the package with `pip install -e .` "
            "from the repository root."
        )
    return load(
        name=name,
        sources=[str(s) for s in sources],
        build_directory=str(_cache_dir(name)),
        verbose=False,
    )
