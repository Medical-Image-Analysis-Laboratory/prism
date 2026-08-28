#!/usr/bin/env python
"""Turn a training checkpoint into a small, portable weights file.

A Lightning training checkpoint is ~1 GB: it carries optimizer and scheduler
state, callback state, and ``hyper_parameters`` that *pickle* the training-only
perceptual loss (whose MedicalNet backbone lives in an external repository, so
the file cannot even be opened without it).

This script keeps only the model tensors, so the result is ~350 MB, loads with
``torch.load(..., weights_only=True)`` anywhere, and is what should be published
alongside the paper.

Examples
--------
    python scripts/export_release_checkpoint.py \\
        logs/1svort_sqm_reg3/runs/<date>/checkpoints/epoch_987_....ckpt \\
        prism_weights.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import rootutils
import torch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from synthgen.utils.checkpoint import load_state_dict_only  # noqa: E402

# Training-only parameters that inference never uses.
DROP_PREFIXES = ("perceptual_loss.",)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("checkpoint", type=Path, help="training .ckpt to export")
    p.add_argument("output", type=Path, help="output .pth")
    p.add_argument("--keep-perceptual", action="store_true",
                   help="keep the perceptual-loss weights as well")
    p.add_argument("--half", action="store_true",
                   help="store float16 weights (half the size, slightly lossy)")
    args = p.parse_args(argv)

    if not args.checkpoint.is_file():
        raise SystemExit(f"no such checkpoint: {args.checkpoint}")

    state = load_state_dict_only(args.checkpoint)
    if not args.keep_perceptual:
        state = {
            k: v for k, v in state.items() if not k.startswith(DROP_PREFIXES)
        }
    if args.half:
        state = {
            k: (v.half() if v.is_floating_point() else v)
            for k, v in state.items()
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)

    n_params = sum(v.numel() for v in state.values())
    print(
        f"{args.checkpoint.name} "
        f"({args.checkpoint.stat().st_size / 1e6:.0f} MB)\n"
        f"  -> {args.output} "
        f"({args.output.stat().st_size / 1e6:.0f} MB, "
        f"{len(state)} tensors, {n_params / 1e6:.1f}M parameters)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
