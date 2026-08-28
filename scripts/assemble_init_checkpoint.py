#!/usr/bin/env python
"""Assemble the initialisation checkpoint for PRISM's end-to-end fine-tuning.

The three components of PRISM are pre-trained separately (Sec. 3.5 of the
paper), then combined into a single checkpoint that every fine-tuning run starts
from (``ckpt_path=<this file>`` with ``ckpt_load_mode: weights``):

* ``--svort``  the retrained SVoRT backbone            (``experiment=prism/0svort``)
* ``--sqm``    the pre-trained slice-quality module    (``experiment=prism/0sqm``)
* ``--reg``    ONE pre-trained residual denoiser, copied into every
               per-iteration denoiser of the model    (``experiment=prism/0reg1ch``)

Any component may be omitted, in which case whatever the base checkpoint already
holds is kept.

Examples
--------
    python scripts/assemble_init_checkpoint.py \\
        --svort logs/0svort/runs/<date>/checkpoints/epoch_2695_....ckpt \\
        --sqm   logs/0sqm/runs/<date>/checkpoints/epoch_1175_....ckpt \\
        --reg   logs/0reg1ch/runs/<date>/checkpoints/epoch_1299_....ckpt \\
        --n-denoisers 3 \\
        --output prism_init.ckpt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import rootutils
import torch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from synthgen.utils.checkpoint import load_state_dict_only  # noqa: E402

SVORT_PREFIXES = ("svrnet1.", "svrnet2.")
SQM_PREFIXES = ("sqm.",)


def _copy(dst: dict, src: dict, prefixes: tuple[str, ...], what: str) -> int:
    keys = [k for k in src if k.startswith(prefixes)]
    if not keys:
        raise SystemExit(f"no {what} weights (prefixes {prefixes}) in the source")
    for k in keys:
        dst[k] = src[k]
    print(f"  {what}: copied {len(keys)} tensors")
    return len(keys)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--base", type=Path, default=None,
                   help="checkpoint to start from (default: --svort, so the "
                        "backbone run provides the skeleton)")
    p.add_argument("--svort", type=Path, default=None,
                   help="checkpoint of the retrained SVoRT backbone")
    p.add_argument("--sqm", type=Path, default=None,
                   help="checkpoint of the pre-trained slice-quality module")
    p.add_argument("--reg", type=Path, default=None,
                   help="checkpoint of the pre-trained single denoiser; its "
                        "`regularizer.0.*` weights are copied into every "
                        "denoiser of the target model")
    p.add_argument("--n-denoisers", type=int, default=3,
                   help="number of per-iteration denoisers to fill (default: 3)")
    p.add_argument("-o", "--output", type=Path, required=True,
                   help="output checkpoint")
    args = p.parse_args(argv)

    base = args.base or args.svort
    if base is None:
        raise SystemExit("pass --base and/or --svort")

    print(f"Base: {base}")
    state = dict(load_state_dict_only(base))

    if args.svort is not None and args.svort != base:
        print(f"SVoRT backbone: {args.svort}")
        _copy(state, load_state_dict_only(args.svort), SVORT_PREFIXES, "svrnet")

    if args.sqm is not None:
        print(f"SQM: {args.sqm}")
        _copy(state, load_state_dict_only(args.sqm), SQM_PREFIXES, "sqm")

    if args.reg is not None:
        print(f"Regularizer: {args.reg}")
        src = load_state_dict_only(args.reg)
        proto = {
            k[len("regularizer.0."):]: v
            for k, v in src.items()
            if k.startswith("regularizer.0.")
        }
        if not proto:
            raise SystemExit(
                f"no `regularizer.0.*` weights in {args.reg}; was it trained "
                "with `experiment=prism/0reg1ch`?"
            )
        for i in range(args.n_denoisers):
            for k, v in proto.items():
                state[f"regularizer.{i}.{k}"] = v
        print(f"  regularizer: {len(proto)} tensors x {args.n_denoisers} "
              "denoisers")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # A `state_dict`-only checkpoint: `ckpt_load_mode: weights` loads it
    # non-strictly with a fresh optimizer, which is what fine-tuning wants.
    torch.save({"state_dict": state}, args.output)
    print(f"\nWrote {args.output} ({len(state)} tensors, "
          f"{args.output.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
