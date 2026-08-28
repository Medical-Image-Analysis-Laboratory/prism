#!/usr/bin/env python3
"""CHUV (real-data) comparison table -- slice metrics only.

Real acquisitions have no GT motion or GT volume, so only the slice-reconstruction
metrics (recon re-sliced through the estimated transforms vs the input slices) are
available. Reads each model's
``CHUV/derivatives/predicted_srr/<model>/<ts>/test_T2w_maskout/metrics.csv``
(one row per scan) and renders a booktabs table: one row per model, columns
NCC / PSNR / SSIM (+ brain-masked NCC), as mean +- std (std in \\tiny).

Formatting matches make_table.py: best is bold, second best underlined, and a
``*`` marks the best when it is significantly better than the second best
(one-sided paired Wilcoxon signed-rank on the per-scan values, p < 0.05).

    python make_table_chuv.py       # -> results/chuv_table.tex (+ stdout)
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from make_table import MODELS, _fmt  # reuse model naming + number formatting

# Where `synthgen/inference.py --config-name=inferencechuv` wrote the runs:
# `<PRED>/<model>/<timestamp>/<SUBDIR>/metrics.csv`. Override with
# $PRISM_PREDICTIONS or `--pred-root`.
PRED = Path(os.environ.get("PRISM_PREDICTIONS", "predictions")) / "real"
SUBDIR = "test_T2w_maskout"
RESULTS = Path(__file__).resolve().parent / "results"

# (csv column, display, decimals, higher_is_better)
COLS = [
    ("slice_NCC", "NCC", 3, True),
    ("slice_PSNR", "PSNR", 2, True),
    ("slice_SSIM", "SSIM", 3, True),
    ("slice_NCC_brain", r"NCC$_{\mathrm{brain}}$", 3, True),
]


def load():
    frames = []
    for mid in MODELS:
        csv = next((PRED / mid).glob(f"*/{SUBDIR}/metrics.csv"))
        df = pd.read_csv(csv)
        df["model"] = mid
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def agg_and_test(df):
    """Return vals[(model, col)]=(mean, std) and meta[col]={best,second,sig}."""
    vals, meta = {}, {}
    for col, _disp, _dec, hb in COLS:
        wide = df.pivot_table(index="name", columns="model", values=col)
        means = wide.mean().reindex(list(MODELS))
        for m in MODELS:
            vals[(m, col)] = (float(means[m]), float(wide[m].std()))
        order = means.sort_values(ascending=not hb).index.tolist()
        best, second = order[0], order[1]
        pair = wide[[best, second]].dropna()
        alt = "greater" if hb else "less"
        try:
            p = float(wilcoxon(pair[best].values, pair[second].values,
                               alternative=alt).pvalue)
        except ValueError:
            p = float("nan")
        meta[col] = {"best": best, "second": second, "p": p,
                     "sig": bool(np.isfinite(p) and p < 0.05)}
    return vals, meta


def _cell(mid, col, dec, vals, meta, latex):
    mean, std = vals[(mid, col)]
    s = _fmt(mean, dec)
    m = meta[col]
    if mid == m["best"]:
        if latex:
            s = r"\textbf{" + s + "}" + (r"$^{*}$" if m["sig"] else "")
        else:
            s = f"**{s}**" + ("<sup>\\*</sup>" if m["sig"] else "")
    elif mid == m["second"]:
        s = (r"\underline{" + s + "}") if latex else f"<u>{s}</u>"
    s += (r" {\tiny$\pm$" + _fmt(std, dec) + "}") if latex else f" <sub>±{_fmt(std, dec)}</sub>"
    return s


def build_latex(vals, meta):
    ncol = len(COLS)
    lines = [r"\begin{tabular}{l" + "c" * ncol + "}", r"\toprule"]
    lines.append(r"Method & \multicolumn{%d}{c}{Slice $\uparrow$} \\" % ncol)
    lines.append(r"\cmidrule(lr){2-%d}" % (ncol + 1))
    lines.append(" & " + " & ".join(d for _c, d, _dec, _hb in COLS) + r" \\")
    units = {"PSNR": "(dB)"}
    lines.append(" & " + " & ".join(units.get(d.split("$")[0].strip(), "")
                                    for _c, d, _dec, _hb in COLS) + r" \\")
    lines.append(r"\midrule")
    for mid, disp in MODELS.items():
        cells = [_cell(mid, col, dec, vals, meta, True) for col, _d, dec, _hb in COLS]
        lines.append(f"{disp} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def build_markdown(vals, meta):
    hdr = "| Method | " + " | ".join(f"{d}↑" for _c, d, _dec, _hb in COLS) + " |"
    hdr = hdr.replace(r"NCC$_\text{brain}$", "NCC_brain")
    rows = [hdr, "|" + "---|" * (len(COLS) + 1)]
    for mid, disp in MODELS.items():
        name = disp.replace(r"\textbf{", "").replace("}", "")
        cells = [_cell(mid, col, dec, vals, meta, False) for col, _d, dec, _hb in COLS]
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main():
    df = load()
    vals, meta = agg_and_test(df)
    latex = build_latex(vals, meta)
    out = RESULTS / "chuv_table.tex"
    out.write_text(latex + "\n")
    n = df["name"].nunique()
    print(f"CHUV real-data slice metrics ({n} scans):\n")
    print(build_markdown(vals, meta))
    print(f"\n[wrote LaTeX -> {out}]")
    print("[LaTeX preamble needs: \\usepackage{booktabs}]")
    print("[legend: bold = best, underline = 2nd best, "
          "* = best sig. better than 2nd (Wilcoxon signed-rank, one-sided, p<0.05)]")


if __name__ == "__main__":
    main()
