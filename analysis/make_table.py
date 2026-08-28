#!/usr/bin/env python3
"""Render the model-comparison results as a LaTeX table (like the paper figure).

Reads ``results/{slice,motion,volume}_metrics.csv`` (produced by build_metrics.py)
and writes a booktabs table grouped by motion split (Good/Med/Bad), one row per
model, with columns:

    Point (mm) | Trans (mm) | Rot (deg) | Slice: NCC/PSNR/SSIM | Volume: NCC/PSNR/SSIM

Motion columns are errors (lower = better); slice/volume are similarities
(higher = better, except MSE which is not shown). Per (split, column) the best
model is bold, the second best underlined, and a ``*`` marks the best when it is
significantly better than the second best -- a one-sided paired Wilcoxon
signed-rank test on the per-scan values (models share the same scans), p < 0.05.
Also prints a plain-markdown version to stdout.

    python make_table.py            # -> results/comparison_table.tex (+ stdout)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RESULTS = Path(__file__).resolve().parent / "results"

# model id -> display name (paper naming), in the order they should appear
MODELS = {
    "1svort_4cgit": "SVoRT",
    "1svort_sqm": r"SVoRT\,(+Q)",
    "1svort_reg": r"SVoRT\,(+R)",
    "1svort_sqm_reg3": r"\textbf{PRISM}",
}
DATASETS = ["Good", "Med", "Bad"]

# (source df, column, display, decimals, higher_is_better)
COLS = [
    ("motion", "point_rmse", "Point", 2, False),
    ("motion", "trans_err_mean", "Trans", 2, False),
    ("motion", "rot_err_mean", "Rot", 2, False),
    ("slice", "slice_NCC", "NCC", 3, True),
    ("slice", "slice_PSNR", "PSNR", 2, True),
    ("slice", "slice_SSIM", "SSIM", 3, True),
    ("volume", "vol_NCC", "NCC", 3, True),
    ("volume", "vol_PSNR", "PSNR", 2, True),
    ("volume", "vol_SSIM", "SSIM", 3, True),
]


def load():
    slice_df = pd.read_csv(RESULTS / "slice_metrics.csv")
    motion_df = pd.read_csv(RESULTS / "motion_metrics.csv")
    volume_df = pd.read_csv(RESULTS / "volume_metrics.csv")
    if "ok" in volume_df:
        volume_df = volume_df[volume_df["ok"] == True]  # noqa: E712
    return {"slice": slice_df, "motion": motion_df, "volume": volume_df}


def agg_table(dfs):
    """Return {(dataset, model, col): (mean, std)}."""
    vals = {}
    for src, col, _disp, _dec, _hb in COLS:
        pm = dfs[src].pivot_table(col, "dataset", "model", "mean", observed=True)
        ps = dfs[src].pivot_table(col, "dataset", "model", "std", observed=True)
        for ds in DATASETS:
            for m in MODELS:
                vals[(ds, m, col)] = (float(pm.loc[ds, m]), float(ps.loc[ds, m]))
    return vals


def _per_scan_wide(dfs, ds, src, col):
    """Per-scan values for one metric in one split: index=name, columns=model."""
    d = dfs[src]
    d = d[d["dataset"] == ds]
    return d.pivot_table(index="name", columns="model", values=col)


def rank_meta(dfs):
    """For each (dataset, col): best/second model (by mean) + paired test.

    Best vs second-best are evaluated on the SAME scans, so we use a one-sided
    Wilcoxon signed-rank test (best is better) on the per-scan paired values.
    """
    meta = {}
    for src, col, _disp, _dec, hb in COLS:
        for ds in DATASETS:
            wide = _per_scan_wide(dfs, ds, src, col)
            means = wide.mean().reindex(list(MODELS))
            order = means.sort_values(ascending=not hb).index.tolist()  # best first
            best, second = order[0], order[1]
            pair = wide[[best, second]].dropna()
            alt = "greater" if hb else "less"  # H1: best is better than second
            try:
                p = float(wilcoxon(pair[best].values, pair[second].values,
                                   alternative=alt).pvalue)
            except ValueError:
                p = float("nan")
            meta[(ds, col)] = {"best": best, "second": second, "p": p,
                               "sig": bool(np.isfinite(p) and p < 0.05),
                               "n": int(len(pair))}
    return meta


def _fmt(v, dec):
    return f"{v:.{dec}f}"


def build_latex(vals, meta):
    lines = []
    lines.append(r"\begin{tabular}{ll" + "ccc" + "ccc" + "ccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Split & Method & \multicolumn{3}{c}{Motion $\downarrow$} & "
        r"\multicolumn{3}{c}{Slice $\uparrow$} & "
        r"\multicolumn{3}{c}{Volume $\uparrow$} \\"
    )
    lines.append(r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}")
    lines.append(
        r" &  & Point & Trans & Rot & NCC & PSNR & SSIM & NCC & PSNR & SSIM \\"
    )
    lines.append(
        r" &  & (mm) & (mm) & (deg) &  & (dB) &  &  & (dB) &  \\"
    )
    lines.append(r"\midrule")

    for di, ds in enumerate(DATASETS):
        for mi, (mid, disp) in enumerate(MODELS.items()):
            cells = []
            for src, col, _d, dec, hb in COLS:
                mean, std = vals[(ds, mid, col)]
                m = meta[(ds, col)]
                s = _fmt(mean, dec)
                if mid == m["best"]:
                    s = r"\textbf{" + s + "}"
                    if m["sig"]:                       # sig. better than 2nd best
                        s += r"$^{*}$"
                elif mid == m["second"]:
                    s = r"\underline{" + s + "}"
                # std in tiny, non-bold, after the (possibly bold) mean
                s += r" {\tiny$\pm$" + _fmt(std, dec) + "}"
                cells.append(s)
            split_cell = (
                r"\multirow{4}{*}{" + ds + "}" if mi == 0 else ""
            )
            lines.append(f"{split_cell} & {disp} & " + " & ".join(cells) + r" \\")
        if di < len(DATASETS) - 1:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def build_markdown(vals, meta):
    hdr = ("| Split | Method | Point↓ | Trans↓ | Rot↓ | Slice NCC↑ | Slice PSNR↑ "
           "| Slice SSIM↑ | Vol NCC↑ | Vol PSNR↑ | Vol SSIM↑ |")
    sep = "|" + "---|" * 11
    rows = [hdr, sep]
    for ds in DATASETS:
        for mi, (mid, disp) in enumerate(MODELS.items()):
            name = disp.replace(r"\textbf{", "").replace("}", "")
            cells = []
            for src, col, _d, dec, hb in COLS:
                mean, std = vals[(ds, mid, col)]
                m = meta[(ds, col)]
                s = _fmt(mean, dec)
                if mid == m["best"]:
                    s = f"**{s}**" + ("<sup>\\*</sup>" if m["sig"] else "")
                elif mid == m["second"]:
                    s = f"<u>{s}</u>"
                s += f" <sub>±{_fmt(std, dec)}</sub>"
                cells.append(s)
            split = ds if mi == 0 else ""
            rows.append(f"| {split} | {name} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main():
    dfs = load()
    vals = agg_table(dfs)
    meta = rank_meta(dfs)
    latex = build_latex(vals, meta)
    out = RESULTS / "comparison_table.tex"
    out.write_text(latex + "\n")
    print(build_markdown(vals, meta))
    print(f"\n[wrote LaTeX -> {out}]")
    print("[LaTeX preamble needs: \\usepackage{booktabs, multirow}]")
    print("[legend: bold = best, underline = 2nd best, "
          "* = best sig. better than 2nd (Wilcoxon signed-rank, one-sided, p<0.05)]")


if __name__ == "__main__":
    main()
