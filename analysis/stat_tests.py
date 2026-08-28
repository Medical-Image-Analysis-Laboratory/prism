#!/usr/bin/env python3
"""Paired statistical tests for the simulated-dHCP model comparison.

All four models are evaluated on the SAME scans per split, so comparisons are
paired. For every (split, metric) we test the proposed model
``1svort_sqm_reg3`` (+Quality+Reg) against each baseline with a two-sided
Wilcoxon signed-rank test on the per-scan values, and report:

    Δ            mean(proposed − baseline)          (sign is in the metric's raw units)
    better       is the proposed model better?      (accounts for metric direction)
    win%         fraction of scans proposed wins
    dz           paired Cohen's d = mean(diff)/std(diff)
    p            raw p-value
    p_holm       Holm-corrected across the whole family of tests
    sig          p_holm < 0.05 AND proposed is the better one

Outputs ``results/stat_tests.csv`` and prints a compact per-metric summary.
Motion metrics are identical after the CG-only re-run, but are included for
completeness. Full pairwise (all 6 model pairs) is printed for the headline
metrics when run with ``--pairwise``.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from make_table import load, _per_scan_wide, COLS, DATASETS, MODELS

PROPOSED = "1svort_sqm_reg3"
BASELINES = [m for m in MODELS if m != PROPOSED]
SHORT = {"1svort_4cgit": "+4CG", "1svort_reg": "+Reg",
         "1svort_sqm": "+Quality", "1svort_sqm_reg3": "+Quality+Reg"}
FAMILY = {"motion": "Motion", "slice": "Slice", "volume": "Volume"}


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adj[idx] = running
    return adj


def paired_test(x, y, higher_better):
    """proposed=x vs baseline=y (paired, aligned, NaN-free). Returns dict."""
    diff = x - y
    n = len(diff)
    try:
        p = float(wilcoxon(x, y).pvalue)          # two-sided, drops zero-diffs
    except ValueError:
        p = np.nan
    better_dir = (diff > 0) if higher_better else (diff < 0)
    return {
        "n": n,
        "delta": float(diff.mean()),
        "proposed_better": bool((diff.mean() > 0) == higher_better),
        "win_pct": float(better_dir.mean() * 100),
        "dz": float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else np.nan,
        "p": p,
    }


def run(dfs):
    rows = []
    for src, col, disp, dec, hb in COLS:
        for ds in DATASETS:
            wide = _per_scan_wide(dfs, ds, src, col)
            for b in BASELINES:
                pair = wide[[PROPOSED, b]].dropna()
                r = paired_test(pair[PROPOSED].values, pair[b].values, hb)
                r.update({"family": FAMILY[src], "metric": disp, "dataset": ds,
                          "baseline": SHORT[b],
                          "mean_proposed": float(wide[PROPOSED].mean()),
                          "mean_baseline": float(wide[b].mean()), "dec": dec})
                rows.append(r)
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"].fillna(1.0).values)
    out["sig"] = (out["p_holm"] < 0.05) & out["proposed_better"]
    return out


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def print_summary(out):
    forder = {"Motion": 0, "Slice": 1, "Volume": 2}
    metrics = (out[["family", "metric"]].drop_duplicates()
               .sort_values(by=["family", "metric"],
                            key=lambda s: s.map(forder) if s.name == "family" else s))
    print(f"\nProposed = +Quality+Reg. Cells: Δ  win%  p_holm(stars). "
          f"'better' sign already applied; ns = not significant after Holm.\n")
    for _, mr in metrics.iterrows():
        fam, met = mr["family"], mr["metric"]
        print(f"### {fam} · {met}")
        sub = out[(out.family == fam) & (out.metric == met)]
        hdr = f"{'split':5s} | " + " | ".join(f"vs {b:12s}" for b in
                                              [SHORT[x] for x in BASELINES])
        print(hdr)
        for ds in DATASETS:
            cells = []
            for b in [SHORT[x] for x in BASELINES]:
                r = sub[(sub.dataset == ds) & (sub.baseline == b)].iloc[0]
                dec = int(r["dec"])
                sign = "+" if r["proposed_better"] else "−"
                cells.append(f"{sign}{abs(r['delta']):.{dec}f} {r['win_pct']:3.0f}% "
                             f"{stars(r['p_holm']):>3s}")
            print(f"{ds:5s} | " + " | ".join(f"{c:15s}" for c in cells))
        print()


def print_pairwise(dfs, keys):
    print("\n=== Full pairwise (win% of ROW over COL; * = ROW Holm-sig better) ===")
    for src, col, disp, hb in keys:
        for ds in DATASETS:
            wide = _per_scan_wide(dfs, ds, src, col)
            names = list(MODELS)
            pairs = [(a, c) for a in names for c in names if a != c]
            cells = {}
            for a, c in pairs:
                pr = wide[[a, c]].dropna()
                cells[(a, c)] = paired_test(pr[a].values, pr[c].values, hb)
            padj = dict(zip(pairs, holm([cells[k]["p"] for k in pairs])))
            print(f"\n{FAMILY[src]}·{disp} — {ds}")
            print("row\\col       " + " ".join(f"{SHORT[c]:>12s}" for c in names))
            for a in names:
                line = [f"{SHORT[a]:12s}"]
                for c in names:
                    if a == c:
                        line.append(f"{'—':>12s}")
                    else:
                        r = cells[(a, c)]
                        beats = padj[(a, c)] < 0.05 and ((r["delta"] > 0) == hb)
                        line.append(f"{r['win_pct']:10.0f}%{'*' if beats else ' '}")
                print(" ".join(line))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairwise", action="store_true",
                    help="also print full 4x4 pairwise win-rate matrices")
    args = ap.parse_args()
    dfs = load()
    out = run(dfs)
    out.to_csv("results/stat_tests.csv", index=False)
    print_summary(out)
    n_sig = int(out.sig.sum())
    print(f"[{n_sig}/{len(out)} proposed-vs-baseline tests significant after Holm; "
          f"full table -> results/stat_tests.csv]")
    if args.pairwise:
        print_pairwise(dfs, [("slice", "slice_NCC_brain", "NCC_brain", True),
                             ("volume", "vol_NCC", "NCC", True),
                             ("motion", "rot_err_mean", "Rot", False)])


if __name__ == "__main__":
    main()
