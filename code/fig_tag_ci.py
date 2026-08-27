"""Per-tag monthly shares WITH confidence intervals, plus a focused figure.

Reviewer requirement: the per-tag series were previously plotted as point estimates only, which
makes a noisy rare tag look like a trend. Every panel now carries a 95% Wilson band, so a reader
can see which movements the data actually supports. Wilson rather than normal-approximation
because several tags run near 1% and the normal interval goes negative there.

Outputs
  fig2_tag_trends_ci.png   all 16 tags, small multiples, with bands  (appendix figure)
  fig4_headline_tags.png   the four tags carrying a reported finding, larger, with bands
  tag_monthly_ci.csv       month x tag shares with lower/upper bounds

Usage: .venv/Scripts/python.exe code/fig_tag_ci.py [--in F]
"""
import csv, math, os, sys
from collections import defaultdict

import numpy as np
import pandas as pd

csv.field_size_limit(2**31 - 1)
OUT = "results_temporal"
PAL = ["#1B6CA8", "#C1440E", "#12876A", "#9B5DE5", "#B07A00", "#A0335E"]
INK, MUTED, GRID, SURFACE = "#1F2328", "#5C6670", "#E3E6E8", "#FCFCFB"

# The tags promoted to their own paragraph in the results. Chosen because each carries a finding
# that survives the composition control, not because they are the largest.
HEADLINE = [
    ("adhd_neurodivergence", "ADHD / neurodivergence", "+24.6%/yr adjusted (95% CI 22.0-27.2)"),
    ("help_seeking", "Help-seeking", "+5.8%/yr adjusted (95% CI 4.8-6.9)"),
    ("depression", "Depression", "-12.7%/yr adjusted (95% CI -13.4 to -11.9)"),
    ("performance_psych", "Performance psychology", "crude +8.5%/yr, adjusted -4.0%/yr"),
]


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    inp = sys.argv[sys.argv.index("--in") + 1] if "--in" in sys.argv \
        else "data/classified/final_dataset_tagged_v3b.csv"
    counts = defaultdict(lambda: defaultdict(int))
    total = defaultdict(int)
    tags = []
    with open(inp, encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        tags = [c[4:] for c in (rd.fieldnames or []) if c.startswith("tag_")]
        for r in rd:
            if r.get("arm") != "matched":
                continue
            m = (r.get("time") or "")[:7]
            if len(m) != 7:
                continue
            total[m] += 1
            for t in tags:
                if r.get(f"tag_{t}") == "1":
                    counts[t][m] += 1
    months = sorted(m for m in total if total[m] >= 30)
    print(f"[ci] {sum(total.values()):,} comments | {len(tags)} tags | {len(months)} months")

    rows = []
    for m in months:
        row = {"month": m, "n": total[m]}
        for t in tags:
            k = counts[t][m]
            lo, hi = wilson(k, total[m])
            row[f"{t}"] = k / total[m]
            row[f"{t}__lo"] = lo
            row[f"{t}__hi"] = hi
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/tag_monthly_ci.csv", index=False)

    # median half-width per tag, a compact way to report precision in the caption
    print(f"\n  {'tag':24} {'mean %':>8} {'median CI half-width (pp)':>26}")
    for t in sorted(tags, key=lambda x: -df[x].mean()):
        hw = ((df[f"{t}__hi"] - df[f"{t}__lo"]) / 2 * 100).median()
        print(f"  {t:24} {df[t].mean()*100:7.2f} {hw:25.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    })
    x = np.arange(len(months))
    ticks = [i for i, m in enumerate(months) if m.endswith("-01")]
    labs = [months[i][:4] for i in ticks]

    # ---- all 16, with bands
    order = sorted(tags, key=lambda t: -df[t].mean())
    nrow = int(np.ceil(len(order) / 4))
    fig, axes = plt.subplots(nrow, 4, figsize=(9.6, 1.85 * nrow), sharex=True)
    axes = np.atleast_2d(axes).ravel()
    for j, t in enumerate(order):
        ax = axes[j]
        c = PAL[j % len(PAL)]
        ax.fill_between(x, df[f"{t}__lo"] * 100, df[f"{t}__hi"] * 100,
                        color=c, alpha=0.20, lw=0)
        ax.plot(x, df[t] * 100, lw=1.5, color=c)
        ax.set_title(t, loc="left", fontsize=8.5, color=INK, pad=3)
        ax.set_xticks(ticks); ax.set_xticklabels(labs, fontsize=7)
        ax.tick_params(labelsize=7); ax.set_ylim(bottom=0)
    for j in range(len(order), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Share of relevant comments carrying each tag (%), with 95% Wilson intervals",
                 x=0.008, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{OUT}/fig2_tag_trends_ci.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- the four headline tags, larger
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.0), sharex=True)
    axes = axes.ravel()
    for j, (t, label, sub) in enumerate(HEADLINE):
        ax = axes[j]
        c = PAL[j % len(PAL)]
        ax.fill_between(x, df[f"{t}__lo"] * 100, df[f"{t}__hi"] * 100,
                        color=c, alpha=0.20, lw=0)
        ax.plot(x, df[t] * 100, lw=1.8, color=c)
        b, a = np.polyfit(x / 12.0, df[t].values * 100, 1)
        ax.plot(x, a + b * (x / 12.0), lw=1.1, ls="--", color=c, alpha=0.85)
        ax.set_title(label, loc="left", fontsize=9.5, color=INK, pad=14)
        ax.text(0.0, 1.012, sub, transform=ax.transAxes, fontsize=7.5, color=MUTED,
                va="bottom", ha="left")
        ax.set_xticks(ticks); ax.set_xticklabels(labs, fontsize=8)
        ax.set_ylim(bottom=0)
        if j % 2 == 0:
            ax.set_ylabel("% of relevant comments")
    fig.tight_layout(h_pad=2.2)
    fig.savefig(f"{OUT}/fig4_headline_tags.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[ci] wrote {OUT}/fig2_tag_trends_ci.png, fig4_headline_tags.png, tag_monthly_ci.csv")


if __name__ == "__main__":
    main()
