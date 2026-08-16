"""Directly standardised relevance rate, and the figure that explains it.

The GLM in rate_adjusted.py says the naive trend and the within-community trend have opposite
signs. Direct standardisation makes the same point without asking the reader to interpret a
coefficient: hold the community mix fixed at the pooled six-year composition and recompute the
monthly rate from each community's own rate that month.

    crude(m)        = sum_s k[m,s] / sum_s n[m,s]
    standardised(m) = sum_s w_s * (k[m,s] / n[m,s]),   w_s = pooled share of community s

If the two series diverge, the difference is the changing mix, not changing behaviour.

Usage: .venv/Scripts/python.exe code/standardize_rate.py
"""
import json, os

import numpy as np
import pandas as pd

OUT = "results_temporal"
PAL = ["#1B6CA8", "#C1440E", "#12876A", "#9B5DE5", "#B07A00", "#A0335E"]
INK, MUTED, GRID, SURFACE = "#1F2328", "#5C6670", "#E3E6E8", "#FCFCFB"
SPECTATOR = {"nba", "CollegeBasketball", "Basketball", "tennis"}


def main():
    d = pd.read_csv(f"{OUT}/rate_month_subreddit.csv")
    months = sorted(d.month.unique())
    subs = sorted(d.subreddit.unique())

    # pooled weights across the whole period
    pooled = d.groupby("subreddit").n_total.sum()
    w = (pooled / pooled.sum()).to_dict()

    piv_k = d.pivot_table(index="month", columns="subreddit", values="n_relevant",
                          aggfunc="sum").reindex(months).fillna(0)
    piv_n = d.pivot_table(index="month", columns="subreddit", values="n_total",
                          aggfunc="sum").reindex(months).fillna(0)

    crude = piv_k.sum(axis=1) / piv_n.sum(axis=1)

    # standardised: only over communities present that month, weights renormalised so the
    # series is not dragged down by a community that simply had no posts
    std, eff_w = [], []
    for m in months:
        n = piv_n.loc[m]
        present = n[n >= 20].index  # need a usable denominator to estimate a rate
        ww = np.array([w[s] for s in present], float)
        if ww.sum() == 0:
            std.append(np.nan); eff_w.append(0.0); continue
        ww = ww / ww.sum()
        p = np.array([piv_k.loc[m, s] / n[s] for s in present], float)
        std.append(float((ww * p).sum()))
        eff_w.append(float(sum(w[s] for s in present)))

    out = pd.DataFrame({"month": months, "crude": crude.values, "standardized": std,
                        "weight_covered": eff_w})
    out["n_total"] = piv_n.sum(axis=1).values
    out["n_relevant"] = piv_k.sum(axis=1).values
    out.to_csv(f"{OUT}/standardized_rate.csv", index=False)

    t = np.arange(len(months)) / 12.0
    def slope(y):
        ok = ~np.isnan(y)
        b = np.polyfit(t[ok], np.asarray(y)[ok], 1)[0]
        return float(b)
    print(f"[std] crude        {crude.iloc[:12].mean()*100:.2f}% -> "
          f"{crude.iloc[-12:].mean()*100:.2f}%   slope {slope(crude.values)*100:+.3f} pp/yr")
    s = np.array(std, float)
    print(f"[std] standardised {np.nanmean(s[:12])*100:.2f}% -> "
          f"{np.nanmean(s[-12:])*100:.2f}%   slope {slope(s)*100:+.3f} pp/yr")

    # ---------------- figure
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

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1], "hspace": 0.2})

    # (a) composition
    ax = axes[0]
    top = pooled.sort_values(ascending=False).head(5).index.tolist()
    shares = (piv_n[top].T / piv_n.sum(axis=1).values).T
    other = 1 - shares.sum(axis=1)
    ax.stackplot(x, *[shares[s].values for s in top], other.values,
                 labels=top + ["other"],
                 colors=PAL[:len(top)] + ["#C9CDD1"], edgecolor=SURFACE, linewidth=0.4)
    ax.set_ylim(0, 1); ax.set_ylabel("Share of monthly volume")
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="lower left")
    ax.set_title("Which communities are posting (the mix moves a lot)",
                 loc="left", fontsize=10, color=INK, pad=6)

    # (b) crude vs standardised
    ax = axes[1]
    ax.plot(x, crude.values * 100, lw=2, color=PAL[1], label="Crude rate")
    ax.plot(x, s * 100, lw=2, color=PAL[0], label="Standardised to a fixed community mix")
    for y, c in ((crude.values, PAL[1]), (s, PAL[0])):
        ok = ~np.isnan(y)
        b, a = np.polyfit(t[ok], np.asarray(y)[ok], 1)
        ax.plot(x, (a + b * t) * 100, lw=1.2, ls="--", color=c, alpha=0.85)
    ax.set_ylabel("% of comments flagged relevant")
    ax.set_xticks(ticks); ax.set_xticklabels(labs)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("Crude rate falls; the same corpus rises once the mix is held fixed",
                 loc="left", fontsize=10, color=INK, pad=6)
    fig.savefig(f"{OUT}/fig3_composition.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    with open(f"{OUT}/standardized_summary.json", "w", encoding="utf-8") as fh:
        json.dump({"crude_first12_pct": float(crude.iloc[:12].mean() * 100),
                   "crude_last12_pct": float(crude.iloc[-12:].mean() * 100),
                   "std_first12_pct": float(np.nanmean(s[:12]) * 100),
                   "std_last12_pct": float(np.nanmean(s[-12:]) * 100),
                   "crude_slope_pp_per_year": slope(crude.values) * 100,
                   "std_slope_pp_per_year": slope(s) * 100}, fh, indent=2)
    print(f"[std] wrote {OUT}/standardized_rate.csv, fig3_composition.png")


if __name__ == "__main__":
    main()
