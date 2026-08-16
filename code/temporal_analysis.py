"""Monthly temporal analysis of the athlete mental-health corpus.

Follows the analysis brief:
  * x axis = month over year, y = number of documents
  * report PROPORTIONS as well as raw frequencies (Reddit volume grows/shrinks over time,
    so a raw count of mental-health talk moves with the platform whether or not anything
    about mental health changed)
  * always compare against a BASELINE -- here two of them: the keyword-matched arm's own
    total volume (the denominator) and the non-keyword control arm from the same subreddits
  * look for spikes and check them against real events

Inputs
  run_logs/classify_dapt.log   per-month `N rows, M relevant` for all three arms, written by
                               code/classify_corpus.py during the deliverable (DAPT) run.
  data/classified/final_dataset_tagged.csv   (optional) per-comment tags, for the per-tag series.

Outputs -> results_temporal/
  monthly_series.csv, tag_monthly.csv, stats.json, spikes.csv, and the figures.

Usage: .venv/Scripts/python.exe code/temporal_analysis.py [--tagged PATH]
"""
import csv, json, os, re, sys, math
from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.api as sm

csv.field_size_limit(2**31 - 1)

LOG = "run_logs/classify_dapt.log"
OUT = "results_temporal"
TAGGED_DEFAULT = "data/classified/final_dataset_tagged.csv"

# arm index in the log -> label. classify_dapt.log runs three arms in this order.
ARM = {1: "matched", 2: "matched", 3: "baseline"}

# Events to test the series against. These are the athlete-mental-health moments of the
# period plus the pandemic shock; chosen a priori, not after looking at the spikes.
EVENTS = [
    ("2020-03", "COVID-19 declared a pandemic; gyms and competition shut down"),
    ("2020-06", "Post-lockdown reopening in much of the US"),
    ("2021-05", "Naomi Osaka withdraws from the French Open citing mental health"),
    ("2021-07", "Simone Biles withdraws from Olympic finals citing mental health"),
    ("2022-01", "New-year training surge"),
]

# Validated categorical palette, fixed order (dataviz six checks all PASS on light surface).
PAL = ["#1B6CA8", "#C1440E", "#12876A", "#9B5DE5", "#B07A00", "#A0335E"]
INK, MUTED, GRID, SURFACE = "#1F2328", "#5C6670", "#E3E6E8", "#FCFCFB"


# ----------------------------------------------------------------- parsing
def parse_log(path=LOG):
    """-> DataFrame[arm, month, n_total, n_relevant] from the classify run log."""
    pat = re.compile(
        r"^\[(\d+):\d+/\d+\]\s+RC_(\d{4}-\d{2})\.csv:\s+([\d,]+) rows,\s+([\d,]+) relevant")
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = pat.match(line.strip())
            if not m:
                continue
            arm_i, month, tot, rel = int(m.group(1)), m.group(2), m.group(3), m.group(4)
            rows.append({"arm": ARM.get(arm_i, f"arm{arm_i}"), "month": month,
                         "n_total": int(tot.replace(",", "")),
                         "n_relevant": int(rel.replace(",", ""))})
    df = pd.DataFrame(rows)
    # arms 1 and 2 are both `matched` (2018-2022 and 2023); sum in case of overlap
    df = df.groupby(["arm", "month"], as_index=False)[["n_total", "n_relevant"]].sum()
    return df.sort_values(["arm", "month"]).reset_index(drop=True)


# ----------------------------------------------------------------- statistics
def wilson(k, n, z=1.96):
    """Wilson score interval. Correct for proportions near 0 (self-harm at ~1%) where the
    normal approximation gives negative lower bounds."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def binomial_trend(k, n, t):
    """Binomial GLM: logit(p) ~ t, with t in years. Returns the multiplicative change in
    odds per year and its 95% CI.

    A GLM on the counts rather than OLS on the proportions, because the monthly denominators
    differ by a factor of ~3 across the series and OLS would weight a 6k-comment month the
    same as an 18k one.
    """
    X = sm.add_constant(np.asarray(t, float))
    y = np.column_stack([np.asarray(k, float), np.asarray(n, float) - np.asarray(k, float)])
    res = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    b, se = res.params[1], res.bse[1]
    return {"or_per_year": float(np.exp(b)),
            "ci_low": float(np.exp(b - 1.96 * se)), "ci_high": float(np.exp(b + 1.96 * se)),
            "p_value": float(res.pvalues[1]), "z": float(res.tvalues[1]),
            "pct_change_per_year": float((np.exp(b) - 1) * 100)}


def seasonality(k, n, months):
    """Month-of-year fixed effects on top of a linear trend. Likelihood-ratio test against
    the trend-only model tells us whether the series has a seasonal shape at all."""
    t = np.arange(len(k)) / 12.0
    moy = np.array([int(m.split("-")[1]) for m in months])
    D = pd.get_dummies(moy, drop_first=True, prefix="m").astype(float).values
    y = np.column_stack([np.asarray(k, float), np.asarray(n, float) - np.asarray(k, float)])
    base = sm.GLM(y, sm.add_constant(t), family=sm.families.Binomial()).fit()
    full = sm.GLM(y, np.column_stack([np.ones(len(t)), t, D]),
                  family=sm.families.Binomial()).fit()
    lr = 2 * (full.llf - base.llf)
    from scipy import stats as st
    pval = float(st.chi2.sf(lr, D.shape[1]))
    # per-month effect relative to January, as an odds ratio
    eff = {}
    for j in range(D.shape[1]):
        eff[j + 2] = float(np.exp(full.params[2 + j]))
    return {"lr_stat": float(lr), "df": int(D.shape[1]), "p_value": pval,
            "odds_vs_january": eff}


def find_spikes(months, prop, window=13, thresh=2.5):
    """Deviation from a centred rolling median, scaled by the median absolute deviation.

    A rolling median rather than a mean because a real spike would drag a mean baseline up
    and hide itself; MAD rather than SD for the same reason.
    """
    s = pd.Series(prop, index=months)
    base = s.rolling(window, center=True, min_periods=5).median()
    resid = s - base
    mad = float(np.nanmedian(np.abs(resid - np.nanmedian(resid))))
    scale = mad * 1.4826 if mad > 0 else float(np.nanstd(resid))
    z = resid / scale if scale > 0 else resid * 0
    out = pd.DataFrame({"month": months, "prop": prop, "baseline": base.values,
                        "resid": resid.values, "robust_z": z.values})
    out["spike"] = out["robust_z"].abs() >= thresh
    return out


def interrupted(k, n, months, cut="2020-03"):
    """Interrupted time series around a cut month: level shift + slope change on top of the
    pre-period trend."""
    t = np.arange(len(k)) / 12.0
    idx = months.index(cut) if cut in months else None
    if idx is None:
        return None
    post = (np.arange(len(k)) >= idx).astype(float)
    tpost = np.where(post > 0, t - t[idx], 0.0)
    X = np.column_stack([np.ones(len(t)), t, post, tpost])
    y = np.column_stack([np.asarray(k, float), np.asarray(n, float) - np.asarray(k, float)])
    r = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    return {"cut": cut,
            "level_shift_or": float(np.exp(r.params[2])), "level_p": float(r.pvalues[2]),
            "slope_change_or_per_year": float(np.exp(r.params[3])),
            "slope_p": float(r.pvalues[3])}


# ----------------------------------------------------------------- tag series
def tag_series(path, months):
    """Per-month positive counts for each tag column in the tagged dataset."""
    if not os.path.exists(path):
        print(f"[temporal] no tagged dataset at {path}; skipping per-tag series")
        return None, []
    counts = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)
    with open(path, encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        tags = [c for c in rd.fieldnames if c.startswith("tag_")]
        for r in rd:
            if r.get("arm") != "matched":
                continue
            tm = (r.get("time") or "")[:7]
            if len(tm) != 7:
                continue
            totals[tm] += 1
            for t in tags:
                if r.get(t) == "1":
                    counts[t][tm] += 1
    rows = []
    for m in months:
        row = {"month": m, "n_relevant_tagged": totals.get(m, 0)}
        for t in tags:
            row[t] = counts[t].get(m, 0)
        rows.append(row)
    return pd.DataFrame(rows), tags


# ----------------------------------------------------------------- main
def main():
    tagged = TAGGED_DEFAULT
    if "--tagged" in sys.argv:
        tagged = sys.argv[sys.argv.index("--tagged") + 1]
    os.makedirs(OUT, exist_ok=True)

    df = parse_log()
    # classify_corpus.py drops the dedicated mental-health subreddits before running the
    # models, but the log's row count is taken BEFORE that drop. Those comments can never be
    # flagged relevant, so leaving them in the denominator dilutes every rate (85,452 comments
    # in the matched arm, 65,486 in the control). Where the corrected per-month totals exist,
    # use them; fall back to the log otherwise.
    corrected = f"{OUT}/arm_denominators.json"
    if os.path.exists(corrected):
        adj = json.load(open(corrected, encoding="utf-8"))
        for arm in ("matched", "baseline"):
            pm = adj.get(arm, {}).get("per_month", {})
            sel = df.arm == arm
            df.loc[sel, "n_total"] = [pm.get(mm, t) for mm, t in
                                      zip(df.loc[sel, "month"], df.loc[sel, "n_total"])]
        print(f"[temporal] using drop-corrected denominators from {corrected}")

    matched = df[df.arm == "matched"].sort_values("month").reset_index(drop=True)
    baseline = df[df.arm == "baseline"].sort_values("month").reset_index(drop=True)
    months = matched.month.tolist()
    print(f"[temporal] matched: {len(matched)} months {months[0]}..{months[-1]}  "
          f"{matched.n_total.sum():,} comments, {matched.n_relevant.sum():,} relevant")
    print(f"[temporal] baseline: {len(baseline)} months  {baseline.n_total.sum():,} comments, "
          f"{baseline.n_relevant.sum():,} relevant")

    # ---- monthly series with Wilson CIs
    m = matched.copy()
    m["prop_relevant"] = m.n_relevant / m.n_total
    ci = [wilson(k, n) for k, n in zip(m.n_relevant, m.n_total)]
    m["ci_low"], m["ci_high"] = [c[0] for c in ci], [c[1] for c in ci]
    b = baseline.set_index("month")
    m["baseline_total"] = m.month.map(b.n_total).astype("float")
    m["baseline_relevant"] = m.month.map(b.n_relevant).astype("float")
    m["baseline_prop"] = m.baseline_relevant / m.baseline_total
    # relevant comments as a share of ALL sports-subreddit activity we observe that month
    m["all_activity"] = m.n_total + m.baseline_total.fillna(0)
    m["prop_of_all_activity"] = m.n_relevant / m.all_activity
    m["separation_x"] = m.prop_relevant / m.baseline_prop
    m.to_csv(f"{OUT}/monthly_series.csv", index=False)

    stats = {"n_months": len(m),
             "span": [months[0], months[-1]],
             "matched_total": int(m.n_total.sum()),
             "matched_relevant": int(m.n_relevant.sum()),
             "overall_prop": float(m.n_relevant.sum() / m.n_total.sum()),
             "baseline_total": int(baseline.n_total.sum()),
             "baseline_relevant": int(baseline.n_relevant.sum()),
             "baseline_prop": float(baseline.n_relevant.sum() / baseline.n_total.sum())}
    stats["overall_separation_x"] = stats["overall_prop"] / stats["baseline_prop"]

    t_years = np.arange(len(m)) / 12.0
    # The control arm only covers 2018-2022. Any statistic that divides by "all activity" must
    # be restricted to those months -- otherwise 2023 has a control volume of 0, the matched
    # share is forced to 1.0, and a spurious growth trend appears out of nothing.
    both = m.baseline_total.notna()
    stats["trend_volume"] = binomial_trend(
        m.n_total[both], m.all_activity[both], t_years[both.values])
    stats["trend_volume_note"] = "restricted to 2018-2022 (months with a control arm)"
    stats["trend_relevance_rate"] = binomial_trend(m.n_relevant, m.n_total, t_years)
    stats["trend_baseline_rate"] = binomial_trend(
        m.baseline_relevant[both], m.baseline_total[both], t_years[both.values])
    stats["seasonality_relevance"] = seasonality(m.n_relevant, m.n_total, months)
    stats["its_covid"] = interrupted(m.n_relevant, m.n_total, months, "2020-03")

    sp = find_spikes(months, m.prop_relevant.values)
    sp.to_csv(f"{OUT}/spikes.csv", index=False)
    flagged = sp[sp.spike]
    stats["spikes"] = [{"month": r.month, "prop": float(r.prop),
                        "robust_z": float(r.robust_z)} for r in flagged.itertuples()]

    # a-priori event windows: is the event month elevated vs its own rolling baseline?
    zmap = dict(zip(sp.month, sp.robust_z))
    stats["events"] = []
    for em, label in EVENTS:
        w = [zmap.get(em), zmap.get(_shift(em, 1))]
        w = [x for x in w if x is not None and not np.isnan(x)]
        stats["events"].append({"month": em, "label": label,
                                "robust_z": None if not w else float(max(w))})

    # ---- per-tag
    tdf, tags = tag_series(tagged, months)
    if tdf is not None and len(tags):
        tdf.to_csv(f"{OUT}/tag_monthly.csv", index=False)
        stats["tag_trends"] = {}
        for t in tags:
            k = tdf[t].values
            n = tdf.n_relevant_tagged.values
            if n.sum() == 0 or k.sum() < 30:
                continue
            keep = n > 0
            tr = binomial_trend(k[keep], n[keep], (np.arange(len(k)) / 12.0)[keep])
            tr["overall_prop"] = float(k.sum() / max(n.sum(), 1))
            tr["n_positive"] = int(k.sum())
            spt = find_spikes(months, np.divide(k, np.maximum(n, 1)))
            tr["spikes"] = [{"month": r.month, "robust_z": float(r.robust_z)}
                            for r in spt[spt.spike].itertuples()]
            stats["tag_trends"][t.replace("tag_", "")] = tr

    with open(f"{OUT}/stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    _report(stats, m)
    _figures(m, months, sp, tdf, tags)
    print(f"\n[temporal] wrote {OUT}/monthly_series.csv, spikes.csv, stats.json, figures")
    return stats


def _shift(ym, k):
    y, mo = int(ym[:4]), int(ym[5:])
    mo += k
    y += (mo - 1) // 12
    mo = (mo - 1) % 12 + 1
    return f"{y:04d}-{mo:02d}"


def _report(s, m):
    print("\n" + "=" * 78)
    print("TEMPORAL ANALYSIS")
    print("=" * 78)
    print(f"span {s['span'][0]} .. {s['span'][1]}  ({s['n_months']} months)")
    print(f"matched   {s['matched_total']:>9,} comments  {s['matched_relevant']:>8,} relevant "
          f"({s['overall_prop']*100:.2f}%)")
    print(f"baseline  {s['baseline_total']:>9,} comments  {s['baseline_relevant']:>8,} relevant "
          f"({s['baseline_prop']*100:.2f}%)")
    print(f"separation {s['overall_separation_x']:.1f}x")

    def line(name, tr):
        star = "***" if tr["p_value"] < .001 else "**" if tr["p_value"] < .01 \
            else "*" if tr["p_value"] < .05 else "ns"
        print(f"  {name:34} {tr['pct_change_per_year']:+6.2f}%/yr  "
              f"OR {tr['or_per_year']:.4f} [{tr['ci_low']:.4f}, {tr['ci_high']:.4f}]  "
              f"p={tr['p_value']:.2e} {star}")

    print("\nTRENDS (binomial GLM, logit ~ time in years)")
    line("matched share of all activity", s["trend_volume"])
    line("relevance rate | matched", s["trend_relevance_rate"])
    line("relevance rate | baseline", s["trend_baseline_rate"])

    se = s["seasonality_relevance"]
    print(f"\nSEASONALITY  LR={se['lr_stat']:.1f} df={se['df']} p={se['p_value']:.2e}")
    top = sorted(se["odds_vs_january"].items(), key=lambda kv: -kv[1])[:3]
    bot = sorted(se["odds_vs_january"].items(), key=lambda kv: kv[1])[:2]
    nm = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    print("  highest vs Jan: " + ", ".join(f"{nm[int(k)]} {v:.3f}" for k, v in top))
    print("  lowest  vs Jan: " + ", ".join(f"{nm[int(k)]} {v:.3f}" for k, v in bot))

    its = s.get("its_covid")
    if its:
        print(f"\nINTERRUPTED TIME SERIES at {its['cut']} (COVID)")
        print(f"  level shift  OR {its['level_shift_or']:.4f}  p={its['level_p']:.2e}")
        print(f"  slope change OR {its['slope_change_or_per_year']:.4f}/yr  p={its['slope_p']:.2e}")

    print(f"\nSPIKES (|robust z| >= 2.5 vs 13-month centred median)")
    if not s["spikes"]:
        print("  none")
    for sp in sorted(s["spikes"], key=lambda d: -abs(d["robust_z"]))[:10]:
        print(f"  {sp['month']}  prop {sp['prop']*100:5.2f}%  z={sp['robust_z']:+.2f}")

    print("\nA-PRIORI EVENT WINDOWS (max robust z in event month or the month after)")
    for e in s["events"]:
        z = e["robust_z"]
        mark = "  <-- elevated" if z is not None and z >= 1.5 else ""
        print(f"  {e['month']}  z={z if z is None else round(z,2)!s:>6}  {e['label'][:52]}{mark}")

    tt = s.get("tag_trends") or {}
    if tt:
        print("\nPER-TAG TRENDS (share of relevant comments carrying the tag)")
        print(f"  {'tag':24} {'overall':>8} {'%/yr':>8} {'p':>10}")
        for k, v in sorted(tt.items(), key=lambda kv: -kv[1]["pct_change_per_year"]):
            star = "***" if v["p_value"] < .001 else "**" if v["p_value"] < .01 \
                else "*" if v["p_value"] < .05 else ""
            print(f"  {k:24} {v['overall_prop']*100:7.2f}% {v['pct_change_per_year']:+7.2f}% "
                  f"{v['p_value']:9.2e} {star}")


def _figures(m, months, sp, tdf, tags):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    })
    x = np.arange(len(months))
    ticks = [i for i, mm in enumerate(months) if mm.endswith("-01")]
    labs = [months[i][:4] for i in ticks]

    # ---- Figure 1: volume (raw) and relevance rate (proportion), stacked, one axis each
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.15], "hspace": 0.18})

    ax = axes[0]
    ax.plot(x, m.n_total, lw=2, color=PAL[0], label="Keyword-matched comments")
    ax.plot(x, m.n_relevant, lw=2, color=PAL[1], label="Athlete mental-health comments")
    ax.set_ylabel("Comments per month")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.set_title("Monthly volume", loc="left", fontsize=10, color=INK, pad=6)

    ax = axes[1]
    ax.fill_between(x, m.ci_low * 100, m.ci_high * 100, color=PAL[1], alpha=0.18, lw=0)
    ax.plot(x, m.prop_relevant * 100, lw=2, color=PAL[1], label="Matched arm")
    ax.plot(x, m.baseline_prop * 100, lw=2, color=PAL[2], label="Control arm (no MH keywords)")
    for r in sp[sp.spike].itertuples():
        i = months.index(r.month)
        ax.scatter([i], [r.prop * 100], s=34, facecolor=SURFACE, edgecolor=PAL[1],
                   zorder=5, linewidth=1.6)
    ax.set_ylabel("% of comments flagged relevant")
    ax.set_xticks(ticks); ax.set_xticklabels(labs)
    ax.legend(frameon=False, loc="center right", fontsize=8)
    ax.set_title("Relevance rate, with 95% Wilson intervals (circles mark spikes)",
                 loc="left", fontsize=10, color=INK, pad=6)
    fig.savefig(f"{OUT}/fig1_volume_and_rate.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: per-tag small multiples (never 12 hues on one axis)
    if tdf is not None and len(tags):
        order = sorted(tags, key=lambda t: -tdf[t].sum())
        n = len(order)
        ncol, nrow = 4, int(np.ceil(n / 4))
        fig, axes = plt.subplots(nrow, ncol, figsize=(9.6, 1.75 * nrow), sharex=True)
        axes = np.atleast_2d(axes).ravel()
        denom = np.maximum(tdf.n_relevant_tagged.values, 1)
        for j, t in enumerate(order):
            ax = axes[j]
            p = tdf[t].values / denom * 100
            ax.plot(x, p, lw=1.6, color=PAL[j % len(PAL)])
            ax.set_title(t.replace("tag_", ""), loc="left", fontsize=8.5, color=INK, pad=3)
            ax.set_xticks(ticks); ax.set_xticklabels(labs, fontsize=7)
            ax.tick_params(labelsize=7)
            ax.set_ylim(bottom=0)
        for j in range(n, len(axes)):
            axes[j].axis("off")
        fig.suptitle("Share of athlete mental-health comments carrying each tag (%)",
                     x=0.008, ha="left", fontsize=10, color=INK)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(f"{OUT}/fig2_tag_trends.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
