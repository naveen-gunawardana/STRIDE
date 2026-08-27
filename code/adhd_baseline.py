"""Is the ADHD rise specific to athlete mental-health discussion, or platform-wide?

The paper's largest per-tag effect is ADHD/neurodivergence at +24.6% per year (adjusted) among
Layer-1-relevant comments. Two explanations fit that equally well:

  (a) athletes increasingly discuss neurodivergence in a mental-health frame, or
  (b) ADHD talk rose everywhere on Reddit over 2018-2023 and the corpus simply inherits it.

The corpus cannot separate these on its own, so this script builds the missing denominator: the
rate of ADHD mentions across ALL comments in the same 19 communities over the same months,
matched and control arms combined. That total is the complete sampled volume of those
communities, so it is unconditioned on mental-health keywords and serves as the community-wide
baseline.

If the baseline rises at the same rate as the tagged series, explanation (b) holds and the
finding should be reported as inherited. If it rises more slowly, the effect is specific to
mental-health discussion.

Outputs -> results_temporal/adhd_baseline.{csv,json} and fig5_adhd_baseline.png

Usage: .venv/Scripts/python.exe code/adhd_baseline.py
"""
import csv, glob, json, os, re, sys
from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classify_corpus import DROP_SUBS

csv.field_size_limit(2**31 - 1)
OUT = "results_temporal"
PAL = ["#1B6CA8", "#C1440E", "#12876A", "#9B5DE5", "#B07A00", "#A0335E"]
INK, MUTED, GRID, SURFACE = "#1F2328", "#5C6670", "#E3E6E8", "#FCFCFB"

RAW = {
    "matched": ["comments/2018-2022/MS_comments_2018_2022/MS_comments_2018_2022/matched/RC_*.csv",
                "comments/2023/comments_2023/filtered_subreddit_keywords/matched/RC_*.csv"],
    "control": ["comments/2018-2022/MS_comments_2018_2022/MS_comments_2018_2022/baseline/RC_*.csv"],
}

# Deliberately the surface vocabulary, not the tag classifier. The question is whether the WORDS
# became more common in these communities, which is what "platform-wide drift" would look like.
# Guarded against the "add weight / add a set" sense that dominates a fitness corpus.
ADHD = re.compile(r"\badhd\b|\ba\.d\.h\.d\b|\bautis(?:m|tic)\b|\basperger|\bneurodiverg|"
                  r"\bon the spectrum\b|\bocd\b|\bexecutive (?:function|dysfunction)\b", re.I)
# A stable comparison term: should be roughly flat if the corpus is not drifting wholesale.
ANX = re.compile(r"\banxiety\b|\banxious\b|\bpanic attack", re.I)


def scan():
    rows = defaultdict(lambda: {"n": 0, "adhd": 0, "anx": 0})
    for arm, pats in RAW.items():
        files = [f for pt in pats for f in sorted(glob.glob(pt))]
        print(f"[adhd] {arm}: {len(files)} files")
        for i, f in enumerate(files, 1):
            month = os.path.basename(f)[3:10]
            with open(f, encoding="utf-8-sig", newline="") as fh:
                for r in csv.DictReader(fh):
                    sub = (r.get("subreddit") or "").strip()
                    if not sub or sub in DROP_SUBS:
                        continue          # same exclusion the classifier applies
                    t = r.get("text") or ""
                    k = rows[(arm, month)]
                    k["n"] += 1
                    if ADHD.search(t):
                        k["adhd"] += 1
                    if ANX.search(t):
                        k["anx"] += 1
            if i % 20 == 0:
                print(f"  ... {i}/{len(files)}", flush=True)
    return rows


def trend(k, n, months):
    t = (pd.PeriodIndex(months, freq="M").astype("int64")).values.astype(float)
    t = (t - t.min()) / 12.0
    X = sm.add_constant(t)
    y = np.column_stack([np.asarray(k, float), np.asarray(n, float) - np.asarray(k, float)])
    r = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    b, se = r.params[1], r.bse[1]
    return {"pct_per_year": float((np.exp(b) - 1) * 100),
            "ci_low": float((np.exp(b - 1.96 * se) - 1) * 100),
            "ci_high": float((np.exp(b + 1.96 * se) - 1) * 100),
            "p": float(r.pvalues[1])}


def main():
    os.makedirs(OUT, exist_ok=True)
    raw = scan()
    recs = []
    for (arm, month), d in raw.items():
        recs.append({"arm": arm, "month": month, **d})
    df = pd.DataFrame(recs).sort_values(["arm", "month"])

    # community-wide = both arms pooled; that total is the full sampled volume of the 19 subs
    tot = df.groupby("month")[["n", "adhd", "anx"]].sum().reset_index()
    tot["adhd_rate"] = tot.adhd / tot.n
    tot["anx_rate"] = tot.anx / tot.n
    tot.to_csv(f"{OUT}/adhd_baseline.csv", index=False)

    months = tot.month.tolist()
    res = {
        "community_wide_adhd": trend(tot.adhd, tot.n, months),
        "community_wide_anxiety": trend(tot.anx, tot.n, months),
        "n_comments": int(tot.n.sum()),
        "adhd_first12_pct": float(tot.adhd.head(12).sum() / tot.n.head(12).sum() * 100),
        "adhd_last12_pct": float(tot.adhd.tail(12).sum() / tot.n.tail(12).sum() * 100),
    }
    for arm in ("matched", "control"):
        a = df[df.arm == arm]
        if len(a) > 12:
            res[f"{arm}_adhd"] = trend(a.adhd, a.n, a.month.tolist())

    # the figure the paper needs: tagged series vs community-wide baseline, indexed to 2018
    tagged = pd.read_csv(f"{OUT}/tag_monthly_ci.csv")[["month", "adhd_neurodivergence"]]
    tagged = tagged.rename(columns={"adhd_neurodivergence": "tagged_rate"})
    m = tot.merge(tagged, on="month", how="inner")
    base_t = m.tagged_rate.head(12).mean()
    base_b = m.adhd_rate.head(12).mean()
    m["tagged_idx"] = m.tagged_rate / base_t * 100
    m["baseline_idx"] = m.adhd_rate / base_b * 100
    res["tagged_index_last12"] = float(m.tagged_idx.tail(12).mean())
    res["baseline_index_last12"] = float(m.baseline_idx.tail(12).mean())

    print("\n" + "=" * 74)
    print("ADHD: ATHLETE-SPECIFIC OR PLATFORM-WIDE?")
    print("=" * 74)
    print(f"scanned {res['n_comments']:,} comments across both arms, {len(months)} months")

    def show(name, d):
        star = "***" if d["p"] < .001 else "**" if d["p"] < .01 else "*" if d["p"] < .05 else "ns"
        print(f"  {name:44} {d['pct_per_year']:+7.2f}%/yr "
              f"[{d['ci_low']:+.2f}, {d['ci_high']:+.2f}] {star}")

    show("ADHD words, all comments in the 19 subs", res["community_wide_adhd"])
    show("  ... matched arm only", res.get("matched_adhd", res["community_wide_adhd"]))
    show("  ... control arm only", res.get("control_adhd", res["community_wide_adhd"]))
    show("Anxiety words, all comments (comparison)", res["community_wide_anxiety"])
    print(f"\n  ADHD word rate {res['adhd_first12_pct']:.3f}% (2018) -> "
          f"{res['adhd_last12_pct']:.3f}% (2023)")
    print(f"\n  Indexed to 2018 = 100:")
    print(f"    tagged ADHD among relevant comments -> {res['tagged_index_last12']:.0f}")
    print(f"    ADHD words community-wide           -> {res['baseline_index_last12']:.0f}")
    ratio = res["tagged_index_last12"] / max(res["baseline_index_last12"], 1e-9)
    res["excess_ratio"] = float(ratio)
    print(f"    excess growth in the tagged series  -> {ratio:.2f}x")

    with open(f"{OUT}/adhd_baseline.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)

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
    x = np.arange(len(m))
    ticks = [i for i, mm in enumerate(m.month) if mm.endswith("-01")]
    labs = [m.month.iloc[i][:4] for i in ticks]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(x, m.tagged_idx, lw=2, color=PAL[0],
            label="ADHD tag, among athlete mental-health comments")
    ax.plot(x, m.baseline_idx, lw=2, color=PAL[2],
            label="ADHD vocabulary, all comments in the same 19 communities")
    ax.axhline(100, color=MUTED, lw=0.9, ls=":")
    ax.set_ylabel("Indexed to 2018 = 100")
    ax.set_xticks(ticks); ax.set_xticklabels(labs)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("ADHD discussion: tagged series against the community-wide baseline",
                 loc="left", fontsize=10, color=INK, pad=6)
    fig.savefig(f"{OUT}/fig5_adhd_baseline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[adhd] wrote {OUT}/adhd_baseline.csv, adhd_baseline.json, fig5_adhd_baseline.png")


if __name__ == "__main__":
    main()
