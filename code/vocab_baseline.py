"""Are the per-tag trends athlete-specific, or inherited from a community-wide vocabulary shift?

Every theme promoted to a finding needs this test. A tagged trend measured among Layer-1-relevant
comments can rise for two very different reasons: athletes increasingly discuss the theme in a
mental-health frame, or the surface vocabulary rose everywhere and the corpus inherits it.

The missing denominator is the rate of each theme's vocabulary across ALL comments in the same 19
communities over the same months, matched and control arms pooled. That pooled total is the
complete sampled volume of those communities, so it is unconditioned on mental-health keywords.

For each theme we index both series to their own 2018 baseline and take the ratio:

    excess = tagged_index_2023 / vocabulary_index_2023

    excess > 1  the theme grew faster inside mental-health discussion than in the community
    excess ~ 1  the tagged trend tracks the community and is inherited
    excess < 1  the tagged trend grew SLOWER than the surrounding community

Deliberately surface regex, not the classifier: the question is whether the words became more
common, which is what a community-wide shift looks like.

Outputs -> results_temporal/vocab_baseline.{csv,json}, fig5_vocab_baseline.png

Usage: .venv/Scripts/python.exe code/vocab_baseline.py
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

PATTERNS = {
    "adhd_neurodivergence": r"\badhd\b|\ba\.d\.h\.d\b|\bautis(?:m|tic)\b|\basperger|"
                            r"\bneurodiverg|\bon the spectrum\b|\bocd\b|"
                            r"\bexecutive (?:function|dysfunction)\b",
    "help_seeking": r"\btherapist\b|\btherapy\b|\bcounsell?or\b|\bcounsell?ing\b|"
                    r"\bpsychiatrist\b|\bpsychologist\b|\bssri\b|\bantidepressant",
    "depression": r"\bdepress(?:ion|ed|ive)\b|\bhopeless\b|\bworthless\b",
    "performance_psych": r"\bperformance anxiety\b|\bthe yips\b|\bmental block\b|"
                         r"\bself.?doubt\b|\bchok(?:e|ed|ing) under pressure\b",
    "anxiety": r"\banxiety\b|\banxious\b|\bpanic attack",
    "self_harm_suicide": r"\bsuicid(?:e|al)\b|\bself.?harm\b",
    "loneliness_isolation": r"\blonel(?:y|iness)\b|\bisolat(?:ed|ion)\b",
}
RX = {k: re.compile(v, re.I) for k, v in PATTERNS.items()}


def scan():
    rows = defaultdict(lambda: dict({"n": 0}, **{k: 0 for k in RX}))
    for arm, pats in RAW.items():
        files = [f for pt in pats for f in sorted(glob.glob(pt))]
        print(f"[vocab] {arm}: {len(files)} files", flush=True)
        for i, f in enumerate(files, 1):
            month = os.path.basename(f)[3:10]
            with open(f, encoding="utf-8-sig", newline="") as fh:
                for r in csv.DictReader(fh):
                    sub = (r.get("subreddit") or "").strip()
                    if not sub or sub in DROP_SUBS:
                        continue
                    t = r.get("text") or ""
                    k = rows[(arm, month)]
                    k["n"] += 1
                    for name, rx in RX.items():
                        if rx.search(t):
                            k[name] += 1
            if i % 24 == 0:
                print(f"  ... {i}/{len(files)}", flush=True)
    return rows


def trend(k, n, months):
    t = (pd.PeriodIndex(months, freq="M").astype("int64")).values.astype(float)
    t = (t - t.min()) / 12.0
    y = np.column_stack([np.asarray(k, float), np.asarray(n, float) - np.asarray(k, float)])
    r = sm.GLM(y, sm.add_constant(t), family=sm.families.Binomial()).fit()
    b, se = r.params[1], r.bse[1]
    return {"pct_per_year": float((np.exp(b) - 1) * 100),
            "ci_low": float((np.exp(b - 1.96 * se) - 1) * 100),
            "ci_high": float((np.exp(b + 1.96 * se) - 1) * 100),
            "p": float(r.pvalues[1])}


def main():
    os.makedirs(OUT, exist_ok=True)
    raw = scan()
    df = pd.DataFrame([{"arm": a, "month": m, **d} for (a, m), d in raw.items()])
    cols = list(RX)
    df.to_csv(f"{OUT}/vocab_baseline_by_arm.csv", index=False)

    # The control arm covers 2018-2022 only. Pooling both arms across all 72 months would drop
    # the denominator by about two thirds in 2023 while the numerator (mental-health vocabulary,
    # which the keyword filter routes into the matched arm) stayed put, inflating every 2023 rate
    # roughly threefold. Restrict to months where both arms are present.
    both = (df.groupby("month").arm.nunique() == 2)
    keep_months = sorted(both[both].index)
    dropped = sorted(set(df.month) - set(keep_months))
    if dropped:
        print(f"[vocab] restricting to {len(keep_months)} months present in BOTH arms; "
              f"excluding {len(dropped)} ({dropped[0]}..{dropped[-1]}) where only the matched "
              f"arm exists")
    df = df[df.month.isin(keep_months)]
    tot = df.groupby("month")[["n"] + cols].sum().reset_index().sort_values("month")
    for c in cols:
        tot[f"{c}_rate"] = tot[c] / tot.n
    tot.to_csv(f"{OUT}/vocab_baseline.csv", index=False)
    months = tot.month.tolist()

    tagged = pd.read_csv(f"{OUT}/tag_monthly_ci.csv")
    res = {"n_comments": int(tot.n.sum()), "n_months": len(months), "themes": {}}

    print("\n" + "=" * 92)
    print("IS EACH TREND ATHLETE-SPECIFIC, OR INHERITED FROM A COMMUNITY-WIDE VOCABULARY SHIFT?")
    print("=" * 92)
    print(f"scanned {res['n_comments']:,} comments in the 19 communities over {len(months)} months\n")
    print(f"  {'theme':22} {'community vocab %/yr':>22} {'vocab idx':>10} "
          f"{'tagged idx':>11} {'excess':>8}  reading")
    print("  " + "-" * 88)

    for c in cols:
        if c not in tagged.columns:
            continue
        vt = trend(tot[c], tot.n, months)
        m = tot[["month", f"{c}_rate"]].merge(tagged[["month", c]], on="month", how="inner")
        bv, bt = m[f"{c}_rate"].head(12).mean(), m[c].head(12).mean()
        if bv <= 0 or bt <= 0:
            continue
        iv = float(m[f"{c}_rate"].tail(12).mean() / bv * 100)
        it_ = float(m[c].tail(12).mean() / bt * 100)
        ex = it_ / iv
        reading = ("inherited, attenuated" if ex < 0.75 else
                   "tracks community" if ex < 1.25 else "athlete-specific")
        res["themes"][c] = {
            "community_vocab_trend": vt, "vocab_index_2023": iv, "tagged_index_2023": it_,
            "excess_ratio": ex, "reading": reading,
            "vocab_rate_2018_pct": float(bv * 100),
            "vocab_rate_2023_pct": float(m[f"{c}_rate"].tail(12).mean() * 100)}
        print(f"  {c:22} {vt['pct_per_year']:+21.2f} {iv:10.0f} {it_:11.0f} {ex:8.2f}  {reading}")

    print("\n  Index: 2018 = 100. Excess = tagged index / vocabulary index.")
    print("  Excess below 1 means the tagged series grew more slowly than the surrounding")
    print("  community, so the trend is inherited rather than specific to athlete discussion.")

    with open(f"{OUT}/vocab_baseline.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)

    # figure: the two clearest cases side by side
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
    show = [c for c in ["adhd_neurodivergence", "help_seeking", "depression", "anxiety"]
            if c in res["themes"]]
    fig, axes = plt.subplots(1, len(show), figsize=(3.0 * len(show), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for j, c in enumerate(show):
        ax = axes[j]
        m = tot[["month", f"{c}_rate"]].merge(tagged[["month", c]], on="month", how="inner")
        x = np.arange(len(m))
        vi = m[f"{c}_rate"] / m[f"{c}_rate"].head(12).mean() * 100
        ti = m[c] / m[c].head(12).mean() * 100
        ax.plot(x, vi, lw=1.8, color=PAL[2], label="community vocabulary")
        ax.plot(x, ti, lw=1.8, color=PAL[0], label="tagged, relevant comments")
        ax.axhline(100, color=MUTED, lw=0.9, ls=":")
        ticks = [i for i, mm in enumerate(m.month) if mm.endswith("-01")]
        ax.set_xticks(ticks); ax.set_xticklabels([m.month.iloc[i][:4] for i in ticks], fontsize=7)
        ax.set_title(c, loc="left", fontsize=9, color=INK, pad=4)
        if j == 0:
            ax.set_ylabel("Indexed to 2018 = 100")
            ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_vocab_baseline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[vocab] wrote {OUT}/vocab_baseline.csv, vocab_baseline.json, fig5_vocab_baseline.png")


if __name__ == "__main__":
    main()
