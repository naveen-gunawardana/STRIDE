"""Subreddit composition of the relevant corpus, overall and over time.

Two questions a reviewer will ask about any temporal claim from this dataset:

  1. What is the corpus actually made of? (the recreational-vs-elite scope statement)
  2. Did the population change underneath the trend? If r/xxfitness grew from 5% to 25% of
     the corpus, a rise in body-image talk says something about sampling, not about athletes.

This writes the composition table and a per-month share for the largest communities, plus a
Herfindahl concentration index per month so the shift can be summarised in one number.

Usage: .venv/Scripts/python.exe code/subreddit_composition.py [--in F] [--top N]
"""
import csv, json, os, sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

csv.field_size_limit(2**31 - 1)
OUT = "results_temporal"


def main():
    inp = sys.argv[sys.argv.index("--in") + 1] if "--in" in sys.argv \
        else "data/classified/final_dataset.csv"
    top_n = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 20
    os.makedirs(OUT, exist_ok=True)

    overall = Counter()
    by_month = defaultdict(Counter)
    months = set()
    n = 0
    with open(inp, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("arm") != "matched":
                continue
            sub = (r.get("subreddit") or "").strip()
            mth = (r.get("time") or "")[:7]
            if not sub or len(mth) != 7:
                continue
            n += 1
            overall[sub] += 1
            by_month[mth][sub] += 1
            months.add(mth)
    months = sorted(months)
    print(f"[subs] {n:,} relevant comments across {len(overall):,} subreddits, "
          f"{len(months)} months")

    # ---- overall composition
    rows = []
    cum = 0
    for sub, c in overall.most_common(top_n):
        cum += c
        rows.append({"subreddit": sub, "n": c, "pct": c / n * 100, "cum_pct": cum / n * 100})
    comp = pd.DataFrame(rows)
    comp.to_csv(f"{OUT}/subreddit_composition.csv", index=False)
    print(f"\n  {'subreddit':28} {'n':>8} {'%':>6} {'cum%':>7}")
    for r in rows:
        print(f"  {r['subreddit']:28} {r['n']:8,} {r['pct']:5.1f}% {r['cum_pct']:6.1f}%")

    top4 = [s for s, _ in overall.most_common(4)]
    share4 = sum(overall[s] for s in top4) / n * 100
    print(f"\n  top-4 ({', '.join(top4)}) = {share4:.1f}% of the relevant corpus")

    # ---- composition over time for the biggest communities
    keep = [s for s, _ in overall.most_common(8)]
    mrows = []
    for m in months:
        tot = sum(by_month[m].values())
        row = {"month": m, "n": tot}
        for s in keep:
            row[s] = by_month[m][s] / max(tot, 1)
        # Herfindahl index over ALL subreddits that month: 1/HHI = effective no. of communities
        shares = np.array(list(by_month[m].values()), float) / max(tot, 1)
        row["hhi"] = float((shares ** 2).sum())
        row["effective_subs"] = float(1.0 / max((shares ** 2).sum(), 1e-12))
        mrows.append(row)
    md = pd.DataFrame(mrows)
    md.to_csv(f"{OUT}/subreddit_monthly.csv", index=False)

    print(f"\n  concentration: effective number of communities "
          f"{md.effective_subs.iloc[0]:.1f} ({months[0]}) -> "
          f"{md.effective_subs.iloc[-1]:.1f} ({months[-1]})")
    print(f"\n  share drift over the period (first 12 months -> last 12 months)")
    print(f"  {'subreddit':28} {'start':>7} {'end':>7} {'delta':>8}")
    drift = {}
    for s in keep:
        a, b = md[s].head(12).mean() * 100, md[s].tail(12).mean() * 100
        drift[s] = {"start_pct": float(a), "end_pct": float(b), "delta_pp": float(b - a)}
        print(f"  {s:28} {a:6.1f}% {b:6.1f}% {b-a:+7.1f}pp")

    with open(f"{OUT}/subreddit_stats.json", "w", encoding="utf-8") as fh:
        json.dump({"n_relevant": n, "n_subreddits": len(overall),
                   "top4": top4, "top4_share_pct": share4,
                   "composition": rows, "drift": drift,
                   "effective_subs_start": float(md.effective_subs.iloc[0]),
                   "effective_subs_end": float(md.effective_subs.iloc[-1])}, fh, indent=2)
    print(f"\n[subs] wrote {OUT}/subreddit_composition.csv, subreddit_monthly.csv, "
          f"subreddit_stats.json")


if __name__ == "__main__":
    main()
