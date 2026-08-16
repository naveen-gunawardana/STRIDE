"""Per-tag temporal trends, with and without controlling for subreddit composition.

The corpus is drawn from a fixed set of 19 communities, but their relative sizes move a lot
over the six years (r/xxfitness and r/Fitness each shed ~11 percentage points of share, while
r/tennis and r/crossfit gain). A raw trend in, say, body-image talk could therefore be a fact
about athletes or a fact about which subreddit happened to be posting that month.

So every tag gets two numbers:

  naive     logit(p) ~ time                        -- what a raw plot shows
  adjusted  logit(p) ~ time + subreddit            -- within-community change, composition held
                                                      fixed by subreddit fixed effects

When the two disagree, the raw trend was composition. Reporting only the naive number would be
the single easiest way to publish an artefact from this dataset.

A third column drops r/nba, which is 15% of the relevant corpus but is a spectator community:
comments there are largely fans discussing professional players rather than athletes talking
about themselves.

Usage: .venv/Scripts/python.exe code/tag_trends.py [--in F]
"""
import csv, json, os, sys
from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.api as sm

csv.field_size_limit(2**31 - 1)
OUT = "results_temporal"
SPECTATOR = {"nba", "CollegeBasketball", "Basketball", "tennis"}


def load_cells(path):
    """-> DataFrame with one row per (month, subreddit): n and per-tag positives."""
    cells = defaultdict(lambda: defaultdict(int))
    tags = []
    with open(path, encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        tags = [c[4:] for c in (rd.fieldnames or []) if c.startswith("tag_")]
        for r in rd:
            if r.get("arm") != "matched":
                continue
            m = (r.get("time") or "")[:7]
            s = (r.get("subreddit") or "").strip()
            if len(m) != 7 or not s:
                continue
            key = (m, s)
            cells[key]["n"] += 1
            for t in tags:
                if r.get(f"tag_{t}") == "1":
                    cells[key][t] += 1
    rows = []
    for (m, s), d in cells.items():
        row = {"month": m, "subreddit": s, "n": d["n"]}
        for t in tags:
            row[t] = d[t]
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["month", "subreddit"]).reset_index(drop=True)
    return df, tags


def fit(df, tag, with_sub):
    """Binomial GLM on (k, n-k) cells. Time in years. Returns %/yr and p."""
    d = df[df.n > 0].copy()
    k = d[tag].values.astype(float)
    n = d["n"].values.astype(float)
    if k.sum() < 50:
        return None
    t = (pd.PeriodIndex(d.month, freq="M").astype("int64")).values.astype(float)
    t = (t - t.min()) / 12.0
    X = [np.ones(len(d)), t]
    if with_sub:
        D = pd.get_dummies(d.subreddit, drop_first=True).astype(float).values
        X.append(D)
    X = np.column_stack(X)
    y = np.column_stack([k, n - k])
    try:
        r = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    except Exception:
        return None
    b, se = r.params[1], r.bse[1]
    return {"pct_per_year": float((np.exp(b) - 1) * 100),
            "ci_low": float((np.exp(b - 1.96 * se) - 1) * 100),
            "ci_high": float((np.exp(b + 1.96 * se) - 1) * 100),
            "p": float(r.pvalues[1])}


def main():
    inp = sys.argv[sys.argv.index("--in") + 1] if "--in" in sys.argv \
        else "data/classified/final_dataset_tagged_v3.csv"
    os.makedirs(OUT, exist_ok=True)
    df, tags = load_cells(inp)
    tot = df.n.sum()
    print(f"[trends] {tot:,} relevant comments | {len(tags)} tags | "
          f"{df.month.nunique()} months | {df.subreddit.nunique()} subreddits")

    no_spec = df[~df.subreddit.isin(SPECTATOR)]
    print(f"[trends] excluding spectator communities ({', '.join(sorted(SPECTATOR))}): "
          f"{no_spec.n.sum():,} comments remain "
          f"({no_spec.n.sum()/tot*100:.1f}%)")

    res = {}
    print(f"\n{'tag':24} {'prev%':>7} {'naive %/yr':>12} {'adjusted %/yr':>15} "
          f"{'no-spectator':>14}")
    print("-" * 78)
    for t in sorted(tags, key=lambda x: -df[x].sum()):
        prev = df[t].sum() / tot * 100
        a = fit(df, t, False)
        b = fit(df, t, True)
        c = fit(no_spec, t, True)
        if a is None:
            continue
        res[t] = {"prevalence_pct": float(prev), "naive": a, "adjusted": b,
                  "adjusted_no_spectator": c}

        def f(x):
            if x is None:
                return "        n/a"
            star = "***" if x["p"] < .001 else "**" if x["p"] < .01 \
                else "*" if x["p"] < .05 else " "
            return f"{x['pct_per_year']:+7.2f}{star:<3}"
        print(f"{t:24} {prev:6.2f}% {f(a):>12} {f(b):>15} {f(c):>14}")

    # how much of each naive trend survives the composition control
    print("\nCOMPOSITION CHECK (adjusted / naive; near 1 = real, near 0 = composition)")
    for t, v in sorted(res.items(), key=lambda kv: -abs(kv[1]["naive"]["pct_per_year"])):
        if v["adjusted"] is None or abs(v["naive"]["pct_per_year"]) < 1:
            continue
        ratio = v["adjusted"]["pct_per_year"] / v["naive"]["pct_per_year"]
        flag = "  <-- mostly composition" if ratio < 0.5 else ""
        print(f"  {t:24} naive {v['naive']['pct_per_year']:+6.2f}  "
              f"adj {v['adjusted']['pct_per_year']:+6.2f}  ratio {ratio:+.2f}{flag}")

    # per-subreddit prevalence, to support the scope statement
    big = df.groupby("subreddit").n.sum().sort_values(ascending=False).head(8).index.tolist()
    print(f"\nTAG PREVALENCE BY COMMUNITY (top 8, % of that community's relevant comments)")
    hdr = f"  {'tag':24}" + "".join(f"{s[:11]:>12}" for s in big)
    print(hdr)
    bysub = {}
    for t in sorted(tags, key=lambda x: -df[x].sum())[:8]:
        line = f"  {t:24}"
        bysub[t] = {}
        for s in big:
            d = df[df.subreddit == s]
            p = d[t].sum() / max(d.n.sum(), 1) * 100
            bysub[t][s] = float(p)
            line += f"{p:11.1f}%"
        print(line)

    with open(f"{OUT}/tag_trends.json", "w", encoding="utf-8") as fh:
        json.dump({"n_relevant": int(tot), "trends": res, "by_subreddit": bysub,
                   "spectator_excluded": sorted(SPECTATOR)}, fh, indent=2)
    df.to_csv(f"{OUT}/tag_month_subreddit.csv", index=False)
    print(f"\n[trends] wrote {OUT}/tag_trends.json, tag_month_subreddit.csv")


if __name__ == "__main__":
    main()
