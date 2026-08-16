"""Composition-adjusted relevance rate over time.

The naive monthly relevance rate mixes two things: how often athletes actually discuss mental
health, and which of the 19 communities happened to be posting that month. The two are not
separable in the raw series, and in this corpus the second effect is large -- r/nba is 15% of
the relevant corpus and its share collapses in March 2020 when the NBA suspended its season,
which by itself pushes the corpus toward communities that discuss mental health more.

This script builds the (month, subreddit) denominators from the RAW keyword-matched monthly
files, joins the relevant counts from the classified dataset, and fits

    naive      logit(relevant) ~ time
    adjusted   logit(relevant) ~ time + subreddit

plus the COVID interrupted time series in both forms. Where the two disagree, the naive series
was describing sampling rather than behaviour.

Usage: .venv/Scripts/python.exe code/rate_adjusted.py
"""
import csv, glob, json, os, sys
from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.api as sm

csv.field_size_limit(2**31 - 1)
OUT = "results_temporal"
RAW = ["comments/2018-2022/MS_comments_2018_2022/MS_comments_2018_2022/matched/RC_*.csv",
       "comments/2023/comments_2023/filtered_subreddit_keywords/matched/RC_*.csv"]
RELEVANT = "data/classified/final_dataset.csv"
COVID = "2020-03"

# Communities whose comments are predominantly spectator discussion of professional athletes
# rather than people writing about their own training. Kept in the corpus but reported
# separately, because "athlete mental health" means something different in each.
SPECTATOR = {"nba", "CollegeBasketball", "Basketball", "tennis"}

# classify_corpus.py drops dedicated mental-health subreddits BEFORE running the models, so
# those comments can never be flagged relevant. They must come out of the denominator too or
# every rate is diluted by a block of structural zeros -- r/Anxiety alone is 34k comments with
# 0 relevant by construction, which is not evidence about anything.
from classify_corpus import DROP_SUBS  # noqa: E402


def denominators():
    """-> {(month, subreddit): n} over the raw keyword-matched arm."""
    den = defaultdict(int)
    files = []
    for pat in RAW:
        files += sorted(glob.glob(pat))
    print(f"[rate] scanning {len(files)} raw matched month files")
    for i, f in enumerate(files, 1):
        month = os.path.basename(f)[3:10]
        with open(f, encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                s = (r.get("subreddit") or "").strip()
                if s and s not in DROP_SUBS:
                    den[(month, s)] += 1
        if i % 20 == 0:
            print(f"  ... {i}/{len(files)}", flush=True)
    return den


def numerators():
    """-> {(month, subreddit): n_relevant} from the classified (pruned) dataset."""
    num = defaultdict(int)
    with open(RELEVANT, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("arm") != "matched":
                continue
            m = (r.get("time") or "")[:7]
            s = (r.get("subreddit") or "").strip()
            if len(m) == 7 and s:
                num[(m, s)] += 1
    return num


def fit(d, with_sub, extra=None):
    k = d.n_relevant.values.astype(float)
    n = d.n_total.values.astype(float)
    t = (pd.PeriodIndex(d.month, freq="M").astype("int64")).values.astype(float)
    t = (t - t.min()) / 12.0
    cols = [np.ones(len(d)), t]
    names = ["const", "time"]
    if extra == "its":
        post = (np.asarray(d.month) >= COVID).astype(float)
        tp = np.where(post > 0, t - t[np.argmax(np.asarray(d.month) >= COVID)], 0.0)
        cols += [post, tp]
        names += ["post", "post_slope"]
    if with_sub:
        D = pd.get_dummies(d.subreddit, drop_first=True).astype(float).values
        cols.append(D)
    X = np.column_stack(cols)
    y = np.column_stack([k, n - k])
    r = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    out = {}
    for j, nm in enumerate(names):
        if nm == "const":
            continue
        b, se = r.params[j], r.bse[j]
        out[nm] = {"pct_per_year" if "slope" in nm or nm == "time" else "or":
                   float((np.exp(b) - 1) * 100) if ("slope" in nm or nm == "time")
                   else float(np.exp(b)),
                   "or": float(np.exp(b)),
                   "ci_low_or": float(np.exp(b - 1.96 * se)),
                   "ci_high_or": float(np.exp(b + 1.96 * se)),
                   "p": float(r.pvalues[j])}
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    den, num = denominators(), numerators()
    rows = []
    for (m, s), n in den.items():
        rows.append({"month": m, "subreddit": s, "n_total": n,
                     "n_relevant": min(num.get((m, s), 0), n)})
    d = pd.DataFrame(rows).sort_values(["month", "subreddit"]).reset_index(drop=True)
    d = d[d.n_total > 0]
    d.to_csv(f"{OUT}/rate_month_subreddit.csv", index=False)
    print(f"[rate] {d.n_total.sum():,} matched comments, {d.n_relevant.sum():,} relevant, "
          f"{d.month.nunique()} months, {d.subreddit.nunique()} subreddits")

    res = {}
    res["naive"] = fit(d, False)
    res["adjusted"] = fit(d, True)
    res["its_naive"] = fit(d, False, "its")
    res["its_adjusted"] = fit(d, True, "its")

    ns = d[~d.subreddit.isin(SPECTATOR)]
    sp = d[d.subreddit.isin(SPECTATOR)]
    res["adjusted_participant_only"] = fit(ns, True)
    res["adjusted_spectator_only"] = fit(sp, True) if sp.subreddit.nunique() > 1 else None
    res["participant_share_pct"] = float(ns.n_total.sum() / d.n_total.sum() * 100)
    res["participant_rate_pct"] = float(ns.n_relevant.sum() / ns.n_total.sum() * 100)
    res["spectator_rate_pct"] = float(sp.n_relevant.sum() / sp.n_total.sum() * 100)

    def show(name, r, key="time"):
        v = r[key]
        star = "***" if v["p"] < .001 else "**" if v["p"] < .01 else "*" if v["p"] < .05 else "ns"
        print(f"  {name:38} {v['pct_per_year']:+6.2f}%/yr  OR {v['or']:.4f} "
              f"[{v['ci_low_or']:.4f}, {v['ci_high_or']:.4f}]  p={v['p']:.2e} {star}")

    print("\n" + "=" * 78)
    print("COMPOSITION-ADJUSTED RELEVANCE RATE")
    print("=" * 78)
    show("naive (time only)", res["naive"])
    show("adjusted (+ subreddit fixed effects)", res["adjusted"])
    show("adjusted, participant communities", res["adjusted_participant_only"])
    if res["adjusted_spectator_only"]:
        show("adjusted, spectator communities", res["adjusted_spectator_only"])
    print(f"\n  participant communities = {res['participant_share_pct']:.1f}% of matched volume; "
          f"relevance rate {res['participant_rate_pct']:.2f}% "
          f"vs spectator {res['spectator_rate_pct']:.2f}%")

    print(f"\nCOVID INTERRUPTED TIME SERIES (cut {COVID})")
    for nm, r in (("naive", res["its_naive"]), ("adjusted", res["its_adjusted"])):
        lv, sl = r["post"], r["post_slope"]
        print(f"  {nm:9} level OR {lv['or']:.4f} [{lv['ci_low_or']:.4f}, {lv['ci_high_or']:.4f}] "
              f"p={lv['p']:.1e}   slope {sl['pct_per_year']:+6.2f}%/yr p={sl['p']:.1e}")

    with open(f"{OUT}/rate_adjusted.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print(f"\n[rate] wrote {OUT}/rate_adjusted.json, rate_month_subreddit.csv")


if __name__ == "__main__":
    main()
