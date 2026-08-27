"""Layer-1 inter-rater agreement, to the same design as the Layer-2 study.

Rater A = the primary relevance labels (`all_labels.csv`).
Rater B = an independent second pass over a subset (`labels_passB.csv`), blind to A.

Reports, for each dimension (mh, sport) and for the AND-ed relevance decision:
  * Cohen's kappa on the FULL double-rated overlap, with a bootstrap 95% interval
  * the same on a stratified 20% subsample carrying every label combination present,
    which is the design specified for the Layer-2 study, so the two layers are comparable
  * raw agreement and each rater's positive rate, since kappa alone hides a base-rate shift

Usage: .venv/Scripts/python.exe code/metrics_interrater_l1.py [--pct 0.20]
"""
import csv, json, os, random, sys
from collections import Counter

import numpy as np
from sklearn.metrics import cohen_kappa_score

csv.field_size_limit(2**31 - 1)
D = "data/data_relevance_ratings/comments"
OUT = "results_temporal"
SEED = 20260818
DIMS = ["mh", "sport"]


def read(path, cols):
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            rid = (r.get("random_id") or "").strip()
            if not rid:
                continue
            vals = {}
            ok = True
            for c in cols:
                v = (r.get(c) or "").strip()
                if v not in ("0", "1"):
                    ok = False
                    break
                vals[c] = int(v)
            if ok:
                out[rid] = vals
    return out


def kappa_ci(a, b, n_boot=2000, seed=SEED):
    a, b = np.asarray(a), np.asarray(b)
    k = cohen_kappa_score(a, b)
    rs = np.random.RandomState(seed)
    n = len(a)
    boots = []
    for _ in range(n_boot):
        idx = rs.randint(0, n, n)
        if len(set(a[idx])) < 2 or len(set(b[idx])) < 2:
            continue
        boots.append(cohen_kappa_score(a[idx], b[idx]))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return float(k), float(lo), float(hi)


def score(ids, A, Bl, label):
    print(f"\n===== {label}  (n = {len(ids)}) =====")
    print(f"  {'decision':12} {'kappa':>7} {'95% CI':>18} {'raw agr':>9} "
          f"{'A pos':>7} {'B pos':>7}")
    res = {}
    for d in DIMS + ["relevant"]:
        if d == "relevant":
            a = [A[i]["mh"] & A[i]["sport"] for i in ids]
            b = [Bl[i]["mh"] & Bl[i]["sport"] for i in ids]
        else:
            a = [A[i][d] for i in ids]
            b = [Bl[i][d] for i in ids]
        k, lo, hi = kappa_ci(a, b)
        raw = float(np.mean(np.asarray(a) == np.asarray(b)))
        res[d] = {"kappa": k, "ci_low": lo, "ci_high": hi, "raw_agreement": raw,
                  "a_pos": int(sum(a)), "b_pos": int(sum(b)), "n": len(ids)}
        print(f"  {d:12} {k:7.3f} [{lo:6.3f}, {hi:6.3f}] {raw:9.3f} "
              f"{sum(a):7} {sum(b):7}")
    return res


def main():
    pct = float(sys.argv[sys.argv.index("--pct") + 1]) if "--pct" in sys.argv else 0.20
    os.makedirs(OUT, exist_ok=True)
    A = read(f"{D}/all_labels.csv", DIMS)
    Bl = read(f"{D}/labels_passB.csv", DIMS)
    ids = sorted(set(A) & set(Bl))
    print(f"[l1-kappa] rater A {len(A):,} | rater B {len(Bl):,} | double-rated overlap {len(ids):,}")

    full = score(ids, A, Bl, "FULL double-rated overlap")

    # stratified subsample: proportional across the four (mh, sport) cells, so every label
    # combination is represented, matching the Layer-2 sampling design
    cells = {}
    for i in ids:
        cells.setdefault((A[i]["mh"], A[i]["sport"]), []).append(i)
    rng = random.Random(SEED)
    sub = []
    for cell, members in sorted(cells.items()):
        rng.shuffle(members)
        take = max(2, int(round(len(members) * pct)))
        sub.extend(members[:take])
    print(f"\n[l1-kappa] stratified {pct:.0%} subsample: {len(sub)} rows across "
          f"{len(cells)} (mh, sport) cells "
          + ", ".join(f"{k}:{min(max(2,int(round(len(v)*pct))), len(v))}"
                      for k, v in sorted(cells.items())))
    strat = score(sub, A, Bl, f"STRATIFIED {pct:.0%} SUBSAMPLE")

    out = {"n_overlap": len(ids), "pct": pct, "full": full, "stratified": strat,
           "cells": {str(k): len(v) for k, v in sorted(cells.items())}}
    with open(f"{OUT}/kappa_layer1.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  Interpretation: >=0.8 almost perfect, 0.6-0.8 substantial, 0.4-0.6 moderate.")
    print(f"  Layer-2 pooled kappa was 0.879 (mean per-tag 0.862) for reference.")
    print(f"\n[l1-kappa] wrote {OUT}/kappa_layer1.json")


if __name__ == "__main__":
    main()
