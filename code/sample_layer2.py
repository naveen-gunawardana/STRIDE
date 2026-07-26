"""Draw a stratified random sample from the Layer-1-relevant dataset for Layer-2 tag labeling.

Stratification: year x sport-family, proportional allocation (largest remainder), fixed seed.
Rationale: the corpus is dominated by a few large subreddits and spans 2018-2023; a plain random
draw would over-represent xxfitness/running and cluster on whichever years are largest. Stratifying
on both keeps sport type and time period represented in proportion, so per-tag prevalence measured
on the sample is an honest estimate for the corpus.

Writes a BLINDED rating file (text only -- subreddit/author/date hidden, as in the Layer-1
protocol) plus a local-only key mapping row_id -> provenance.

Usage:
  .venv/Scripts/python.exe code/sample_layer2.py --n 100 --out data/data_layer2_ratings/layer2_sample_100
  optional: --exclude <csv>  (comma-sep rating files whose ids must not be re-drawn)
"""
import csv, os, random, sys
from collections import defaultdict
csv.field_size_limit(2**31 - 1)

DATASET = "data/classified/final_dataset.csv"
SEED = 20260725

FAMILY = {
    "xxfitness": "strength_fitness", "Fitness": "strength_fitness",
    "bodybuilding": "strength_fitness", "weightroom": "strength_fitness",
    "crossfit": "strength_fitness",
    "running": "endurance", "triathlon": "endurance", "Swimming": "endurance",
    "Rowing": "endurance", "trackandfield": "endurance",
    "nba": "team_ball", "CollegeBasketball": "team_ball", "Basketball": "team_ball",
    "BasketballTips": "team_ball", "volleyball": "team_ball",
    "tennis": "individual_skill", "climbing": "individual_skill",
    "Gymnastics": "individual_skill", "sportspsychology": "individual_skill",
}
TAGS = ["depression", "anxiety", "stress_pressure", "burnout_motivation", "performance_psych",
        "body_image_eating", "injury_distress", "self_harm_suicide", "help_seeking",
        "exercise_coping"]

def family(sub):
    return FAMILY.get(sub, "individual_skill")

def allocate(strata_sizes, n):
    """Proportional allocation with largest-remainder so the parts sum to exactly n."""
    total = sum(strata_sizes.values())
    exact = {k: v / total * n for k, v in strata_sizes.items()}
    base = {k: int(v) for k, v in exact.items()}
    left = n - sum(base.values())
    for k, _ in sorted(exact.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True)[:left]:
        base[k] += 1
    return base

def main():
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 100
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else \
        f"data/data_layer2_ratings/layer2_sample_{n}"
    exclude = set()
    if "--exclude" in sys.argv:
        for p in sys.argv[sys.argv.index("--exclude") + 1].split(","):
            if os.path.exists(p):
                with open(p, encoding="utf-8", newline="") as fh:
                    for r in csv.DictReader(fh):
                        exclude.add(r.get("comment_id", ""))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    # Pass 1: bucket row offsets by stratum (matched arm only -- baseline is the control, not the study set)
    strata = defaultdict(list)
    with open(DATASET, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("arm") != "matched":
                continue
            t = (r.get("text") or "").strip()
            if len(t) < 20 or r.get("id", "") in exclude:      # skip stubs the rater can't judge
                continue
            year = (r.get("time") or "")[:4]
            strata[(year, family(r.get("subreddit", "")))].append(r)

    sizes = {k: len(v) for k, v in strata.items()}
    alloc = allocate(sizes, n)
    rng = random.Random(SEED)
    picked = []
    for k in sorted(strata):
        take = min(alloc[k], len(strata[k]))
        picked += rng.sample(strata[k], take)
    rng.shuffle(picked)

    print(f"[sample] {sum(sizes.values()):,} eligible matched-arm comments in {len(sizes)} strata")
    print(f"[sample] drew {len(picked)} (seed {SEED})\n")
    print("  stratum                        pool      drawn")
    for k in sorted(sizes):
        if alloc[k]:
            print(f"  {k[0]} {k[1]:<20} {sizes[k]:8,}  {alloc[k]:5}")

    rated, key = out + "_rated.csv", out + "_key.csv"
    with open(rated, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row_id", "text"] + TAGS + ["notes"])
        for i, r in enumerate(picked, 1):
            w.writerow([i, r.get("text", "")] + [""] * len(TAGS) + [""])
    with open(key, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["row_id", "comment_id", "subreddit", "time",
                                           "year", "family", "p_mh", "p_sport"])
        w.writeheader()
        for i, r in enumerate(picked, 1):
            w.writerow(dict(row_id=i, comment_id=r.get("id", ""), subreddit=r.get("subreddit", ""),
                            time=r.get("time", ""), year=(r.get("time") or "")[:4],
                            family=family(r.get("subreddit", "")),
                            p_mh=r.get("p_mh", ""), p_sport=r.get("p_sport", "")))
    print(f"\n[sample] blinded rating file -> {rated}")
    print(f"[sample] provenance key      -> {key}  (gitignored)")

if __name__ == "__main__":
    main()
