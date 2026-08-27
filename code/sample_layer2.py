"""Draw a stratified sample from the Layer-1-relevant dataset for Layer-2 tag labeling.

Two sampling modes:

--mode prop   (default) year x sport-family, PROPORTIONAL allocation, fixed seed.
              The corpus is dominated by a few large subreddits and spans 2018-2023; a plain
              random draw would over-represent xxfitness/running and cluster on the largest
              years. Proportional stratification keeps sport type and period represented in
              proportion, so tag prevalence measured here is an honest corpus estimate. This is
              the natural-distribution TEST set.

--mode tag    THEME-stratified: quota per tag using the high-recall lexicon cues as a candidate
              finder, plus a cue-free block. Rationale: at natural prevalence a 100-comment
              sample yields ~2-3 positives for the rarer tags (self-harm, injury distress),
              which cannot support an honest per-tag recall estimate at the 0.8 bar. Drawing on
              a high-RECALL cue (not the high-precision one used for silver labels) enriches
              positives without presupposing the answer -- the rater still decides, and the
              cue-free block measures what the cues miss.
              NOTE: metrics on this set are per-tag estimates on an enriched sample; corpus-level
              prevalence must be read off the --mode prop sample instead.

Both write a BLINDED rating file (text only -- subreddit/author/date hidden, as in the Layer-1
protocol) plus a local-only key mapping row_id -> provenance.

Usage:
  .venv/Scripts/python.exe code/sample_layer2.py --n 100 --out data/data_layer2_ratings/layer2_sample_100
  .venv/Scripts/python.exe code/sample_layer2.py --mode tag --per-tag 30 --out .../layer2_tagstrat
  optional: --exclude <csv,...>  (rating/key files whose comment ids must not be re-drawn)
"""
import csv, os, random, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
csv.field_size_limit(2**31 - 1)
import layer2_lexicon as L

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
# Kept in sync with layer2_lexicon.TAGS so the v3 themes are drawn too; an earlier hardcoded
# list covered only the original ten, which left the newer themes without enrichment.
TAGS = list(L.TAGS)

def family(sub):
    return FAMILY.get(sub, "individual_skill")

# Mid-precision candidate finders for tags whose HIGH-RECALL cue is swamped by wrong-sense text.
# --mode tag drew only 6 self-harm positives in 330 because its HR net ("kill", "dead", "die")
# is mostly sports slang; 6 positives cannot support a recall estimate at the 0.8 bar. These
# patterns sit BETWEEN HR and HP: narrow enough to surface real positives, still broad enough
# that the rater -- not the rule -- decides, so recall is not measured on the rule's own hits.
FOCUS = {
    "self_harm_suicide": r"suicid|kill (?:my|him|her|them)self|self.?harm|\bcutting\b|end (?:it all|my life)|want(?:ed)? to die|took (?:his|her|their) own life|overdose|hotline",
    "injury_distress": r"(?:\binjur|\bacl\b|surgery|\brehab|sidelined|torn|fracture|physio)[\s\S]{0,200}(?:depress|anxi|frustrat|miserabl|\bsad\b|devastat|mental|upset|\bdown\b|discourag|scared|afraid)|(?:depress|anxi|frustrat|miserabl|devastat|mental|discourag)[\s\S]{0,200}(?:\binjur|\bacl\b|surgery|\brehab|sidelined)",
    "performance_psych": r"(?:chok|yips|confidence|nerves|nervous|self.?doubt|mental block|overthink|performance anxiety|in my head|pressure)[\s\S]{0,200}(?:race|game|meet|comp|match|perform|lift|shot|season|tournament)|(?:race|game|meet|comp|match|tournament)[\s\S]{0,200}(?:chok|yips|nerves|self.?doubt|mental block|performance anxiety)",
    "burnout_motivation": r"burn(?:ed|t)? ?out|burnout|\bmotivat|dread|mentally (?:exhaust|drain|fried)|going through the motions",
    "stress_pressure": r"stressed|overwhelm|too much pressure|under pressure|burden|expectations",
}

def sample_focus(pool, tags, per_tag, rng, seen):
    """Targeted draw for tags the theme-stratified pass left with too few positives."""
    import re as _re
    picked = []
    for t in tags:
        rx = _re.compile(FOCUS[t], _re.I)
        cand = [r for r in pool if r.get("id", "") not in seen and rx.search(r.get("text") or "")]
        rng.shuffle(cand)
        got = cand[:per_tag]
        for r in got:
            seen.add(r.get("id", ""))
        picked += got
        print(f"  focus {t:22} {len(cand):8,} candidates -> drew {len(got)}")
    return picked

def sample_by_tag(pool, per_tag, rng):
    """Quota per tag using HIGH-RECALL cues as a candidate finder + a cue-free block.

    Deliberately uses HR (not the high-precision HP that generates silver labels): HR is a broad
    "could plausibly be this theme" net, so the drawn candidates are genuinely ambiguous and the
    human rating is doing real work. Drawing on HP would sample exactly the cases the silver
    rules already get right and inflate every metric.
    """
    import layer2_lexicon as L
    taken, picked = set(), []
    for t in TAGS:
        cand = [r for r in pool if id(r) not in taken and L.hr_match(r.get("text") or "", t)]
        rng.shuffle(cand)
        got = cand[:per_tag]
        for r in got:
            taken.add(id(r))
        picked += got
        print(f"  {t:22} {len(cand):8,} candidates -> drew {len(got)}")
    # cue-free block: no HR cue for ANY tag. Measures false positives on genuinely off-theme text.
    nocue = [r for r in pool if id(r) not in taken and not L.any_hr(r.get("text") or "")]
    rng.shuffle(nocue)
    k = max(10, per_tag)
    picked += nocue[:k]
    print(f"  {'(no cue for any tag)':22} {len(nocue):8,} candidates -> drew {min(k, len(nocue))}")
    return picked

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
    mode = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "prop"
    per_tag = int(sys.argv[sys.argv.index("--per-tag") + 1]) if "--per-tag" in sys.argv else 30
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
    pool = []
    with open(DATASET, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("arm") != "matched":
                continue
            t = (r.get("text") or "").strip()
            if len(t) < 20 or r.get("id", "") in exclude:      # skip stubs the rater can't judge
                continue
            year = (r.get("time") or "")[:4]
            strata[(year, family(r.get("subreddit", "")))].append(r)
            pool.append(r)

    rng = random.Random(SEED if mode == "prop" else SEED + 1)
    if mode == "focus":
        tags = (sys.argv[sys.argv.index("--tags") + 1].split(",")
                if "--tags" in sys.argv else list(FOCUS))
        picked = sample_focus(pool, tags, per_tag, rng, set(exclude))
    elif mode == "tag":
        picked = sample_by_tag(pool, per_tag, rng)
    else:
        sizes = {k: len(v) for k, v in strata.items()}
        alloc = allocate(sizes, n)
        picked = []
        for k in sorted(strata):
            picked += rng.sample(strata[k], min(alloc[k], len(strata[k])))
        print(f"[sample] {sum(sizes.values()):,} eligible matched-arm comments in {len(sizes)} strata")
        print("  stratum                        pool      drawn")
        for k in sorted(sizes):
            if alloc[k]:
                print(f"  {k[0]} {k[1]:<20} {sizes[k]:8,}  {alloc[k]:5}")
    rng.shuffle(picked)
    print(f"\n[sample] drew {len(picked)} (mode={mode}, seed {SEED})")

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
