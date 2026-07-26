"""Exploratory pass over the Layer-1-relevant corpus to ground the Layer-2 tag taxonomy.

Reports (a) subreddit mix, (b) frequency of candidate theme cues, (c) co-occurrence, so the
tag set is derived from what the data actually contains rather than invented a priori.

Usage: .venv/Scripts/python.exe code/explore_tags.py <classified_dir> [--n N]
"""
import csv, glob, os, random, re, sys
from collections import Counter
csv.field_size_limit(2**31 - 1)

# Candidate themes drawn from the sport-psychology / athlete-MH literature. These are ONLY
# exploratory probes to measure prevalence -- not the final labeling rule.
PROBES = {
    "depression":      r"\bdepress(ed|ion|ive)?\b|\bhopeless\b|\bworthless\b|\bempty inside\b|\bno joy\b",
    "anxiety":         r"\banxi(ety|ous)\b|\bpanic attack\b|\bnervous\b|\bon edge\b|\bdread\b",
    "stress":          r"\bstress(ed|ful|ing)?\b|\boverwhelm(ed|ing)\b|\bpressure\b",
    "burnout":         r"\bburn(ed|t)? ?out\b|\bburnout\b|\bexhaust(ed|ion)\b|\bno motivation\b|\blost motivation\b|\bdon'?t want to train\b",
    "perf_anxiety":    r"\bchoke(d|s|ing)?\b|\bperformance anxiety\b|\byips\b|\bmental block\b|\bfear of failure\b|\bconfidence\b|\bself.?doubt\b|\bovcerthink|\boverthink(ing)?\b",
    "body_image":      r"\beating disorder\b|\banorexi|\bbulimi|\bbinge\b|\bpurge\b|\bbody image\b|\bhate my body\b|\bfat\b|\bweight\b|\bcalorie",
    "sleep":           r"\binsomnia\b|\bcan'?t sleep\b|\bsleepless\b|\btrouble sleeping\b|\bsleep problems?\b",
    "injury_distress": r"\binjur(y|ies|ed)\b|\btorn\b|\bacl\b|\bsurgery\b|\brehab\b|\bsidelined\b",
    "substance":       r"\bdrink(ing)?\b|\balcohol\b|\bdrunk\b|\bweed\b|\bdrugs?\b|\badderall\b|\bsober\b",
    "self_harm":       r"\bsuicid(e|al)\b|\bkill myself\b|\bself.?harm\b|\bcutting\b|\bend it all\b",
    "loneliness":      r"\blonely\b|\bloneliness\b|\bisolat(ed|ion)\b|\bno friends\b|\balone\b",
    "trauma":          r"\bptsd\b|\btrauma(tic|tized)?\b|\babus(e|ive|ed)\b|\bflashback",
    "adhd_neuro":      r"\badhd\b|\badd\b|\bautis(m|tic)\b|\bocd\b|\bbipolar\b",
    "therapy_help":    r"\btherap(y|ist)\b|\bcounsel(l)?or\b|\bpsychiatrist\b|\bmedication\b|\bssri\b|\bantidepressant",
    "identity":        r"\bretire(ment|d)?\b|\bquit(ting)?\b|\bwho am i\b|\bidentity\b|\bwithout (the )?sport\b",
}

def main():
    d = sys.argv[1]
    n_show = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 0
    rx = {k: re.compile(v, re.I) for k, v in PROBES.items()}
    subs, hits, total = Counter(), Counter(), 0
    pairs = Counter()
    examples = {k: [] for k in PROBES}
    rng = random.Random(1)
    for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
        with open(f, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                t = r.get("text") or ""
                total += 1
                subs[r.get("subreddit", "")] += 1
                got = [k for k, p in rx.items() if p.search(t)]
                for k in got:
                    hits[k] += 1
                    if len(examples[k]) < 400 and rng.random() < 0.02:
                        examples[k].append(t[:400])
                for i, a in enumerate(got):
                    for b in got[i + 1:]:
                        pairs[tuple(sorted((a, b)))] += 1
    print(f"TOTAL relevant comments scanned: {total:,}\n")
    print("== top 30 subreddits ==")
    for s, c in subs.most_common(30):
        print(f"  {c:8,}  {c/total*100:5.1f}%  {s}")
    print("\n== candidate theme prevalence (regex probe, high-recall) ==")
    for k, c in hits.most_common():
        print(f"  {c:8,}  {c/total*100:5.1f}%  {k}")
    print("\n== top 25 theme co-occurrences ==")
    for (a, b), c in pairs.most_common(25):
        print(f"  {c:7,}  {a} + {b}")
    if n_show:
        print("\n== sample texts per theme ==")
        for k in PROBES:
            print(f"\n--- {k} ---")
            for t in examples[k][:n_show]:
                print("   *", t.replace("\n", " ")[:300])

if __name__ == "__main__":
    main()
