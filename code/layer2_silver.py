"""Build the Layer-2 multi-label silver training set from the Layer-1-relevant corpus.

Applies the weak-supervision labeling functions (code/layer2_lexicon.py) to every relevant
comment, then samples a training file that keeps ALL rule positives (the rare tags need them)
plus a random pool of comments that are clean negatives, so the class balance is workable
without discarding signal.

Abstains are written as -1 and masked out of the loss by the trainer.

Gold comment ids are excluded so the held-out evaluation stays honest.

Usage:
  .venv/Scripts/python.exe code/layer2_silver.py [--max-neg N] [--out data/layer2/silver.csv]
"""
import csv, glob, os, random, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layer2_lexicon as L
csv.field_size_limit(2**31 - 1)

DATASET = "data/classified/final_dataset.csv"
GOLD_KEYS = "data/data_layer2_ratings/*_key.csv"
SEED = 20260725

GOLD_RATED = "data/data_layer2_ratings/*_rated.csv"

def gold_ids():
    ids = set()
    for p in glob.glob(GOLD_KEYS):
        with open(p, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                ids.add(r.get("comment_id", ""))
    return ids

def gold_texts():
    """Also exclude by TEXT, not just id: the corpus contains duplicate comment bodies under
    different ids (reposts, automod copies of a post). Excluding by id alone left 3 gold-test
    rows in the silver training set -- small, but it is exactly the kind of leak that quietly
    inflates a held-out score."""
    txt = set()
    for p in glob.glob(GOLD_RATED):
        with open(p, encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                t = (r.get("text") or "").strip()
                if t:
                    txt.add(t)
    return txt

def main():
    max_neg = int(sys.argv[sys.argv.index("--max-neg") + 1]) if "--max-neg" in sys.argv else 45_000
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "data/layer2/silver.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    excl = gold_ids()
    excl_txt = gold_texts()
    print(f"[silver] excluding {len(excl):,} gold comment ids and {len(excl_txt):,} gold texts",
          flush=True)

    pos_rows, neg_rows = [], []
    counts = Counter(); abst = Counter(); negc = Counter()
    n = 0
    rng = random.Random(SEED)
    with open(DATASET, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("arm") != "matched" or r.get("id", "") in excl:
                continue
            t = (r.get("text") or "").strip()
            if len(t) < 20 or t in excl_txt:
                continue
            n += 1
            lab = L.label(t)
            row = [t] + [(-1 if lab[k] is None else lab[k]) for k in L.TAGS]
            any_pos = False
            for k in L.TAGS:
                if lab[k] == 1:
                    counts[k] += 1; any_pos = True
                elif lab[k] is None:
                    abst[k] += 1
                else:
                    negc[k] += 1
            (pos_rows if any_pos else neg_rows).append(row)
            if n % 25000 == 0:
                print(f"  ... {n:,} scanned, {len(pos_rows):,} with >=1 rule positive", flush=True)

    rng.shuffle(neg_rows)
    keep_neg = neg_rows[:max_neg]
    rows = pos_rows + keep_neg
    rng.shuffle(rows)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["text"] + L.TAGS)
        w.writerows(rows)

    print(f"\n[silver] scanned {n:,} relevant comments")
    print(f"[silver] {len(pos_rows):,} have >=1 rule positive; kept {len(keep_neg):,} of "
          f"{len(neg_rows):,} all-negative comments")
    print(f"[silver] wrote {len(rows):,} rows -> {out}\n")
    print(f"  {'tag':22} {'pos':>8} {'neg':>9} {'abstain':>9}   pos% of relevant")
    for k in L.TAGS:
        print(f"  {k:22} {counts[k]:8,} {negc[k]:9,} {abst[k]:9,}   {counts[k]/max(n,1)*100:5.2f}%")

if __name__ == "__main__":
    main()
