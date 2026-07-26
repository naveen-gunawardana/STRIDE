"""Split the hand-labeled Layer-2 gold into train / dev / test.

Two rules that keep the evaluation honest:

  1. The PROPORTIONAL sample (layer2_prop100) goes entirely into TEST. It is the only set drawn
     at natural corpus prevalence, so it is the one that answers "how does this behave on the
     real dataset?" -- it must never be trained or tuned on.
  2. The theme-stratified and focus samples are split by ITERATIVE STRATIFICATION rather than
     at random. With 10 correlated labels and tags as rare as self-harm, a random split can
     easily leave a tag with 4 positives in test; iterative stratification allocates each row to
     whichever split is currently furthest from its quota on that row's rarest label, so every
     tag keeps proportional support everywhere.

Usage: .venv/Scripts/python.exe code/split_gold.py [--test 0.40] [--dev 0.20]
"""
import csv, glob, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layer2_lexicon import TAGS
csv.field_size_limit(2**31 - 1)

RATED = "data/data_layer2_ratings"
OUT = "data/layer2"
SEED = 20260726

def read(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return [r for r in csv.DictReader(fh) if (r.get("text") or "").strip()]

def npos(rows, t):
    return sum(1 for r in rows if r.get(t) == "1")

def main():
    p_test = float(sys.argv[sys.argv.index("--test") + 1]) if "--test" in sys.argv else 0.40
    p_dev = float(sys.argv[sys.argv.index("--dev") + 1]) if "--dev" in sys.argv else 0.20
    os.makedirs(OUT, exist_ok=True)

    prop = read(f"{RATED}/layer2_prop100_rated.csv")
    rest = []
    for p in sorted(glob.glob(f"{RATED}/*_rated.csv")):
        if "prop100" in p:
            continue
        rest.append((os.path.basename(p), read(p)))
    pool = [r for _, rows in rest for r in rows]
    print(f"[split] prop100 (natural prevalence) -> TEST only: {len(prop)} rows")
    for name, rows in rest:
        print(f"[split] {name}: {len(rows)} rows -> stratified across train/dev/test")

    # iterative stratification: place each row into the split most starved of its rarest label
    rng = random.Random(SEED)
    rng.shuffle(pool)
    target = {"train": 1 - p_test - p_dev, "dev": p_dev, "test": p_test}
    splits = {k: [] for k in target}
    # rarest-label-first ordering makes the scarce tags get placed while there is still freedom
    rarity = {t: npos(pool, t) for t in TAGS}
    def key(r):
        pos = [t for t in TAGS if r.get(t) == "1"]
        return min((rarity[t] for t in pos), default=10**9)
    pool.sort(key=key)

    want = {k: {t: target[k] * npos(pool, t) for t in TAGS} for k in target}
    have = {k: {t: 0 for t in TAGS} for k in target}
    for r in pool:
        pos = [t for t in TAGS if r.get(t) == "1"]
        if pos:
            t0 = min(pos, key=lambda t: rarity[t])
            best = max(target, key=lambda k: (want[k][t0] - have[k][t0],
                                              target[k] * len(pool) - len(splits[k])))
        else:
            best = max(target, key=lambda k: target[k] * len(pool) - len(splits[k]))
        splits[best].append(r)
        for t in pos:
            have[best][t] += 1

    splits["test"] += prop
    fields = ["text"] + TAGS
    for k, rows in splits.items():
        rng.shuffle(rows)
        with open(f"{OUT}/gold_{k}.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\n[split] gold_{k}.csv: {len(rows)} rows")

    print(f"\n  {'tag':22} {'train+':>7} {'dev+':>6} {'test+':>7}   test share")
    for t in TAGS:
        a, b, c = (npos(splits[k], t) for k in ("train", "dev", "test"))
        tot = a + b + c
        print(f"  {t:22} {a:7} {b:6} {c:7}   {c/max(tot,1)*100:4.0f}%  (total {tot})")

if __name__ == "__main__":
    main()
