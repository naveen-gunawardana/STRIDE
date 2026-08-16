"""Stage a blinded second-rater pass for Layer-2 inter-rater kappa.

Rater A = the existing gold labels (Claude, from the written rubric).
Rater B = a second INDEPENDENT blinded pass (no access to A's labels).

Samples a subset of gold-test (stratified so every tag has positives present), writes:
  data/layer2/kappa_raterA.csv   id + the 16-tag gold labels (held back from rater B)
  data/layer2/kappa_blinded.csv  id + text only  -> what rater B sees

Then metrics_interrater_l2.py scores per-tag Cohen's kappa on the shared ids.
"""
import csv, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layer2_lexicon as L
csv.field_size_limit(2**31 - 1)

GOLD = "data/layer2/gold_test.csv"
TAGS = L.TAGS  # 16 (identity dropped)
N_SAMPLE = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 160
SEED = 20260810

def main():
    with open(GOLD, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for i, r in enumerate(rows):
        r["_id"] = i
    rng = random.Random(SEED)

    # Stratified pick: guarantee at least a few positives per tag, then fill randomly.
    picked, picked_ids = [], set()
    for t in TAGS:
        pos = [r for r in rows if str(r.get(t, "")).strip() == "1" and r["_id"] not in picked_ids]
        rng.shuffle(pos)
        for r in pos[:6]:
            picked.append(r); picked_ids.add(r["_id"])
    pool = [r for r in rows if r["_id"] not in picked_ids]
    rng.shuffle(pool)
    for r in pool:
        if len(picked) >= N_SAMPLE:
            break
        picked.append(r); picked_ids.add(r["_id"])
    picked.sort(key=lambda r: r["_id"])

    os.makedirs("data/layer2", exist_ok=True)
    with open("data/layer2/kappa_raterA.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh); w.writerow(["id"] + TAGS)
        for r in picked:
            w.writerow([r["_id"]] + [1 if str(r.get(t, "")).strip() == "1" else 0 for t in TAGS])
    with open("data/layer2/kappa_blinded.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh); w.writerow(["id", "text"])
        for r in picked:
            w.writerow([r["_id"], (r.get("text") or "").strip()])

    # prevalence report
    print(f"[kappa] sampled {len(picked)} of {len(rows)} gold-test records (seed {SEED})")
    print(f"  {'tag':22} {'rater-A pos':>11}")
    for t in TAGS:
        p = sum(1 for r in picked if str(r.get(t, "")).strip() == "1")
        print(f"  {t:22} {p:11}")

if __name__ == "__main__":
    main()
