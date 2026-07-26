"""Active-learning draw for Layer 2: label where the model is actually confused.

Gold fine-tuning is the lever that moved Layer 2 (macro F1 0.68 -> 0.71 off 204 rows), so more
gold is worth more than more silver. Rather than sampling more at random, this harvests the
model's own uncertainty on the tags that are still weak, which is the same active-learning move
that fixed Layer-1 over-flagging (process doc 6e).

Per requested tag it draws three bands, because they fix different errors:
  * UNCERTAIN     -- p closest to the operating threshold: the decision boundary itself
  * CONFIDENT-POS -- p well above threshold: finds false positives (fixes precision)
  * CUE-BUT-QUIET -- lexicon HR cue present yet p well below threshold: finds false negatives
                     the model is confidently missing (fixes recall)

Usage:
  .venv/Scripts/python.exe code/al_sample_layer2.py --model DIR --thr FILE
      --tags injury_distress,performance_psych --per-band 12 --pool 40000 --out PATH
"""
import csv, glob, json, os, random, sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layer2_lexicon as L
csv.field_size_limit(2**31 - 1)

DATASET = "data/classified/final_dataset.csv"
GOLD_KEYS = "data/data_layer2_ratings/*_key.csv"
GOLD_RATED = "data/data_layer2_ratings/*_rated.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 20260726

def arg(flag, default=None, cast=None):
    if flag not in sys.argv:
        return default
    return (cast or str)(sys.argv[sys.argv.index(flag) + 1])

def excluded():
    ids, txt = set(), set()
    for p in glob.glob(GOLD_KEYS):
        with open(p, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                ids.add(r.get("comment_id", ""))
    for p in glob.glob(GOLD_RATED):
        with open(p, encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                t = (r.get("text") or "").strip()
                if t:
                    txt.add(t)
    return ids, txt

def main():
    model_dir = arg("--model", "models/layer2_tags_r3a")
    thr_file = arg("--thr")
    tags = arg("--tags", ",".join(L.TAGS)).split(",")
    per_band = arg("--per-band", 12, int)
    pool_n = arg("--pool", 40000, int)
    out = arg("--out", "data/data_layer2_ratings/layer2_al")
    thr = json.load(open(thr_file, encoding="utf-8")) if thr_file and os.path.exists(thr_file) \
        else {t: 0.5 for t in L.TAGS}

    ids, txt = excluded()
    rng = random.Random(SEED)
    pool = []
    with open(DATASET, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("arm") != "matched" or r.get("id", "") in ids:
                continue
            t = (r.get("text") or "").strip()
            if len(t) < 20 or t in txt:
                continue
            pool.append(r)
    rng.shuffle(pool)
    pool = pool[:pool_n]
    print(f"[al] scoring {len(pool):,} candidate comments on {DEVICE}", flush=True)

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE).eval()
    if DEVICE == "cuda":
        model = model.half()
    P = []
    with torch.no_grad():
        for s in range(0, len(pool), 64):
            enc = tok([(r.get("text") or "") for r in pool[s:s + 64]], truncation=True,
                      padding=True, max_length=256, return_tensors="pt").to(DEVICE)
            P.append(torch.sigmoid(model(**enc).logits).float().cpu())
    P = torch.cat(P).numpy()

    taken, picked = set(), []
    for t in tags:
        j = L.TAGS.index(t)
        c = thr.get(t, 0.5)
        avail = [i for i in range(len(pool)) if i not in taken]
        bands = {
            "uncertain": sorted(avail, key=lambda i: abs(P[i, j] - c))[:per_band],
            "confident_pos": sorted(avail, key=lambda i: -P[i, j])[:per_band],
            "cue_but_quiet": [i for i in sorted(avail, key=lambda i: P[i, j])
                              if L.hr_match(pool[i].get("text") or "", t)][:per_band],
        }
        for b, idxs in bands.items():
            for i in idxs:
                if i in taken:
                    continue
                taken.add(i); picked.append((pool[i], t, b, float(P[i, j])))
        print(f"  {t:22} -> {sum(len(v) for v in bands.values())} drawn "
              f"(thr {c:.2f}, bands uncertain/confident/quiet)")

    rng.shuffle(picked)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out + "_rated.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row_id", "text"] + L.TAGS + ["notes"])
        for i, (r, _, _, _) in enumerate(picked, 1):
            w.writerow([i, r.get("text", "")] + [""] * len(L.TAGS) + [""])
    with open(out + "_key.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["row_id", "comment_id", "subreddit", "time",
                                           "target_tag", "band", "p"])
        w.writeheader()
        for i, (r, t, b, p) in enumerate(picked, 1):
            w.writerow(dict(row_id=i, comment_id=r.get("id", ""), subreddit=r.get("subreddit", ""),
                            time=r.get("time", ""), target_tag=t, band=b, p=f"{p:.4f}"))
    print(f"\n[al] {len(picked)} rows -> {out}_rated.csv")

if __name__ == "__main__":
    main()
