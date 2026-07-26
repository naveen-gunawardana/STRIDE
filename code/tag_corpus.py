"""Apply the Layer-2 tag model to the Layer-1-relevant dataset (the full cascade output).

Layer 1 (mh AND sport) already pruned the corpus to athlete-mental-health comments; this adds
one column per tag (probability) plus a binary column at the tuned operating threshold, so the
study dataset can be filtered/counted by theme.

Usage:
  .venv/Scripts/python.exe code/tag_corpus.py [--in F] [--out F] [--model DIR] [--thr FILE]
"""
import csv, json, os, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layer2_lexicon import TAGS
csv.field_size_limit(2**31 - 1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 64

def arg(flag, default=None, cast=None):
    if flag not in sys.argv:
        return default
    return (cast or str)(sys.argv[sys.argv.index(flag) + 1])

def main():
    inp = arg("--in", "data/classified/final_dataset.csv")
    out = arg("--out", "data/classified/final_dataset_tagged.csv")
    model_dir = arg("--model", "models/layer2_tags")
    thr_file = arg("--thr", "models/layer2_tags/thresholds.json")
    thr = json.load(open(thr_file, encoding="utf-8")) if os.path.exists(str(thr_file)) \
        else {t: 0.5 for t in TAGS}
    print(f"[tag] model {model_dir} | thresholds {thr}", flush=True)

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE).eval()
    if DEVICE == "cuda":
        model = model.half()

    n = 0
    t0 = time.time()
    counts = {t: 0 for t in TAGS}
    with open(inp, encoding="utf-8", newline="") as fin, \
         open(out, "w", encoding="utf-8", newline="") as fout:
        rd = csv.DictReader(fin)
        fields = list(rd.fieldnames or []) + [f"p_{t}" for t in TAGS] + [f"tag_{t}" for t in TAGS]
        w = csv.DictWriter(fout, fieldnames=fields)
        w.writeheader()
        buf = []

        def flush(buf):
            nonlocal n
            if not buf:
                return
            texts = [(r.get("text") or "") for r in buf]
            with torch.no_grad():
                enc = tok(texts, truncation=True, padding=True, max_length=256,
                          return_tensors="pt").to(DEVICE)
                P = torch.sigmoid(model(**enc).logits).float().cpu().numpy()
            for r, p in zip(buf, P):
                for j, t in enumerate(TAGS):
                    r[f"p_{t}"] = f"{p[j]:.4f}"
                    hit = int(p[j] >= thr[t])
                    r[f"tag_{t}"] = hit
                    counts[t] += hit
                w.writerow(r)
            n += len(buf)
            if n % 20000 < BATCH:
                el = time.time() - t0
                print(f"  ... {n:,} tagged  {n/max(el,1):.0f} rows/s", flush=True)

        for r in rd:
            buf.append(r)
            if len(buf) >= BATCH:
                flush(buf); buf = []
        flush(buf)

    print(f"\n[tag] {n:,} comments tagged -> {out}  ({(time.time()-t0)/60:.1f} min)\n")
    print(f"  {'tag':22} {'count':>9}   % of relevant")
    for t in TAGS:
        print(f"  {t:22} {counts[t]:9,}   {counts[t]/max(n,1)*100:5.1f}%")

if __name__ == "__main__":
    main()
