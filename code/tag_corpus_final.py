"""Build the FINAL released AMHC corpus: tag the gate-relevant comments with the chosen Layer-2
model, keep only the tags that passed the quality bar, and filter to comments carrying >=1 kept tag.

Design decisions (2026-08-10, with Naveen):
  * Model = models/layer2_tags_v3 (round-1). It beat the round-2 retrains (v3b/v3c/v3d) on the kept
    tag set; the rule-broadening experiments regressed the shared encoder. v3 is a 17-head model
    (its thresholds.json carries the head order, incl. identity_retirement at 15); we map heads by
    that file, so alignment is exact.
  * Kept tags = the 12 with gold-test F1 >= 0.65. Dropped: exercise_coping, substance_use,
    trauma_ptsd, exercise_dependence (all < 0.65). Dropping is output-only -- no retrain.
  * Final corpus = matched arm only, filtered to >=1 kept tag (multi-label ok). Every row keeps its
    p_<tag> probability so users can re-threshold.
  * The control/baseline arm is written separately (control_baseline.csv) for base-rate comparison.

Usage:
  .venv/Scripts/python.exe code/tag_corpus_final.py [--model models/layer2_tags_v3]
"""
import csv, json, os, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
csv.field_size_limit(2**31 - 1)

def arg(flag, default=None, cast=None):
    if flag not in sys.argv:
        return default
    v = sys.argv[sys.argv.index(flag) + 1]
    return cast(v) if cast else v

MODEL = arg("--model", "models/layer2_tags_v3")
INP = arg("--in", "data/classified/final_dataset.csv")
OUT = arg("--out", "data/classified/final_dataset_tagged.csv")
CTRL_OUT = arg("--control-out", "data/classified/control_baseline.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The 12 tags that passed F1 >= 0.65 on gold-test (v3). Order = descending F1, for readability.
KEEP = ["anxiety", "depression", "self_harm_suicide", "burnout_motivation", "help_seeking",
        "loneliness_isolation", "body_image_eating", "sleep", "stress_pressure",
        "adhd_neurodivergence", "performance_psych", "injury_distress"]

def main():
    thr = json.load(open(os.path.join(MODEL, "thresholds.json"), encoding="utf-8"))
    order = list(thr.keys())                      # model head order (17 for v3)
    keep_idx = {t: order.index(t) for t in KEEP}  # exact head index per kept tag
    print(f"[final] model {MODEL} on {DEVICE} | heads {len(order)} | keeping {len(KEEP)} tags",
          flush=True)
    print(f"[final] dropped (F1<0.65): {[t for t in order if t not in KEEP and t!='identity_retirement']}",
          flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(DEVICE).eval()
    if DEVICE == "cuda":
        model = model.half()

    @torch.no_grad()
    def probs(texts):
        out = []
        for i in range(0, len(texts), 128):
            enc = tok(texts[i:i+128], truncation=True, padding=True, max_length=256,
                      return_tensors="pt").to(DEVICE)
            out.extend(torch.sigmoid(model(**enc).logits).float().cpu().tolist())
        return out

    fin = open(INP, encoding="utf-8", newline="")
    rd = csv.DictReader(fin)
    base_fields = [c for c in (rd.fieldnames or []) if c not in ("p_mh", "p_sport")]
    extra = ["p_mh", "p_sport"] + [f"p_{t}" for t in KEEP] + [f"tag_{t}" for t in KEEP]
    fields = base_fields + extra

    fout = open(OUT, "w", encoding="utf-8", newline="")
    fctrl = open(CTRL_OUT, "w", encoding="utf-8", newline="")
    w = csv.DictWriter(fout, fieldnames=fields); w.writeheader()
    wc = csv.DictWriter(fctrl, fieldnames=fields); wc.writeheader()

    n = kept = ctrl = dropped0 = 0
    tagcount = {t: 0 for t in KEEP}
    buf, bufrows = [], []
    t0 = time.time()

    def flush():
        nonlocal kept, ctrl, dropped0
        if not buf:
            return
        P = probs(buf)
        for row, p in zip(bufrows, P):
            hits = 0
            for t in KEEP:
                pv = p[keep_idx[t]]
                row[f"p_{t}"] = f"{pv:.4f}"
                fire = 1 if pv >= thr[t] else 0
                row[f"tag_{t}"] = fire
                if fire:
                    hits += 1; tagcount[t] += 1
            if row.get("arm") == "matched":
                if hits >= 1:
                    w.writerow({k: row.get(k, "") for k in fields}); kept += 1
                else:
                    dropped0 += 1
            else:
                wc.writerow({k: row.get(k, "") for k in fields}); ctrl += 1
        buf.clear(); bufrows.clear()

    for r in rd:
        t = (r.get("text") or "").strip()
        n += 1
        buf.append(t); bufrows.append(dict(r))
        if len(buf) >= 512:
            flush()
            if n % 20480 == 0:
                print(f"  {n:,} scanned, {kept:,} kept, {dropped0:,} dropped(0-tag) "
                      f"({time.time()-t0:.0f}s)", flush=True)
    flush()
    fin.close(); fout.close(); fctrl.close()

    print(f"\n[final] scanned {n:,} rows")
    print(f"[final] FINAL CORPUS (matched + >=1 kept tag): {kept:,} -> {OUT}")
    print(f"[final] dropped matched rows with 0 kept tags: {dropped0:,}")
    print(f"[final] control baseline rows: {ctrl:,} -> {CTRL_OUT}")
    print(f"\n  per-tag positives across all scanned:")
    for t in KEEP:
        print(f"    {t:22} {tagcount[t]:7,}")

if __name__ == "__main__":
    main()
