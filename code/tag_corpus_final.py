"""Build the FINAL released AMHC corpus: tag the gate-relevant comments with the chosen Layer-2
model, keep only the tags that passed the quality bar, and filter to comments carrying >=1 kept tag.

Design decisions (updated 2026-08-18):
  * Model = models/layer2_tags_v3b. The broadened-rule retrain beats round 1 on the pooled
    metrics (micro F1 0.79 vs 0.76, macro 0.77 vs 0.73) and lifts the weakest heads sharply
    (exercise_dependence 0.20 -> 0.73, trauma_ptsd 0.48 -> 0.63). Head order is read from its
    thresholds.json so alignment is exact.
  * ALL 16 tags are released, with their per-tag precision, recall, F1 and inter-rater kappa,
    rather than filtered to a quality bar. A single global F1 cutoff would silently decide for
    the user which constructs are usable; for subjective constructs of this kind the per-tag
    figures are the honest interface, and the reviewer preference is explicitly against a hard
    0.8 gate. Users who want a stricter subset can threshold on the published table.
  * Final corpus = matched arm, filtered to comments carrying >=1 tag (multi-label). Every row
    keeps its p_<tag> probability so users can re-threshold.
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

MODEL = arg("--model", "models/layer2_tags_v3b")
INP = arg("--in", "data/classified/final_dataset.csv")
OUT = arg("--out", "data/classified/stride_release.csv")
CTRL_OUT = arg("--control-out", "data/classified/control_baseline.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# All 16 released tags, ordered by gold-test F1 on v3b for readability. No quality filter is
# applied here; see the module docstring.
KEEP = ["depression", "anxiety", "self_harm_suicide", "loneliness_isolation", "help_seeking",
        "adhd_neurodivergence", "burnout_motivation", "body_image_eating", "sleep",
        "stress_pressure", "exercise_dependence", "exercise_coping", "performance_psych",
        "substance_use", "trauma_ptsd", "injury_distress"]

def main():
    thr = json.load(open(os.path.join(MODEL, "thresholds.json"), encoding="utf-8"))
    order = list(thr.keys())                      # model head order (17 for v3)
    keep_idx = {t: order.index(t) for t in KEEP}  # exact head index per kept tag
    print(f"[final] model {MODEL} on {DEVICE} | heads {len(order)} | keeping {len(KEEP)} tags",
          flush=True)
    missing = [t for t in KEEP if t not in order]
    if missing:
        raise SystemExit(f"[final] tags absent from {MODEL}/thresholds.json: {missing}")

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
