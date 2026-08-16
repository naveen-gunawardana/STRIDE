"""End-to-end (joint) evaluation of the full cascade: Layer-1 gate -> Layer-2 tags.

Motivation: the per-tag Layer-2 numbers are measured on *already-relevant* gold comments, and the
"pipeline" figure quoted in the report (0.89 x 0.76) is only a composition of two separately measured
numbers. A reviewer will ask for a real joint measurement on raw, ungated comments. This does that.

Test set = the Layer-2 gold-test records (truly relevant, with gold tags)
         + a sample of control-arm comments (treated as truly irrelevant -> all tags 0).

For every comment we run the ACTUAL cascade:
  1. gate: relevant iff P(mh) >= MH_THR AND P(sport) >= SP_THR
  2. if relevant -> Layer-2 sigmoid >= per-tag threshold gives the predicted tags
     if not relevant -> the comment never reaches Layer 2, so every tag is predicted 0

Then per-tag joint P/R/F1 over the whole set. A truly-relevant comment the gate drops becomes a
false negative for each of its gold tags (that is the cost the composition hides); an irrelevant
comment the gate passes that then fires a tag becomes a false positive.

Usage:
  .venv/Scripts/python.exe code/eval_joint.py --l2-model models/layer2_tags_v3 [--neg 300]
"""
import csv, glob, json, os, sys, random
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
csv.field_size_limit(2**31 - 1)

# The 12 released tags (F1 >= 0.65 on v3). Joint eval measures the full cascade on these.
TAGS = ["anxiety", "depression", "self_harm_suicide", "burnout_motivation", "help_seeking",
        "loneliness_isolation", "body_image_eating", "sleep", "stress_pressure",
        "adhd_neurodivergence", "performance_psych", "injury_distress"]
MH_THR, SP_THR = 0.50, 0.40
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def arg(flag, default=None, cast=None):
    if flag not in sys.argv:
        return default
    v = sys.argv[sys.argv.index(flag) + 1]
    return cast(v) if cast else v

GOLD = "data/layer2/gold_test.csv"
DATASET = "data/classified/final_dataset.csv"

def load_gate(group):
    p = f"models/filter_relevance_{group}"
    return (AutoTokenizer.from_pretrained(p),
            AutoModelForSequenceClassification.from_pretrained(p).to(DEVICE).eval())

@torch.no_grad()
def gate_prob(model_tok, texts, bs=32):
    tok, model = model_tok
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i+bs], truncation=True, padding=True, max_length=512,
                  return_tensors="pt").to(DEVICE)
        out.extend(model(**enc).logits.softmax(1)[:, 1].float().cpu().tolist())
    return out

@torch.no_grad()
def l2_probs(model_tok, texts, bs=32):
    tok, model = model_tok
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i+bs], truncation=True, padding=True, max_length=256,
                  return_tensors="pt").to(DEVICE)
        out.extend(torch.sigmoid(model(**enc).logits).float().cpu().tolist())
    return out

def main():
    l2_dir = arg("--l2-model", "models/layer2_tags_v3")
    thr_file = arg("--l2-thr", os.path.join(l2_dir, "thresholds.json"))
    n_neg = arg("--neg", 300, int)
    thr = json.load(open(thr_file, encoding="utf-8"))
    order = list(thr.keys())                       # model head order (17 for v3)
    keep_idx = {t: order.index(t) for t in TAGS}   # exact head index per released tag

    # relevant, tagged
    with open(GOLD, encoding="utf-8-sig", newline="") as fh:
        gold = list(csv.DictReader(fh))
    texts = [(r.get("text") or "").strip() for r in gold]
    Y = [[1 if str(r.get(t, "")).strip() == "1" else 0 for t in TAGS] for r in gold]
    rel_truth = [1] * len(gold)

    # True negatives = the Layer-1 relevance holdout rows labelled relevant=0. These are genuinely
    # irrelevant (human/model-labelled), so the gate SHOULD reject them; that is what lets us measure
    # the gate's precision contribution to joint error. (final_dataset.csv and the baseline_* files
    # only hold gate-POSITIVE comments, which is why an earlier attempt showed 100% false-pass.)
    rng = random.Random(20260810)
    ctrl = []
    with open("data/data_relevance_ratings/comments/holdout_labeled.csv",
              encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            if str(r.get("relevant", "")).strip() == "1":
                continue
            t = (r.get("text") or "").strip()
            if len(t) >= 20:
                ctrl.append(t)
    rng.shuffle(ctrl)
    ctrl = ctrl[:n_neg]
    texts += ctrl
    Y += [[0] * len(TAGS) for _ in ctrl]
    rel_truth += [0] * len(ctrl)
    print(f"[joint] {len(gold)} relevant gold + {len(ctrl)} control negatives = {len(texts)} on {DEVICE}",
          flush=True)

    print("[joint] loading models ...", flush=True)
    mh, sport = load_gate("mh"), load_gate("sport")
    l2 = (AutoTokenizer.from_pretrained(l2_dir),
          AutoModelForSequenceClassification.from_pretrained(l2_dir).to(DEVICE).eval())

    print("[joint] running gate ...", flush=True)
    p_mh = gate_prob(mh, texts)
    p_sp = gate_prob(sport, texts)
    passed = [(p_mh[i] >= MH_THR and p_sp[i] >= SP_THR) for i in range(len(texts))]

    # gate diagnostics
    rel_idx = [i for i in range(len(texts)) if rel_truth[i] == 1]
    irr_idx = [i for i in range(len(texts)) if rel_truth[i] == 0]
    gate_recall = sum(passed[i] for i in rel_idx) / max(len(rel_idx), 1)
    gate_fpr = sum(passed[i] for i in irr_idx) / max(len(irr_idx), 1)
    print(f"[joint] gate recall on relevant: {gate_recall:.3f} | "
          f"gate false-pass on irrelevant: {gate_fpr:.3f}\n", flush=True)

    print("[joint] running Layer 2 on gate-survivors ...", flush=True)
    surv = [i for i in range(len(texts)) if passed[i]]
    probs = l2_probs(l2, [texts[i] for i in surv]) if surv else []
    pred = [[0] * len(TAGS) for _ in texts]
    for k, i in enumerate(surv):
        pred[i] = [1 if probs[k][keep_idx[TAGS[j]]] >= thr[TAGS[j]] else 0 for j in range(len(TAGS))]

    # per-tag joint P/R/F1
    print(f"\n===== JOINT (cascade) per-tag, n={len(texts)} =====")
    print(f"  {'tag':22} {'P':>6} {'R':>6} {'F1':>6} {'gold+':>6}")
    TP = FP = FN = 0
    for j, t in enumerate(TAGS):
        tp = sum(1 for i in range(len(texts)) if pred[i][j] == 1 and Y[i][j] == 1)
        fp = sum(1 for i in range(len(texts)) if pred[i][j] == 1 and Y[i][j] == 0)
        fn = sum(1 for i in range(len(texts)) if pred[i][j] == 0 and Y[i][j] == 1)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        sup = sum(Y[i][j] for i in range(len(texts)))
        flag = "" if (p >= 0.8 and r >= 0.8 and f >= 0.8) else "  <0.8"
        print(f"  {t:22} {p:6.2f} {r:6.2f} {f:6.2f} {sup:6}{flag}")
        TP += tp; FP += fp; FN += fn
    micro_p = TP / (TP + FP) if TP + FP else 0.0
    micro_r = TP / (TP + FN) if TP + FN else 0.0
    micro_f = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0
    print(f"\n  {'MICRO (joint, pooled)':22} {micro_p:6.2f} {micro_r:6.2f} {micro_f:6.2f}")
    print(f"\n  Note: gate recall {gate_recall:.2f} caps joint recall -- a relevant comment the gate")
    print(f"  drops loses all its tags. Compare joint micro-F1 to the Layer-2-only micro (0.76).")

if __name__ == "__main__":
    main()
