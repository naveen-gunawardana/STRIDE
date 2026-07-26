"""Score the Layer-2 tag model against hand-labeled gold, per tag.

Reports precision / recall / F1 / support for every tag plus micro and macro averages, at
either fixed thresholds or per-tag thresholds tuned on a dev file. Cells the rater marked `x`
(unclear) are excluded from that tag's metrics, matching the Layer-1 protocol.

Usage:
  .venv/Scripts/python.exe code/eval_layer2.py --gold F [--dev F] [--model DIR]
                                               [--thr FILE|0.5] [--save-thr FILE] [--errors N]
"""
import csv, json, os, sys
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layer2_lexicon import TAGS
csv.field_size_limit(2**31 - 1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Bootstrap resamples for threshold selection (0 = plain argmax-F1). See tune().
BOOTSTRAP = int(os.environ.get("L2_BOOTSTRAP", "200"))
# "f1" = argmax-F1 on dev; "rate" = prevalence matching (default -- see tune()).
THR_MODE = os.environ.get("L2_THR_MODE", "rate")

def arg(flag, default=None, cast=None):
    if flag not in sys.argv:
        return default
    return (cast or str)(sys.argv[sys.argv.index(flag) + 1])

def read_gold(path):
    texts, Y = [], []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            t = (r.get("text") or "").strip()
            if not t:
                continue
            texts.append(t)
            Y.append([1.0 if (r.get(k) or "").strip() == "1"
                      else 0.0 if (r.get(k) or "").strip() == "0" else np.nan for k in TAGS])
    return texts, np.array(Y)

def predict(model_dir, texts, bs=32):
    tok = AutoTokenizer.from_pretrained(model_dir)
    m = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE).eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(texts), bs):
            enc = tok(texts[s:s + bs], truncation=True, padding=True, max_length=256,
                      return_tensors="pt").to(DEVICE)
            out.append(torch.sigmoid(m(**enc).logits).float().cpu().numpy())
    return np.vstack(out) if out else np.zeros((0, len(TAGS)))

def prf(y, p):
    """y, p are 1-D arrays over the rows where the gold label is known."""
    tp = float(((p == 1) & (y == 1)).sum())
    fp = float(((p == 1) & (y == 0)).sum())
    fn = float(((p == 0) & (y == 1)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return pr, rc, f1, int(y.sum()), tp, fp, fn

def tune(P, Y):
    """Per-tag threshold maximising F1 on the dev set (ties -> higher threshold = safer).

    Candidates are the observed dev probabilities themselves, not a fixed 0.05..0.95 grid.
    The heads are strongly bimodal (median ~0.01, p90 ~0.999), so a fixed grid puts almost no
    candidates where the decision boundary actually lies and every tag pegs at the grid edge --
    which is what the first tuning run did (8/10 tags returned exactly 0.95).
    """
    thr = {}
    for j, t in enumerate(TAGS):
        known = ~np.isnan(Y[:, j])
        y, p = Y[known, j], P[known, j]
        if y.sum() == 0:
            thr[t] = 0.5
            continue
        cands = np.unique(np.concatenate([p, [0.0, 1.0]]))
        # midpoints between observed values -> a threshold strictly between two scores
        cands = np.unique(np.concatenate([cands, (cands[:-1] + cands[1:]) / 2]))

        def best_thr(idx):
            yy, pp = y[idx], p[idx]
            best, bt = -1.0, 0.5
            for c in cands:
                _, _, f1, *_ = prf(yy, (pp >= c).astype(float))
                if f1 >= best:
                    best, bt = f1, float(c)
            return bt

        if THR_MODE == "rate":
            # Prevalence matching: set the threshold so the model predicts positive at the rate
            # the gold says the tag actually occurs. Argmax-F1 needs a good estimate of the
            # P/R trade-off curve, which ~9 dev positives cannot give; a prevalence estimate is
            # far more stable at the same sample size, and it directly fixes the P>>R imbalance
            # that argmax-F1 produced (macro P 0.83 vs R 0.69).
            k = int(round(y.sum()))
            thr[t] = float(f"{np.sort(p)[::-1][min(k, len(p)) - 1]:.6g}") if k else 0.5
            continue
        if BOOTSTRAP:
            # A single argmax-F1 threshold on ~100 dev rows is very noisy and biases HIGH: with
            # few positives, pushing the threshold up removes false positives faster than it
            # loses true ones. On the first tuned run this cost ~0.12 macro recall on test.
            # Median over bootstrap resamples is far more stable.
            rs = np.random.RandomState(7)
            n = len(y)
            picks = [best_thr(rs.randint(0, n, n)) for _ in range(BOOTSTRAP)]
            bt = float(np.median(picks))
        else:
            bt = best_thr(np.arange(len(y)))
        thr[t] = float(f"{bt:.6g}")
    return thr

def report(P, Y, thr, title):
    print(f"\n===== {title} =====")
    print(f"  {'tag':22} {'thr':>5} {'P':>6} {'R':>6} {'F1':>6} {'n+':>5} {'TP':>5} {'FP':>5} {'FN':>5}")
    rows, mtp = [], [0.0, 0.0, 0.0]
    for j, t in enumerate(TAGS):
        known = ~np.isnan(Y[:, j])
        y = Y[known, j]
        pr, rc, f1, sup, tp, fp, fn = prf(y, (P[known, j] >= thr[t]).astype(float))
        rows.append((t, pr, rc, f1, sup))
        mtp[0] += tp; mtp[1] += fp; mtp[2] += fn
        flag = "" if (pr >= 0.8 and rc >= 0.8 and f1 >= 0.8) else "  <-- below 0.8"
        print(f"  {t:22} {thr[t]:5.2f} {pr:6.2f} {rc:6.2f} {f1:6.2f} {sup:5} {int(tp):5} "
              f"{int(fp):5} {int(fn):5}{flag}")
    mp = mtp[0] / (mtp[0] + mtp[1]) if mtp[0] + mtp[1] else 0.0
    mr = mtp[0] / (mtp[0] + mtp[2]) if mtp[0] + mtp[2] else 0.0
    mf = 2 * mp * mr / (mp + mr) if mp + mr else 0.0
    MP = float(np.mean([r[1] for r in rows])); MR = float(np.mean([r[2] for r in rows]))
    MF = float(np.mean([r[3] for r in rows]))
    print(f"\n  {'MICRO (pooled)':22} {'':5} {mp:6.2f} {mr:6.2f} {mf:6.2f}")
    print(f"  {'MACRO (mean of tags)':22} {'':5} {MP:6.2f} {MR:6.2f} {MF:6.2f}")
    worst = min(rows, key=lambda r: min(r[1], r[2], r[3]))
    print(f"\n  weakest tag: {worst[0]}  (P {worst[1]:.2f} / R {worst[2]:.2f} / F1 {worst[3]:.2f})")
    allpass = all(r[1] >= 0.8 and r[2] >= 0.8 and r[3] >= 0.8 for r in rows)
    print(f"  ALL TAGS >= 0.8 on P, R and F1: {'YES' if allpass else 'NO'}")
    return {"micro": (mp, mr, mf), "macro": (MP, MR, MF), "rows": rows, "all_pass": allpass}

def main():
    gold = arg("--gold")
    if not gold or not os.path.exists(gold):
        sys.exit(f"--gold is required and must exist (got {gold!r})")
    dev = arg("--dev")
    model_dir = arg("--model", "models/layer2_tags")
    thr_arg = arg("--thr", "0.5")
    n_err = arg("--errors", 0, int)

    texts, Y = read_gold(gold)
    print(f"[eval] model {model_dir} | gold {gold}: {len(texts)} rows", flush=True)
    P = predict(model_dir, texts)

    if os.path.exists(str(thr_arg)):
        thr = json.load(open(thr_arg, encoding="utf-8"))
    elif dev:
        dt, dY = read_gold(dev)
        thr = tune(predict(model_dir, dt), dY)
        print(f"[eval] tuned thresholds on dev {dev} ({len(dt)} rows): {thr}")
    else:
        thr = {t: float(thr_arg) for t in TAGS}

    res = report(P, Y, thr, f"{os.path.basename(gold)} @ {model_dir}")
    save_thr = arg("--save-thr")
    if save_thr:
        with open(save_thr, "w", encoding="utf-8") as fh:
            json.dump(thr, fh, indent=1)
        print(f"\n[eval] thresholds -> {save_thr}")

    if n_err:
        print(f"\n===== error samples (up to {n_err} per tag) =====")
        for j, t in enumerate(TAGS):
            known = np.where(~np.isnan(Y[:, j]))[0]
            fps = [i for i in known if P[i, j] >= thr[t] and Y[i, j] == 0]
            fns = [i for i in known if P[i, j] < thr[t] and Y[i, j] == 1]
            if not fps and not fns:
                continue
            print(f"\n--- {t} ---")
            for i in fps[:n_err]:
                print(f"  FP p={P[i,j]:.2f}  {texts[i][:200]}")
            for i in fns[:n_err]:
                print(f"  FN p={P[i,j]:.2f}  {texts[i][:200]}")
    return res

if __name__ == "__main__":
    main()
