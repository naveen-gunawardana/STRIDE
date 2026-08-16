"""Per-tag inter-rater agreement for the Layer-2 multi-label tags.

Rater A = gold labels (data/layer2/kappa_raterA.csv).
Rater B = second independent blinded pass (data/layer2/kappa_batches/raterB_batch*.csv, merged).

For each of the 16 tags, computes Cohen's kappa between the two raters on the shared ids, plus
raw agreement and each rater's positive rate. This is the Layer-2 analogue of the Layer-1
relevance kappa (code/metrics_interrater.py), generalised to multi-label: one kappa per tag.

A per-tag kappa is undefined (nan) when a rater used only one class for that tag; we report the
raw agreement in that case and flag it, since kappa's chance correction degenerates at zero
variance.

Usage: .venv/Scripts/python.exe code/metrics_interrater_l2.py
"""
import csv, glob, os, sys
from sklearn.metrics import cohen_kappa_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layer2_lexicon as L
csv.field_size_limit(2**31 - 1)

TAGS = L.TAGS
A_FILE = "data/layer2/kappa_raterA.csv"
B_GLOB = "data/layer2/kappa_batches/raterB_batch*.csv"

def read_labels(path):
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            rid = str(r.get("id", "")).strip()
            if not rid:
                continue
            out[int(rid)] = {t: (1 if str(r.get(t, "")).strip() == "1" else 0) for t in TAGS}
    return out

def main():
    A = read_labels(A_FILE)
    B = {}
    files = sorted(glob.glob(B_GLOB))
    for p in files:
        B.update(read_labels(p))
    print(f"[kappa-l2] rater A: {len(A)} records | rater B: {len(B)} records "
          f"(from {len(files)} batch files)")

    ids = sorted(set(A) & set(B))
    missing_b = sorted(set(A) - set(B))
    if missing_b:
        print(f"  WARNING: {len(missing_b)} ids rated by A but missing from B: {missing_b[:15]}"
              f"{'...' if len(missing_b) > 15 else ''}")
    print(f"[kappa-l2] scoring {len(ids)} shared records\n")

    print(f"  {'tag':22} {'kappa':>7} {'raw-agr':>8} {'A pos':>6} {'B pos':>6} {'note'}")
    kappas = []
    for t in TAGS:
        a = [A[i][t] for i in ids]
        b = [B[i][t] for i in ids]
        agree = sum(1 for x, y in zip(a, b) if x == y) / len(ids)
        ap, bp = sum(a), sum(b)
        if len(set(a)) < 2 or len(set(b)) < 2:
            note = "kappa undefined (single-class); raw agr shown"
            print(f"  {t:22} {'--':>7} {agree:8.3f} {ap:6} {bp:6}  {note}")
            continue
        k = cohen_kappa_score(a, b)
        kappas.append(k)
        flag = "" if k >= 0.6 else ("  weak (<0.6)" if k >= 0.4 else "  poor (<0.4)")
        print(f"  {t:22} {k:7.3f} {agree:8.3f} {ap:6} {bp:6}{flag}")

    # Pooled: flatten every (record, tag) decision into one long vector -> overall kappa.
    fa = [A[i][t] for i in ids for t in TAGS]
    fb = [B[i][t] for i in ids for t in TAGS]
    pooled = cohen_kappa_score(fa, fb)
    pooled_agree = sum(1 for x, y in zip(fa, fb) if x == y) / len(fa)
    print(f"\n  {'POOLED (all tag decisions)':22} {pooled:7.3f} {pooled_agree:8.3f}")
    if kappas:
        print(f"  {'MEAN of per-tag kappas':22} {sum(kappas)/len(kappas):7.3f}   "
              f"(over {len(kappas)} tags with both classes present)")
    print(f"\n  Interpretation: >=0.8 almost perfect, 0.6-0.8 substantial, 0.4-0.6 moderate.")
    print(f"  Layer-1 relevance kappa was 0.81 / 0.86 for reference.")

if __name__ == "__main__":
    main()
