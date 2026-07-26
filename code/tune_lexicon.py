"""Score the weak-supervision labeling functions directly against hand-labeled gold.

Round 1 showed a near-monotonic relationship between (silver prevalence / gold prevalence) and
per-tag test F1: every tag whose rules under-fired badly relative to the rubric scored under 0.6,
because the model faithfully learned the rules' narrower concept. So the rules themselves are the
thing to fix, and they need to be measurable.

This reports each tag's HIGH-PRECISION rule as if it were a classifier, on gold TRAIN+DEV only
(test is never touched). Target: keep rule precision high (weak supervision tolerates some noise)
while lifting recall, so the silver positives cover the rubric's concept rather than a corner of it.

Usage: .venv/Scripts/python.exe code/tune_lexicon.py [--show TAG] [--fp] [--fn]
"""
import csv, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layer2_lexicon as L
csv.field_size_limit(2**31 - 1)

def main():
    rows = []
    for s in ("train", "dev"):
        with open(f"data/layer2/gold_{s}.csv", encoding="utf-8-sig", newline="") as fh:
            rows += list(csv.DictReader(fh))
    show = sys.argv[sys.argv.index("--show") + 1] if "--show" in sys.argv else None
    print(f"[lex] {len(rows)} gold train+dev rows\n")
    print(f"  {'tag':22} {'P':>6} {'R':>6} {'F1':>6} {'gold+':>6} {'rulepos':>8} {'abstain':>8}")
    macro = []
    for t in L.TAGS:
        tp = fp = fn = gold = rulepos = abst = 0
        fps, fns = [], []
        for r in rows:
            g = r.get(t)
            lab = L.label(r["text"])[t]
            if lab == 1:
                rulepos += 1
            if lab is None:
                abst += 1
            if g not in ("0", "1"):
                continue
            gold += (g == "1")
            if lab == 1 and g == "1":
                tp += 1
            elif lab == 1 and g == "0":
                fp += 1; fps.append(r["text"])
            elif lab != 1 and g == "1":
                fn += 1; fns.append(r["text"])
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
        macro.append(f1)
        print(f"  {t:22} {p:6.2f} {rc:6.2f} {f1:6.2f} {gold:6} {rulepos:8} {abst:8}")
        if show == t:
            rng = random.Random(3)
            if "--fp" in sys.argv:
                print(f"\n  --- {t}: rule says YES, gold says NO ({len(fps)}) ---")
                for x in rng.sample(fps, min(12, len(fps))):
                    print("   FP:", " ".join(x.split())[:200])
            if "--fn" in sys.argv:
                print(f"\n  --- {t}: gold says YES, rule misses ({len(fns)}) ---")
                for x in rng.sample(fns, min(12, len(fns))):
                    print("   FN:", " ".join(x.split())[:200])
    print(f"\n  macro rule F1: {sum(macro)/len(macro):.3f}")

if __name__ == "__main__":
    main()
