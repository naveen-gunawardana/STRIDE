"""Draw the comments our own rules may have wrongly suppressed, for a recall audit.

Why this exists. The paper's central comparison is that sleep appears in 2.2% of athlete
mental-health comments against 26.4% prevalence in screening meta-analyses, and substance use in
1.6% against 18.8%. Both tags carry wrong-sense kill rules written specifically to suppress the
fitness senses -- recovery sleep, drinking water, pre-workout. Those rules were correct for
building a precise classifier, and they bias exactly the comparison being made. Before that gap
can be reported as a fact about athletes it has to be separated from a fact about our regexes.

Design. For each audited tag, sample from three strata inside the Layer-1-relevant corpus:

  killed    the high-precision cue fired but a KILL pattern vetoed it
  abstain   only the broad high-recall cue fired, so the rule abstained
  low_prob  no cue fired, but the model's probability sits just under threshold

Every stratum is a place a true positive could have been lost, and each is sampled separately so
the correction can be weighted by how many corpus comments each stratum actually contains. Rows
are written blinded (text only) with the stratum held back in a separate key, matching the
labelling protocol used for the gold sets.

Usage: .venv/Scripts/python.exe code/sample_missed.py [--tags sleep,substance_use] [--n 80]
"""
import csv, json, os, random, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layer2_lexicon as L

csv.field_size_limit(2**31 - 1)
IN = "data/classified/final_dataset_tagged_v3b.csv"
OUTDIR = "data/layer2_audit"
SEED = 20260826


def stratum(text, tag, tagged, prob, thr):
    """Which recovery stratum, if any, does this comment fall in?"""
    if tagged == 1:
        return None                      # already captured
    hp = L._HP[tag].search(text)
    if hp:
        ctx = text[max(0, hp.start() - 60):hp.end() + 60]
        if any(k.search(ctx) for k in L._KILLS.get(tag, ())):
            return "killed"              # cue fired, a kill rule vetoed it
    if L._HR[tag].search(text):
        return "abstain"                 # plausible but unproven
    if prob is not None and thr is not None and prob >= thr * 0.5:
        return "low_prob"                # model was warm but under threshold
    return None


def main():
    tags = (sys.argv[sys.argv.index("--tags") + 1].split(",")
            if "--tags" in sys.argv else ["sleep", "substance_use"])
    n_per = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 80
    os.makedirs(OUTDIR, exist_ok=True)
    thr = json.load(open("models/layer2_tags_v3b/thresholds.json", encoding="utf-8"))

    pools = {t: defaultdict(list) for t in tags}
    counts = {t: Counter() for t in tags}
    n_rel = 0
    with open(IN, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("arm") != "matched":
                continue
            txt = (r.get("text") or "").strip()
            if len(txt) < 25:
                continue
            n_rel += 1
            for t in tags:
                tagged = 1 if r.get(f"tag_{t}") == "1" else 0
                if tagged:
                    counts[t]["tagged"] += 1
                    continue
                try:
                    prob = float(r.get(f"p_{t}") or 0)
                except ValueError:
                    prob = 0.0
                st = stratum(txt, t, tagged, prob, thr.get(t))
                if st:
                    counts[t][st] += 1
                    pools[t][st].append((r.get("id", ""), txt))

    rng = random.Random(SEED)
    summary = {"n_relevant": n_rel, "tags": {}}
    for t in tags:
        print(f"\n=== {t} ===  (relevant corpus n = {n_rel:,})")
        print(f"  tagged positive by the model : {counts[t]['tagged']:>7,} "
              f"({counts[t]['tagged']/n_rel*100:.2f}%)")
        rows, key = [], []
        for st in ("killed", "abstain", "low_prob"):
            pool = pools[t][st]
            print(f"  {st:<12} candidates      : {len(pool):>7,} "
                  f"({len(pool)/n_rel*100:.2f}%)")
            if not pool:
                continue
            rng.shuffle(pool)
            take = pool[:n_per]
            for cid, txt in take:
                rid = len(rows) + 1
                rows.append({"random_id": rid, "text": txt, t: ""})
                key.append({"random_id": rid, "comment_id": cid, "stratum": st})
        rng.shuffle(rows)
        blinded = f"{OUTDIR}/audit_{t}_blinded.csv"
        keyf = f"{OUTDIR}/audit_{t}_key.csv"
        with open(blinded, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["random_id", "text", t]); w.writeheader()
            w.writerows(rows)
        with open(keyf, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["random_id", "comment_id", "stratum"])
            w.writeheader(); w.writerows(key)
        summary["tags"][t] = {"tagged": counts[t]["tagged"],
                              "strata": {s: counts[t][s] for s in
                                         ("killed", "abstain", "low_prob")},
                              "sampled": len(rows)}
        print(f"  -> {len(rows)} blinded rows -> {blinded}")

    with open(f"{OUTDIR}/audit_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[audit] wrote {OUTDIR}/audit_summary.json")


if __name__ == "__main__":
    main()
