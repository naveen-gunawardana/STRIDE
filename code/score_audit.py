"""Recall-corrected prevalence for the audited tags.

The classifier reports sleep in 2.2% of relevant comments and substance use in 1.7%, against
26.4% and 18.8% in the clinical prevalence literature. Before that gap can be attributed to
athletes rather than to our own rules, we need to know how many true positives sit outside the
classifier's positive set.

Method. Three recovery strata were sampled (see code/sample_missed.py): comments where a kill
rule vetoed a fired cue, comments where the rule abstained, and comments the model scored warm
but under threshold. Each was hand-labelled against the published rubric. The corrected count is

    corrected = tagged + SUM over strata of  stratum_size * (positives / sampled)

with a Wilson interval on each stratum rate carried through, so the correction has an interval
rather than a point value.

This bounds how much of the discourse-versus-screening gap is measurement. It does not close it:
the labels come from the same rater who produced the gold set, so this is a recall audit, not an
independent validation.

Usage: .venv/Scripts/python.exe code/score_audit.py
"""
import csv, json, math, os
from collections import defaultdict

D = "data/layer2_audit"
OUT = "results_temporal"
N_RELEVANT = 129430

# Published prevalence for the same construct, for the comparison the paper makes.
LIT = {"sleep": 26.4, "substance_use": 18.8}

# Positive ids assigned by reading each blinded sample against docs/methods/layer2_tag_rubric.md.
LABELS = {
    "sleep": {5, 51, 14, 67, 88, 61, 118, 39, 98, 89, 29},
    "substance_use": {108, 72, 112, 23, 101, 143, 85, 60, 96, 121, 92, 55, 107, 149,
                      99, 20, 134, 131, 67, 17, 32, 139,
                      140, 151, 89, 88, 26, 105, 148, 122, 98, 115, 118},
}
# Of the substance positives, those resting only on tobacco or nicotine. The v3 rubric names
# alcohol and drugs and is silent on tobacco, so this subset is reported separately rather than
# silently included or excluded.
TOBACCO_ONLY = {108, 143, 96, 149, 99, 134, 131, 32, 140, 151, 88, 105, 148, 98, 115}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    summary = json.load(open(f"{D}/audit_summary.json", encoding="utf-8"))
    report = {}
    for tag, pos in LABELS.items():
        key = {}
        with open(f"{D}/audit_{tag}_key.csv", encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                key[int(r["random_id"])] = r["stratum"]
        info = summary["tags"][tag]
        tagged = info["tagged"]
        strata = info["strata"]

        by = defaultdict(lambda: [0, 0])          # stratum -> [positives, sampled]
        for rid, st in key.items():
            by[st][1] += 1
            if rid in pos:
                by[st][0] += 1

        print(f"\n===== {tag} =====")
        print(f"  model-tagged positives            {tagged:>8,}  "
              f"({tagged / N_RELEVANT * 100:.2f}% of relevant)")
        print(f"  {'stratum':<10} {'size':>8} {'sampled':>8} {'pos':>5} {'rate':>7} "
              f"{'-> est. missed':>16}")
        add, add_lo, add_hi = 0.0, 0.0, 0.0
        for st in ("killed", "abstain", "low_prob"):
            size = strata.get(st, 0)
            k, n = by[st]
            if n == 0 or size == 0:
                continue
            rate = k / n
            lo, hi = wilson(k, n)
            add += size * rate
            add_lo += size * lo
            add_hi += size * hi
            print(f"  {st:<10} {size:>8,} {n:>8} {k:>5} {rate:>6.1%} "
                  f"{size * rate:>10,.0f} [{size*lo:,.0f}-{size*hi:,.0f}]")

        corr = tagged + add
        corr_lo, corr_hi = tagged + add_lo, tagged + add_hi
        p, plo, phi = (corr / N_RELEVANT * 100, corr_lo / N_RELEVANT * 100,
                       corr_hi / N_RELEVANT * 100)
        lit = LIT[tag]
        print(f"\n  corrected prevalence  {p:.2f}%  (95% CI {plo:.2f}-{phi:.2f})")
        print(f"  literature            {lit:.1f}%")
        print(f"  remaining gap         {lit / p:.1f}x  "
              f"(was {lit / (tagged / N_RELEVANT * 100):.1f}x before correction)")

        report[tag] = {
            "tagged": tagged, "tagged_pct": tagged / N_RELEVANT * 100,
            "corrected": corr, "corrected_pct": p,
            "corrected_pct_ci": [plo, phi],
            "literature_pct": lit,
            "gap_before": lit / (tagged / N_RELEVANT * 100), "gap_after": lit / p,
            "strata": {st: {"size": strata.get(st, 0), "sampled": by[st][1],
                            "positives": by[st][0]} for st in by},
        }

        if tag == "substance_use":
            npos = {i for i in pos if i not in TOBACCO_ONLY}
            by2 = defaultdict(lambda: [0, 0])
            for rid, st in key.items():
                by2[st][1] += 1
                if rid in npos:
                    by2[st][0] += 1
            add2 = sum(strata.get(st, 0) * (by2[st][0] / by2[st][1])
                       for st in by2 if by2[st][1])
            p2 = (tagged + add2) / N_RELEVANT * 100
            print(f"\n  excluding tobacco-only positives ({len(TOBACCO_ONLY)} of {len(pos)} "
                  f"sampled positives rest on smoking or nicotine, which the v3 rubric does not "
                  f"name): corrected prevalence {p2:.2f}%, remaining gap {lit / p2:.1f}x")
            report[tag]["corrected_pct_excl_tobacco"] = p2
            report[tag]["gap_after_excl_tobacco"] = lit / p2

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/audit_corrected.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n[audit] wrote {OUT}/audit_corrected.json")


if __name__ == "__main__":
    main()
