"""Write Layer-2 gold labels into a blinded rating CSV from a compact spec.

The rater reads the blinded file and records one line per row:

    <row_id>: <code>[,<code>...]        e.g.  17: dep,anx,help
    <row_id>: none                      no tag applies
    <row_id>: dep,anx?                  trailing '?' marks that tag UNCLEAR -> written as 'x'
                                        (excluded from metrics, as in the Layer-1 protocol)

Every tag not listed is written 0, so a spec line is a complete labeling of that row. Rows with
no spec line are left blank and are skipped by the evaluator.

Usage:
  .venv/Scripts/python.exe code/apply_labels.py <rated.csv> <spec.txt> [--notes notes.txt]
"""
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layer2_lexicon import TAGS
csv.field_size_limit(2**31 - 1)

CODES = {
    "dep": "depression", "anx": "anxiety", "str": "stress_pressure",
    "bur": "burnout_motivation", "perf": "performance_psych", "body": "body_image_eating",
    "inj": "injury_distress", "sh": "self_harm_suicide", "help": "help_seeking",
    "cope": "exercise_coping",
}

def parse(spec_path):
    out = {}
    with open(spec_path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith("#") or ":" not in ln:
                continue
            rid, rest = ln.split(":", 1)
            rid = rid.strip()
            vals = {t: "0" for t in TAGS}
            for c in rest.split(","):
                c = c.strip().lower()
                if not c or c == "none":
                    continue
                unclear = c.endswith("?")
                c = c.rstrip("?")
                if c not in CODES:
                    sys.exit(f"unknown code {c!r} on line: {ln}")
                vals[CODES[c]] = "x" if unclear else "1"
            out[rid] = vals
    return out

def main():
    rated, spec = sys.argv[1], sys.argv[2]
    labels = parse(spec)
    with open(rated, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
        fields = list(rows[0].keys()) if rows else []
    hit = 0
    for r in rows:
        v = labels.get(str(r.get("row_id", "")).strip())
        if v:
            r.update(v); hit += 1
    with open(rated, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    missing = [r.get("row_id") for r in rows if str(r.get("row_id", "")).strip() not in labels]
    print(f"[labels] wrote {hit}/{len(rows)} rows in {rated}")
    if missing:
        print(f"[labels] UNLABELED row_ids ({len(missing)}): {missing[:40]}")
    pos = {t: sum(1 for r in rows if r.get(t) == "1") for t in TAGS}
    unc = {t: sum(1 for r in rows if r.get(t) == "x") for t in TAGS}
    print(f"\n  {'tag':22} {'pos':>5} {'unclear':>8}   prevalence")
    for t in TAGS:
        print(f"  {t:22} {pos[t]:5} {unc[t]:8}   {pos[t]/max(hit,1)*100:5.1f}%")
    ntag = [sum(1 for t in TAGS if r.get(t) == "1") for r in rows if r.get("depression") in ("0", "1", "x")]
    if ntag:
        print(f"\n  mean tags per labeled comment: {sum(ntag)/len(ntag):.2f}")
        print(f"  comments with zero tags: {sum(1 for k in ntag if k == 0)}/{len(ntag)}")

if __name__ == "__main__":
    main()
