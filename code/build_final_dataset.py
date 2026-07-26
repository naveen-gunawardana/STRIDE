"""Concatenate the per-month pruned arm outputs into the single study dataset + a counts report.

Previously done ad hoc; scripted here so the dataset is reproducible from the pipeline
(`driver_classify.py` -> per-month CSVs -> this).

Output:
  data/classified/final_dataset.csv   arm, year_range + all original columns + p_mh, p_sport
  stdout: per-arm processed/relevant/% table (the separation statistic for the paper)

Usage: .venv/Scripts/python.exe code/build_final_dataset.py [--out PATH]
"""
import csv, glob, os, sys
csv.field_size_limit(2**31 - 1)

# (arm label, year_range label, classified dir, source dir for the processed-row count)
ARMS = [
    ("matched",  "2018-2022", "data/classified/matched_2018_2022",
     "comments/2018-2022/MS_comments_2018_2022/MS_comments_2018_2022/matched"),
    ("matched",  "2023",      "data/classified/matched_2023",
     "comments/2023/comments_2023/filtered_subreddit_keywords/matched"),
    ("baseline", "2018-2022", "data/classified/baseline_2018_2022",
     "comments/2018-2022/MS_comments_2018_2022/MS_comments_2018_2022/baseline"),
]

def count_rows(d):
    """Rows read from the raw arm (denominator incl. comments dropped by the subreddit filter)."""
    n = 0
    for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
        with open(f, encoding="utf-8-sig", newline="") as fh:
            n += sum(1 for _ in csv.DictReader(fh))
    return n

def main():
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else \
        "data/classified/final_dataset.csv"
    writer = fh_out = None
    report = []
    for arm, yr, cdir, sdir in ARMS:
        files = sorted(glob.glob(os.path.join(cdir, "*.csv")))
        if not files:
            print(f"[build] WARNING: no classified files in {cdir} -- skipping", flush=True)
            continue
        rel = 0
        for f in files:
            with open(f, encoding="utf-8", newline="") as fin:
                rd = csv.DictReader(fin)
                for r in rd:
                    if writer is None:
                        fh_out = open(out, "w", encoding="utf-8", newline="")
                        writer = csv.DictWriter(
                            fh_out, fieldnames=["arm", "year_range"] + list(rd.fieldnames or []))
                        writer.writeheader()
                    r["arm"], r["year_range"] = arm, yr
                    writer.writerow(r)
                    rel += 1
        proc = count_rows(sdir)
        report.append((arm, yr, proc, rel))
        print(f"[build] {arm} {yr}: {proc:,} processed -> {rel:,} relevant ({rel/max(proc,1)*100:.2f}%)",
              flush=True)
    if fh_out:
        fh_out.close()

    print(f"\n[build] wrote {out}  ({os.path.getsize(out)/1e6:.0f} MB)\n")
    print("| arm | processed | relevant | % |")
    print("|---|---|---|---|")
    mp = mr = bp = br = 0
    for arm, yr, p, r in report:
        print(f"| {arm} {yr} | {p:,} | {r:,} | {r/max(p,1)*100:.2f}% |")
        if arm == "matched":
            mp += p; mr += r
        else:
            bp += p; br += r
    if mp:
        print(f"| **matched total (the dataset)** | **{mp:,}** | **{mr:,}** | **{mr/mp*100:.2f}%** |")
    if bp:
        print(f"| baseline (control) | {bp:,} | {br:,} | {br/bp*100:.2f}% |")
    if mp and bp and br:
        print(f"\nseparation: matched {mr/mp*100:.2f}% vs baseline {br/bp*100:.2f}% "
              f"= {(mr/mp)/(br/bp):.0f}x")

if __name__ == "__main__":
    main()
