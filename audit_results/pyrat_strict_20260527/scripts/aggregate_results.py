#!/usr/bin/env python3
"""
Aggregate per-benchmark CSVs into a single summary, side-by-side with the
report's PyRAT scored/unscored numbers from arXiv-2512.19007v1.

Output:
  results_pure/_summary.csv
  results_pure/_summary.md
"""
import csv
import re
from collections import Counter
from pathlib import Path

PYRAT_DIR = Path(__file__).resolve().parent.parent
RES_DIR = PYRAT_DIR / "results_pure"

# Report numbers extracted from arXiv-2512.19007v1/generated/2025/zero_tol/scored.tex
# and unscored.tex (PyRAT rows). Order is the benchmark name -> (verified, falsified).
REPORT = {
    # scored
    "acasxu_2023":                     (139, 46),
    "cersyve":                         (2, 6),
    "cgan_2023":                       (9, 12),
    "cifar100_2024":                   (61, 25),
    "collins_rul_cnn_2022":            (39, 23),
    "cora_2024":                       (20, 128),
    "dist_shift_2023":                 (64, 7),
    "linearizenn_2024":                (59, 1),
    "malbeware":                       (121, 18),
    "metaroom_2023":                   (97, 3),
    "nn4sys":                          (40, 0),
    "safenlp_2024":                    (331, 647),
    "sat_relu":                        (9, 50),
    "soundnessbench":                  (68, 35),
    "tinyimagenet_2024":               (15, 17),
    "tllverifybench_2023":             None,  # confirm if rerun lands one
    # unscored / extended
    "cctsdb_yolo_2023":                (11, 28),
    "collins_aerospace_benchmark":     (0, 6),
    "lsnc_relu":                       (36, 3),
    "ml4acopf_2024":                   (61, 16),
    "relusplitter":                    (13, 1),
    "traffic_signs_recognition_2023":  (83, 0),
    "vggnet16_2022":                   (40, 0),
    "vit_2023":                        None,
    "yolo_2023":                       None,
}


def load_one(bench_dir):
    csv_p = bench_dir / "results.csv"
    if not csv_p.exists():
        return None
    verdicts = Counter()
    total_wall = 0.0
    with open(csv_p) as fh:
        rdr = csv.reader(fh)
        next(rdr, None)  # header
        for row in rdr:
            if len(row) < 7:
                continue
            verdicts[row[3]] += 1
            try:
                total_wall += float(row[4])
            except Exception:
                pass
    return verdicts, total_wall


def main():
    rows_out = []
    for bench_dir in sorted(RES_DIR.iterdir()):
        if not bench_dir.is_dir() or bench_dir.name.startswith("_"):
            continue
        loaded = load_one(bench_dir)
        if loaded is None:
            continue
        verdicts, wall = loaded
        bench = bench_dir.name
        v = verdicts.get("verified", 0)
        f = verdicts.get("falsified", 0)
        u = verdicts.get("unknown", 0)
        t = verdicts.get("timeout", 0)
        e = verdicts.get("error", 0)
        rep = REPORT.get(bench)
        rep_str = f"{rep[0]}/{rep[1]}" if rep else "-"
        rows_out.append((bench, v, f, u, t, e, wall, rep_str))

    summary_csv = RES_DIR / "_summary.csv"
    summary_md = RES_DIR / "_summary.md"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["benchmark", "verified", "falsified", "unknown",
                    "timeout", "error", "wall_s", "report_v/f"])
        for r in rows_out:
            w.writerow(r)

    with open(summary_md, "w") as fh:
        fh.write("# PyRAT 2025 — strict no-adv-search reproduction\n\n")
        fh.write("|benchmark|verified|falsified|unknown|timeout|error|wall(s)|report v/f|\n")
        fh.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows_out:
            fh.write(f"|{r[0]}|{r[1]}|{r[2]}|{r[3]}|{r[4]}|{r[5]}|{r[6]:.0f}|{r[7]}|\n")

    print(f"wrote {summary_csv}")
    print(f"wrote {summary_md}")


if __name__ == "__main__":
    main()
