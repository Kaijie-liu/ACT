#!/usr/bin/env python3
"""Compare decisive ACT outcomes with VNN-COMP report labels.

This is a label-consistency audit, not a replacement for strict receipts:
CERTIFIED must align with official UNSAT and FALSIFIED with official SAT.
UNKNOWN and bounded watchdog outcomes make no semantic claim.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OFFICIAL = Path("/data1/Kane/HyZor/arXiv-2512.19007v1/generated/2025")
REPORT_NAMES = {
    "acasxu_2023": "Acasxu 2023",
    "cersyve": "Cersyve",
    "cifar100_2024": "Cifar100 2024",
    "collins_rul_cnn_2022": "Collins Rul Cnn 2022",
    "cora_2024": "Cora 2024",
    "dist_shift_2023": "Dist Shift 2023",
    "linearizenn_2024": "Linearizenn 2024",
    "malbeware": "Malbeware",
    "metaroom_2023": "Metaroom 2023",
    "nn4sys": "Nn4sys",
    "safenlp_2024": "Safenlp 2024",
    "sat_relu": "Sat Relu",
    "soundnessbench": "Soundnessbench",
    "tinyimagenet_2024": "Tinyimagenet 2024",
    "tllverifybench_2023": "Tllverifybench 2023",
}
# Only compare designated result sources.  Earlier configuration sweeps and
# pre-fix/error runs remain archived in CSVs, but are not reportable outcomes.
SOURCES = {
    "acasxu_2023": {"cpu_auto", "gpu"},
    "cersyve": {"cpu_native_r2"},
    "cifar100_2024": {"cpu_smoke"},
    "collins_rul_cnn_2022": {"cpu", "gpu"},
    "cora_2024": {"cpu_smoke"},
    "dist_shift_2023": {"cpu"},
    "linearizenn_2024": {"cpu_R9", "gpu"},
    "malbeware": {"cpu", "gpu"},
    "metaroom_2023": {"cpu_smoke"},
    "nn4sys": {"cpu", "cpu2"},
    "safenlp_2024": {"cpu_auto"},
    "sat_relu": {"cpu", "gpu"},
    "soundnessbench": {"cpu_smoke"},
    "tinyimagenet_2024": {"cpu_smoke"},
    "tllverifybench_2023": {"cpu_witness"},
}
PATTERN = re.compile(r"^2025\s+(.+?)\s+&\s+(\S+)\s+&\s+~\\textsc\{(sat|unsat)\}")


def parse_official(tolerance: str) -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    with (OFFICIAL / tolerance / "longtable.tex").open() as fp:
        for line in fp:
            match = PATTERN.search(line)
            if match:
                labels.setdefault(match.group(1).strip(), {})[match.group(2)] = match.group(3)
    return labels


def classify(verdict: str, official: str) -> str:
    if verdict == "CERTIFIED":
        return "agree_cert" if official == "unsat" else "disagreement"
    if verdict == "FALSIFIED":
        return "agree_fal" if official == "sat" else "disagreement"
    return "no_claim"


def audit(tolerance: str) -> tuple[dict[str, Any], list[dict[str, str]]]:
    labels = parse_official(tolerance)
    details: dict[str, Any] = {}
    disagreements: list[dict[str, str]] = []
    for benchmark, sources in SOURCES.items():
        report_name = REPORT_NAMES[benchmark]
        official = labels.get(report_name, {})
        csv_path = ROOT / benchmark / "per_instance.csv"
        counts: Counter[str] = Counter()
        if not csv_path.exists() or not official:
            details[benchmark] = {"official_labels": len(official), "rows_checked": 0}
            continue
        with csv_path.open(newline="") as fp:
            for row in csv.DictReader(fp):
                if row["source"] not in sources or row["iid"] not in official:
                    continue
                outcome = classify(row["verdict"], official[row["iid"]])
                counts[outcome] += 1
                counts["rows_checked"] += 1
                if outcome == "disagreement":
                    disagreements.append(
                        {
                            "tolerance": tolerance,
                            "benchmark": benchmark,
                            "source": row["source"],
                            "iid": row["iid"],
                            "act_verdict": row["verdict"],
                            "official_label": official[row["iid"]],
                            "json_path": row["json_path"],
                        }
                    )
        distinct = sorted(
            {row["iid"] for row in disagreements if row["benchmark"] == benchmark and row["tolerance"] == tolerance},
            key=lambda iid: int(iid),
        )
        details[benchmark] = {
            "official_labels": len(official),
            "sources": sorted(sources),
            "rows_checked": counts["rows_checked"],
            "agree_cert": counts["agree_cert"],
            "agree_fal": counts["agree_fal"],
            "no_claim": counts["no_claim"],
            "disagreement_rows": counts["disagreement"],
            "disagreement_iids": distinct,
        }
    return details, disagreements


def main() -> None:
    summary = {}
    disagreement_rows = []
    for tolerance in ("zero_tol", "small_tol"):
        summary[tolerance], current = audit(tolerance)
        disagreement_rows.extend(current)
    out_json = ROOT / "OFFICIAL_CROSSCHECK_SUMMARY.json"
    out_csv = ROOT / "OFFICIAL_CROSSCHECK_DISAGREEMENTS.csv"
    with out_json.open("w") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)
        fp.write("\n")
    fields = ["tolerance", "benchmark", "source", "iid", "act_verdict", "official_label", "json_path"]
    with out_csv.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(disagreement_rows)
    for tolerance in ("zero_tol", "small_tol"):
        print(f"=== {tolerance} ===")
        for benchmark, record in summary[tolerance].items():
            if record.get("rows_checked", 0):
                print(
                    f"{benchmark:30s} rows={record['rows_checked']:4d} "
                    f"CERT_ok={record['agree_cert']:3d} FAL_ok={record['agree_fal']:3d} "
                    f"no_claim={record['no_claim']:4d} "
                    f"disagree={record['disagreement_rows']:2d} {record['disagreement_iids']}"
                )
    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
