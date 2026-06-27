# ===- act/pipeline/hybridz_results.py - HybridZ run reporting ----------===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#

"""CSV reporting helpers for strict HybridZ frontend runs."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Mapping, Optional

from act.back_end.hybridz_config import (
    CROSS_TOOL_NAMES,
    FROZEN_COMPETITOR_COUNTS,
)
from act.util.stats import VerifyResult, VerifyStatus


_VERDICT_BY_STATUS = {
    VerifyStatus.CERTIFIED: "CERT",
    VerifyStatus.FALSIFIED: "ADV",
    VerifyStatus.TIMEOUT: "TIMEOUT",
    VerifyStatus.VERIFIER_ERROR: "ERROR",
    VerifyStatus.MODEL_INFER_FAILURE: "ERROR",
    VerifyStatus.UNKNOWN: "UNKNOWN",
}


def _metadata_p0(meta: Mapping[str, Any]) -> bool:
    """Return whether result metadata explicitly reports a P0 soundness flag."""

    for key in ("p0", "P0"):
        value = meta.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into dictionaries, returning an empty list if missing."""

    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_sha256_manifest(root: Path, manifest_name: str = "_MANIFEST.sha256") -> Path:
    """Write a deterministic SHA256 manifest for files under ``root``."""

    root = Path(root)
    manifest = root / manifest_name
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path == manifest:
            continue
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        rows.append(f"{h.hexdigest()}  {path.relative_to(root).as_posix()}")
    manifest.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return manifest


def int_field(row: Mapping[str, object], key: str) -> int:
    """Parse an integer CSV field defensively."""

    try:
        return int(row.get(key, 0) or 0)
    except Exception:
        return 0


@dataclass
class HybridZRunRow:
    bench: str
    tag: str
    lane: int
    status: str
    verdict: str
    wall_s: float
    reason: str = ""
    hz_verdict: str = ""
    engine: str = ""
    p0: bool = False
    metadata_json: str = ""


@dataclass
class HybridZRunRecorder:
    """Collect per-instance HybridZ frontend results and write CSV summaries."""

    bench: str
    rows: list[HybridZRunRow] = field(default_factory=list)

    def add_results(self, tag: str, results: Iterable[VerifyResult], *, wall_s: float) -> None:
        for lane, result in enumerate(results):
            meta: Mapping[str, Any] = result.metadata or {}
            self.rows.append(
                HybridZRunRow(
                    bench=self.bench,
                    tag=tag,
                    lane=int(meta.get("lane", lane)),
                    status=result.status.name,
                    verdict=_VERDICT_BY_STATUS.get(result.status, "UNKNOWN"),
                    wall_s=float(wall_s),
                    reason=str(meta.get("reason", "")),
                    hz_verdict=str(meta.get("hz_verdict", "")),
                    engine=str(meta.get("engine", "")),
                    p0=_metadata_p0(meta),
                    metadata_json=json.dumps(meta, sort_keys=True, default=str),
                )
            )

    def summary(self) -> dict[str, int | str]:
        counts = {"CERT": 0, "ADV": 0, "TIMEOUT": 0, "UNKNOWN": 0, "ERROR": 0}
        for row in self.rows:
            counts[row.verdict] = counts.get(row.verdict, 0) + 1
        cert = int(counts.get("CERT", 0))
        adv = int(counts.get("ADV", 0))
        timeout = int(counts.get("TIMEOUT", 0))
        unknown = int(counts.get("UNKNOWN", 0))
        error = int(counts.get("ERROR", 0))
        p0 = sum(1 for row in self.rows if row.p0)
        return {
            "Bench": self.bench,
            "N": len(self.rows),
            "CERT": cert,
            "ADV": adv,
            "V+A": cert + adv,
            "TIMEOUT": timeout,
            "UNKNOWN": unknown,
            "ERROR": error,
            "P0": p0,
            "unsolved": timeout + unknown + error,
        }

    def write(self, out_dir: str | Path) -> tuple[Path, Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        detail_path = out / f"{self.bench}_hybridz_detail.csv"
        summary_path = out / f"{self.bench}_hybridz_summary.csv"

        row_fields = [
            "bench",
            "tag",
            "lane",
            "status",
            "verdict",
            "wall_s",
            "reason",
            "hz_verdict",
            "engine",
            "p0",
            "metadata_json",
        ]
        with detail_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row_fields)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({name: getattr(row, name) for name in row_fields})

        summary = self.summary()
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            writer.writeheader()
            writer.writerow(summary)
        return detail_path, summary_path


def _ours_counts_from_summary(row: Mapping[str, object]) -> tuple[int, int, int, int, int]:
    return (
        int_field(row, "CERT"),
        int_field(row, "ADV"),
        int_field(row, "TIMEOUT"),
        int_field(row, "UNKNOWN"),
        int_field(row, "ERROR"),
    )


def _va_count(counts: tuple[int, int, int, int, int]) -> int:
    return int(counts[0]) + int(counts[1])


def build_cross_tool_rows(
    summary_rows: Iterable[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build frozen-suite ranking and cross-tool comparison rows."""

    ranked_rows: list[dict[str, object]] = []
    cross_rows: list[dict[str, object]] = []

    for row in summary_rows:
        bench = str(row.get("Bench", ""))
        if bench == "TOTAL" or bench not in FROZEN_COMPETITOR_COUNTS:
            continue
        tool_counts = {"OURS": _ours_counts_from_summary(row)}
        tool_counts.update(FROZEN_COMPETITOR_COUNTS[bench])
        tool_va = {tool: _va_count(tool_counts[tool]) for tool in CROSS_TOOL_NAMES}
        ours_va = tool_va["OURS"]
        best_va = max(tool_va.values())
        best_tools = [tool for tool in CROSS_TOOL_NAMES if tool_va[tool] == best_va]

        ranked = {
            "Bench": bench,
            "N": int_field(row, "N"),
            "CERT": int_field(row, "CERT"),
            "ADV": int_field(row, "ADV"),
            "V+A": ours_va,
            "TIMEOUT": int_field(row, "TIMEOUT"),
            "UNKNOWN": int_field(row, "UNKNOWN"),
            "ERROR": int_field(row, "ERROR"),
            "P0": int_field(row, "P0"),
            "unsolved": int_field(row, "unsolved"),
            "rank_competition": 1 + sum(1 for value in tool_va.values() if value > ours_va),
            "rank_dense": 1 + len({value for value in tool_va.values() if value > ours_va}),
            "best_V+A": best_va,
            "best_tools": "+".join(best_tools),
            "gap_to_best": best_va - ours_va,
        }
        ranked.update({tool: tool_va[tool] for tool in CROSS_TOOL_NAMES})
        ranked_rows.append(ranked)

        cross = {"Bench": bench, "N": int_field(row, "N")}
        for tool in CROSS_TOOL_NAMES:
            unsat, sat, timeout, unknown, error = tool_counts[tool]
            cross.update(
                {
                    tool: unsat + sat,
                    f"{tool}_unsat": unsat,
                    f"{tool}_sat": sat,
                    f"{tool}_timeout": timeout,
                    f"{tool}_unknown": unknown,
                    f"{tool}_error": error,
                }
            )
        cross_rows.append(cross)
    return ranked_rows, cross_rows


def write_suite_cross_tool_outputs(
    out_dir: Path,
    suite_summary_path: Path,
    *,
    max_instances: Optional[int] = None,
) -> tuple[Path, Path, Path] | tuple[()]:
    """Write frozen-suite final ranking and cross-tool comparison CSVs."""

    if max_instances is not None:
        return ()
    ranked_rows, cross_rows = build_cross_tool_rows(read_csv_rows(suite_summary_path))
    if not ranked_rows:
        return ()

    out_dir = Path(out_dir)
    ranking_fields = [
        "Bench",
        "N",
        "CERT",
        "ADV",
        "V+A",
        "TIMEOUT",
        "UNKNOWN",
        "ERROR",
        "P0",
        "unsolved",
        "rank_competition",
        "rank_dense",
        "best_V+A",
        "best_tools",
        "gap_to_best",
        *CROSS_TOOL_NAMES,
    ]
    final_results = out_dir / "FINAL_HYBRIDZ_RESULTS.csv"
    final_ranking = out_dir / "FINAL_CROSS_TOOL_RANKING.csv"
    for path in (final_results, final_ranking):
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ranking_fields)
            writer.writeheader()
            writer.writerows(
                {name: row.get(name, "") for name in ranking_fields}
                for row in ranked_rows
            )

    cross_fields = ["Bench", "N"]
    for tool in CROSS_TOOL_NAMES:
        cross_fields.extend(
            [
                tool,
                f"{tool}_unsat",
                f"{tool}_sat",
                f"{tool}_timeout",
                f"{tool}_unknown",
                f"{tool}_error",
            ]
        )
    cross_path = out_dir / "_CROSS_TOOL_SUMMARY.csv"
    with cross_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cross_fields)
        writer.writeheader()
        writer.writerows({name: row.get(name, "") for name in cross_fields} for row in cross_rows)
    return final_results, final_ranking, cross_path


def _test_hybridz_results() -> None:  # pragma: no cover
    rec = HybridZRunRecorder("toy")
    rec.add_results(
        "iid0",
        [
            VerifyResult(VerifyStatus.CERTIFIED, metadata={"engine": "hybridz"}),
            VerifyResult(
                VerifyStatus.FALSIFIED,
                metadata={"reason": "adv", "lane": 7, "hz_verdict": "ADV"},
            ),
            VerifyResult(VerifyStatus.UNKNOWN, metadata={"reason": "timeout"}),
            VerifyResult(VerifyStatus.TIMEOUT, metadata={"hz_verdict": "UNKNOWN"}),
            VerifyResult(VerifyStatus.MODEL_INFER_FAILURE, metadata={"reason": "bad model"}),
        ],
        wall_s=1.25,
    )
    s = rec.summary()
    assert s["N"] == 5
    assert s["CERT"] == 1
    assert s["ADV"] == 1
    assert s["V+A"] == 2
    assert s["UNKNOWN"] == 1
    assert s["TIMEOUT"] == 1
    assert s["ERROR"] == 1
    assert s["P0"] == 0
    assert s["unsolved"] == 3
    assert rec.rows[1].lane == 7
    assert rec.rows[1].reason == "adv"
    assert rec.rows[3].hz_verdict == "UNKNOWN"

    rec_p0 = HybridZRunRecorder("toy")
    rec_p0.add_results(
        "iid1",
        [VerifyResult(VerifyStatus.UNKNOWN, metadata={"p0": True})],
        wall_s=0.1,
    )
    assert rec_p0.summary()["P0"] == 1

    with TemporaryDirectory() as tmp:
        detail_path, summary_path = rec.write(tmp)
        with detail_path.open(newline="") as f:
            detail_rows = list(csv.DictReader(f))
        with summary_path.open(newline="") as f:
            summary_rows = list(csv.DictReader(f))
        assert detail_rows[1]["lane"] == "7"
        assert detail_rows[1]["verdict"] == "ADV"
        assert json.loads(detail_rows[0]["metadata_json"])["engine"] == "hybridz"
        assert len(summary_rows) == 1
        assert summary_rows[0]["V+A"] == "2"
        assert summary_rows[0]["P0"] == "0"
        assert summary_rows[0]["unsolved"] == "3"

        manifest = write_sha256_manifest(Path(tmp))
        assert manifest.exists()
        assert "toy_hybridz_summary.csv" in manifest.read_text(encoding="utf-8")

    ranked_rows, cross_rows = build_cross_tool_rows(
        [
            {
                "Bench": "safenlp_2024",
                "N": "1080",
                "CERT": "432",
                "ADV": "647",
                "TIMEOUT": "0",
                "UNKNOWN": "1",
                "ERROR": "0",
                "P0": "0",
                "unsolved": "1",
            }
        ]
    )
    assert ranked_rows[0]["rank_competition"] == 2
    assert ranked_rows[0]["best_tools"] == "abCROWN"
    assert cross_rows[0]["OURS_sat"] == 647


if __name__ == "__main__":  # pragma: no cover
    _test_hybridz_results()
    print("PASS _test_hybridz_results")
