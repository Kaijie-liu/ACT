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
import json
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Mapping

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


if __name__ == "__main__":  # pragma: no cover
    _test_hybridz_results()
    print("PASS _test_hybridz_results")
