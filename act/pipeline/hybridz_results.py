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
    FROZEN_BENCHMARK_SUITE,
    FROZEN_COMPETITOR_COUNTS,
    FROZEN_SUMMARY_FIELDS,
    frozen_hybridz_expected_summary,
    get_bench_profile,
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
FROZEN_REPRO_MATCH_FIELDS = ("N", "CERT", "ADV", "V+A", "ERROR", "P0", "unsolved")


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


def run_verify_time_s(run: Mapping[str, object]) -> float:
    """Extract the verification wall time from a runner result row."""

    for row in run.get("detail_rows", []) or []:
        if not isinstance(row, Mapping):
            continue
        try:
            return float(row.get("wall_s") or run.get("wall_s") or 0.0)
        except Exception:
            break
    try:
        return float(run.get("wall_s") or 0.0)
    except Exception:
        return 0.0


def truthy_flag(value: object) -> bool:
    """Parse bool-like CSV/JSON fields used by reporting checks."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return False
        try:
            return float(text) != 0.0
        except ValueError:
            return text in {"true", "yes", "y"}
    return False


def row_has_p0(row: Mapping[str, object]) -> bool:
    """Return whether a detail/summary CSV row reports a P0 flag."""

    if truthy_flag(row.get("p0")) or truthy_flag(row.get("P0")):
        return True
    raw_meta = row.get("metadata_json")
    if not raw_meta:
        return False
    try:
        meta = json.loads(str(raw_meta))
    except Exception:
        return False
    if not isinstance(meta, Mapping):
        return False
    return truthy_flag(meta.get("p0")) or truthy_flag(meta.get("P0"))


def run_has_p0(run: Mapping[str, object]) -> bool:
    """Return whether a runner result or any attached CSV row reports P0."""

    if truthy_flag(run.get("p0")):
        return True
    payload = run.get("module_payload")
    if isinstance(payload, Mapping) and (
        truthy_flag(payload.get("p0")) or truthy_flag(payload.get("P0"))
    ):
        return True
    for key in ("summary_rows", "detail_rows"):
        for row in run.get(key, []) or []:
            if isinstance(row, Mapping) and row_has_p0(row):
                return True
    return False


def _json_field(value: object) -> str:
    return json.dumps(value if value is not None else {}, sort_keys=True, default=str)


def _detail_reason(run: Mapping[str, object]) -> str:
    for row in run.get("detail_rows", []) or []:
        if not isinstance(row, Mapping):
            continue
        reason = str(row.get("reason", "")).strip()
        if reason:
            return reason[:300]
    return ""


def _tail_text(run: Mapping[str, object]) -> str:
    return (str(run.get("stderr_tail", "")) or str(run.get("stdout_tail", "")))[:300]


def taxonomy_class(run: Mapping[str, object]) -> tuple[str, str]:
    """Classify unresolved strict-HybridZ runner results for audit CSVs."""

    verdict = str(run.get("verdict", "ERROR"))
    portfolio_done = run.get("portfolio_done") or {}
    branches = [str(x) for x in run.get("portfolio_branches", []) or []]
    err = _tail_text(run)
    reason = _detail_reason(run)
    text = f"{reason}\n{err}".lower()

    if verdict in {"CERT", "ADV"}:
        return "verified", "pure exact-HZ solver verdict"
    if verdict == "TIMEOUT":
        return "official_wall_timeout", "official per-instance wall exhausted"
    if "unsupported" in text:
        return "unsupported_operator", (reason or err or "unsupported operator")[:300]
    if verdict == "UNKNOWN":
        if any(str(v) == "TIMEOUT" for v in getattr(portfolio_done, "values", lambda: [])()):
            if any(name.startswith("sparse") for name in branches):
                return "sparse_portfolio_wall", "sparse pure fallback exhausted its portfolio"
            return "portfolio_wall", "pure-HZ formulation portfolio exhausted its wall"
        if "drop" in text or "dropped" in text:
            return "representation_wall", "HZ representation dropped before a counted proof"
        return "engine_unknown", reason or "pure HZ engine returned UNKNOWN"
    if verdict == "ERROR":
        if str(run.get("branch", "")) == "missing_downloaded_vnnlib_instances":
            return "missing_downloaded_data", err or reason or "downloaded VNNLIB instances missing"
        if "no downloaded vnnlib instances found" in text:
            return "missing_downloaded_data", err or reason
        return "engine_error", err
    return "other", f"unhandled verdict {verdict}"


def write_suite_failure_taxonomy(
    out_dir: str | Path,
    run_rows: Iterable[Mapping[str, object]],
) -> tuple[Path, Path]:
    """Write per-instance failure taxonomy and per-benchmark summary CSVs."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in run_rows:
        result_class, note = taxonomy_class(run)
        rows.append(
            {
                "bench": str(run.get("bench", "")),
                "iid": int(run.get("index", -1)),
                "verdict": str(run.get("verdict", "ERROR")),
                "result_class": result_class,
                "branch": str(run.get("branch", "")),
                "portfolio_done": _json_field(run.get("portfolio_done", {})),
                "portfolio_branches": ";".join(str(x) for x in run.get("portfolio_branches", []) or []),
                "time_s": f"{run_verify_time_s(run):.2f}",
                "returncode": str(run.get("returncode", "")),
                "p0": int(run_has_p0(run)),
                "reason": _detail_reason(run),
                "err": _tail_text(run),
                "note": note,
            }
        )
    rows.sort(key=lambda row: (row["bench"], int(row["iid"])))

    detail_path = out / "failure_taxonomy_detail.csv"
    detail_fields = [
        "bench",
        "iid",
        "verdict",
        "result_class",
        "branch",
        "portfolio_done",
        "portfolio_branches",
        "time_s",
        "returncode",
        "p0",
        "reason",
        "err",
        "note",
    ]
    with detail_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows({name: row.get(name, "") for name in detail_fields} for row in rows)

    counts_by_bench: dict[str, dict[str, int]] = {}
    for row in rows:
        bench = str(row["bench"])
        counts = counts_by_bench.setdefault(bench, {"N": 0, "V+A": 0, "P0": 0})
        counts["N"] += 1
        if row["verdict"] in {"CERT", "ADV"} and row["result_class"] == "verified":
            counts["V+A"] += 1
        if truthy_flag(row.get("p0")):
            counts["P0"] += 1
        counts[str(row["result_class"])] = counts.get(str(row["result_class"]), 0) + 1

    classes = sorted({
        key
        for counts in counts_by_bench.values()
        for key in counts
        if key not in {"N", "V+A", "P0"}
    })
    summary_path = out / "failure_taxonomy_summary.csv"
    summary_fields = ["bench", "N", "V+A", "P0", *classes]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for bench in sorted(counts_by_bench):
            counts = counts_by_bench[bench]
            writer.writerow({field: bench if field == "bench" else counts.get(field, 0) for field in summary_fields})
    return detail_path, summary_path


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


def verdict_to_icse_result(verdict: str) -> str:
    """Map strict HybridZ verdict tokens to ICSE/VNN-COMP CSV tokens."""

    if verdict == "CERT":
        return "unsat"
    if verdict == "ADV":
        return "sat"
    if verdict == "TIMEOUT":
        return "timeout"
    if verdict == "UNKNOWN":
        return "unknown"
    return "error"


def write_icse_benchmark_outputs(
    bench: str,
    out_dir: str | Path,
    run_rows: Iterable[Mapping[str, object]],
) -> tuple[Path, Path, Path]:
    """Write per-benchmark ICSE/VNN-COMP style CSV exports."""

    out = Path(out_dir)
    rows = list(run_rows)
    bench_path = out / f"{bench}.csv"
    index_path = out / f"{bench}_icse_index.csv"
    detail_path = out / f"{bench}_icse_detail.csv"

    counts = {"unsat": 0, "sat": 0, "timeout": 0, "unknown": 0, "unsupported": 0, "error": 0}
    total_time = 0.0
    with bench_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["onnx", "vnnlib", "result", "time_sec"])
        writer.writeheader()
        for run in rows:
            result = verdict_to_icse_result(str(run.get("verdict", "ERROR")))
            time_s = run_verify_time_s(run)
            total_time += time_s
            counts[result] = counts.get(result, 0) + 1
            writer.writerow(
                {
                    "onnx": str(run.get("onnx_model", "")),
                    "vnnlib": str(run.get("vnnlib_spec", "")),
                    "result": result,
                    "time_sec": f"{time_s:.2f}",
                }
            )

    index_row = {
        "benchmark": bench,
        "N": len(rows),
        "unsat": counts["unsat"],
        "sat": counts["sat"],
        "timeout": counts["timeout"],
        "unknown": counts["unknown"],
        "unsupported": counts["unsupported"],
        "error": counts["error"],
        "total_time_sec": f"{total_time:.1f}",
    }
    index_fields = [
        "benchmark",
        "N",
        "unsat",
        "sat",
        "timeout",
        "unknown",
        "unsupported",
        "error",
        "total_time_sec",
    ]
    with index_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=index_fields)
        writer.writeheader()
        writer.writerow(index_row)

    detail_fields = [
        "benchmark",
        "iid",
        "onnx",
        "vnnlib",
        "csv_timeout",
        "result",
        "time_sec",
        "raw_verdict",
        "branch",
        "portfolio_branches",
        "portfolio_done",
        "p0",
        "err",
    ]
    with detail_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        for run in rows:
            result = verdict_to_icse_result(str(run.get("verdict", "ERROR")))
            time_s = run_verify_time_s(run)
            err = run.get("stderr_tail", "") or run.get("stdout_tail", "")
            writer.writerow(
                {
                    "benchmark": bench,
                    "iid": int(run.get("index", -1)),
                    "onnx": str(run.get("onnx_model", "")),
                    "vnnlib": str(run.get("vnnlib_spec", "")),
                    "csv_timeout": run.get("timeout_s", ""),
                    "result": result,
                    "time_sec": f"{time_s:.2f}",
                    "raw_verdict": str(run.get("verdict", "ERROR")),
                    "branch": str(run.get("branch", "")),
                    "portfolio_branches": ";".join(str(x) for x in run.get("portfolio_branches", [])),
                    "portfolio_done": str(run.get("portfolio_done", "")),
                    "p0": int(run_has_p0(run)),
                    "err": str(err)[:300],
                }
            )
    return bench_path, index_path, detail_path


def profile_json(bench: str) -> dict[str, object]:
    """Serialize the HybridZ benchmark profile used for a benchmark."""

    profile = get_bench_profile(bench)
    return {
        "workers": profile.workers,
        "mem_gb": profile.mem_gb,
        "mem_floor_gb": profile.mem_floor_gb,
        "milp_fraction": profile.milp_fraction,
        "milp_timeout_cap": profile.milp_timeout_cap,
        "sigmoid_k": profile.sigmoid_k,
        "cell_budget": profile.cell_budget,
        "compressed_relu": profile.compressed_relu,
        "relu_valid_cuts": profile.relu_valid_cuts,
        "cutoff_row": profile.cutoff_row,
        "sparse_first": profile.sparse_first,
        "sparse_fallback": profile.sparse_fallback,
        "parallel_cutoff_portfolio": profile.parallel_cutoff_portfolio,
        "distshift_elim_portfolio": profile.distshift_elim_portfolio,
        "acasxu_scip_witness_fallback": profile.acasxu_scip_witness_fallback,
        "query_workers": profile.query_workers,
        "milp_threads": profile.milp_threads,
        "milp_env": dict(profile.milp_env or {}),
    }


def write_benchmark_json_summary(
    bench: str,
    out_dir: str | Path,
    summary: Mapping[str, object],
    run_rows: Iterable[Mapping[str, object]],
) -> Path:
    """Write a per-benchmark JSON summary for HybridZ runner output."""

    out = Path(out_dir)
    rows = list(run_rows)
    payload = {
        "bench": bench,
        "out_dir": str(out),
        "summary": dict(summary),
        "profile": profile_json(bench),
        "instances": [
            {
                "iid": int(row.get("index", -1)),
                "verdict": str(row.get("verdict", "ERROR")),
                "branch": str(row.get("branch", "")),
                "onnx": str(row.get("onnx_model", "")),
                "vnnlib": str(row.get("vnnlib_spec", "")),
                "time_sec": run_verify_time_s(row),
                "portfolio_done": row.get("portfolio_done", {}),
            }
            for row in rows
        ],
    }
    path = out / f"{bench}_run_summary.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_combined(
    bench: str,
    out_dir: str | Path,
    results: Iterable[Mapping[str, object]],
) -> tuple[Path, Path]:
    """Write benchmark detail/summary CSVs plus ICSE, JSON, and manifest exports."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    detail_path = out / f"{bench}_hybridz_detail.csv"
    summary_path = out / f"{bench}_hybridz_summary.csv"
    run_rows = list(results)

    detail_fields = [
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
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        for run in run_rows:
            rows = run.get("detail_rows", []) or []
            for row in rows:
                if isinstance(row, Mapping):
                    writer.writerow({name: row.get(name, "") for name in detail_fields})
            if rows:
                continue
            payload = run.get("module_payload")
            metadata = {
                "index": run.get("index"),
                "onnx_model": run.get("onnx_model", ""),
                "vnnlib_spec": run.get("vnnlib_spec", ""),
                "timeout_s": run.get("timeout_s", ""),
                "returncode": run.get("returncode", ""),
                "portfolio_done": run.get("portfolio_done", {}),
                "portfolio_branches": run.get("portfolio_branches", []),
            }
            if isinstance(payload, Mapping):
                metadata["module_payload"] = payload
            try:
                idx = int(run.get("index", -1))
                tag = f"iid{idx:05d}"
            except Exception:
                tag = str(run.get("index", ""))
            writer.writerow(
                {
                    "bench": bench,
                    "tag": tag,
                    "lane": str(run.get("branch", "")),
                    "status": str(run.get("verdict", "ERROR")),
                    "verdict": str(run.get("verdict", "ERROR")),
                    "wall_s": f"{run_verify_time_s(run):.2f}",
                    "reason": str(run.get("stderr_tail", "") or run.get("stdout_tail", ""))[:300],
                    "hz_verdict": str(run.get("verdict", "ERROR")),
                    "engine": str(payload.get("mode", "hybridz") if isinstance(payload, Mapping) else "hybridz"),
                    "p0": int(run_has_p0(run)),
                    "metadata_json": json.dumps(metadata, sort_keys=True),
                }
            )

    counts = {"CERT": 0, "ADV": 0, "TIMEOUT": 0, "UNKNOWN": 0, "ERROR": 0}
    for run in run_rows:
        verdict = str(run.get("verdict", "ERROR"))
        counts[verdict] = counts.get(verdict, 0) + 1
    p0 = sum(1 for run in run_rows if run_has_p0(run))
    summary = {
        "Bench": bench,
        "N": len(run_rows),
        "CERT": counts["CERT"],
        "ADV": counts["ADV"],
        "V+A": counts["CERT"] + counts["ADV"],
        "TIMEOUT": counts["TIMEOUT"],
        "UNKNOWN": counts["UNKNOWN"],
        "ERROR": counts["ERROR"],
        "P0": p0,
        "unsolved": counts["TIMEOUT"] + counts["UNKNOWN"] + counts["ERROR"],
    }
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    write_icse_benchmark_outputs(bench, out, run_rows)
    write_benchmark_json_summary(bench, out, summary, run_rows)
    write_sha256_manifest(out)
    return detail_path, summary_path


def write_suite_combined(
    out_dir: str | Path,
    bench_summaries: Iterable[Mapping[str, object]],
    detail_paths: Iterable[Path],
) -> tuple[Path, Path]:
    """Write suite-level detail and summary CSVs."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summaries = list(bench_summaries)
    detail_path = out / "hybridz_suite_detail.csv"
    summary_path = out / "hybridz_suite_summary.csv"

    detail_fields = [
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
    with detail_path.open("w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=detail_fields)
        writer.writeheader()
        for path in detail_paths:
            for row in read_csv_rows(Path(path)):
                writer.writerow({name: row.get(name, "") for name in detail_fields})

    fields = [
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
    ]
    total = {key: 0 for key in fields if key != "Bench"}
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            out_row = {key: row.get(key, 0 if key != "Bench" else "") for key in fields}
            writer.writerow(out_row)
            for key in total:
                total[key] += int(out_row.get(key, 0) or 0)
        if summaries:
            writer.writerow({"Bench": "TOTAL", **total})
    return detail_path, summary_path


def write_suite_icse_outputs(
    out_dir: str | Path,
    benches: Iterable[str],
) -> tuple[Path, Path]:
    """Write suite-level ICSE/VNN-COMP aggregate CSV exports."""

    out = Path(out_dir)
    index_path = out / "_INDEX.csv"
    detail_path = out / "_DETAIL.csv"
    index_fields = [
        "benchmark",
        "N",
        "unsat",
        "sat",
        "timeout",
        "unknown",
        "unsupported",
        "error",
        "total_time_sec",
    ]
    detail_fields = [
        "benchmark",
        "iid",
        "onnx",
        "vnnlib",
        "csv_timeout",
        "result",
        "time_sec",
        "raw_verdict",
        "branch",
        "portfolio_branches",
        "portfolio_done",
        "p0",
        "err",
    ]
    index_rows: list[dict[str, str]] = []
    detail_rows: list[dict[str, str]] = []
    for bench in benches:
        bench_dir = out / bench
        bench_csv = bench_dir / f"{bench}.csv"
        if bench_csv.exists():
            (out / f"{bench}.csv").write_text(
                bench_csv.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        index_rows.extend(read_csv_rows(bench_dir / f"{bench}_icse_index.csv"))
        detail_rows.extend(read_csv_rows(bench_dir / f"{bench}_icse_detail.csv"))

    with index_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=index_fields)
        writer.writeheader()
        for row in index_rows:
            writer.writerow({name: row.get(name, "") for name in index_fields})

    with detail_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        for row in detail_rows:
            writer.writerow({name: row.get(name, "") for name in detail_fields})

    readme = out / "README_REPRODUCIBILITY.md"
    readme.write_text(
        "# ACT HybridZ Benchmark Runner Export\n\n"
        "Generated by `python -m act.pipeline --verify hybridz-benchmark`. "
        "Per-benchmark CSVs use the ICSE/VNN-COMP style columns "
        "`onnx,vnnlib,result,time_sec`; strict HybridZ result tokens map "
        "`CERT -> unsat`, `ADV -> sat`, `TIMEOUT -> timeout`, "
        "`UNKNOWN -> unknown`, and errors to `error`.\n",
        encoding="utf-8",
    )
    return index_path, detail_path


def write_suite_json_summary(
    out_dir: str | Path,
    benches: tuple[str, ...],
    suite_summary_path: Path,
    *,
    max_instances: Optional[int] = None,
    workers: Optional[int] = None,
    timeout_cap_s: float = 900.0,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
) -> Path:
    """Write the suite-level HybridZ JSON summary."""

    out = Path(out_dir)
    rows = read_csv_rows(suite_summary_path)
    bench_rows = [row for row in rows if row.get("Bench") != "TOTAL"]
    total_rows = [row for row in rows if row.get("Bench") == "TOTAL"]
    payload = {
        "suite": "frozen" if benches == FROZEN_BENCHMARK_SUITE else "custom",
        "out_dir": str(out),
        "benchmarks": bench_rows,
        "total": total_rows[0] if total_rows else {},
        "profiles": {bench: profile_json(bench) for bench in benches},
        "config": {
            "benches": list(benches),
            "max_instances": max_instances,
            "workers": workers,
            "timeout_cap_s": timeout_cap_s,
            "device": device,
            "dtype": dtype,
        },
    }
    path = out / "hybridz_suite_summary.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


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


def build_frozen_repro_rows(
    summary_rows: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Build current-vs-frozen comparison rows for the frozen benchmark suite."""

    current_by_bench = {
        str(row.get("Bench", "")): row
        for row in summary_rows
        if row.get("Bench") and row.get("Bench") != "TOTAL"
    }
    rows: list[dict[str, object]] = []
    for bench in FROZEN_BENCHMARK_SUITE:
        expected = frozen_hybridz_expected_summary(bench)
        current = current_by_bench.get(bench)
        out: dict[str, object] = {"Bench": bench}
        mismatch = current is None
        for field in FROZEN_SUMMARY_FIELDS:
            current_value = 0 if current is None else int_field(current, field)
            expected_value = int(expected[field])
            delta = current_value - expected_value
            out[f"current_{field}"] = current_value
            out[f"expected_{field}"] = expected_value
            out[f"delta_{field}"] = delta
            if field in FROZEN_REPRO_MATCH_FIELDS:
                mismatch = mismatch or delta != 0
        out["status"] = "missing" if current is None else ("mismatch" if mismatch else "match")
        rows.append(out)

    for bench in sorted(set(current_by_bench) - set(FROZEN_BENCHMARK_SUITE)):
        current = current_by_bench[bench]
        out = {"Bench": bench, "status": "unexpected"}
        for field in FROZEN_SUMMARY_FIELDS:
            out[f"current_{field}"] = int_field(current, field)
            out[f"expected_{field}"] = ""
            out[f"delta_{field}"] = ""
        rows.append(out)
    return rows


def write_frozen_repro_check(
    out_dir: Path,
    suite_summary_path: Path,
    *,
    benches: tuple[str, ...],
    max_instances: Optional[int] = None,
) -> tuple[Path, Path, bool] | tuple[()]:
    """Write frozen reproduction comparison CSV/JSON for full frozen-suite runs."""

    if max_instances is not None or benches != FROZEN_BENCHMARK_SUITE:
        return ()

    out_dir = Path(out_dir)
    rows = build_frozen_repro_rows(read_csv_rows(suite_summary_path))
    fields = ["Bench", "status"]
    for field in FROZEN_SUMMARY_FIELDS:
        fields.extend([f"current_{field}", f"expected_{field}", f"delta_{field}"])

    csv_path = out_dir / "FROZEN_REPRO_COMPARISON.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({name: row.get(name, "") for name in fields} for row in rows)

    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", ""))
        status_counts[status] = status_counts.get(status, 0) + 1
    payload = {
        "ok": status_counts == {"match": len(FROZEN_BENCHMARK_SUITE)},
        "status_counts": status_counts,
        "expected_source": "FINAL_HYBRIDZ_RESULTS_20260627_FINAL.csv",
        "match_fields": list(FROZEN_REPRO_MATCH_FIELDS),
        "audit_only_fields": [
            field for field in FROZEN_SUMMARY_FIELDS if field not in FROZEN_REPRO_MATCH_FIELDS
        ],
        "rows": rows,
    }
    json_path = out_dir / "FROZEN_REPRO_COMPARISON.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return csv_path, json_path, bool(payload["ok"])


def enforce_frozen_match(repro_check: tuple[Path, Path, bool] | tuple[()]) -> None:
    """Raise if the frozen reproduction comparison is missing or mismatched."""

    if not repro_check:
        raise RuntimeError(
            "--hybridz-require-frozen-match requires a full frozen suite "
            "without --max-instances"
        )
    _, repro_json, repro_ok = repro_check
    if not repro_ok:
        raise RuntimeError(f"frozen HybridZ reproduction mismatch; see {repro_json}")


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

    frozen_rows = [
        {
            "Bench": "safenlp_2024",
            "N": "1080",
            "CERT": "432",
            "ADV": "647",
            "V+A": "1079",
            "TIMEOUT": "0",
            "UNKNOWN": "1",
            "ERROR": "0",
            "P0": "0",
            "unsolved": "1",
        },
        {
            "Bench": "linearizenn_2024",
            "N": "60",
            "CERT": "39",
            "ADV": "0",
            "V+A": "39",
            "TIMEOUT": "21",
            "UNKNOWN": "0",
            "ERROR": "0",
            "P0": "0",
            "unsolved": "21",
        },
        {
            "Bench": "dist_shift_2023",
            "N": "72",
            "CERT": "70",
            "ADV": "0",
            "V+A": "70",
            "TIMEOUT": "2",
            "UNKNOWN": "0",
            "ERROR": "0",
            "P0": "0",
            "unsolved": "2",
        },
    ]
    repro_rows = build_frozen_repro_rows(frozen_rows)
    by_bench = {str(row["Bench"]): row for row in repro_rows}
    assert by_bench["safenlp_2024"]["status"] == "match"
    assert by_bench["linearizenn_2024"]["status"] == "mismatch"
    assert by_bench["linearizenn_2024"]["delta_ADV"] == -1
    assert by_bench["dist_shift_2023"]["status"] == "match"
    assert by_bench["dist_shift_2023"]["delta_TIMEOUT"] == 2
    assert by_bench["dist_shift_2023"]["delta_UNKNOWN"] == -2
    assert by_bench["cgan_2023"]["status"] == "missing"

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        summary = tmp_path / "frozen_check_summary.csv"
        with summary.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(frozen_rows[0].keys()))
            writer.writeheader()
            writer.writerows(frozen_rows)
        repro_paths = write_frozen_repro_check(
            tmp_path,
            summary,
            benches=FROZEN_BENCHMARK_SUITE,
        )
        assert len(repro_paths) == 3
        payload = json.loads((tmp_path / "FROZEN_REPRO_COMPARISON.json").read_text(encoding="utf-8"))
        assert payload["ok"] is False
        assert payload["match_fields"] == list(FROZEN_REPRO_MATCH_FIELDS)
        assert "TIMEOUT" in payload["audit_only_fields"]
        try:
            enforce_frozen_match(repro_paths)
            raise AssertionError("expected frozen mismatch gate to fail")
        except RuntimeError as exc:
            assert "reproduction mismatch" in str(exc)

        full_match_summary = tmp_path / "frozen_full_match_summary.csv"
        full_match_rows = [
            {"Bench": bench, **frozen_hybridz_expected_summary(bench)}
            for bench in FROZEN_BENCHMARK_SUITE
        ]
        with full_match_summary.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Bench", *FROZEN_SUMMARY_FIELDS])
            writer.writeheader()
            writer.writerows(full_match_rows)
        full_match_paths = write_frozen_repro_check(
            tmp_path,
            full_match_summary,
            benches=FROZEN_BENCHMARK_SUITE,
        )
        assert len(full_match_paths) == 3
        enforce_frozen_match(full_match_paths)
        try:
            enforce_frozen_match(())
            raise AssertionError("expected missing frozen check to fail")
        except RuntimeError as exc:
            assert "requires a full frozen suite" in str(exc)


if __name__ == "__main__":  # pragma: no cover
    _test_hybridz_results()
    print("PASS _test_hybridz_results")
