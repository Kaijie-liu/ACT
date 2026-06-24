# ===- act/back_end/hybridz_config.py - HybridZ product config ----------===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   Benchmark-wide HybridZ profiles shared by the production backend and the
#   mainline benchmark runner. These settings are scheduling/formulation choices
#   for the strict pure-HZ engine; they must not encode per-instance rescue.
#
# ===---------------------------------------------------------------------===#

"""HybridZ benchmark profiles.

The values here are intentionally benchmark-wide. They keep the frozen HybridZ
configuration on ACT's backend surface, so frontends can reproduce the frozen
artifact without depending on local script constants.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, FrozenSet, Mapping, Tuple


@dataclass(frozen=True)
class HybridZBenchProfile:
    """Static per-benchmark HybridZ scheduling profile.

    This structure records knobs that do not alter the mathematical reachable
    set: solver backend order, wall-time slicing, concurrency, memory backstop,
    and exact-valid formulation choices. The verifier still owns the soundness
    checks for CERT/ADV.
    """

    wall_timeout_s: int
    workers: int | None = None
    mem_gb: float | None = None
    mem_floor_gb: float | None = None
    milp_fraction: float | None = None
    milp_timeout_cap: float | None = None
    sigmoid_k: int | None = None
    cell_budget: int | None = None
    compressed_relu: bool = False
    relu_valid_cuts: bool = False
    cutoff_row: bool = False
    sparse_first: bool = False
    sparse_fallback: bool = False
    parallel_cutoff_portfolio: bool = False
    distshift_elim_portfolio: bool = False
    acasxu_scip_witness_fallback: bool = False
    query_workers: int | None = None
    milp_threads: int | None = None
    milp_env: Mapping[str, str] | None = None


# Supported benchmarks present in the current VNN-COMP 2025 set, with
# per-benchmark fallback wall timeout in seconds. Official per-row timeouts from
# instances.csv remain authoritative when available.
BENCHMARKS: Dict[str, int] = {
    "malbeware": 180,
    "cersyve": 180,
    "safenlp_2024": 120,
    "sat_relu": 120,
    "tllverifybench_2023": 320,
    "acasxu_2023": 300,
    "cora_2024": 250,
    "lsnc_relu": 120,
    "linearizenn_2024": 180,
    "soundnessbench": 180,
    "relusplitter": 320,
    "dist_shift_2023": 300,
    "metaroom_2023": 250,
    "cgan_2023": 300,
    "collins_aerospace_benchmark": 180,
    "cifar100_2024": 250,
    "tinyimagenet_2024": 250,
    "yolo_2023": 300,
}

FROZEN_BENCHMARK_SUITE: Tuple[str, ...] = (
    "safenlp_2024",
    "metaroom_2023",
    "sat_relu",
    "malbeware",
    "cersyve",
    "acasxu_2023",
    "linearizenn_2024",
    "dist_shift_2023",
    "tllverifybench_2023",
    "cora_2024",
    "relusplitter",
    "cgan_2023",
)

CROSS_TOOL_NAMES = (
    "OURS",
    "abCROWN",
    "NeuralSAT",
    "nnenum",
    "PyRAT",
    "PyRAT-hz",
    "NNV",
)

# Tuple order is (unsat, sat, timeout, unknown, error). These frozen comparison
# counts are reporting/ranking metadata only. Current HybridZ run results always
# supply the OURS counts.
FROZEN_COMPETITOR_COUNTS: Dict[str, Dict[str, Tuple[int, int, int, int, int]]] = {
    "acasxu_2023": {
        "abCROWN": (139, 0, 47, 0, 0),
        "NeuralSAT": (137, 0, 46, 0, 3),
        "nnenum": (139, 47, 0, 0, 0),
        "PyRAT": (138, 40, 8, 0, 0),
        "PyRAT-hz": (49, 6, 131, 0, 0),
        "NNV": (75, 0, 104, 6, 1),
    },
    "cersyve": {
        "abCROWN": (6, 0, 5, 0, 1),
        "NeuralSAT": (0, 0, 8, 0, 4),
        "nnenum": (0, 0, 0, 0, 12),
        "PyRAT": (3, 4, 5, 0, 0),
        "PyRAT-hz": (1, 0, 11, 0, 0),
        "NNV": (0, 0, 0, 0, 0),
    },
    "cgan_2023": {
        "abCROWN": (9, 0, 9, 0, 3),
        "NeuralSAT": (10, 0, 0, 0, 11),
        "nnenum": (0, 0, 0, 0, 21),
        "PyRAT": (9, 10, 0, 0, 2),
        "PyRAT-hz": (7, 0, 11, 0, 3),
        "NNV": (0, 0, 0, 0, 19),
    },
    "cora_2024": {
        "abCROWN": (22, 0, 158, 0, 0),
        "NeuralSAT": (22, 0, 118, 40, 0),
        "nnenum": (19, 1, 160, 0, 0),
        "PyRAT": (20, 0, 160, 0, 0),
        "PyRAT-hz": (19, 0, 156, 4, 1),
        "NNV": (18, 0, 161, 1, 0),
    },
    "dist_shift_2023": {
        "abCROWN": (65, 0, 6, 0, 1),
        "NeuralSAT": (65, 0, 7, 0, 0),
        "nnenum": (0, 0, 0, 0, 72),
        "PyRAT": (64, 3, 5, 0, 0),
        "PyRAT-hz": (55, 0, 0, 17, 0),
        "NNV": (0, 0, 0, 0, 72),
    },
    "linearizenn_2024": {
        "abCROWN": (59, 0, 1, 0, 0),
        "NeuralSAT": (59, 0, 1, 0, 0),
        "nnenum": (59, 1, 0, 0, 0),
        "PyRAT": (59, 1, 0, 0, 0),
        "PyRAT-hz": (13, 0, 47, 0, 0),
        "NNV": (0, 0, 0, 0, 0),
    },
    "malbeware": {
        "abCROWN": (131, 18, 0, 0, 1),
        "NeuralSAT": (127, 1, 15, 0, 7),
        "nnenum": (88, 3, 10, 0, 49),
        "PyRAT": (125, 0, 24, 1, 0),
        "PyRAT-hz": (64, 0, 37, 1, 48),
        "NNV": (49, 0, 90, 1, 10),
    },
    "metaroom_2023": {
        "abCROWN": (94, 0, 6, 0, 0),
        "NeuralSAT": (94, 0, 6, 0, 0),
        "nnenum": (14, 0, 0, 0, 86),
        "PyRAT": (60, 0, 1, 0, 39),
        "PyRAT-hz": (60, 0, 1, 0, 39),
        "NNV": (93, 0, 5, 2, 0),
    },
    "relusplitter": {
        "abCROWN": (113, 0, 47, 0, 60),
        "NeuralSAT": (62, 0, 138, 0, 20),
        "nnenum": (15, 0, 100, 0, 105),
        "PyRAT": (41, 0, 115, 0, 64),
        "PyRAT-hz": (20, 0, 125, 49, 26),
        "NNV": (0, 0, 3, 137, 80),
    },
    "safenlp_2024": {
        "abCROWN": (433, 647, 0, 0, 0),
        "NeuralSAT": (425, 433, 221, 0, 1),
        "nnenum": (285, 624, 171, 0, 0),
        "PyRAT": (417, 55, 400, 208, 0),
        "PyRAT-hz": (189, 0, 0, 891, 0),
        "NNV": (162, 0, 910, 0, 8),
    },
    "sat_relu": {
        "abCROWN": (50, 49, 0, 0, 1),
        "NeuralSAT": (50, 50, 0, 0, 0),
        "nnenum": (9, 36, 36, 0, 19),
        "PyRAT": (12, 0, 88, 0, 0),
        "PyRAT-hz": (20, 0, 79, 0, 1),
        "NNV": (4, 0, 94, 2, 0),
    },
    "tllverifybench_2023": {
        "abCROWN": (15, 0, 17, 0, 0),
        "NeuralSAT": (15, 0, 4, 0, 13),
        "nnenum": (2, 17, 9, 0, 4),
        "PyRAT": (15, 15, 2, 0, 0),
        "PyRAT-hz": (0, 0, 20, 0, 12),
        "NNV": (0, 0, 19, 13, 0),
    },
}

FROZEN_SUMMARY_FIELDS = (
    "N",
    "CERT",
    "ADV",
    "V+A",
    "TIMEOUT",
    "UNKNOWN",
    "ERROR",
    "P0",
    "unsolved",
)

# Tuple order follows FROZEN_SUMMARY_FIELDS. This is an acceptance/reporting
# oracle for frozen-suite reproduction, not an input to the verifier.
FROZEN_HYBRIDZ_EXPECTED_COUNTS: Dict[str, Tuple[int, int, int, int, int, int, int, int, int]] = {
    "safenlp_2024": (1080, 432, 647, 1079, 0, 1, 0, 0, 1),
    "metaroom_2023": (100, 94, 1, 95, 5, 0, 0, 0, 5),
    "sat_relu": (100, 50, 50, 100, 0, 0, 0, 0, 0),
    "malbeware": (150, 131, 19, 150, 0, 0, 0, 0, 0),
    "cersyve": (12, 5, 6, 11, 0, 1, 0, 0, 1),
    "acasxu_2023": (186, 86, 34, 120, 6, 60, 0, 0, 66),
    "linearizenn_2024": (60, 39, 1, 40, 20, 0, 0, 0, 20),
    "dist_shift_2023": (72, 70, 0, 70, 2, 0, 0, 0, 2),
    "tllverifybench_2023": (32, 5, 12, 17, 14, 1, 0, 0, 15),
    "cora_2024": (180, 19, 6, 25, 154, 1, 0, 0, 155),
    "relusplitter": (220, 41, 2, 43, 108, 69, 0, 0, 177),
    "cgan_2023": (21, 5, 8, 13, 0, 8, 0, 0, 8),
}

FROZEN_HYBRIDZ_TOTAL_COUNTS: Tuple[int, int, int, int, int, int, int, int, int] = (
    2213,
    977,
    786,
    1763,
    309,
    141,
    0,
    0,
    450,
)

# Empty: LP-tight [alpha,beta] is now ON for every benchmark.
NO_TIGHT: FrozenSet[str] = frozenset()

# Wide benches whose bottleneck is the 2n tight-bounds LPs. This is only
# scheduling; the bounds are bit-identical to serial execution.
PARALLEL_TIGHT: FrozenSet[str] = frozenset({"cora_2024", "relusplitter"})
PARALLEL_TIGHT_CONFIG: Dict[str, Tuple[int, int]] = {
    "cora_2024": (1, 8),
    "relusplitter": (1, 6),
}
PT_WORKERS, PT_THREADS = 3, 6

MILP_FRACTION: Dict[str, float] = {
    "safenlp_2024": 0.95,
    "sat_relu": 0.96,
    "dist_shift_2023": 0.40,
    "linearizenn_2024": 0.75,
}
MILP_TIMEOUT_CAP: Dict[str, float] = {"linearizenn_2024": 650}
SIGMOID_K: Dict[str, int] = {"dist_shift_2023": 2}
CELL_BUDGET: Dict[str, int] = {"dist_shift_2023": 800_000_000}
RELU_VALID_CUTS: FrozenSet[str] = frozenset()
COMPRESSED_RELU_DEFAULT: FrozenSet[str] = frozenset(
    {"linearizenn_2024", "cgan_2023"}
)

MILP_ENV_DEFAULTS: Dict[str, Dict[str, str]] = {
    "safenlp_2024": {
        "HZ_MILP_ELIM_SINGLETONS": "1",
    },
    "acasxu_2023": {
        "HZ_MILP_BACKEND": "portfolio",
        "HZ_MILP_START": "lp_binary",
        "HZ_MILP_HEURISTIC": "1.0",
    },
    "cgan_2023": {
        "HZ_MILP_BACKEND": "portfolio",
        "HZ_MILP_START": "lp_binary",
        "HZ_MILP_HEURISTIC": "1.0",
    },
}

MILP_CUTOFF_ROW: FrozenSet[str] = frozenset({"sat_relu"})
PARALLEL_CUTOFF_PORTFOLIO: FrozenSet[str] = frozenset({"safenlp_2024"})
SPARSE_PURE_FALLBACK: FrozenSet[str] = frozenset(
    {"metaroom_2023", "malbeware", "cersyve", "tllverifybench_2023", "relusplitter"}
)
SPARSE_FIRST: FrozenSet[str] = frozenset({"metaroom_2023", "tllverifybench_2023"})
ACASXU_SCIP_WITNESS_FALLBACK: FrozenSet[str] = frozenset({"acasxu_2023"})
ACASXU_SCIP_WITNESS_MILP_TIMEOUT = 1.0

DISTSHIFT_ELIM_PORTFOLIO: FrozenSet[str] = frozenset({"dist_shift_2023"})

BENCH_WORKERS: Dict[str, int] = {
    "safenlp_2024": 1,
    "dist_shift_2023": 3,
    "linearizenn_2024": 4,
    "tllverifybench_2023": 6,
    "cgan_2023": 2,
}
BENCH_MEM_GB: Dict[str, float] = {
    "cgan_2023": 32.0,
}
MEM_FLOOR_GB: float | None = 20.0
QUERY_WORKERS: Dict[str, int] = {"linearizenn_2024": 2, "relusplitter": 9}
MILP_THREADS: Dict[str, int] = {"linearizenn_2024": 4}


def get_bench_profile(bench: str) -> HybridZBenchProfile:
    """Return the static HybridZ profile for ``bench``.

    Unknown benches get a conservative default wall and no special portfolio
    branches. This keeps the profile queryable from backend/front-end code
    without adding benchmark-specific rescue logic.
    """

    return HybridZBenchProfile(
        wall_timeout_s=BENCHMARKS.get(bench, 250),
        workers=BENCH_WORKERS.get(bench),
        mem_gb=BENCH_MEM_GB.get(bench),
        mem_floor_gb=MEM_FLOOR_GB,
        milp_fraction=MILP_FRACTION.get(bench),
        milp_timeout_cap=MILP_TIMEOUT_CAP.get(bench),
        sigmoid_k=SIGMOID_K.get(bench),
        cell_budget=CELL_BUDGET.get(bench),
        compressed_relu=bench in COMPRESSED_RELU_DEFAULT,
        relu_valid_cuts=bench in RELU_VALID_CUTS,
        cutoff_row=bench in MILP_CUTOFF_ROW,
        sparse_first=bench in SPARSE_FIRST,
        sparse_fallback=bench in SPARSE_PURE_FALLBACK,
        parallel_cutoff_portfolio=bench in PARALLEL_CUTOFF_PORTFOLIO,
        distshift_elim_portfolio=bench in DISTSHIFT_ELIM_PORTFOLIO,
        acasxu_scip_witness_fallback=bench in ACASXU_SCIP_WITNESS_FALLBACK,
        query_workers=QUERY_WORKERS.get(bench),
        milp_threads=MILP_THREADS.get(bench),
        milp_env=MILP_ENV_DEFAULTS.get(bench),
    )


def frozen_hybridz_expected_summary(bench: str) -> Dict[str, int]:
    """Return the frozen expected HybridZ summary row for ``bench``."""

    values = FROZEN_HYBRIDZ_EXPECTED_COUNTS[bench]
    return {field: int(value) for field, value in zip(FROZEN_SUMMARY_FIELDS, values)}


def frozen_hybridz_total_summary() -> Dict[str, int]:
    """Return the frozen strict-HybridZ suite total summary."""

    return {
        field: int(value)
        for field, value in zip(FROZEN_SUMMARY_FIELDS, FROZEN_HYBRIDZ_TOTAL_COUNTS)
    }


def validate_frozen_hybridz_results_csv(path: str | Path) -> None:
    """Assert that a frozen HybridZ result CSV matches the config oracle."""

    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    data_rows = [row for row in rows if row.get("Bench") and row.get("Bench") != "TOTAL"]
    order = tuple(str(row.get("Bench", "")) for row in data_rows)
    assert order == FROZEN_BENCHMARK_SUITE, (csv_path, order)
    expected_benches = set(FROZEN_BENCHMARK_SUITE)
    assert set(order) == expected_benches, csv_path

    for row in data_rows:
        bench = str(row["Bench"])
        expected = frozen_hybridz_expected_summary(bench)
        for field, expected_value in expected.items():
            actual_value = int(row[field])
            assert actual_value == expected_value, (
                csv_path,
                bench,
                field,
                actual_value,
                expected_value,
            )

    total_rows = [row for row in rows if row.get("Bench") == "TOTAL"]
    if total_rows:
        total = frozen_hybridz_total_summary()
        for field, expected_value in total.items():
            actual_value = int(total_rows[0][field])
            assert actual_value == expected_value, (csv_path, "TOTAL", field, actual_value, expected_value)


def _test_hybridz_config() -> None:  # pragma: no cover
    frozen_set = set(FROZEN_BENCHMARK_SUITE)
    assert len(frozen_set) == len(FROZEN_BENCHMARK_SUITE)
    assert frozen_set <= set(BENCHMARKS)
    assert set(FROZEN_HYBRIDZ_EXPECTED_COUNTS) == frozen_set
    assert set(FROZEN_COMPETITOR_COUNTS) == frozen_set
    assert len(FROZEN_HYBRIDZ_TOTAL_COUNTS) == len(FROZEN_SUMMARY_FIELDS)
    assert CROSS_TOOL_NAMES[0] == "OURS"
    assert len(set(CROSS_TOOL_NAMES)) == len(CROSS_TOOL_NAMES)
    competitor_tools = set(CROSS_TOOL_NAMES) - {"OURS"}
    totals = [0 for _ in FROZEN_SUMMARY_FIELDS]
    for bench in FROZEN_BENCHMARK_SUITE:
        expected = FROZEN_HYBRIDZ_EXPECTED_COUNTS[bench]
        assert len(expected) == len(FROZEN_SUMMARY_FIELDS), bench
        totals = [total + int(value) for total, value in zip(totals, expected)]
        row = dict(zip(FROZEN_SUMMARY_FIELDS, expected))
        assert frozen_hybridz_expected_summary(bench) == row, bench
        assert row["N"] > 0, bench
        assert row["V+A"] == row["CERT"] + row["ADV"], bench
        assert row["unsolved"] == row["N"] - row["V+A"], bench
        assert row["P0"] == 0, bench
        assert set(FROZEN_COMPETITOR_COUNTS[bench]) == competitor_tools, bench
        for tool, counts in FROZEN_COMPETITOR_COUNTS[bench].items():
            assert len(counts) == 5, (bench, tool)
            assert all(int(value) >= 0 for value in counts), (bench, tool)
        profile = get_bench_profile(bench)
        assert profile.wall_timeout_s > 0, bench
        env_text = " ".join(f"{k}={v}" for k, v in (profile.milp_env or {}).items()).lower()
        assert "gurobi" not in env_text, bench
    assert tuple(totals) == FROZEN_HYBRIDZ_TOTAL_COUNTS
    total_row = frozen_hybridz_total_summary()
    assert total_row["N"] == 2213
    assert total_row["CERT"] == 977
    assert total_row["ADV"] == 786
    assert total_row["V+A"] == 1763
    assert total_row["P0"] == 0
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "frozen.csv"
        fields = ["Bench", *FROZEN_SUMMARY_FIELDS]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for bench in FROZEN_BENCHMARK_SUITE:
                writer.writerow({"Bench": bench, **frozen_hybridz_expected_summary(bench)})
            writer.writerow({"Bench": "TOTAL", **frozen_hybridz_total_summary()})
        validate_frozen_hybridz_results_csv(path)


__all__ = [
    "ACASXU_SCIP_WITNESS_FALLBACK",
    "ACASXU_SCIP_WITNESS_MILP_TIMEOUT",
    "BENCHMARKS",
    "BENCH_MEM_GB",
    "BENCH_WORKERS",
    "CELL_BUDGET",
    "COMPRESSED_RELU_DEFAULT",
    "CROSS_TOOL_NAMES",
    "DISTSHIFT_ELIM_PORTFOLIO",
    "FROZEN_BENCHMARK_SUITE",
    "FROZEN_COMPETITOR_COUNTS",
    "FROZEN_HYBRIDZ_EXPECTED_COUNTS",
    "FROZEN_HYBRIDZ_TOTAL_COUNTS",
    "FROZEN_SUMMARY_FIELDS",
    "HybridZBenchProfile",
    "MILP_CUTOFF_ROW",
    "MILP_ENV_DEFAULTS",
    "MILP_FRACTION",
    "MEM_FLOOR_GB",
    "MILP_THREADS",
    "MILP_TIMEOUT_CAP",
    "NO_TIGHT",
    "PARALLEL_CUTOFF_PORTFOLIO",
    "PARALLEL_TIGHT",
    "PARALLEL_TIGHT_CONFIG",
    "PT_THREADS",
    "PT_WORKERS",
    "QUERY_WORKERS",
    "RELU_VALID_CUTS",
    "SIGMOID_K",
    "SPARSE_FIRST",
    "SPARSE_PURE_FALLBACK",
    "frozen_hybridz_expected_summary",
    "frozen_hybridz_total_summary",
    "get_bench_profile",
    "validate_frozen_hybridz_results_csv",
]


if __name__ == "__main__":  # pragma: no cover
    import sys

    _test_hybridz_config()
    print("PASS _test_hybridz_config")
    for arg in sys.argv[1:]:
        validate_frozen_hybridz_results_csv(arg)
        print(f"PASS validate_frozen_hybridz_results_csv {arg}")
