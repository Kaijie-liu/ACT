# ===- act/pipeline/hybridz_benchmark_runner.py - HybridZ runner -------===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#

"""Mainline benchmark runner for strict HybridZ frontend verification.

The runner invokes the normal ACT frontend path per VNNLIB instance:

``python -m act.pipeline --verify vnnlib --instance-index I --solvers hybridz``

It deliberately does not call the legacy out-of-tree worker scripts.  Process-
level isolation, per-instance official walls, and CSV aggregation live in ACT
package code.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from act.back_end.hybridz_config import (
    ACASXU_SCIP_WITNESS_MILP_TIMEOUT,
    CROSS_TOOL_NAMES,
    FROZEN_BENCHMARK_SUITE,
    FROZEN_COMPETITOR_COUNTS,
    FROZEN_SUMMARY_FIELDS,
    frozen_hybridz_expected_summary,
    get_bench_profile,
)
from act.front_end.vnnlib_loader.data_model_loader import list_downloaded_pairs


@dataclass(frozen=True)
class HybridZBenchmarkInstance:
    index: int
    timeout_s: float
    onnx_model: str
    vnnlib_spec: str


@dataclass(frozen=True)
class HybridZBenchmarkConfig:
    bench: str
    out_dir: Path
    max_instances: Optional[int] = None
    workers: Optional[int] = None
    timeout_cap_s: float = 900.0
    device: Optional[str] = None
    dtype: Optional[str] = None
    python: str = sys.executable


@dataclass(frozen=True)
class HybridZBenchmarkSuiteConfig:
    benches: tuple[str, ...]
    out_dir: Path
    max_instances: Optional[int] = None
    workers: Optional[int] = None
    timeout_cap_s: float = 900.0
    device: Optional[str] = None
    dtype: Optional[str] = None
    python: str = sys.executable
    require_frozen_match: bool = False


@dataclass(frozen=True)
class HybridZRunBranch:
    name: str
    set_env: dict[str, str]
    unset_env: tuple[str, ...] = ()
    extra_args: tuple[str, ...] = ()
    module: str | None = None
    module_args: tuple[str, ...] = ()
    timeout_override_s: Optional[float] = None
    accept_verdicts: tuple[str, ...] = ("CERT", "ADV")


FULL_WORKER_MODULE = "act.pipeline.hybridz_full_worker"
SPARSE_WORKER_MODULE = "act.pipeline.hybridz_sparse_worker"
SEQUENTIAL_PORTFOLIO_BENCHES = frozenset({
    "metaroom_2023",
    "malbeware",
    "tllverifybench_2023",
    "relusplitter",
    "cgan_2023",
})
FROZEN_REPRO_MATCH_FIELDS = ("N", "CERT", "ADV", "V+A", "ERROR", "P0", "unsolved")
HIGHS_HEURISTIC_OPTIONS = (
    "mip_heuristic_effort=1.0",
    "mip_heuristic_run_shifting=true",
    "mip_heuristic_run_zi_round=true",
)
HIGHS_HEURISTIC_ENV = ",".join(HIGHS_HEURISTIC_OPTIONS)


def _highs_option_args(options: Iterable[str]) -> tuple[str, ...]:
    args: list[str] = []
    for option in options:
        args.extend(["--highs-option", str(option)])
    return tuple(args)


def _profile_milp_timeout(bench: str, official_timeout_s: float) -> float:
    profile = get_bench_profile(bench)
    fraction = profile.milp_fraction if profile.milp_fraction is not None else 0.4
    cap = profile.milp_timeout_cap if profile.milp_timeout_cap is not None else 250.0
    return float(min(int(float(official_timeout_s) * float(fraction)), float(cap)))


def _full_worker_args(cfg: HybridZBenchmarkConfig, inst: HybridZBenchmarkInstance) -> tuple[str, ...]:
    profile = get_bench_profile(cfg.bench)
    args: list[str] = [
        "--cap",
        "4096",
        "--mem-gb",
        str(profile.mem_gb if profile.mem_gb is not None else 20.0),
        "--milp-timeout",
        str(_profile_milp_timeout(cfg.bench, inst.timeout_s)),
        "--sigmoid-k",
        str(profile.sigmoid_k if profile.sigmoid_k is not None else 2),
    ]
    if profile.cell_budget is not None:
        args.extend(["--cell-budget", str(profile.cell_budget)])
    if profile.compressed_relu:
        args.append("--compressed-relu")
    if profile.relu_valid_cuts:
        args.append("--relu-valid-cuts")
    return tuple(args)


def _full_worker_branch(
    name: str = "normal",
    *,
    env: Optional[dict[str, str]] = None,
    module_args: tuple[str, ...] = (),
) -> HybridZRunBranch:
    return HybridZRunBranch(
        name,
        set_env=dict(env or {}),
        module=FULL_WORKER_MODULE,
        module_args=module_args,
    )


def _sparse_worker_branch(
    name: str,
    *args: str,
    env: Optional[dict[str, str]] = None,
    accept_verdicts: tuple[str, ...] = ("CERT", "ADV"),
) -> HybridZRunBranch:
    return HybridZRunBranch(
        name,
        set_env=dict(env or {}),
        module=SPARSE_WORKER_MODULE,
        module_args=tuple(str(arg) for arg in args),
        accept_verdicts=accept_verdicts,
    )


def _distshift_scurve_branch(k: int, *, cutrow: bool = False) -> HybridZRunBranch:
    suffix = "_cutrow" if cutrow else ""
    args = [
        "--milp-timeout",
        str({2: 70, 4: 90, 6: 120, 8: 140}.get(k, 90)),
        "--lp-queries",
        "1",
        "--compressed-relu",
        "--compressed-sigmoid",
        "--sigmoid-prune-degenerate",
        "--sigmoid-k",
        str(k),
        "--tanh-k",
        "1",
        "--scurve-domain-cuts",
        "--scurve-graph-cuts",
        "--mip-solver",
        "scip",
    ]
    if cutrow:
        args.append("--cutoff-row")
    return _sparse_worker_branch(
        f"scurve_graph_k{k}{suffix}_scip",
        *args,
    )


def resolve_hybridz_benchmark_categories(category: str) -> tuple[str, ...]:
    """Resolve a CLI category token into one or more HybridZ benchmarks."""

    key = str(category).strip()
    if key in {"frozen", "hybridz_frozen", "hybridz-suite", "suite"}:
        return FROZEN_BENCHMARK_SUITE
    return (key,)


def list_hybridz_benchmark_instances(bench: str) -> list[HybridZBenchmarkInstance]:
    instances = []
    for item in list_downloaded_pairs():
        if item["category"] != bench:
            continue
        instances.append(
            HybridZBenchmarkInstance(
                index=int(item.get("index", len(instances))),
                timeout_s=float(item.get("timeout") or get_bench_profile(bench).wall_timeout_s),
                onnx_model=str(item["onnx_model"]),
                vnnlib_spec=str(item["vnnlib_spec"]),
            )
        )
    instances.sort(key=lambda inst: inst.index)
    return instances


def _downloaded_benchmark_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in list_downloaded_pairs():
        category = str(item.get("category", ""))
        counts[category] = counts.get(category, 0) + 1
    return counts


def _raise_if_strict_suite_data_missing(benches: tuple[str, ...]) -> None:
    counts = _downloaded_benchmark_counts()
    missing = [bench for bench in benches if counts.get(bench, 0) <= 0]
    if missing:
        raise RuntimeError(
            "missing downloaded VNNLIB instances for strict HybridZ suite run: "
            + ", ".join(missing)
        )


def _profiled_timeout(bench: str, official_timeout_s: float, cap_s: float) -> float:
    profile = get_bench_profile(bench)
    timeout = min(float(official_timeout_s), float(cap_s))
    if profile.milp_fraction is not None:
        timeout *= float(profile.milp_fraction)
    if profile.milp_timeout_cap is not None:
        timeout = min(timeout, float(profile.milp_timeout_cap))
    return max(1.0, timeout)


def _instance_wall_limit(cfg: HybridZBenchmarkConfig, inst: HybridZBenchmarkInstance) -> float:
    # Some benchmark CSV rows carry a lower timeout than the wall used for the
    # frozen pure-HZ run, while others carry the official cap.  The benchmark
    # profile is a lower bound, the frontend cap is an upper bound.
    profile_wall = float(get_bench_profile(cfg.bench).wall_timeout_s or inst.timeout_s)
    return min(max(float(inst.timeout_s), profile_wall), float(cfg.timeout_cap_s)) + 30.0


def _instance_command(
    cfg: HybridZBenchmarkConfig,
    inst: HybridZBenchmarkInstance,
    instance_dir: Path,
    branch: Optional[HybridZRunBranch] = None,
) -> list[str]:
    if branch is not None and branch.timeout_override_s is not None:
        timeout_s = float(branch.timeout_override_s)
    else:
        timeout_s = _profiled_timeout(cfg.bench, inst.timeout_s, cfg.timeout_cap_s)
    if branch is not None and branch.module:
        module_device = "cpu" if branch.module in {
            FULL_WORKER_MODULE,
            SPARSE_WORKER_MODULE,
            "act.pipeline.hybridz_projected_relu_mip",
        } else str(cfg.device or "cpu")
        module_args = tuple(branch.module_args)
        if branch.module == FULL_WORKER_MODULE:
            module_args = _full_worker_args(cfg, inst) + module_args
        elif branch.module == SPARSE_WORKER_MODULE:
            module_args = ("--worker-timeout", str(_instance_wall_limit(cfg, inst)), *module_args)
        return [
            cfg.python,
            "-m",
            branch.module,
            "--bench",
            cfg.bench,
            "--iid",
            str(inst.index),
            "--device",
            module_device,
            *module_args,
        ]
    cmd = [
        cfg.python,
        "-m",
        "act.pipeline",
        "--verify",
        "vnnlib",
        "--category",
        cfg.bench,
        "--instance-index",
        str(inst.index),
        "--solvers",
        "hybridz",
        "--hybridz-timeout",
        str(timeout_s),
        "--hybridz-results-dir",
        str(instance_dir),
    ]
    if cfg.device:
        cmd.extend(["--device", cfg.device])
    if cfg.dtype:
        cmd.extend(["--dtype", cfg.dtype])
    if branch is not None:
        cmd.extend(branch.extra_args)
    return cmd


def _branch_plan(cfg: HybridZBenchmarkConfig) -> list[HybridZRunBranch]:
    profile = get_bench_profile(cfg.bench)
    normal_env = {"HZ_MILP_CUTOFF_ROW": "1"} if profile.cutoff_row else {}
    normal = _full_worker_branch("normal", env=normal_env)
    sparse = HybridZRunBranch(
        "sparse",
        set_env={},
        extra_args=("--hybridz-engine", "sparse_hz_objbound"),
    )
    if cfg.bench == "malbeware":
        sparse = _sparse_worker_branch(
            "sparse",
            "--milp-timeout",
            "180",
            "--lp-queries",
            "99",
            "--cutoff-row",
        )
    elif cfg.bench == "tllverifybench_2023":
        return [
            _sparse_worker_branch(
                "sparse_tll_cutrow_eqsubst",
                "--milp-timeout",
                "120",
                "--lp-queries",
                "99",
                "--cutoff-row",
                "--compressed-relu",
                "--elim-eq-subst",
                "--skip-lp-before-milp",
            ),
            _sparse_worker_branch(
                "sparse_tll_objtarget_comprelu",
                "--milp-timeout",
                "120",
                "--lp-queries",
                "99",
                "--compressed-relu",
            ),
            _sparse_worker_branch(
                "sparse_tll_cutrow_relu_cuts_eqsubst",
                "--milp-timeout",
                "120",
                "--lp-queries",
                "99",
                "--cutoff-row",
                "--compressed-relu",
                "--elim-eq-subst",
                "--skip-lp-before-milp",
                "--relu-cuts",
            ),
            normal,
        ]
    elif cfg.bench == "cersyve":
        return [
            normal,
            _sparse_worker_branch(
                "cersyve_highs_cuts_fbbt",
                "--milp-timeout",
                "90",
                "--lp-queries",
                "99",
                "--compressed-relu",
                "--relu-cuts",
                "--fbbt-passes",
                "5",
                "--relax-precheck-timeout",
                "3",
                "--mip-start",
                "base-binary",
                "--mip-solver",
                "highs",
            ),
            _sparse_worker_branch(
                "cersyve_scip_cuts_fbbt",
                "--milp-timeout",
                "90",
                "--lp-queries",
                "99",
                "--compressed-relu",
                "--relu-cuts",
                "--fbbt-passes",
                "5",
                "--relax-precheck-timeout",
                "3",
                "--mip-start",
                "base-binary",
                "--mip-solver",
                "scip",
            ),
        ]
    elif cfg.bench == "cgan_2023":
        return [
            normal,
            _sparse_worker_branch(
                "sparse_exact_milp_witness",
                "--milp-timeout",
                "180",
                "--lp-queries",
                "2",
                "--query-indices",
                "1,0",
                "--compressed-relu",
                "--compressed-sigmoid",
                "--sigmoid-prune-degenerate",
                "--sigmoid-k",
                "2",
                "--tanh-k",
                "2",
                "--scurve-domain-cuts",
                "--scurve-graph-cuts",
                "--connected-presolve",
                "--mip-start",
                "base",
                "--skip-lp-before-milp",
                "--no-elim-singletons",
            ),
        ]
    elif cfg.bench == "acasxu_2023":
        return [
            normal,
            _sparse_worker_branch(
                "acasxu_cuts_fbbt",
                "--milp-timeout",
                "90",
                "--lp-queries",
                "99",
                "--compressed-relu",
                "--relu-cuts",
                "--fbbt-passes",
                "5",
                "--relax-precheck-timeout",
                "3",
                "--mip-start",
                "base-binary",
                "--mip-solver",
                "highs",
            ),
            _sparse_worker_branch(
                "acasxu_scip_witness",
                "--milp-timeout",
                str(ACASXU_SCIP_WITNESS_MILP_TIMEOUT),
                "--lp-queries",
                "99",
                "--compressed-relu",
                "--skip-lp-before-milp",
                "--mip-solver",
                "scip",
                accept_verdicts=("ADV",),
            ),
        ]
    elif cfg.bench == "dist_shift_2023":
        branches = [
            normal,
            _full_worker_branch("elim_singletons", env={"HZ_MILP_ELIM_SINGLETONS": "1"}),
            _distshift_scurve_branch(2),
            _distshift_scurve_branch(4),
            _distshift_scurve_branch(6),
            _distshift_scurve_branch(4, cutrow=True),
            _distshift_scurve_branch(8),
        ]
        return branches
    elif cfg.bench == "linearizenn_2024":
        return [
            normal,
            HybridZRunBranch(
                "linear_portfolio_m360",
                set_env={"HZ_MILP_BACKEND": "portfolio"},
                module=FULL_WORKER_MODULE,
                module_args=("--milp-timeout", "360"),
            ),
        ]
    elif cfg.bench == "cora_2024":
        return [
            normal,
            _sparse_worker_branch(
                "sparse_non_mnist_set_m10heur",
                "--milp-timeout",
                "10",
                "--lp-queries",
                "3",
                "--compressed-relu",
                "--mip-start",
                "base-binary",
                "--mip-solver",
                "highs",
                *_highs_option_args(HIGHS_HEURISTIC_OPTIONS),
            ),
        ]
    if profile.sparse_first:
        branches = [sparse, normal]
    else:
        branches = [normal]
    if profile.parallel_cutoff_portfolio:
        branches.append(_full_worker_branch("cutrow", env={"HZ_MILP_CUTOFF_ROW": "1"}))
    if cfg.bench == "safenlp_2024":
        worker_env = {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "",
        }
        branches.append(
            _full_worker_branch(
                "normal_pscost1",
                env={"HZ_HIGHS_OPTIONS": "mip_pscost_minreliable=1"},
                module_args=("--milp-timeout", "25"),
            )
        )
        branches.append(
            _full_worker_branch(
                "normal_seed2",
                # Fixed safenlp-wide HiGHS portfolio branch for B&B edge cases.
                env={"HZ_HIGHS_OPTIONS": "random_seed=2"},
            )
        )
        sparse_base = (
            "--milp-timeout",
            "18",
            "--lp-queries",
            "1",
            "--compressed-relu",
        )
        branches.append(
            _sparse_worker_branch(
                "sparse_comprelu",
                *sparse_base,
                env=worker_env,
            )
        )
        branches.append(
            _sparse_worker_branch(
                "sparse_comprelu_heur",
                *sparse_base,
                *_highs_option_args(HIGHS_HEURISTIC_OPTIONS),
                env={
                    **worker_env,
                    "HZ_HIGHS_OPTIONS": HIGHS_HEURISTIC_ENV,
                },
            )
        )
        projected_base = (
            "--lp-queries",
            "1",
            "--milp-timeout",
            "18",
            *_highs_option_args(HIGHS_HEURISTIC_OPTIONS),
            "--check-witness",
            "--stop-on-unsafe",
        )
        branches.append(
            HybridZRunBranch(
                "projected_relu_mip",
                set_env=worker_env,
                module="act.pipeline.hybridz_projected_relu_mip",
                module_args=projected_base,
            )
        )
        for solver_name in ("scip-bigm", "scip-indicator"):
            branches.append(
                HybridZRunBranch(
                    f"projected_relu_{solver_name}",
                    set_env=worker_env,
                    module="act.pipeline.hybridz_projected_relu_mip",
                    module_args=(
                        "--lp-queries",
                        "1",
                        "--milp-timeout",
                        "16",
                        "--mip-solver",
                        solver_name,
                        "--check-witness",
                        "--stop-on-unsafe",
                    ),
                )
            )
    if profile.sparse_fallback and not profile.sparse_first:
        branches.append(sparse)
    if profile.distshift_elim_portfolio:
        branches.extend(
            (
                _distshift_scurve_branch(2),
                _distshift_scurve_branch(4),
                _distshift_scurve_branch(6),
                _distshift_scurve_branch(4, cutrow=True),
                _distshift_scurve_branch(8),
            )
        )
    if profile.acasxu_scip_witness_fallback:
        branches.append(
            HybridZRunBranch(
                "sparse_scip_witness",
                set_env={"HZ_MILP_BACKEND": "scip"},
                extra_args=(
                    "--hybridz-engine",
                    "sparse_hz_objbound",
                    "--hybridz-compressed-relu",
                ),
                timeout_override_s=ACASXU_SCIP_WITNESS_MILP_TIMEOUT,
                accept_verdicts=("ADV",),
            )
        )
    return branches


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_sha256_manifest(root: Path, manifest_name: str = "_MANIFEST.sha256") -> Path:
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


def _branch_env(cfg: HybridZBenchmarkConfig, branch: HybridZRunBranch) -> dict[str, str]:
    env = os.environ.copy()
    profile = get_bench_profile(cfg.bench)
    if profile.mem_gb is not None:
        env.setdefault("ACT_HYBRIDZ_RLIMIT_AS_GB", str(profile.mem_gb))
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    if profile.query_workers is not None:
        env.setdefault("HZ_QUERY_WORKERS", str(profile.query_workers))
    if profile.milp_threads is not None:
        env.setdefault("HZ_MILP_THREADS", str(profile.milp_threads))
    for key, value in (profile.milp_env or {}).items():
        env.setdefault(str(key), str(value))
    for key in branch.unset_env:
        env.pop(key, None)
    env.update(branch.set_env)
    return env


def _available_memory_gb() -> Optional[float]:
    """Return host MemAvailable in GiB, or None if the platform cannot report it."""

    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            vals = {line.split(":", 1)[0]: int(line.split()[1]) for line in f}
        return vals.get("MemAvailable", 0) / 1024**2
    except Exception:
        return None


def _memory_floor_gb(cfg: HybridZBenchmarkConfig) -> float:
    raw = os.environ.get("ACT_HYBRIDZ_MEM_FLOOR_GB")
    if raw is not None and raw.strip() != "":
        return max(0.0, float(raw))
    profile = get_bench_profile(cfg.bench)
    return max(0.0, float(profile.mem_floor_gb or 0.0))


def _wait_for_memory_headroom(cfg: HybridZBenchmarkConfig) -> None:
    floor = _memory_floor_gb(cfg)
    if floor <= 0.0:
        return
    try:
        poll_s = max(1.0, float(os.environ.get("ACT_HYBRIDZ_MEM_POLL_S", "10") or 10))
    except Exception:
        poll_s = 10.0
    while True:
        avail = _available_memory_gb()
        if avail is None or avail >= floor:
            return
        print(
            f"[hybridz-benchmark] free RAM {avail:.1f}GB < {floor:.1f}GB; "
            f"pausing launch for {poll_s:.0f}s",
            flush=True,
        )
        time.sleep(poll_s)


def _branch_accepts(branch: HybridZRunBranch, result: dict[str, object]) -> bool:
    return str(result.get("verdict")) in set(branch.accept_verdicts)


def _verdict_to_icse_result(verdict: str) -> str:
    if verdict == "CERT":
        return "unsat"
    if verdict == "ADV":
        return "sat"
    if verdict == "TIMEOUT":
        return "timeout"
    if verdict == "UNKNOWN":
        return "unknown"
    return "error"


def _run_verify_time_s(run: dict[str, object]) -> float:
    for row in run.get("detail_rows", []):
        try:
            return float(row.get("wall_s") or run.get("wall_s") or 0.0)
        except Exception:
            break
    try:
        return float(run.get("wall_s") or 0.0)
    except Exception:
        return 0.0


def _truthy_flag(value: object) -> bool:
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


def _row_has_p0(row: dict[str, object]) -> bool:
    if _truthy_flag(row.get("p0")) or _truthy_flag(row.get("P0")):
        return True
    raw_meta = row.get("metadata_json")
    if not raw_meta:
        return False
    try:
        meta = json.loads(str(raw_meta))
    except Exception:
        return False
    if not isinstance(meta, dict):
        return False
    return _truthy_flag(meta.get("p0")) or _truthy_flag(meta.get("P0"))


def _run_has_p0(run: dict[str, object]) -> bool:
    if _truthy_flag(run.get("p0")):
        return True
    payload = run.get("module_payload")
    if isinstance(payload, dict) and (
        _truthy_flag(payload.get("p0")) or _truthy_flag(payload.get("P0"))
    ):
        return True
    for key in ("summary_rows", "detail_rows"):
        for row in run.get(key, []) or []:
            if isinstance(row, dict) and _row_has_p0(row):
                return True
    return False


def _kill_process_group(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    except ProcessLookupError:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.communicate(timeout=2)
    except Exception:
        pass


def _run_command_process_group(
    cmd: list[str],
    *,
    env: dict[str, str],
    timeout_s: float,
) -> tuple[int, str, str, float]:
    """Run ``cmd`` in its own process group and kill the group on timeout."""

    started = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=Path(__file__).resolve().parents[2],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=float(timeout_s))
        return int(proc.returncode or 0), stdout, stderr, time.time() - started
    except subprocess.TimeoutExpired as exc:
        _kill_process_group(proc)
        try:
            stdout, stderr = proc.communicate(timeout=2)
        except Exception:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
        stderr = (stderr or "") + "\nsubprocess timeout"
        return 124, stdout or "", stderr, time.time() - started


def _classify_branch_result(
    cfg: HybridZBenchmarkConfig,
    inst: HybridZBenchmarkInstance,
    branch: HybridZRunBranch,
    branch_dir: Path,
    *,
    returncode: int,
    stdout: str,
    stderr: str,
    wall_s: float,
) -> dict[str, object]:
    detail_path = branch_dir / f"{cfg.bench}_hybridz_detail.csv"
    summary_path = branch_dir / f"{cfg.bench}_hybridz_summary.csv"
    detail_rows = _read_csv_rows(detail_path)
    summary_rows = _read_csv_rows(summary_path)
    verdict = "ERROR"
    module_payload: dict[str, object] = {}
    if summary_rows:
        row = summary_rows[0]
        verdict = _instance_verdict_from_summary_row(row)
    elif branch.module:
        for line in reversed(str(stdout).splitlines()):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                module_payload = payload
                break
        raw = str(module_payload.get("verdict", "")).upper()
        if bool(module_payload.get("p0", False)):
            verdict = "ERROR"
        elif raw in {"CERT", "ADV", "UNKNOWN", "TIMEOUT", "ERROR"}:
            verdict = raw
        elif returncode == 124:
            verdict = "TIMEOUT"
        elif returncode == 0:
            verdict = "UNKNOWN"
    elif returncode == 124:
        verdict = "TIMEOUT"
    if module_payload and module_payload.get("time_s") is not None:
        try:
            wall_s = float(module_payload["time_s"])
        except Exception:
            pass
    p0 = any(_row_has_p0(row) for row in summary_rows + detail_rows) or (
        _truthy_flag(module_payload.get("p0")) or _truthy_flag(module_payload.get("P0"))
    )

    return {
        "index": inst.index,
        "verdict": verdict,
        "branch": branch.name,
        "returncode": returncode,
        "wall_s": wall_s,
        "onnx_model": inst.onnx_model,
        "vnnlib_spec": inst.vnnlib_spec,
        "timeout_s": inst.timeout_s,
        "detail_rows": detail_rows,
        "summary_rows": summary_rows,
        "module_payload": module_payload,
        "p0": p0,
        "stdout_tail": "\n".join(str(stdout).splitlines()[-20:]),
        "stderr_tail": "\n".join(str(stderr).splitlines()[-20:]),
    }


def _instance_verdict_from_summary_row(row: dict[str, object]) -> str:
    """Collapse per-query frontend summary counts into one VNNLIB verdict.

    A VNNLIB instance may be split into multiple unsafe disjunct queries.  The
    instance is falsified if any disjunct is falsified, and it is certified only
    if every emitted query is certified.  Partial CERT rows plus an unresolved
    or ADV row must never be promoted to an instance-level CERT.
    """

    n = _int_field(row, "N")
    cert = _int_field(row, "CERT")
    adv = _int_field(row, "ADV")
    timeout = _int_field(row, "TIMEOUT")
    unknown = _int_field(row, "UNKNOWN")
    error = _int_field(row, "ERROR")
    if adv > 0:
        return "ADV"
    if n > 0 and cert == n:
        return "CERT"
    if timeout > 0:
        return "TIMEOUT"
    if unknown > 0:
        return "UNKNOWN"
    if error > 0:
        return "ERROR"
    if cert > 0:
        return "UNKNOWN"
    return "ERROR"


def _run_one_branch(
    cfg: HybridZBenchmarkConfig,
    inst: HybridZBenchmarkInstance,
    branch: HybridZRunBranch,
    branch_dir: Path,
    wall_limit: float,
) -> dict[str, object]:
    branch_dir.mkdir(parents=True, exist_ok=True)
    cmd = _instance_command(cfg, inst, branch_dir, branch)
    returncode, stdout, stderr, wall_s = _run_command_process_group(
        cmd,
        env=_branch_env(cfg, branch),
        timeout_s=wall_limit,
    )
    return _classify_branch_result(
        cfg,
        inst,
        branch,
        branch_dir,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        wall_s=wall_s,
    )


def _run_one(cfg: HybridZBenchmarkConfig, inst: HybridZBenchmarkInstance) -> dict[str, object]:
    instance_dir = cfg.out_dir / f"iid_{inst.index:05d}"
    instance_dir.mkdir(parents=True, exist_ok=True)
    wall_limit = _instance_wall_limit(cfg, inst)
    branches = _branch_plan(cfg)
    if len(branches) == 1:
        return _run_one_branch(cfg, inst, branches[0], instance_dir, wall_limit)
    if cfg.bench in SEQUENTIAL_PORTFOLIO_BENCHES:
        done: dict[str, dict[str, object]] = {}
        winner: Optional[dict[str, object]] = None
        deadline = time.time() + wall_limit
        for branch in branches:
            remaining = max(0.1, deadline - time.time())
            branch_dir = instance_dir / branch.name
            result = _run_one_branch(cfg, inst, branch, branch_dir, remaining)
            done[branch.name] = result
            if _branch_accepts(branch, result):
                winner = result
                break
            if time.time() >= deadline:
                break
        if winner is None:
            winner = done.get("normal") or next(iter(done.values()))
        winner["portfolio_done"] = {name: row.get("verdict") for name, row in done.items()}
        winner["portfolio_branches"] = [branch.name for branch in branches]
        winner["portfolio_mode"] = "sequential_sparse"
        return winner

    procs: dict[str, tuple[HybridZRunBranch, Path, subprocess.Popen, float]] = {}
    for branch in branches:
        branch_dir = instance_dir / branch.name
        branch_dir.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(
            _instance_command(cfg, inst, branch_dir, branch),
            cwd=Path(__file__).resolve().parents[2],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_branch_env(cfg, branch),
            start_new_session=True,
        )
        procs[branch.name] = (branch, branch_dir, proc, time.time())

    deadline = time.time() + wall_limit
    done: dict[str, dict[str, object]] = {}
    winner: Optional[dict[str, object]] = None
    while procs and time.time() < deadline:
        for name, (branch, branch_dir, proc, started) in list(procs.items()):
            if proc.poll() is None:
                continue
            stdout, stderr = proc.communicate()
            result = _classify_branch_result(
                cfg,
                inst,
                branch,
                branch_dir,
                returncode=int(proc.returncode or 0),
                stdout=stdout,
                stderr=stderr,
                wall_s=time.time() - started,
            )
            done[name] = result
            procs.pop(name)
            if _branch_accepts(branch, result) and winner is None:
                winner = result
        if winner is not None:
            break
        time.sleep(0.05)

    for name, (branch, branch_dir, proc, started) in list(procs.items()):
        _kill_process_group(proc)
        stdout, stderr = proc.communicate()
        done[name] = _classify_branch_result(
            cfg,
            inst,
            branch,
            branch_dir,
            returncode=124,
            stdout=stdout,
            stderr=stderr or "portfolio branch killed after winner/timeout",
            wall_s=time.time() - started,
        )

    if winner is None:
        winner = done.get("normal") or next(iter(done.values()))
    winner["portfolio_done"] = {name: row.get("verdict") for name, row in done.items()}
    winner["portfolio_branches"] = [branch.name for branch in branches]
    return winner


def _write_icse_benchmark_outputs(
    cfg: HybridZBenchmarkConfig,
    run_rows: Iterable[dict[str, object]],
) -> tuple[Path, Path, Path]:
    rows = list(run_rows)
    bench_path = cfg.out_dir / f"{cfg.bench}.csv"
    index_path = cfg.out_dir / f"{cfg.bench}_icse_index.csv"
    detail_path = cfg.out_dir / f"{cfg.bench}_icse_detail.csv"

    counts = {"unsat": 0, "sat": 0, "timeout": 0, "unknown": 0, "unsupported": 0, "error": 0}
    total_time = 0.0
    with bench_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["onnx", "vnnlib", "result", "time_sec"])
        writer.writeheader()
        for run in rows:
            result = _verdict_to_icse_result(str(run.get("verdict", "ERROR")))
            time_s = _run_verify_time_s(run)
            total_time += time_s
            counts[result] = counts.get(result, 0) + 1
            writer.writerow({
                "onnx": str(run.get("onnx_model", "")),
                "vnnlib": str(run.get("vnnlib_spec", "")),
                "result": result,
                "time_sec": f"{time_s:.2f}",
            })

    index_row = {
        "benchmark": cfg.bench,
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
            result = _verdict_to_icse_result(str(run.get("verdict", "ERROR")))
            time_s = _run_verify_time_s(run)
            err = run.get("stderr_tail", "") or run.get("stdout_tail", "")
            writer.writerow({
                "benchmark": cfg.bench,
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
                "p0": int(_run_has_p0(run)),
                "err": str(err)[:300],
            })
    return bench_path, index_path, detail_path


def _profile_json(bench: str) -> dict[str, object]:
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


def _write_benchmark_json_summary(
    cfg: HybridZBenchmarkConfig,
    summary: dict[str, object],
    run_rows: Iterable[dict[str, object]],
) -> Path:
    rows = list(run_rows)
    payload = {
        "bench": cfg.bench,
        "out_dir": str(cfg.out_dir),
        "summary": summary,
        "profile": _profile_json(cfg.bench),
        "instances": [
            {
                "iid": int(row.get("index", -1)),
                "verdict": str(row.get("verdict", "ERROR")),
                "branch": str(row.get("branch", "")),
                "onnx": str(row.get("onnx_model", "")),
                "vnnlib": str(row.get("vnnlib_spec", "")),
                "time_sec": _run_verify_time_s(row),
                "portfolio_done": row.get("portfolio_done", {}),
            }
            for row in rows
        ],
    }
    path = cfg.out_dir / f"{cfg.bench}_run_summary.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_combined(cfg: HybridZBenchmarkConfig, results: Iterable[dict[str, object]]) -> tuple[Path, Path]:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = cfg.out_dir / f"{cfg.bench}_hybridz_detail.csv"
    summary_path = cfg.out_dir / f"{cfg.bench}_hybridz_summary.csv"
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
            if isinstance(payload, dict):
                metadata["module_payload"] = payload
            try:
                idx = int(run.get("index", -1))
                tag = f"iid{idx:05d}"
            except Exception:
                tag = str(run.get("index", ""))
            writer.writerow({
                "bench": cfg.bench,
                "tag": tag,
                "lane": str(run.get("branch", "")),
                "status": str(run.get("verdict", "ERROR")),
                "verdict": str(run.get("verdict", "ERROR")),
                "wall_s": f"{_run_verify_time_s(run):.2f}",
                "reason": str(run.get("stderr_tail", "") or run.get("stdout_tail", ""))[:300],
                "hz_verdict": str(run.get("verdict", "ERROR")),
                "engine": str(payload.get("mode", "hybridz") if isinstance(payload, dict) else "hybridz"),
                "p0": int(_run_has_p0(run)),
                "metadata_json": json.dumps(metadata, sort_keys=True),
            })

    counts = {"CERT": 0, "ADV": 0, "TIMEOUT": 0, "UNKNOWN": 0, "ERROR": 0}
    for run in run_rows:
        verdict = str(run.get("verdict", "ERROR"))
        counts[verdict] = counts.get(verdict, 0) + 1
    p0 = sum(1 for run in run_rows if _run_has_p0(run))
    summary = {
        "Bench": cfg.bench,
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
    _write_icse_benchmark_outputs(cfg, run_rows)
    _write_benchmark_json_summary(cfg, summary, run_rows)
    _write_sha256_manifest(cfg.out_dir)
    return detail_path, summary_path


def _write_suite_combined(
    cfg: HybridZBenchmarkSuiteConfig,
    bench_summaries: Iterable[dict[str, object]],
    detail_paths: Iterable[Path],
) -> tuple[Path, Path]:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = list(bench_summaries)
    detail_path = cfg.out_dir / "hybridz_suite_detail.csv"
    summary_path = cfg.out_dir / "hybridz_suite_summary.csv"

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
            for row in _read_csv_rows(path):
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
            out = {key: row.get(key, 0 if key != "Bench" else "") for key in fields}
            writer.writerow(out)
            for key in total:
                total[key] += int(out.get(key, 0) or 0)
        if summaries:
            writer.writerow({"Bench": "TOTAL", **total})
    return detail_path, summary_path


def _int_field(row: dict[str, object], key: str) -> int:
    try:
        return int(row.get(key, 0) or 0)
    except Exception:
        return 0


def _reuse_frozen_benchmark_outputs(
    cfg: HybridZBenchmarkConfig,
) -> tuple[Path, Path, list[dict[str, object]]] | None:
    """Reuse an already completed frozen bench only when its summary matches.

    This is a resume guard for long frozen-suite runs.  It is deliberately
    stricter than the public frozen comparison: every summary field must match
    the frozen HybridZ oracle before the bench is skipped.
    """

    try:
        expected = frozen_hybridz_expected_summary(cfg.bench)
    except KeyError:
        return None

    detail_path = cfg.out_dir / f"{cfg.bench}_hybridz_detail.csv"
    summary_path = cfg.out_dir / f"{cfg.bench}_hybridz_summary.csv"
    if not detail_path.exists() or not summary_path.exists():
        return None

    rows = _read_csv_rows(summary_path)
    if len(rows) != 1:
        return None
    row = rows[0]
    for field in FROZEN_SUMMARY_FIELDS:
        if _int_field(row, field) != int(expected[field]):
            return None
    return detail_path, summary_path, []


def _ours_counts_from_summary(row: dict[str, object]) -> tuple[int, int, int, int, int]:
    return (
        _int_field(row, "CERT"),
        _int_field(row, "ADV"),
        _int_field(row, "TIMEOUT"),
        _int_field(row, "UNKNOWN"),
        _int_field(row, "ERROR"),
    )


def _va_count(counts: tuple[int, int, int, int, int]) -> int:
    return int(counts[0]) + int(counts[1])


def _build_cross_tool_rows(
    summary_rows: Iterable[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
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
            "N": _int_field(row, "N"),
            "CERT": _int_field(row, "CERT"),
            "ADV": _int_field(row, "ADV"),
            "V+A": ours_va,
            "TIMEOUT": _int_field(row, "TIMEOUT"),
            "UNKNOWN": _int_field(row, "UNKNOWN"),
            "ERROR": _int_field(row, "ERROR"),
            "P0": _int_field(row, "P0"),
            "unsolved": _int_field(row, "unsolved"),
            "rank_competition": 1 + sum(1 for value in tool_va.values() if value > ours_va),
            "rank_dense": 1 + len({value for value in tool_va.values() if value > ours_va}),
            "best_V+A": best_va,
            "best_tools": "+".join(best_tools),
            "gap_to_best": best_va - ours_va,
        }
        ranked.update({tool: tool_va[tool] for tool in CROSS_TOOL_NAMES})
        ranked_rows.append(ranked)

        cross = {"Bench": bench, "N": _int_field(row, "N")}
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


def _write_suite_cross_tool_outputs(
    cfg: HybridZBenchmarkSuiteConfig,
    suite_summary_path: Path,
) -> tuple[Path, Path, Path] | tuple[()]:
    if cfg.max_instances is not None:
        return ()
    ranked_rows, cross_rows = _build_cross_tool_rows(_read_csv_rows(suite_summary_path))
    if not ranked_rows:
        return ()

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
    final_results = cfg.out_dir / "FINAL_HYBRIDZ_RESULTS.csv"
    final_ranking = cfg.out_dir / "FINAL_CROSS_TOOL_RANKING.csv"
    for path in (final_results, final_ranking):
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ranking_fields)
            writer.writeheader()
            writer.writerows({name: row.get(name, "") for name in ranking_fields} for row in ranked_rows)

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
    cross_path = cfg.out_dir / "_CROSS_TOOL_SUMMARY.csv"
    with cross_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cross_fields)
        writer.writeheader()
        writer.writerows({name: row.get(name, "") for name in cross_fields} for row in cross_rows)
    return final_results, final_ranking, cross_path


def _build_frozen_repro_rows(summary_rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
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
            current_value = 0 if current is None else _int_field(current, field)
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
            out[f"current_{field}"] = _int_field(current, field)
            out[f"expected_{field}"] = ""
            out[f"delta_{field}"] = ""
        rows.append(out)
    return rows


def _write_frozen_repro_check(
    cfg: HybridZBenchmarkSuiteConfig,
    suite_summary_path: Path,
) -> tuple[Path, Path, bool] | tuple[()]:
    if cfg.max_instances is not None or cfg.benches != FROZEN_BENCHMARK_SUITE:
        return ()

    rows = _build_frozen_repro_rows(_read_csv_rows(suite_summary_path))
    fields = ["Bench", "status"]
    for field in FROZEN_SUMMARY_FIELDS:
        fields.extend([f"current_{field}", f"expected_{field}", f"delta_{field}"])

    csv_path = cfg.out_dir / "FROZEN_REPRO_COMPARISON.csv"
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
    json_path = cfg.out_dir / "FROZEN_REPRO_COMPARISON.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return csv_path, json_path, bool(payload["ok"])


def _enforce_frozen_match(repro_check: tuple[Path, Path, bool] | tuple[()]) -> None:
    if not repro_check:
        raise RuntimeError(
            "--hybridz-require-frozen-match requires a full frozen suite "
            "without --max-instances"
        )
    _, repro_json, repro_ok = repro_check
    if not repro_ok:
        raise RuntimeError(f"frozen HybridZ reproduction mismatch; see {repro_json}")


def _json_field(value: object) -> str:
    return json.dumps(value if value is not None else {}, sort_keys=True, default=str)


def _detail_reason(run: dict[str, object]) -> str:
    for row in run.get("detail_rows", []):
        reason = str(row.get("reason", "")).strip()
        if reason:
            return reason[:300]
    return ""


def _tail_text(run: dict[str, object]) -> str:
    return (str(run.get("stderr_tail", "")) or str(run.get("stdout_tail", "")))[:300]


def _taxonomy_class(run: dict[str, object]) -> tuple[str, str]:
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


def _write_suite_failure_taxonomy(
    cfg: HybridZBenchmarkSuiteConfig,
    run_rows: Iterable[dict[str, object]],
) -> tuple[Path, Path]:
    rows = []
    for run in run_rows:
        result_class, note = _taxonomy_class(run)
        rows.append(
            {
                "bench": str(run.get("bench", "")),
                "iid": int(run.get("index", -1)),
                "verdict": str(run.get("verdict", "ERROR")),
                "result_class": result_class,
                "branch": str(run.get("branch", "")),
                "portfolio_done": _json_field(run.get("portfolio_done", {})),
                "portfolio_branches": ";".join(str(x) for x in run.get("portfolio_branches", []) or []),
                "time_s": f"{_run_verify_time_s(run):.2f}",
                "returncode": str(run.get("returncode", "")),
                "p0": int(_run_has_p0(run)),
                "reason": _detail_reason(run),
                "err": _tail_text(run),
                "note": note,
            }
        )
    rows.sort(key=lambda row: (row["bench"], int(row["iid"])))

    detail_path = cfg.out_dir / "failure_taxonomy_detail.csv"
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
        if _truthy_flag(row.get("p0")):
            counts["P0"] += 1
        counts[str(row["result_class"])] = counts.get(str(row["result_class"]), 0) + 1

    classes = sorted({key for counts in counts_by_bench.values() for key in counts if key not in {"N", "V+A", "P0"}})
    summary_path = cfg.out_dir / "failure_taxonomy_summary.csv"
    summary_fields = ["bench", "N", "V+A", "P0", *classes]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for bench in sorted(counts_by_bench):
            counts = counts_by_bench[bench]
            writer.writerow({field: bench if field == "bench" else counts.get(field, 0) for field in summary_fields})
    return detail_path, summary_path


def _write_suite_icse_outputs(cfg: HybridZBenchmarkSuiteConfig) -> tuple[Path, Path]:
    index_path = cfg.out_dir / "_INDEX.csv"
    detail_path = cfg.out_dir / "_DETAIL.csv"
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
    for bench in cfg.benches:
        bench_dir = cfg.out_dir / bench
        bench_csv = bench_dir / f"{bench}.csv"
        if bench_csv.exists():
            (cfg.out_dir / f"{bench}.csv").write_text(
                bench_csv.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        index_rows.extend(_read_csv_rows(bench_dir / f"{bench}_icse_index.csv"))
        detail_rows.extend(_read_csv_rows(bench_dir / f"{bench}_icse_detail.csv"))

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

    readme = cfg.out_dir / "README_REPRODUCIBILITY.md"
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


def _write_suite_json_summary(
    cfg: HybridZBenchmarkSuiteConfig,
    suite_summary_path: Path,
) -> Path:
    rows = _read_csv_rows(suite_summary_path)
    bench_rows = [row for row in rows if row.get("Bench") != "TOTAL"]
    total_rows = [row for row in rows if row.get("Bench") == "TOTAL"]
    payload = {
        "suite": "frozen" if cfg.benches == FROZEN_BENCHMARK_SUITE else "custom",
        "out_dir": str(cfg.out_dir),
        "benchmarks": bench_rows,
        "total": total_rows[0] if total_rows else {},
        "profiles": {bench: _profile_json(bench) for bench in cfg.benches},
        "config": {
            "benches": list(cfg.benches),
            "max_instances": cfg.max_instances,
            "workers": cfg.workers,
            "timeout_cap_s": cfg.timeout_cap_s,
            "device": cfg.device,
            "dtype": cfg.dtype,
        },
    }
    path = cfg.out_dir / "hybridz_suite_summary.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_hybridz_benchmark(cfg: HybridZBenchmarkConfig) -> tuple[Path, Path, list[dict[str, object]]]:
    instances = list_hybridz_benchmark_instances(cfg.bench)
    if cfg.max_instances is not None:
        instances = instances[: max(0, int(cfg.max_instances))]
    if not instances:
        raise RuntimeError(f"no downloaded VNNLIB instances found for {cfg.bench!r}")

    workers = cfg.workers
    if workers is None:
        workers = get_bench_profile(cfg.bench).workers or 1
    workers = max(1, min(int(workers), len(instances)))

    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        inst_iter = iter(instances)
        pending = {}

        def _submit_next() -> bool:
            try:
                inst = next(inst_iter)
            except StopIteration:
                return False
            _wait_for_memory_headroom(cfg)
            pending[pool.submit(_run_one, cfg, inst)] = inst
            return True

        for _ in range(workers):
            if not _submit_next():
                break
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                pending.pop(fut)
                results.append(fut.result())
            while len(pending) < workers:
                if not _submit_next():
                    break
    results.sort(key=lambda row: int(row["index"]))
    detail_path, summary_path = _write_combined(cfg, results)
    return detail_path, summary_path, results


def _missing_benchmark_results(cfg: HybridZBenchmarkConfig, reason: str) -> tuple[Path, Path, list[dict[str, object]]]:
    row = {
        "index": -1,
        "verdict": "ERROR",
        "branch": "missing_downloaded_vnnlib_instances",
        "returncode": 1,
        "wall_s": 0.0,
        "onnx_model": "",
        "vnnlib_spec": "",
        "timeout_s": 0.0,
        "detail_rows": [
            {
                "bench": cfg.bench,
                "tag": "missing_downloaded_vnnlib_instances",
                "lane": 0,
                "status": "ERROR",
                "verdict": "ERROR",
                "wall_s": "0.0",
                "reason": reason,
                "hz_verdict": "",
                "engine": "hybridz_benchmark",
                "metadata_json": json.dumps({"reason": reason}, sort_keys=True),
            }
        ],
        "summary_rows": [],
        "stdout_tail": "",
        "stderr_tail": reason,
    }
    detail_path, summary_path = _write_combined(cfg, [row])
    return detail_path, summary_path, [row]


def run_hybridz_benchmark_suite(
    cfg: HybridZBenchmarkSuiteConfig,
) -> tuple[Path, Path, list[dict[str, object]]]:
    """Run a benchmark suite through the normal strict-HybridZ frontend path."""

    if cfg.max_instances is None or cfg.require_frozen_match:
        _raise_if_strict_suite_data_missing(cfg.benches)

    summaries: list[dict[str, object]] = []
    detail_paths: list[Path] = []
    all_results: list[dict[str, object]] = []
    for bench in cfg.benches:
        bench_cfg = HybridZBenchmarkConfig(
            bench=bench,
            out_dir=cfg.out_dir / bench,
            max_instances=cfg.max_instances,
            workers=cfg.workers,
            timeout_cap_s=cfg.timeout_cap_s,
            device=cfg.device,
            dtype=cfg.dtype,
            python=cfg.python,
        )
        reused = None
        if cfg.max_instances is None and cfg.require_frozen_match:
            reused = _reuse_frozen_benchmark_outputs(bench_cfg)
        if reused is not None:
            detail_path, summary_path, results = reused
            detail_paths.append(detail_path)
            rows = _read_csv_rows(summary_path)
            if rows:
                summaries.append(rows[0])
            continue
        try:
            detail_path, summary_path, results = run_hybridz_benchmark(bench_cfg)
        except RuntimeError as exc:
            reason = str(exc)
            is_missing_data = reason.startswith("no downloaded VNNLIB instances found")
            if cfg.max_instances is None or cfg.require_frozen_match or not is_missing_data:
                raise
            detail_path, summary_path, results = _missing_benchmark_results(bench_cfg, reason)
        detail_paths.append(detail_path)
        rows = _read_csv_rows(summary_path)
        if rows:
            summaries.append(rows[0])
        all_results.extend({**row, "bench": bench} for row in results)
    suite_detail, suite_summary = _write_suite_combined(cfg, summaries, detail_paths)
    _write_suite_icse_outputs(cfg)
    _write_suite_cross_tool_outputs(cfg, suite_summary)
    repro_check = _write_frozen_repro_check(cfg, suite_summary)
    _write_suite_failure_taxonomy(cfg, all_results)
    _write_suite_json_summary(cfg, suite_summary)
    _write_sha256_manifest(cfg.out_dir)
    if cfg.require_frozen_match:
        _enforce_frozen_match(repro_check)
    return suite_detail, suite_summary, all_results


def _test_hybridz_benchmark_runner() -> None:  # pragma: no cover
    vals = list_hybridz_benchmark_instances("acasxu_2023")
    assert vals and vals[0].index == 0
    assert resolve_hybridz_benchmark_categories("acasxu_2023") == ("acasxu_2023",)
    frozen = resolve_hybridz_benchmark_categories("frozen")
    assert frozen[0] == "safenlp_2024"
    assert "cgan_2023" in frozen
    safenlp_branches = [
        "normal",
        "cutrow",
        "normal_pscost1",
        "normal_seed2",
        "sparse_comprelu",
        "sparse_comprelu_heur",
        "projected_relu_mip",
        "projected_relu_scip-bigm",
        "projected_relu_scip-indicator",
    ]
    expected_frozen_branches = {
        "safenlp_2024": safenlp_branches,
        "metaroom_2023": ["sparse", "normal"],
        "sat_relu": ["normal"],
        "malbeware": ["normal", "sparse"],
        "cersyve": ["normal", "cersyve_highs_cuts_fbbt", "cersyve_scip_cuts_fbbt"],
        "acasxu_2023": ["normal", "acasxu_cuts_fbbt", "acasxu_scip_witness"],
        "dist_shift_2023": [
            "normal",
            "elim_singletons",
            "scurve_graph_k2_scip",
            "scurve_graph_k4_scip",
            "scurve_graph_k6_scip",
            "scurve_graph_k4_cutrow_scip",
            "scurve_graph_k8_scip",
        ],
        "linearizenn_2024": ["normal", "linear_portfolio_m360"],
        "tllverifybench_2023": [
            "sparse_tll_cutrow_eqsubst",
            "sparse_tll_objtarget_comprelu",
            "sparse_tll_cutrow_relu_cuts_eqsubst",
            "normal",
        ],
        "cora_2024": ["normal", "sparse_non_mnist_set_m10heur"],
        "relusplitter": ["normal", "sparse"],
        "cgan_2023": ["normal", "sparse_exact_milp_witness"],
    }
    for bench in FROZEN_BENCHMARK_SUITE:
        plan_cfg = HybridZBenchmarkConfig(bench=bench, out_dir=Path("/tmp/hz_runner_test"))
        branches = _branch_plan(plan_cfg)
        assert [b.name for b in branches] == expected_frozen_branches[bench]
        dummy_inst = HybridZBenchmarkInstance(
            index=0,
            timeout_s=20.0,
            onnx_model="dummy.onnx",
            vnnlib_spec="dummy.vnnlib",
        )
        for branch in branches:
            cmd = _instance_command(plan_cfg, dummy_inst, Path("/tmp/hz_runner_test/iid_00000"), branch)
            joined_cmd = " ".join(cmd)
            branch_policy_text = " ".join(
                [
                    branch.name,
                    *branch.extra_args,
                    *branch.module_args,
                    *(f"{key}={value}" for key, value in branch.set_env.items()),
                ]
            ).lower()
            assert "iid" not in branch_policy_text, (bench, branch.name, branch_policy_text)
            assert "instance-index" not in branch_policy_text, (bench, branch.name, branch_policy_text)
            assert "scripts/" not in joined_cmd and "/scripts" not in joined_cmd, (bench, branch.name, cmd)
            env_text = " ".join(f"{key}={value}" for key, value in branch.set_env.items()).lower()
            assert "gurobi" not in env_text, (bench, branch.name, branch.set_env)
            if branch.module:
                assert branch.module.startswith("act.pipeline."), (bench, branch.name, branch.module)
                assert cmd[1] == "-m" and cmd[2] == branch.module
            else:
                assert cmd[1] == "-m" and cmd[2] == "act.pipeline"
                assert "--solvers" in cmd and "hybridz" in cmd
    expected_profile_bits = {
        "safenlp_2024": {"workers": 1, "milp_fraction": 0.95, "parallel_cutoff_portfolio": True},
        "sat_relu": {"milp_fraction": 0.96, "cutoff_row": True},
        "dist_shift_2023": {"workers": 3, "milp_fraction": 0.40, "sigmoid_k": 2, "cell_budget": 800_000_000},
        "linearizenn_2024": {
            "workers": 4,
            "milp_fraction": 1.00,
            "milp_timeout_cap": 900,
            "compressed_relu": True,
            "query_workers": 2,
            "milp_threads": 4,
            "milp_env": {
                "HZ_MILP_BACKEND": "highs",
                "HZ_MILP_START": "lp_binary",
                "HZ_MILP_HEURISTIC": "1.0",
            },
        },
        "tllverifybench_2023": {"workers": 6, "sparse_first": True, "sparse_fallback": True},
        "relusplitter": {"sparse_fallback": True, "query_workers": 9},
        "cgan_2023": {"workers": 2, "mem_gb": 32.0, "compressed_relu": True},
    }
    for bench, expected in expected_profile_bits.items():
        profile = get_bench_profile(bench)
        for key, value in expected.items():
            assert getattr(profile, key) == value, (bench, key, getattr(profile, key), value)
    cfg = HybridZBenchmarkConfig(bench="acasxu_2023", out_dir=Path("/tmp/hz_runner_test"))
    cmd = _instance_command(cfg, vals[0], Path("/tmp/hz_runner_test/iid_00000"))
    assert "--instance-index" in cmd
    assert "0" in cmd
    acas_branches = _branch_plan(cfg)
    assert [b.name for b in acas_branches] == ["normal", "acasxu_cuts_fbbt", "acasxu_scip_witness"]
    assert acas_branches[2].accept_verdicts == ("ADV",)
    acas_cmd = _instance_command(
        cfg,
        vals[0],
        Path("/tmp/hz_runner_test/iid_00000"),
        acas_branches[2],
    )
    assert acas_cmd[2] == SPARSE_WORKER_MODULE
    assert "--compressed-relu" in acas_cmd
    assert str(ACASXU_SCIP_WITNESS_MILP_TIMEOUT) in acas_cmd
    mal_cfg = HybridZBenchmarkConfig(bench="malbeware", out_dir=Path("/tmp/hz_runner_test"))
    mal_sparse = _branch_plan(mal_cfg)[1]
    mal_cmd = _instance_command(
        mal_cfg,
        vals[0],
        Path("/tmp/hz_runner_test/iid_00000"),
        mal_sparse,
    )
    assert mal_cmd[2] == SPARSE_WORKER_MODULE
    assert mal_cmd[mal_cmd.index("--milp-timeout") + 1] == "180"
    assert "--worker-timeout" in mal_cmd
    safenlp_cfg = HybridZBenchmarkConfig(bench="safenlp_2024", out_dir=Path("/tmp/hz_runner_test"))
    safenlp_plan = _branch_plan(safenlp_cfg)
    assert [b.name for b in safenlp_plan] == safenlp_branches
    projected_branch = next(b for b in safenlp_plan if b.name == "projected_relu_mip")
    assert _instance_command(
        safenlp_cfg,
        vals[0],
        Path("/tmp/hz_runner_test/iid_00000"),
        projected_branch,
    )[2] == "act.pipeline.hybridz_projected_relu_mip"
    metaroom_cfg = HybridZBenchmarkConfig(bench="metaroom_2023", out_dir=Path("/tmp/hz_runner_test"))
    branches = _branch_plan(metaroom_cfg)
    assert [b.name for b in branches] == ["sparse", "normal"]
    sparse_cmd = _instance_command(
        metaroom_cfg,
        vals[0],
        Path("/tmp/hz_runner_test/iid_00000"),
        branches[0],
    )
    assert "--hybridz-engine" in sparse_cmd and "sparse_hz_objbound" in sparse_cmd
    relu_cfg = HybridZBenchmarkConfig(bench="relusplitter", out_dir=Path("/tmp/hz_runner_test"))
    assert [b.name for b in _branch_plan(relu_cfg)] == ["normal", "sparse"]
    cgan_cfg = HybridZBenchmarkConfig(bench="cgan_2023", out_dir=Path("/tmp/hz_runner_test"))
    cgan_env = _branch_env(cgan_cfg, _branch_plan(cgan_cfg)[0])
    assert cgan_env["ACT_HYBRIDZ_RLIMIT_AS_GB"] == "32.0"
    cgan_sparse = _branch_plan(cgan_cfg)[1]
    assert cgan_sparse.module == SPARSE_WORKER_MODULE
    cgan_cmd = _instance_command(
        cgan_cfg,
        vals[0],
        Path("/tmp/hz_runner_test/iid_00000"),
        cgan_sparse,
    )
    assert "--worker-timeout" in cgan_cmd
    assert _instance_verdict_from_summary_row(
        {"N": "19", "CERT": "17", "ADV": "2", "TIMEOUT": "0", "UNKNOWN": "0", "ERROR": "0"}
    ) == "ADV"
    assert _instance_verdict_from_summary_row(
        {"N": "19", "CERT": "18", "ADV": "0", "TIMEOUT": "1", "UNKNOWN": "0", "ERROR": "0"}
    ) == "TIMEOUT"
    assert _instance_verdict_from_summary_row(
        {"N": "19", "CERT": "19", "ADV": "0", "TIMEOUT": "0", "UNKNOWN": "0", "ERROR": "0"}
    ) == "CERT"
    assert _run_has_p0({"module_payload": {"p0": True}})
    assert _run_has_p0({"summary_rows": [{"P0": "2"}]})
    assert _run_has_p0({"detail_rows": [{"metadata_json": json.dumps({"P0": 1})}]})
    assert not _run_has_p0({"detail_rows": [{"metadata_json": json.dumps({"p0": False})}]})
    rc, stdout, stderr, wall = _run_command_process_group(
        [sys.executable, "-c", "print('hybridz-runner-ok')"],
        env=os.environ.copy(),
        timeout_s=5,
    )
    assert rc == 0, stderr
    assert "hybridz-runner-ok" in stdout
    assert wall >= 0.0
    rc, stdout, stderr, wall = _run_command_process_group(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import act.back_end.solver; import act.pipeline; "
                "assert 'act.back_end.solver.solver_gurobi' not in sys.modules; "
                "assert 'act.pipeline.verification.torch2act' not in sys.modules; "
                "assert 'act.pipeline.verification.model_factory' not in sys.modules; "
                "assert 'act.pipeline.verification.utils' not in sys.modules; "
                "assert 'act.pipeline.verification.validate_verifier' not in sys.modules; "
                "assert 'act.pipeline.verification.llm_probe' not in sys.modules; "
                "print('pure-hybridz-import-ok')"
            ),
        ],
        env=os.environ.copy(),
        timeout_s=10,
    )
    assert rc == 0, stderr
    assert "pure-hybridz-import-ok" in stdout
    assert "Gurobi license" not in stdout + stderr
    assert "gurobipy" not in stdout + stderr
    rc, stdout, stderr, wall = _run_command_process_group(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        env=os.environ.copy(),
        timeout_s=0.2,
    )
    assert rc == 124
    assert "subprocess timeout" in stderr
    assert wall < 3.0
    old_floor = os.environ.get("ACT_HYBRIDZ_MEM_FLOOR_GB")
    try:
        os.environ["ACT_HYBRIDZ_MEM_FLOOR_GB"] = "0"
        assert _memory_floor_gb(cgan_cfg) == 0.0
    finally:
        if old_floor is None:
            os.environ.pop("ACT_HYBRIDZ_MEM_FLOOR_GB", None)
        else:
            os.environ["ACT_HYBRIDZ_MEM_FLOOR_GB"] = old_floor
    dist_cfg = HybridZBenchmarkConfig(bench="dist_shift_2023", out_dir=Path("/tmp/hz_runner_test"))
    dist_branches = _branch_plan(dist_cfg)
    assert [b.name for b in dist_branches] == [
        "normal",
        "elim_singletons",
        "scurve_graph_k2_scip",
        "scurve_graph_k4_scip",
        "scurve_graph_k6_scip",
        "scurve_graph_k4_cutrow_scip",
        "scurve_graph_k8_scip",
    ]
    assert dist_branches[1].module == FULL_WORKER_MODULE
    assert "--scurve-graph-cuts" in dist_branches[2].module_args
    assert "--cutoff-row" in dist_branches[5].module_args
    assert "8" in dist_branches[6].module_args
    suite_dir = Path("/tmp/hz_runner_test_suite")
    fake_summary = [
        {
            "Bench": "toy_a",
            "N": "2",
            "CERT": "1",
            "ADV": "1",
            "V+A": "2",
            "TIMEOUT": "0",
            "UNKNOWN": "0",
            "ERROR": "0",
            "P0": "0",
            "unsolved": "0",
        },
        {
            "Bench": "toy_b",
            "N": "1",
            "CERT": "0",
            "ADV": "0",
            "V+A": "0",
            "TIMEOUT": "1",
            "UNKNOWN": "0",
            "ERROR": "0",
            "P0": "1",
            "unsolved": "1",
        },
    ]
    fake_detail = suite_dir / "toy_detail.csv"
    suite_dir.mkdir(parents=True, exist_ok=True)
    with fake_detail.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bench",
                "tag",
                "lane",
                "status",
                "verdict",
                "wall_s",
                "reason",
                "hz_verdict",
                "engine",
                "metadata_json",
            ],
        )
        writer.writeheader()
        writer.writerow({"bench": "toy_a", "tag": "iid0", "verdict": "CERT"})
    _, suite_summary = _write_suite_combined(
        HybridZBenchmarkSuiteConfig(benches=("toy_a", "toy_b"), out_dir=suite_dir),
        fake_summary,
        [fake_detail],
    )
    rows = _read_csv_rows(suite_summary)
    assert rows[-1]["Bench"] == "TOTAL"
    assert rows[-1]["N"] == "3"
    assert rows[-1]["V+A"] == "2"
    assert rows[-1]["P0"] == "1"
    suite_json = _write_suite_json_summary(
        HybridZBenchmarkSuiteConfig(benches=("toy_a", "toy_b"), out_dir=suite_dir),
        suite_summary,
    )
    suite_payload = json.loads(suite_json.read_text(encoding="utf-8"))
    assert suite_payload["total"]["Bench"] == "TOTAL"
    assert set(suite_payload["profiles"]) == {"toy_a", "toy_b"}
    assert "milp_env" in suite_payload["profiles"]["toy_a"]
    assert "compressed_relu" in suite_payload["profiles"]["toy_b"]
    ranking_input = [
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
            "ADV": "1",
            "V+A": "40",
            "TIMEOUT": "20",
            "UNKNOWN": "0",
            "ERROR": "0",
            "P0": "0",
            "unsolved": "20",
        },
    ]
    ranked_rows, cross_rows = _build_cross_tool_rows(ranking_input)
    assert ranked_rows[0]["rank_competition"] == 2
    assert ranked_rows[0]["best_tools"] == "abCROWN"
    assert ranked_rows[1]["rank_competition"] == 5
    assert ranked_rows[1]["rank_dense"] == 3
    assert cross_rows[0]["OURS_sat"] == 647
    ranking_summary = suite_dir / "ranking_summary.csv"
    with ranking_summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ranking_input[0].keys()))
        writer.writeheader()
        writer.writerows(ranking_input)
    ranking_paths = _write_suite_cross_tool_outputs(
        HybridZBenchmarkSuiteConfig(
            benches=("safenlp_2024", "linearizenn_2024"),
            out_dir=suite_dir,
        ),
        ranking_summary,
    )
    assert len(ranking_paths) == 3
    assert (suite_dir / "FINAL_HYBRIDZ_RESULTS.csv").exists()
    assert _read_csv_rows(suite_dir / "_CROSS_TOOL_SUMMARY.csv")[0]["abCROWN"] == "1080"
    frozen_check_input = suite_dir / "frozen_check_summary.csv"
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
    with frozen_check_input.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(frozen_rows[0].keys()))
        writer.writeheader()
        writer.writerows(frozen_rows)
    repro_rows = _build_frozen_repro_rows(frozen_rows)
    by_bench = {str(row["Bench"]): row for row in repro_rows}
    assert by_bench["safenlp_2024"]["status"] == "match"
    assert by_bench["linearizenn_2024"]["status"] == "mismatch"
    assert by_bench["linearizenn_2024"]["delta_ADV"] == -1
    assert by_bench["dist_shift_2023"]["status"] == "match"
    assert by_bench["dist_shift_2023"]["delta_TIMEOUT"] == 2
    assert by_bench["dist_shift_2023"]["delta_UNKNOWN"] == -2
    assert by_bench["cgan_2023"]["status"] == "missing"
    repro_paths = _write_frozen_repro_check(
        HybridZBenchmarkSuiteConfig(benches=FROZEN_BENCHMARK_SUITE, out_dir=suite_dir),
        frozen_check_input,
    )
    assert len(repro_paths) == 3
    repro_payload = json.loads((suite_dir / "FROZEN_REPRO_COMPARISON.json").read_text(encoding="utf-8"))
    assert repro_payload["ok"] is False
    assert repro_payload["match_fields"] == list(FROZEN_REPRO_MATCH_FIELDS)
    assert "TIMEOUT" in repro_payload["audit_only_fields"]
    assert repro_payload["status_counts"]["match"] == 2
    try:
        _enforce_frozen_match(repro_paths)
        raise AssertionError("expected frozen mismatch gate to fail")
    except RuntimeError as exc:
        assert "reproduction mismatch" in str(exc)
    full_match_summary = suite_dir / "frozen_full_match_summary.csv"
    full_match_rows = [
        {"Bench": bench, **frozen_hybridz_expected_summary(bench)}
        for bench in FROZEN_BENCHMARK_SUITE
    ]
    with full_match_summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Bench", *FROZEN_SUMMARY_FIELDS])
        writer.writeheader()
        writer.writerows(full_match_rows)
    full_match_paths = _write_frozen_repro_check(
        HybridZBenchmarkSuiteConfig(benches=FROZEN_BENCHMARK_SUITE, out_dir=suite_dir),
        full_match_summary,
    )
    assert len(full_match_paths) == 3
    _enforce_frozen_match(full_match_paths)
    reuse_dir = suite_dir / "reuse_check" / "safenlp_2024"
    reuse_dir.mkdir(parents=True, exist_ok=True)
    reuse_detail = reuse_dir / "safenlp_2024_hybridz_detail.csv"
    reuse_detail.write_text("bench,tag,lane,status,verdict\n", encoding="utf-8")
    reuse_summary = reuse_dir / "safenlp_2024_hybridz_summary.csv"
    reuse_row = {"Bench": "safenlp_2024", **frozen_hybridz_expected_summary("safenlp_2024")}
    with reuse_summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Bench", *FROZEN_SUMMARY_FIELDS])
        writer.writeheader()
        writer.writerow(reuse_row)
    reuse_cfg = HybridZBenchmarkConfig(bench="safenlp_2024", out_dir=reuse_dir)
    reused = _reuse_frozen_benchmark_outputs(reuse_cfg)
    assert reused is not None
    assert reused[0] == reuse_detail
    assert reused[1] == reuse_summary
    bad_reuse_row = dict(reuse_row)
    bad_reuse_row["ADV"] = 0
    with reuse_summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Bench", *FROZEN_SUMMARY_FIELDS])
        writer.writeheader()
        writer.writerow(bad_reuse_row)
    assert _reuse_frozen_benchmark_outputs(reuse_cfg) is None
    try:
        _enforce_frozen_match(())
        raise AssertionError("expected missing frozen check to fail")
    except RuntimeError as exc:
        assert "requires a full frozen suite" in str(exc)
    taxonomy_cfg = HybridZBenchmarkSuiteConfig(
        benches=("toy_a", "toy_b", "toy_c"),
        out_dir=suite_dir,
    )
    _, taxonomy_summary = _write_suite_failure_taxonomy(
        taxonomy_cfg,
        [
            {"bench": "toy_a", "index": 0, "verdict": "CERT", "detail_rows": []},
            {
                "bench": "toy_a",
                "index": 1,
                "verdict": "UNKNOWN",
                "portfolio_done": {"normal": "UNKNOWN", "sparse": "TIMEOUT"},
                "portfolio_branches": ["normal", "sparse"],
                "detail_rows": [],
            },
            {"bench": "toy_b", "index": 0, "verdict": "TIMEOUT", "p0": True, "detail_rows": []},
            {
                "bench": "toy_c",
                "index": 0,
                "verdict": "UNKNOWN",
                "detail_rows": [{"reason": "unsupported operator Foo"}],
            },
            {
                "bench": "toy_d",
                "index": -1,
                "verdict": "ERROR",
                "branch": "missing_downloaded_vnnlib_instances",
                "stderr_tail": "no downloaded VNNLIB instances found for 'toy_d'",
                "detail_rows": [],
            },
        ],
    )
    taxonomy_rows = _read_csv_rows(taxonomy_summary)
    assert taxonomy_rows[0]["bench"] == "toy_a"
    assert taxonomy_rows[0]["verified"] == "1"
    assert taxonomy_rows[0]["sparse_portfolio_wall"] == "1"
    assert taxonomy_rows[1]["official_wall_timeout"] == "1"
    assert taxonomy_rows[1]["P0"] == "1"
    assert taxonomy_rows[2]["unsupported_operator"] == "1"
    assert taxonomy_rows[3]["missing_downloaded_data"] == "1"
    toy_a_cfg = HybridZBenchmarkConfig(bench="toy_a", out_dir=suite_dir / "toy_a")
    toy_b_cfg = HybridZBenchmarkConfig(bench="toy_b", out_dir=suite_dir / "toy_b")
    _, toy_a_summary = _write_combined(
        toy_a_cfg,
        [
            {
                "index": 0,
                "verdict": "CERT",
                "branch": "normal_pscost1",
                "wall_s": 21.05,
                "onnx_model": "toy_a.onnx",
                "vnnlib_spec": "toy_a.vnnlib",
                "timeout_s": 20.0,
                "portfolio_done": {"normal": "TIMEOUT", "normal_pscost1": "CERT"},
                "portfolio_branches": ["normal", "normal_pscost1"],
                "detail_rows": [
                    {
                        "bench": "toy_a",
                        "tag": "iid0",
                        "wall_s": "16.815699100494385",
                    }
                ],
            },
            {
                "index": 1,
                "verdict": "ADV",
                "branch": "sparse",
                "wall_s": 4.0,
                "onnx_model": "toy_a2.onnx",
                "vnnlib_spec": "toy_a2.vnnlib",
                "timeout_s": 30.0,
                "p0": True,
                "detail_rows": [{"bench": "toy_a", "tag": "iid1", "wall_s": "2.50"}],
            },
        ],
    )
    assert _read_csv_rows(toy_a_summary)[0]["P0"] == "1"
    _write_combined(
        toy_b_cfg,
        [
            {
                "index": 0,
                "verdict": "TIMEOUT",
                "branch": "normal",
                "wall_s": 3.0,
                "onnx_model": "toy_b.onnx",
                "vnnlib_spec": "toy_b.vnnlib",
                "timeout_s": 30.0,
                "detail_rows": [],
            },
        ],
    )
    toy_b_detail = _read_csv_rows(suite_dir / "toy_b" / "toy_b_hybridz_detail.csv")
    assert len(toy_b_detail) == 1
    assert toy_b_detail[0]["tag"] == "iid00000"
    assert toy_b_detail[0]["verdict"] == "TIMEOUT"
    assert toy_b_detail[0]["lane"] == "normal"
    _write_suite_icse_outputs(
        HybridZBenchmarkSuiteConfig(benches=("toy_a", "toy_b"), out_dir=suite_dir)
    )
    icse_rows = _read_csv_rows(suite_dir / "toy_a.csv")
    assert icse_rows[0]["result"] == "unsat"
    assert icse_rows[1]["result"] == "sat"
    assert icse_rows[0]["time_sec"] == "16.82"
    assert icse_rows[0]["time_sec"] != "21.05"
    bench_json = suite_dir / "toy_a" / "toy_a_run_summary.json"
    assert bench_json.exists()
    toy_profile = json.loads(bench_json.read_text(encoding="utf-8"))["profile"]
    for key in (
        "workers",
        "mem_gb",
        "mem_floor_gb",
        "milp_fraction",
        "milp_timeout_cap",
        "sigmoid_k",
        "cell_budget",
        "compressed_relu",
        "relu_valid_cuts",
        "cutoff_row",
        "sparse_first",
        "sparse_fallback",
        "parallel_cutoff_portfolio",
        "distshift_elim_portfolio",
        "acasxu_scip_witness_fallback",
        "query_workers",
        "milp_threads",
        "milp_env",
    ):
        assert key in toy_profile
    bench_manifest = (suite_dir / "toy_a" / "_MANIFEST.sha256").read_text(encoding="utf-8")
    assert "toy_a.csv" in bench_manifest
    index_rows = _read_csv_rows(suite_dir / "_INDEX.csv")
    assert index_rows[0]["benchmark"] == "toy_a"
    assert index_rows[0]["unsat"] == "1"
    assert index_rows[1]["timeout"] == "1"
    detail_rows = _read_csv_rows(suite_dir / "_DETAIL.csv")
    assert detail_rows[0]["raw_verdict"] == "CERT"
    assert detail_rows[1]["p0"] == "1"
    _write_sha256_manifest(suite_dir)
    suite_manifest = (suite_dir / "_MANIFEST.sha256").read_text(encoding="utf-8")
    assert "_INDEX.csv" in suite_manifest
    assert "hybridz_suite_summary.json" in suite_manifest
    missing_dir = suite_dir / "missing_suite"
    _, missing_summary, missing_results = run_hybridz_benchmark_suite(
        HybridZBenchmarkSuiteConfig(
            benches=("definitely_missing_hybridz_bench",),
            out_dir=missing_dir,
            max_instances=1,
        )
    )
    assert missing_results[0]["verdict"] == "ERROR"
    missing_rows = _read_csv_rows(missing_summary)
    assert missing_rows[0]["ERROR"] == "1"
    assert (missing_dir / "definitely_missing_hybridz_bench" / "definitely_missing_hybridz_bench_run_summary.json").exists()
    try:
        run_hybridz_benchmark_suite(
            HybridZBenchmarkSuiteConfig(
                benches=("definitely_missing_hybridz_bench", "also_missing_hybridz_bench"),
                out_dir=missing_dir / "strict_missing",
            )
        )
        raise AssertionError("expected full suite missing benchmark to fail")
    except RuntimeError as exc:
        msg = str(exc)
        assert "missing downloaded VNNLIB instances for strict HybridZ suite run" in msg
        assert "definitely_missing_hybridz_bench" in msg
        assert "also_missing_hybridz_bench" in msg


if __name__ == "__main__":  # pragma: no cover
    _test_hybridz_benchmark_runner()
    print("PASS _test_hybridz_benchmark_runner")
