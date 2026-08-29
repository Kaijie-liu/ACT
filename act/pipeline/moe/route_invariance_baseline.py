# ===- act/pipeline/moe/route_invariance_baseline.py - Baseline ------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#

"""Explicit route-invariance baseline on the frozen confirmatory cohort.

The baseline and Route A share the same downstream guarded HZ, support, and F0
implementation.  The only algorithmic difference is that the baseline stops
with UNKNOWN unless the exact, tie-inclusive unordered top-k route set is
unique over the endpoint box.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import torch

from act.back_end.moe import load_output_moe_checkpoint
from act.back_end.solver.solver_hz import hz_numerical_policy_manifest
from act.pipeline.moe.experiment1 import (
    PROJECT_ROOT,
    WRITE_ROOT,
    _git_value,
    _inside,
    _sha256,
    _write_json,
)
from act.pipeline.moe.experiment1_confirmatory import (
    SEMANTIC_REASONS,
    _run_boundary_with_deadline,
    _save_gate_witness,
)
from act.pipeline.moe.experiment1c import diagnose_radius
from act.pipeline.moe.experiment1f0 import _run_parent_row
from act.pipeline.moe.train import _load_dataset
from act.util.path_config import get_torchvision_data_root


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "act/pipeline/moe/configs/route_invariance_baseline_confirmatory.json"
)
EXPECTED_PYTHON = Path("/data1/Kane/miniconda3/envs/act-py312/bin/python")
CSV_FIELDS = (
    "sample_rank",
    "dataset_index",
    "endpoint_kind",
    "epsilon",
    "route_precondition_status",
    "exact_feasible_pair_count",
    "baseline_status",
    "baseline_reason",
    "route_a_status",
    "route_a_reason",
    "route_a_only_safe",
    "baseline_seconds",
    "route_a_seconds",
    "route_a_source",
    "full_model_witness_valid",
    "error",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _append_json(handle, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _csv_row(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in CSV_FIELDS}


def _load_jsonl(path: Path, expected_hash: str) -> list[dict[str, Any]]:
    if _sha256(path) != expected_hash:
        raise RuntimeError(f"frozen artifact hash changed: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            row = json.loads(raw)
            row["_line_number"] = line_number
            row["_row_sha256"] = _sha256_bytes(raw)
            rows.append(row)
    return rows


def _stable_cap_endpoint(
    parent: dict[str, Any], cap: float
) -> dict[str, Any]:
    search = parent.get("route_search") or {}
    if search.get("status") != "not_found" or search.get("upper") is not None:
        raise RuntimeError("no-boundary parent does not contain a completed search")
    history = search.get("history") or []
    at_cap = [
        item
        for item in history
        if abs(float(item.get("epsilon", -1.0)) - cap) <= 1e-15
    ]
    if not at_cap or at_cap[-1].get("status") != "stable":
        raise RuntimeError("route-invariant cap lacks an exact stable certificate")
    return {
        "endpoint_kind": "NO_BOUNDARY_CAP",
        "epsilon": cap,
        "exact_feasible_pairs": [parent["clean_topk_set"]],
        "route_precondition_status": "INVARIANT",
        "precondition_source": "frozen_exact_route_search",
    }


def _load_selection(config: dict[str, Any]) -> list[dict[str, Any]]:
    parent_path = _inside(Path(config["parent_results_jsonl"]), WRITE_ROOT)
    closure_path = _inside(Path(config["closure_results_jsonl"]), WRITE_ROOT)
    parents = _load_jsonl(parent_path, config["parent_results_sha256"])
    closure = _load_jsonl(closure_path, config["closure_results_sha256"])
    closure_by_rank = {int(row["sample_rank"]): row for row in closure}
    if len(parents) != int(config["expected_samples"]):
        raise RuntimeError("confirmatory parent row count changed")
    cap = float(config["no_boundary_cap"])
    selection: list[dict[str, Any]] = []
    for parent in parents:
        rank = int(parent["sample_rank"])
        if parent.get("reason") == "NO_ROUTE_BOUNDARY_WITHIN_SEARCH":
            endpoint = _stable_cap_endpoint(parent, cap)
        elif parent.get("exact_feasible_pair_count", 0) > 1:
            endpoint = {
                "endpoint_kind": "ROUTE_BOUNDARY_PRIMARY",
                "epsilon": float(parent["epsilon"]),
                "exact_feasible_pairs": parent["exact_feasible_pairs"],
                "route_precondition_status": "UNSTABLE",
                "precondition_source": "frozen_exact_confirmatory_pairs",
            }
        elif parent.get("reason") == "INSTANCE_HARD_DEADLINE":
            followup = closure_by_rank.get(rank)
            if followup is None:
                raise RuntimeError(f"hard-deadline rank {rank} lacks closure data")
            pairs = (followup.get("f0") or {}).get("feasible_pairs") or []
            if len(pairs) <= 1:
                raise RuntimeError(
                    f"hard-deadline rank {rank} lacks exact route-unstable pairs"
                )
            endpoint = {
                "endpoint_kind": "ROUTE_BOUNDARY_PRIMARY",
                "epsilon": float(followup["epsilon"]),
                "exact_feasible_pairs": pairs,
                "route_precondition_status": "UNSTABLE",
                "precondition_source": "frozen_experiment1d_exact_pairs",
            }
        else:
            raise RuntimeError(f"unclassified confirmatory rank {rank}")
        followup = closure_by_rank.get(rank)
        selection.append(
            {
                "sample_rank": rank,
                "dataset_index": int(parent["dataset_index"]),
                "parent_line_number": parent["_line_number"],
                "parent_row_sha256": parent["_row_sha256"],
                "parent": parent,
                "closure": followup,
                **endpoint,
            }
        )
    counts = Counter(item["route_precondition_status"] for item in selection)
    expected = config["expected_precondition_counts"]
    if dict(counts) != expected:
        raise RuntimeError(f"route-precondition counts changed: {dict(counts)}")
    return selection


def _run_invariant_row(
    model,
    dataset,
    selection: dict[str, Any],
    work_dir: Path,
    runtime: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run the shared downstream verifier after an exact invariant precondition."""
    started = time.monotonic()
    if selection["route_precondition_status"] != "INVARIANT":
        raise RuntimeError("property solving is forbidden for a failed precondition")
    rank = int(selection["sample_rank"])
    index = int(selection["dataset_index"])
    epsilon = float(selection["epsilon"])
    image, label = dataset[index]
    x = image.unsqueeze(0).double()
    with torch.no_grad():
        output, route = model.forward_with_routing(x)
    prediction = int(output.argmax(dim=1).item())
    if prediction != int(label):
        raise RuntimeError(f"rank {rank} is no longer clean-correct")
    clean_set = sorted(int(value) for value in route.indices[0].tolist())
    if clean_set != sorted(selection["exact_feasible_pairs"][0]):
        raise RuntimeError("frozen invariant set disagrees with clean routing")

    bracket = {
        "lower": epsilon,
        "upper": epsilon,
        "lower_status": "stable",
        "upper_status": "stable",
        "bisection_complete": True,
        "termination": "registered_no_boundary_cap",
    }
    gate_config = {
        "candidate_query_timeout": config["candidate_query_timeout"],
        "support": config["support"],
        "solver": config["solver"],
        "matched_no_support_solve": False,
        "return_witness_tensor": True,
    }
    gate_started = time.monotonic()
    gate = diagnose_radius(
        model=model,
        x=x,
        label=int(label),
        clean_prediction=prediction,
        clean_set=clean_set,
        epsilon=epsilon,
        epsilon_multiplier=1.0,
        bracket=bracket,
        config=gate_config,
    )
    gate_seconds = time.monotonic() - gate_started
    feasible_pairs = gate.get("feasible_route_sets") or []
    if len(feasible_pairs) != 1 or sorted(feasible_pairs[0]) != clean_set:
        raise RuntimeError("fresh exact analysis did not reproduce route invariance")
    gate_candidate = gate.pop("_counterexample_input", None)
    final_status, final_reason = gate["status"], gate["reason"]
    full_witness = bool(gate["full_model_witness_valid"])
    witness_path = witness_hash = None
    f0 = None
    f0_seconds = 0.0
    if final_status == "SAFE":
        final_reason = "SAFE_GATE_ELIMINATION"
    elif final_status == "UNSAFE":
        if gate_candidate is None or not full_witness:
            raise RuntimeError("gate UNSAFE lacks a validated concrete input")
        witness_path, witness_hash = _save_gate_witness(
            work_dir,
            rank,
            gate_candidate,
            {
                "sample_rank": rank,
                "dataset_index": index,
                "epsilon": epsilon,
                "clean_prediction": prediction,
                "counterexample_prediction": gate["counterexample_prediction"],
                "counterexample_topk_set": gate["counterexample_topk_set"],
            },
        )
        final_reason = "UNSAFE_FULL_FORWARD"
    elif final_reason in SEMANTIC_REASONS:
        f0_started = time.monotonic()
        parent_id = hashlib.sha256(
            f"route-invariance:{runtime['source_config_sha256']}:{rank}:{epsilon:.17g}".encode()
        ).hexdigest()
        f0 = _run_parent_row(
            selection={
                "parent_row_id": parent_id,
                "parent_line_number": rank + 1,
                "parent_row_sha256": selection["parent_row_sha256"],
                "parent_artifact_sha256": config["parent_results_sha256"],
                "parent": {
                    "sample_rank": rank,
                    "dataset_index": index,
                    "epsilon": epsilon,
                    "epsilon_multiplier": 1.0,
                    "clean_prediction": prediction,
                    "clean_topk_set": clean_set,
                    "status": gate["status"],
                    "reason": gate["reason"],
                },
            },
            model=model,
            dataset=dataset,
            output_dir=work_dir,
            config=config["f0"],
        )
        f0_seconds = time.monotonic() - f0_started
        final_status, final_reason = f0["status"], f0["reason"]
        full_witness = bool(f0["full_model_witness_valid"])
        witness_path, witness_hash = f0["witness_path"], f0["witness_sha256"]
    elapsed = time.monotonic() - started
    return {
        "sample_rank": rank,
        "dataset_index": index,
        "endpoint_kind": selection["endpoint_kind"],
        "epsilon": epsilon,
        "route_precondition_status": "INVARIANT",
        "precondition_source": selection["precondition_source"],
        "exact_feasible_pairs": feasible_pairs,
        "exact_feasible_pair_count": 1,
        "baseline_status": final_status,
        "baseline_reason": final_reason,
        "route_a_status": final_status,
        "route_a_reason": final_reason,
        "route_a_only_safe": False,
        "baseline_seconds": elapsed,
        "route_a_seconds": elapsed,
        "route_a_source": "shared_fresh_invariant_endpoint_solve",
        "gate": gate,
        "f0": f0,
        "f0_invoked": f0 is not None,
        "full_model_witness_valid": full_witness,
        "witness_path": witness_path,
        "witness_sha256": witness_hash,
        "gate_seconds": gate_seconds,
        "f0_seconds": f0_seconds,
        "total_seconds": elapsed,
    }


def _reused_unstable_row(selection: dict[str, Any]) -> dict[str, Any]:
    parent = selection["parent"]
    closure = selection.get("closure")
    candidate_seconds = float(parent.get("candidate_seconds", 0.0))
    if not candidate_seconds and closure:
        candidate_seconds = float((closure.get("f0") or {}).get("candidate_seconds", 0.0))
    return {
        "sample_rank": int(selection["sample_rank"]),
        "dataset_index": int(selection["dataset_index"]),
        "endpoint_kind": selection["endpoint_kind"],
        "epsilon": float(selection["epsilon"]),
        "route_precondition_status": "UNSTABLE",
        "precondition_source": selection["precondition_source"],
        "exact_feasible_pairs": selection["exact_feasible_pairs"],
        "exact_feasible_pair_count": len(selection["exact_feasible_pairs"]),
        "baseline_status": "UNKNOWN",
        "baseline_reason": "ROUTE_INVARIANCE_PRECONDITION_FAILED",
        "route_a_status": parent["status"],
        "route_a_reason": parent["reason"],
        "route_a_only_safe": parent["status"] == "SAFE",
        "baseline_seconds": candidate_seconds,
        "route_a_seconds": float(parent.get("total_seconds", 0.0)),
        "route_a_source": "frozen_confirmatory_artifact",
        "route_a_followup_status": closure.get("status") if closure else None,
        "route_a_followup_reason": closure.get("reason") if closure else None,
        "route_a_followup_seconds": closure.get("total_seconds") if closure else None,
        "parent_line_number": selection["parent_line_number"],
        "parent_row_sha256": selection["parent_row_sha256"],
        "full_model_witness_valid": bool(parent.get("full_model_witness_valid")),
        "witness_path": parent.get("witness_path"),
        "witness_sha256": parent.get("witness_sha256"),
        "total_seconds": 0.0,
    }


def _summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    baseline = Counter(row["baseline_status"] for row in rows)
    route_a = Counter(row["route_a_status"] for row in rows)
    followup_solved = sum(
        (row.get("route_a_followup_status") or row["route_a_status"])
        in {"SAFE", "UNSAFE"}
        for row in rows
    )
    return {
        "rows": len(rows),
        "samples": len({int(row["sample_rank"]) for row in rows}),
        "endpoint_counts": dict(Counter(row["endpoint_kind"] for row in rows)),
        "route_precondition_counts": dict(
            Counter(row["route_precondition_status"] for row in rows)
        ),
        "baseline_status_counts": dict(baseline),
        "route_a_status_counts": dict(route_a),
        "baseline_solved": baseline["SAFE"] + baseline["UNSAFE"],
        "route_a_solved": route_a["SAFE"] + route_a["UNSAFE"],
        "route_a_followup_solved": followup_solved,
        "coverage_difference": (
            route_a["SAFE"] + route_a["UNSAFE"]
            - baseline["SAFE"] - baseline["UNSAFE"]
        ),
        "route_a_only_safe": sum(bool(row["route_a_only_safe"]) for row in rows),
        "route_a_only_safe_ranks": [
            int(row["sample_rank"]) for row in rows if row["route_a_only_safe"]
        ],
        "baseline_seconds": sum(float(row["baseline_seconds"]) for row in rows),
        "route_a_seconds": sum(float(row["route_a_seconds"]) for row in rows),
        "runtime_interpretation": (
            "artifact-backed endpoint accounting; invariant rows use one shared "
            "fresh downstream solve and unstable baseline rows stop after the "
            "frozen exact candidate stage"
        ),
    }


def _run(config_path: Path) -> dict[str, Any]:
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError(f"must run with {EXPECTED_PYTHON}, got {sys.executable}")
    if _git_value("rev-parse", "--abbrev-ref", "HEAD") != "feat/moe-route-verification":
        raise RuntimeError("route-invariance baseline requires the feature branch")
    config_path = _inside(config_path, PROJECT_ROOT)
    config = json.load(config_path.open(encoding="utf-8"))
    if config["numerical_safety"] != hz_numerical_policy_manifest():
        raise RuntimeError("tracked numerical policy differs from implementation")
    output_dir = _inside(Path(config["output_dir"]), WRITE_ROOT)
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    selected = _load_selection(config)
    checkpoint = _inside(Path(config["checkpoint"]), WRITE_ROOT)
    if _sha256(checkpoint) != config["checkpoint_sha256"]:
        raise RuntimeError("checkpoint hash changed")
    runtime = {
        "git_head": _git_value("rev-parse", "HEAD"),
        "source_config_sha256": _sha256(config_path),
        "checkpoint_sha256": _sha256(checkpoint),
        "python": sys.executable,
        "torchvision_root": str(get_torchvision_data_root()),
    }
    _write_json(output_dir / "config.json", {**config, "runtime": runtime})
    paths = {
        "jsonl": output_dir / "results.jsonl",
        "csv": output_dir / "results.csv",
        "log": output_dir / "baseline.log",
        "summary": output_dir / "summary.json",
    }
    rows: list[dict[str, Any]] = []
    with (
        paths["jsonl"].open("x", encoding="utf-8") as json_handle,
        paths["csv"].open("x", newline="", encoding="utf-8") as csv_handle,
        paths["log"].open("x", encoding="utf-8") as log_handle,
    ):
        writer = csv.DictWriter(csv_handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        csv_handle.flush()
        os.fsync(csv_handle.fileno())
        for position, selection in enumerate(selected, 1):
            if selection["route_precondition_status"] == "UNSTABLE":
                row = _reused_unstable_row(selection)
            else:
                row = _run_boundary_with_deadline(
                    model=None,
                    dataset=None,
                    selection=selection,
                    stage_dir=output_dir,
                    runtime=runtime,
                    config=config,
                    row_runner=_run_invariant_row,
                )
                if row.get("reason") == "INSTANCE_HARD_DEADLINE":
                    row.update(
                        {
                            "endpoint_kind": selection["endpoint_kind"],
                            "epsilon": selection["epsilon"],
                            "route_precondition_status": "INVARIANT",
                            "exact_feasible_pair_count": 1,
                            "baseline_status": "TIMEOUT",
                            "baseline_reason": "INSTANCE_HARD_DEADLINE",
                            "route_a_status": "TIMEOUT",
                            "route_a_reason": "INSTANCE_HARD_DEADLINE",
                            "route_a_only_safe": False,
                            "baseline_seconds": row["total_seconds"],
                            "route_a_seconds": row["total_seconds"],
                            "route_a_source": "shared_fresh_invariant_endpoint_solve",
                        }
                    )
            rows.append(row)
            _append_json(json_handle, row)
            writer.writerow(_csv_row(row))
            csv_handle.flush()
            os.fsync(csv_handle.fileno())
            log_handle.write(
                f"ROW {position}/100 rank={row['sample_rank']} "
                f"baseline={row['baseline_status']} route_a={row['route_a_status']}\n"
            )
            log_handle.flush()
            os.fsync(log_handle.fileno())
    summary = _summary(rows)
    _write_json(paths["summary"], summary)
    return {
        "output_dir": str(output_dir),
        "summary": summary,
        "manifest": {
            str(path.relative_to(output_dir)): _sha256(path)
            for path in sorted(output_dir.rglob("*"))
            if path.is_file()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    print(json.dumps(_run(Path(args.config)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
