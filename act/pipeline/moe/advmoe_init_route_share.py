"""Measure full-test AdvMoE init routes and construct cross-route line witnesses."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from act.pipeline.moe.advmoe_adapter import construct_official_init, state_dict_sha256
from act.pipeline.moe.advmoe_router_bracket import load_cifar10_test_archive
from act.pipeline.moe.published_moe_router_gradient_audit import (
    MOE_ROOT,
    PROJECT_ROOT,
    _inside,
    _sha256,
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refuses to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(values.min()),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "q75": float(np.quantile(values, 0.75)),
        "maximum": float(values.max()),
        "standard_deviation": float(values.std(ddof=0)),
    }


def _forward_scores(
    router: torch.nn.Module,
    inputs: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = torch.from_numpy(inputs[start : start + batch_size]).to(device)
            rows.append(router(batch).detach().cpu().numpy())
    return np.concatenate(rows, axis=0)


def _cross_route_line_bracket(
    router: torch.nn.Module,
    start: np.ndarray,
    partner: np.ndarray,
    start_route: int,
    *,
    device: torch.device,
    iterations: int,
) -> tuple[dict[str, Any], np.ndarray]:
    x0 = torch.from_numpy(start[None]).to(device)
    x1 = torch.from_numpy(partner[None]).to(device)
    low_t, high_t = 0.0, 1.0
    low_x, high_x = x0.clone(), x1.clone()
    with torch.no_grad():
        if int(router(low_x).argmax(dim=1).item()) != int(start_route):
            raise RuntimeError("line start route changed before bisection")
        if int(router(high_x).argmax(dim=1).item()) == int(start_route):
            raise RuntimeError("line partner is not on the opposite route")
        representation_limited = False
        executed = 0
        for _ in range(int(iterations)):
            midpoint_t = (low_t + high_t) / 2.0
            midpoint_x = x0 + midpoint_t * (x1 - x0)
            if torch.equal(midpoint_x, low_x) or torch.equal(midpoint_x, high_x):
                representation_limited = True
                break
            midpoint_route = int(router(midpoint_x).argmax(dim=1).item())
            executed += 1
            if midpoint_route == int(start_route):
                low_t, low_x = midpoint_t, midpoint_x
            else:
                high_t, high_x = midpoint_t, midpoint_x
        low_scores = router(low_x).reshape(-1)
        high_scores = router(high_x).reshape(-1)
    low_route = int(low_scores.argmax().item())
    high_route = int(high_scores.argmax().item())
    if low_route != int(start_route) or high_route == int(start_route):
        raise RuntimeError("line bracket lost its concrete route endpoints")
    low_linf = float((low_x - x0).abs().max().item())
    high_linf = float((high_x - x0).abs().max().item())
    return (
        {
            "status": "CONCRETE_CROSS_ROUTE_LINE_BRACKET",
            "start_route": int(start_route),
            "opposite_route": high_route,
            "lower_t": low_t,
            "upper_t": high_t,
            "lower_linf": low_linf,
            "upper_linf": high_linf,
            "lower_linf_x255": low_linf * 255.0,
            "upper_linf_x255": high_linf * 255.0,
            "lower_route": low_route,
            "upper_route": high_route,
            "lower_signed_start_margin": float(
                (low_scores[start_route] - low_scores[1 - start_route]).item()
            ),
            "upper_signed_start_margin": float(
                (high_scores[start_route] - high_scores[1 - start_route]).item()
            ),
            "iterations_executed": executed,
            "representation_limited": representation_limited,
        },
        high_x.detach().cpu().numpy()[0],
    )


def run(config_path: Path) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = _inside(Path(config["output_dir"]), MOE_ROOT)
    if output_dir.exists():
        raise RuntimeError(f"refuses to reuse {output_dir}")
    archive = _inside(Path(config["dataset_archive"]), MOE_ROOT)
    if not torch.cuda.is_available():
        raise RuntimeError("registered full-test route diagnostic requires CUDA")
    free, total = torch.cuda.mem_get_info()
    if free < int(config["minimum_free_memory_bytes"]):
        raise RuntimeError("free CUDA memory is below the diagnostic resource gate")
    inputs, labels = load_cifar10_test_archive(archive)
    model, router, _moe_type = construct_official_init(int(config["model_seed"]))
    del model
    router_hash = state_dict_sha256(router)
    device = torch.device(str(config["device"]))
    router = router.to(device).eval()
    started = time.monotonic()
    scores = _forward_scores(
        router,
        inputs,
        device=device,
        batch_size=int(config["batch_size"]),
    )
    forward_seconds = time.monotonic() - started
    routes = scores.argmax(axis=1).astype(np.int64)
    signed_difference = scores[:, 0].astype(np.float64) - scores[:, 1].astype(np.float64)
    selected_margin = np.abs(signed_difference)
    counts = np.bincount(routes, minlength=2)
    signed_std = float(signed_difference.std(ddof=0))
    signed_mean = float(signed_difference.mean())

    line_rows: list[dict[str, Any]] = []
    line_witnesses: list[np.ndarray] = []
    partner_indices: list[int] = []
    for sample_index in [int(value) for value in config["line_sample_indices"]]:
        route = int(routes[sample_index])
        candidates = np.flatnonzero(routes != route)
        if not len(candidates):
            line_rows.append(
                {
                    "sample_index": sample_index,
                    "status": "NO_OPPOSITE_ROUTE_IN_OFFICIAL_TEST_SET",
                    "start_route": route,
                }
            )
            continue
        distances = np.max(
            np.abs(inputs[candidates] - inputs[sample_index]), axis=(1, 2, 3)
        )
        partner = int(candidates[int(np.argmin(distances))])
        row, witness = _cross_route_line_bracket(
            router,
            inputs[sample_index],
            inputs[partner],
            route,
            device=device,
            iterations=int(config["line_bisection_iterations"]),
        )
        row.update(
            {
                "sample_index": sample_index,
                "partner_index": partner,
                "partner_route": int(routes[partner]),
                "endpoint_distance_linf": float(distances.min()),
            }
        )
        line_rows.append(row)
        line_witnesses.append(witness)
        partner_indices.append(partner)

    output_dir.mkdir(parents=True)
    route_path = output_dir / "full_test_scores_routes.npz"
    np.savez_compressed(
        route_path,
        scores=scores,
        routes=routes,
        labels=labels,
        signed_difference=signed_difference,
        selected_margin=selected_margin,
    )
    witness_path = output_dir / "line_witnesses.npz"
    np.savez_compressed(
        witness_path,
        sample_indices=np.asarray(
            [row["sample_index"] for row in line_rows if "partner_index" in row],
            dtype=np.int64,
        ),
        partner_indices=np.asarray(partner_indices, dtype=np.int64),
        witnesses=np.asarray(line_witnesses, dtype=np.float32),
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETED_NOT_INDEPENDENTLY_AUDITED",
        "scope": config["scope"],
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "dataset": {
            "archive": str(archive),
            "sha256": _sha256(archive),
            "ordered_test_size": int(len(inputs)),
        },
        "router_sha256": router_hash,
        "resource_gate": {
            "free_bytes_before": int(free),
            "total_bytes": int(total),
        },
        "forward_seconds": forward_seconds,
        "route_counts": counts.astype(int).tolist(),
        "route_shares": (counts / len(routes)).tolist(),
        "signed_score_difference": {
            **_distribution(signed_difference),
            "absolute_mean_over_standard_deviation": (
                None if signed_std == 0.0 else abs(signed_mean) / signed_std
            ),
        },
        "selected_margin": {
            **_distribution(selected_margin),
            "mean_over_standard_deviation": (
                None
                if float(selected_margin.std(ddof=0)) == 0.0
                else float(selected_margin.mean() / selected_margin.std(ddof=0))
            ),
        },
        "line_partner_policy": config["line_partner_policy"],
        "line_rows": line_rows,
        "artifacts": {
            "scores_routes": {"path": str(route_path), "sha256": _sha256(route_path)},
            "line_witnesses": {"path": str(witness_path), "sha256": _sha256(witness_path)},
        },
        "interpretation": config["interpretation"],
    }
    _write_json(output_dir / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps({
        "route_counts": result["route_counts"],
        "route_shares": result["route_shares"],
        "signed_score_difference": result["signed_score_difference"],
        "selected_margin": result["selected_margin"],
        "line_rows": result["line_rows"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
