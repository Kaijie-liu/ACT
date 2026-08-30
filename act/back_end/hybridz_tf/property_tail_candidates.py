"""Heuristic-only negative-slope candidates for property-tail upper planes.

This module has deliberately no verdict authority.  It minimizes a floating
point proxy for the HybridZ cube support of final property rows by choosing a
lower ReLU facet ``relu(x) >= alpha*x`` for negative property coefficients,
with ``alpha`` projected to ``[0, 1]``.  The caller must reconstruct every
candidate plane with exact endpoint auditing and may use the result only after
an independent soundness-preserving construction.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time
from typing import Any, Dict, Optional

import numpy as np
import scipy.sparse as sp
import torch


@dataclass(frozen=True)
class PropertyTailAlphaCandidates:
    """Candidate alpha matrix plus an auditable, non-authoritative receipt."""

    alpha: np.ndarray
    receipt: Dict[str, Any]


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _zero_result(
    *,
    shape: tuple[int, int],
    status: str,
    started: float,
    receipt: Dict[str, Any],
) -> PropertyTailAlphaCandidates:
    alpha = np.zeros(shape, dtype=np.float64)
    result_receipt = {
        **receipt,
        "status": str(status),
        "completed_steps": 0,
        "elapsed_seconds": float(max(0.0, time.monotonic() - started)),
        "eligible_alpha": int(receipt.get("eligible_alpha", 0)),
        "nonzero_alpha": 0,
        "alpha_max": 0.0,
        "alpha_sha256": _sha256_array(alpha),
    }
    return PropertyTailAlphaCandidates(alpha=alpha, receipt=result_receipt)


def optimize_property_tail_negative_alpha(
    *,
    preactivation_center: np.ndarray,
    preactivation_generators: sp.csr_matrix,
    preactivation_error: np.ndarray,
    baseline_planes: np.ndarray,
    baseline_intercepts: np.ndarray,
    property_coefficients: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    steps: int,
    time_limit: float,
    learning_rate: float = 0.08,
    max_cells: int = 50_000_000,
    deadline: Optional[float] = None,
    device: Optional[str] = None,
) -> PropertyTailAlphaCandidates:
    """Propose per-rival negative ReLU slopes with projected Adam.

    The optimized proxy for each rival is

    ``d@c + k + ||d@G||_1 + |d|@err``,

    where ``d`` is the baseline plane plus ``q*alpha`` on approximately
    negative, cube-unstable coordinates.  Rounding allowances introduced by
    the production affine builder and the exact candidate intercept are not
    modeled here.  Consequently the returned values are candidates only; the
    caller must audit and compare the resulting sound expressions.
    """

    started = time.monotonic()
    requested_steps = int(steps)
    local_seconds = float(time_limit)
    lr = float(learning_rate)
    cell_limit = int(max_cells)
    if requested_steps < 0:
        raise ValueError("property-tail alpha steps must be nonnegative")
    if not math.isfinite(local_seconds) or local_seconds < 0.0:
        raise ValueError(
            "property-tail alpha time limit must be finite and nonnegative"
        )
    if not math.isfinite(lr) or lr <= 0.0:
        raise ValueError(
            "property-tail alpha learning rate must be finite and positive"
        )
    if cell_limit <= 0:
        raise ValueError("property-tail alpha max_cells must be positive")
    if deadline is not None and not math.isfinite(float(deadline)):
        raise ValueError("property-tail alpha deadline must be finite")

    center = np.asarray(preactivation_center, dtype=np.float64).reshape(-1)
    generators = sp.csr_matrix(
        preactivation_generators, dtype=np.float64
    )
    generators.sum_duplicates()
    generators.sort_indices()
    error = np.asarray(preactivation_error, dtype=np.float64).reshape(-1)
    planes = np.asarray(baseline_planes, dtype=np.float64)
    intercepts = np.asarray(
        baseline_intercepts, dtype=np.float64
    ).reshape(-1)
    q = np.asarray(property_coefficients, dtype=np.float64)
    lower_array = np.asarray(lower, dtype=np.float64).reshape(-1)
    upper_array = np.asarray(upper, dtype=np.float64).reshape(-1)

    if (
        planes.ndim != 2
        or q.shape != planes.shape
        or intercepts.size != planes.shape[0]
        or center.size != planes.shape[1]
        or error.size != center.size
        or lower_array.size != center.size
        or upper_array.size != center.size
        or generators.shape[0] != center.size
        or np.any(lower_array > upper_array)
        or np.any(error < 0.0)
        or not np.all(np.isfinite(center))
        or not np.all(np.isfinite(generators.data))
        or not np.all(np.isfinite(error))
        or not np.all(np.isfinite(planes))
        or not np.all(np.isfinite(intercepts))
        or not np.all(np.isfinite(q))
        or not np.all(np.isfinite(lower_array))
        or not np.all(np.isfinite(upper_array))
    ):
        raise ValueError(
            "property-tail alpha candidate inputs are malformed or non-finite"
        )

    rivals, width = planes.shape
    n_cont = int(generators.shape[1])
    support_cells = int(rivals * n_cont)
    work_cells = int(
        support_cells + generators.nnz + rivals * width
    )
    eligible = (
        (q < 0.0)
        & (lower_array.reshape(1, -1) < 0.0)
        & (upper_array.reshape(1, -1) > 0.0)
    )
    eligible_count = int(np.count_nonzero(eligible))
    base_receipt: Dict[str, Any] = {
        "schema": "property_tail_negative_alpha_candidates_v1",
        "proof_authority": False,
        "candidate_only": True,
        "requested_steps": requested_steps,
        "time_limit_seconds": local_seconds,
        "learning_rate": lr,
        "max_cells": cell_limit,
        "work_cells": work_cells,
        "support_cells": support_cells,
        "estimated_peak_bytes": int(
            8
            * (
                6 * support_cells
                + 4 * int(generators.nnz)
                + 12 * rivals * width
            )
        ),
        "rivals": int(rivals),
        "preactivation_rows": int(width),
        "continuous_columns": n_cont,
        "generator_nnz": int(generators.nnz),
        "eligible_alpha": eligible_count,
        "property_coefficients_sha256": _sha256_array(q),
        "baseline_planes_sha256": _sha256_array(planes),
    }
    shape = (rivals, width)
    if requested_steps == 0 or local_seconds == 0.0:
        return _zero_result(
            shape=shape,
            status="disabled",
            started=started,
            receipt=base_receipt,
        )
    if eligible_count == 0:
        return _zero_result(
            shape=shape,
            status="no_eligible_negative_unstable",
            started=started,
            receipt=base_receipt,
        )
    if work_cells > cell_limit:
        return _zero_result(
            shape=shape,
            status="max_cells_fallback_baseline",
            started=started,
            receipt=base_receipt,
        )

    # Candidate work must not consume the shared proof-reconstruction tail.
    # The reserve is intentionally modest and is also bounded by 10% of the
    # local candidate allowance.
    audit_reserve = min(0.25, 0.1 * local_seconds)
    stop_at = started + local_seconds
    if deadline is not None:
        stop_at = min(stop_at, float(deadline) - audit_reserve)
    if time.monotonic() >= stop_at:
        return _zero_result(
            shape=shape,
            status="deadline_fallback_baseline",
            started=started,
            receipt=base_receipt,
        )

    if device is None or str(device).lower() == "auto":
        torch_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        torch_device = torch.device(str(device))
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        return _zero_result(
            shape=shape,
            status="cuda_unavailable_fallback_baseline",
            started=started,
            receipt={**base_receipt, "device": str(torch_device)},
        )

    receipt = {
        **base_receipt,
        "device": str(torch_device),
        "torch_version": str(torch.__version__),
        "cuda_version": (
            str(torch.version.cuda)
            if torch_device.type == "cuda" else None
        ),
        "gpu_name": (
            str(torch.cuda.get_device_name(torch_device))
            if torch_device.type == "cuda" else None
        ),
        "deterministic_algorithms_requested": bool(
            torch_device.type == "cuda"
        ),
        "tf32_disabled": bool(torch_device.type == "cuda"),
        "seed_points": [0.0, 0.5, 1.0],
    }
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = (
        torch.is_deterministic_algorithms_warn_only_enabled()
    )
    previous_tf32 = None
    if torch_device.type == "cuda":
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
    try:
        generator_transpose = generators.transpose().tocoo()
        generator_indices = np.vstack(
            (generator_transpose.row, generator_transpose.col)
        ).astype(np.int64, copy=False)
        center_t = torch.as_tensor(
            center, dtype=torch.float64, device=torch_device
        )
        generator_indices_t = torch.as_tensor(
            generator_indices, dtype=torch.int64, device=torch_device
        )
        generator_values_t = torch.as_tensor(
            generator_transpose.data,
            dtype=torch.float64,
            device=torch_device,
        )
        generators_transpose_t = torch.sparse_coo_tensor(
            generator_indices_t,
            generator_values_t,
            size=(n_cont, width),
            dtype=torch.float64,
            device=torch_device,
        ).coalesce()
        if time.monotonic() >= stop_at:
            return _zero_result(
                shape=shape,
                status="deadline_fallback_baseline",
                started=started,
                receipt=receipt,
            )
        error_t = torch.as_tensor(
            error, dtype=torch.float64, device=torch_device
        )
        planes_t = torch.as_tensor(
            planes, dtype=torch.float64, device=torch_device
        )
        intercepts_t = torch.as_tensor(
            intercepts, dtype=torch.float64, device=torch_device
        )
        q_update_t = torch.as_tensor(
            np.where(eligible, q, 0.0),
            dtype=torch.float64,
            device=torch_device,
        )
        eligible_t = torch.as_tensor(
            eligible, dtype=torch.bool, device=torch_device
        )
        alpha_t = torch.where(
            eligible_t,
            torch.full(
                shape, 0.5, dtype=torch.float64, device=torch_device
            ),
            torch.zeros(shape, dtype=torch.float64, device=torch_device),
        ).requires_grad_()
        zero_alpha_t = torch.zeros(
            shape, dtype=torch.float64, device=torch_device
        )
        optimizer = torch.optim.Adam([alpha_t], lr=lr)

        def proxy(alpha_value: torch.Tensor) -> torch.Tensor:
            d = planes_t + q_update_t * alpha_value
            transformed = torch.sparse.mm(
                generators_transpose_t, d.transpose(0, 1)
            ).transpose(0, 1)
            return (
                d @ center_t
                + intercepts_t
                + torch.sum(torch.abs(transformed), dim=1)
                + torch.abs(d) @ error_t
            )

        with torch.no_grad():
            baseline_objective_t = proxy(zero_alpha_t)
            if not bool(torch.all(torch.isfinite(baseline_objective_t))):
                raise FloatingPointError(
                    "baseline property-tail proxy is non-finite"
                )
            best_objective_t = baseline_objective_t.clone()
            best_alpha_t = zero_alpha_t.clone()
            for seed_alpha_t in (
                alpha_t.detach(),
                eligible_t.to(dtype=torch.float64),
            ):
                seed_objective_t = proxy(seed_alpha_t)
                if not bool(torch.all(torch.isfinite(seed_objective_t))):
                    raise FloatingPointError(
                        "seeded property-tail proxy is non-finite"
                    )
                seed_improved_t = seed_objective_t < best_objective_t
                best_objective_t = torch.where(
                    seed_improved_t,
                    seed_objective_t,
                    best_objective_t,
                )
                best_alpha_t = torch.where(
                    seed_improved_t.reshape(-1, 1),
                    seed_alpha_t,
                    best_alpha_t,
                )

        completed_steps = 0
        with torch.enable_grad():
            for _step in range(requested_steps):
                if torch_device.type == "cuda":
                    torch.cuda.synchronize(torch_device)
                if time.monotonic() >= stop_at:
                    break
                optimizer.zero_grad(set_to_none=True)
                objective_t = proxy(alpha_t)
                if not bool(torch.all(torch.isfinite(objective_t))):
                    raise FloatingPointError(
                        "property-tail alpha proxy became non-finite"
                    )
                torch.sum(objective_t).backward()
                if alpha_t.grad is None or not bool(
                    torch.all(torch.isfinite(alpha_t.grad))
                ):
                    raise FloatingPointError(
                        "property-tail alpha gradient is non-finite"
                    )
                optimizer.step()
                if torch_device.type == "cuda":
                    torch.cuda.synchronize(torch_device)
                if time.monotonic() >= stop_at:
                    # The iterate that crossed the wall is not retained.
                    break
                with torch.no_grad():
                    alpha_t.clamp_(0.0, 1.0)
                    alpha_t.masked_fill_(~eligible_t, 0.0)
                    candidate_objective_t = proxy(alpha_t)
                    if not bool(torch.all(torch.isfinite(candidate_objective_t))):
                        raise FloatingPointError(
                            "property-tail candidate proxy is non-finite"
                        )
                    if time.monotonic() >= stop_at:
                        break
                    improved_t = candidate_objective_t < best_objective_t
                    best_objective_t = torch.where(
                        improved_t, candidate_objective_t, best_objective_t
                    )
                    best_alpha_t = torch.where(
                        improved_t.reshape(-1, 1),
                        alpha_t.detach(),
                        best_alpha_t,
                    )
                completed_steps += 1

        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        best_alpha = (
            best_alpha_t.detach().cpu().numpy().astype(np.float64, copy=True)
        )
        baseline_objective = (
            baseline_objective_t.detach().cpu().numpy().astype(
                np.float64, copy=True
            )
        )
        best_objective = (
            best_objective_t.detach().cpu().numpy().astype(
                np.float64, copy=True
            )
        )
        best_alpha[~eligible] = 0.0
        if (
            not np.all(np.isfinite(best_alpha))
            or np.any(best_alpha < 0.0)
            or np.any(best_alpha > 1.0)
            or not np.all(best_objective <= baseline_objective)
        ):
            raise FloatingPointError(
                "property-tail candidate postcondition failed"
            )
        improved = best_objective < baseline_objective
        result_receipt = {
            **receipt,
            "status": (
                "optimized"
                if completed_steps == requested_steps
                else "time_limit_partial"
                if completed_steps > 0
                else "deadline_fallback_baseline"
            ),
            "completed_steps": int(completed_steps),
            "elapsed_seconds": float(
                max(0.0, time.monotonic() - started)
            ),
            "proxy_improved_rivals": int(np.count_nonzero(improved)),
            "baseline_proxy_upper_min": float(
                np.min(baseline_objective)
            ),
            "baseline_proxy_upper_max": float(
                np.max(baseline_objective)
            ),
            "best_proxy_upper_min": float(np.min(best_objective)),
            "best_proxy_upper_max": float(np.max(best_objective)),
            "proxy_total_improvement": float(
                np.sum(baseline_objective - best_objective)
            ),
            "sparse_spmm": True,
            "nonzero_alpha": int(np.count_nonzero(best_alpha)),
            "alpha_max": float(np.max(best_alpha)),
            "alpha_sha256": _sha256_array(best_alpha),
        }
        return PropertyTailAlphaCandidates(
            alpha=best_alpha,
            receipt=result_receipt,
        )
    except (RuntimeError, FloatingPointError) as exc:
        if torch_device.type == "cuda":
            try:
                torch.cuda.synchronize(torch_device)
            except RuntimeError:
                pass
        return _zero_result(
            shape=shape,
            status="error_fallback_baseline",
            started=started,
            receipt={
                **receipt,
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            },
        )
    finally:
        if torch_device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = bool(previous_tf32)
            torch.use_deterministic_algorithms(
                previous_deterministic,
                warn_only=previous_warn_only,
            )


__all__ = [
    "PropertyTailAlphaCandidates",
    "optimize_property_tail_negative_alpha",
]
