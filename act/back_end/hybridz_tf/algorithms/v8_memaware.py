#===- act/back_end/hybridz_tf/algorithms/v8_memaware.py - Memory-Aware v8 ReLU Dispatch -====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Pick ReLU encoding (eq_native / compact / triangle) and dtype
#   (float64 / float32) under estimated GPU memory budget. OOM-recovery
#   cascade falls back to lighter encodings on RuntimeError.
#
#===---------------------------------------------------------------------===#

"""Memory-aware v8 ReLU dispatch (Y2 Stage 5d port).

Faithful port of HyZor ``HybridZReLU._apply_relu_v8_memaware``
(HybridZVerifier.py:160). Picks between ``eq_native`` / ``compact`` /
``triangle`` and ``float64`` / ``float32`` based on free GPU memory and
estimated peak bytes per mode. Falls through to triangle/f32 if all
heavier modes overflow budget. Includes OOM recovery with retries.

Also exports ``estimate_intersect_box_peak_bytes`` used by the v8
caller to decide whether to skip ``intersect_box`` (which adds 2n new
constraint rows — can be huge for conv layers).
"""
from __future__ import annotations
import os
from typing import Optional, Tuple

import torch

from act.back_end.solver.solver_hz import HZono, _propagate_base
from act.back_end.hybridz_tf.tf_mlp import hz_apply_relu
from act.back_end.hybridz_tf.algorithms.relu_methods import (
    hz_apply_relu_compact, hz_apply_relu_triangle,
)


__all__ = [
    "is_oom_error",
    "fmt_gb",
    "get_v8_mem_budget_bytes",
    "estimate_relu_peak_bytes",
    "estimate_intersect_box_peak_bytes",
    "clone_hz_dtype",
    "apply_relu_v8_memaware",
]


def is_oom_error(e: Exception) -> bool:
    s = str(e).lower()
    return (
        ("out of memory" in s)
        or ("cuda oom" in s)
        or ("cublas_status_alloc_failed" in s)
    )


def fmt_gb(x: int) -> str:
    return f"{float(x) / (1024.0 ** 3):.2f}GB"


def estimate_relu_peak_bytes(hz: HZono, lb: torch.Tensor, ub: torch.Tensor,
                              mode: str, elem_bytes: int) -> int:
    """Conservative peak-bytes estimate for a ReLU mode. HyZor:60-100."""
    n = int(hz.c.shape[0])
    ng0 = int(hz.Gc.shape[1])
    nb0 = int(hz.Gb.shape[1])
    nc0 = int(hz.b.shape[0])
    lbv = lb.view(-1)
    ubv = ub.view(-1)
    k = int(((lbv < 0) & (ubv > 0)).sum().item())

    def b(rows: int, cols: int) -> int:
        return int(max(0, rows) * max(0, cols) * elem_bytes)

    if mode == "eq_native":
        ng = ng0 + 4 * k
        nb = nb0 + k
        r = nc0 + 3 * k
        out_live = b(n, ng) + b(n, nb) + b(n, 1)
        eq_live = b(3 * k, ng) + b(3 * k, nb) + b(3 * k, 1)
        old_ext = b(nc0, ng) + b(nc0, nb)
        cat_live = b(r, ng) + b(r, nb) + b(r, 1)
        peak = max(out_live + eq_live + old_ext, out_live + cat_live)
        return int(1.12 * peak + 256 * 1024 * 1024)
    if mode == "compact":
        ng = ng0 + k
        nb = nb0 + k
        r = nc0 + 2 * k
        out_live = b(n, ng) + b(n, nb) + b(n, 1)
        graph_live = b(2 * k, ng) + b(2 * k, nb) + b(2 * k, 1)
        old_ext = b(nc0, ng) + b(nc0, nb)
        cat_live = b(r, ng) + b(r, nb) + b(r, 1)
        peak = max(out_live + graph_live + old_ext, out_live + cat_live)
        return int(1.10 * peak + 160 * 1024 * 1024)
    # triangle
    ng = ng0 + k
    out_live = b(n, ng) + b(n, nb0) + b(n, 1)
    old_ext = b(nc0, ng) + b(nc0, nb0) + b(nc0, 1)
    peak = out_live + old_ext
    return int(1.08 * peak + 96 * 1024 * 1024)


def get_v8_mem_budget_bytes(hz: HZono) -> Optional[int]:
    """Free-bytes minus safety reserve. Returns None on CPU (no budget)."""
    dev = hz.c.device
    if dev.type != "cuda" or not torch.cuda.is_available():
        return None
    free_b = None
    try:
        free_b, _ = torch.cuda.mem_get_info(dev)
        free_b = int(free_b)
    except Exception:
        try:
            di = dev.index if dev.index is not None else torch.cuda.current_device()
            props = torch.cuda.get_device_properties(di)
            total_b = int(props.total_memory)
            reserved_b = int(torch.cuda.memory_reserved(di))
            allocated_b = int(torch.cuda.memory_allocated(di))
            free_b = max(0, total_b - max(reserved_b, allocated_b))
        except Exception:
            return None
    reserve_gb = 3.0
    env_reserve = os.environ.get("HYZOR_V8_MEM_RESERVE_GB", "").strip()
    if env_reserve:
        try:
            reserve_gb = max(0.5, float(env_reserve))
        except Exception:
            pass
    reserve_b = int(reserve_gb * (1024.0 ** 3))
    budget_b = max(256 * 1024 * 1024, free_b - reserve_b)
    env_budget = os.environ.get("HYZOR_V8_MEM_BUDGET_GB", "").strip()
    if env_budget:
        try:
            budget_b = min(
                budget_b,
                int(max(0.25, float(env_budget)) * (1024.0 ** 3)),
            )
        except Exception:
            pass
    return int(max(128 * 1024 * 1024, budget_b))


def estimate_intersect_box_peak_bytes(hz: HZono, elem_bytes: int) -> int:
    """Peak bytes for ``intersect_box`` (adds 2n new rows). HyZor:138."""
    n = int(hz.c.shape[0])
    ng = int(hz.Gc.shape[1])
    nb = int(hz.Gb.shape[1])
    nc = int(hz.b.shape[0])
    rows_new = nc + 2 * n
    bytes_ac = int(rows_new * ng * elem_bytes)
    bytes_ab = int(rows_new * nb * elem_bytes)
    bytes_b = int(rows_new * elem_bytes)
    return max(bytes_ac, bytes_ab, bytes_b)


def clone_hz_dtype(hz: HZono, dtype: torch.dtype) -> HZono:
    """Recast HZono to ``dtype`` (float32/float64). _base_ng/nb preserved."""
    if hz.c.dtype == dtype:
        return hz
    out = HZono(
        c=hz.c.to(dtype=dtype),
        Gc=hz.Gc.to(dtype=dtype),
        Gb=hz.Gb.to(dtype=dtype),
        Ac=hz.Ac.to(dtype=dtype),
        Ab=hz.Ab.to(dtype=dtype),
        b=hz.b.to(dtype=dtype),
        eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
    )
    _propagate_base(hz, out)
    return out


def apply_relu_v8_memaware(
    hz: HZono, lb: torch.Tensor, ub: torch.Tensor
) -> HZono:
    """Mem-aware dispatch of v8 ReLU encoding. HyZor:160 port.

    Picks the heaviest mode (eq_native > compact > triangle) and dtype
    (f64 > f32) that fits the current free GPU budget. Falls through to
    triangle/f32 if all heavier modes overflow. Includes OOM recovery
    with degradation cascade.
    """
    budget_b = get_v8_mem_budget_bytes(hz)
    dtype_run = hz.c.dtype
    mode = "eq_native"
    est_pick = None

    n = int(hz.c.shape[0])
    ng0 = int(hz.Gc.shape[1])
    lbv = lb.view(-1)
    ubv = ub.view(-1)
    k = int(((lbv < 0) & (ubv > 0)).sum().item())

    if budget_b is not None:
        elem_now = 8 if dtype_run == torch.float64 else 4
        eq_gc_only = int(n * (ng0 + 4 * k) * elem_now)
        cp_gc_only = int(n * (ng0 + k) * elem_now)
        cp32_gc_only = int(n * (ng0 + k) * 4)
        forbid_eq_native = bool(
            (n >= 32768 and k >= 2048)
            or (eq_gc_only > int(0.45 * budget_b))
        )
        forbid_compact64 = bool(
            (n >= 32768 and k >= 4096)
            or (cp_gc_only > int(0.40 * budget_b))
        )
        forbid_compact32 = bool(
            (n >= 49152 and k >= 8192)
            or (cp32_gc_only > int(0.58 * budget_b))
        )

        est64 = {
            "eq_native": estimate_relu_peak_bytes(hz, lb, ub, "eq_native", 8),
            "compact": estimate_relu_peak_bytes(hz, lb, ub, "compact", 8),
            "triangle": estimate_relu_peak_bytes(hz, lb, ub, "triangle", 8),
        }
        est32 = {
            "eq_native": estimate_relu_peak_bytes(hz, lb, ub, "eq_native", 4),
            "compact": estimate_relu_peak_bytes(hz, lb, ub, "compact", 4),
            "triangle": estimate_relu_peak_bytes(hz, lb, ub, "triangle", 4),
        }
        fit_ratio = {"eq_native": 0.78, "compact": 0.82, "triangle": 0.96}

        candidates = []
        if not forbid_eq_native:
            candidates.append(("eq_native", torch.float64, est64["eq_native"]))
        if not forbid_compact64:
            candidates.append(("compact", torch.float64, est64["compact"]))
        if not forbid_eq_native:
            candidates.append(("eq_native", torch.float32, est32["eq_native"]))
        if not forbid_compact32:
            candidates.append(("compact", torch.float32, est32["compact"]))
        candidates.extend([
            ("triangle", torch.float64, est64["triangle"]),
            ("triangle", torch.float32, est32["triangle"]),
        ])

        chosen = None
        for cand_mode, cand_dtype, cand_est in candidates:
            lim = int(float(fit_ratio[cand_mode]) * budget_b)
            if cand_est <= lim:
                chosen = (cand_mode, cand_dtype, cand_est)
                break
        if chosen is None:
            chosen = (
                "triangle",
                torch.float32 if dtype_run == torch.float64 else dtype_run,
                est32["triangle"] if dtype_run == torch.float64 else est64["triangle"],
            )
        mode, dtype_run, est_pick = chosen

    hz_run = clone_hz_dtype(hz, dtype_run)
    lb_run = lb.to(device=hz_run.c.device, dtype=hz_run.c.dtype)
    ub_run = ub.to(device=hz_run.c.device, dtype=hz_run.c.dtype)

    if (
        budget_b is not None
        and est_pick is not None
        and hz_run.c.device.type == "cuda"
        and est_pick >= int(0.60 * budget_b)
    ):
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # Optional override.
    if os.environ.get("HYZOR_FORCE_EQ_NATIVE", "0") == "1":
        mode = "eq_native"
        dtype_run = hz.c.dtype
        hz_run = clone_hz_dtype(hz, dtype_run)
        lb_run = lb.to(device=hz_run.c.device, dtype=hz_run.c.dtype)
        ub_run = ub.to(device=hz_run.c.device, dtype=hz_run.c.dtype)

    try:
        if mode == "eq_native":
            return hz_apply_relu(hz_run, external_bounds=(lb_run, ub_run))
        if mode == "compact":
            return hz_apply_relu_compact(hz_run)
        return hz_apply_relu_triangle(hz_run)
    except RuntimeError as e:
        if not is_oom_error(e):
            raise
        if hz_run.c.device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        # OOM degradation cascade
        try:
            return hz_apply_relu_compact(hz_run)
        except RuntimeError as e2:
            if is_oom_error(e2) and hz_run.c.device.type == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            hz32 = clone_hz_dtype(hz_run, torch.float32)
            try:
                return hz_apply_relu_compact(hz32)
            except RuntimeError as e3:
                if is_oom_error(e3) and hz32.c.device.type == "cuda":
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                return hz_apply_relu_triangle(hz32)
