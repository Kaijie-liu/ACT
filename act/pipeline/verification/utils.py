#===- act/pipeline/utils.py - Pipeline Testing Utilities ---------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Shared utilities for ACT pipeline testing framework. Provides common
#   utilities for parallel execution, performance profiling, logging,
#   and other shared functionality across the pipeline testing system.
#
#===---------------------------------------------------------------------===#

import time
import psutil
import logging
import functools
import threading
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Callable, Any, Dict, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
import torch
import torch.fx as fx
import torch.nn as nn

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Shape utilities
# -----------------------------------------------------------------------------

def _prod(shape_tail: Tuple[int, ...]) -> int:
    """Compute product of shape dimensions."""
    result = 1
    for s in shape_tail:
        result *= int(s)
    return result


def _normalize_tuple(val: Any, default: Tuple[int, int] = (1, 1)) -> Tuple[int, int]:
    """Normalize int, list, or tuple to 2-tuple for kernel_size/stride/padding.

    Lists are accepted because onnx2torch constructs nn.MaxPool2d/AvgPool2d with
    list-typed kernel_size / stride / padding (PyTorch's own constructors
    typically store them as tuples).
    """
    if isinstance(val, (list, tuple)):
        return tuple(val)
    return (val, val) if val is not None else default


def _assert_dag(preds: Dict[int, List[int]], succs: Dict[int, List[int]], n_layers: int) -> None:
    """Kahn's algorithm cycle check. Raises ValueError listing the cycle nodes.

    ``preds`` may legitimately contain duplicates when a layer has two
    operands fed by the same source (e.g. ``Mul(x, x)`` for ``Pow(x, 2)``);
    several TFs index ``preds[L][0]`` and ``preds[L][1]`` so the duplicate
    is positionally meaningful. The DAG check, however, is about the
    structure of the graph, not the multiplicity of edges — so we count
    **unique** predecessors for the in-degree, matching the deduplicated
    successor list built in ``_build_preds_succs``. Without this, a
    Mul(x,x) layer would have in_degree=2 against a succs entry of 1 and
    Kahn would falsely report a cycle.
    """
    if n_layers == 0:
        return
    in_degree = {i: len(set(preds.get(i, []))) for i in range(n_layers)}
    queue = [i for i in range(n_layers) if in_degree[i] == 0]
    visited = 0
    while queue:
        node = queue.pop(0)
        visited += 1
        for succ in succs.get(node, []):
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    if visited != n_layers:
        cycle_nodes = [i for i in range(n_layers) if in_degree[i] > 0]
        raise ValueError(f"Layer graph contains a cycle! Nodes: {cycle_nodes}")


def _normalize_axes(axes: Any, rank: int) -> List[int]:
    """Sort + dedupe + normalise possibly-negative axes against ``rank``."""
    return sorted({(a + rank) if a < 0 else a for a in (int(x) for x in axes)})


def _reduce_output_shape(input_shape: Tuple[int, ...], norm_axes: List[int],
                         keepdims: bool) -> Tuple[int, ...]:
    """Output shape of a reduce-along-axes op. ``norm_axes`` must be already-normalised."""
    if keepdims:
        return tuple(1 if i in norm_axes else int(d) for i, d in enumerate(input_shape))
    return tuple(int(d) for i, d in enumerate(input_shape) if i not in norm_axes) or (1,)


def _compute_slice_output_shape(
    input_shape: Tuple[int, ...],
    starts: List[int], ends: List[int],
    axes: List[int], steps: List[int],
) -> Tuple[Tuple[int, ...], List[int], List[int], List[int]]:
    """ONNX Slice output shape with negative-index + clamp + step semantics.

    Returns ``(output_shape, n_starts, n_ends, n_axes)`` with all indices
    normalised against ``input_shape``. Raises on zero step.
    """
    rank = len(input_shape)
    n_starts: List[int] = []
    n_ends: List[int] = []
    n_axes: List[int] = []
    output_shape = list(input_shape)
    for s, e, ax, st in zip(starts, ends, axes, steps):
        ax = int(ax) + rank if int(ax) < 0 else int(ax)
        dim = int(input_shape[ax])
        st = int(st)
        if st == 0:
            raise ValueError("OnnxSlice: zero step")
        s = int(s) + dim if int(s) < 0 else int(s)
        e = int(e) + dim if int(e) < 0 else int(e)
        if st > 0:
            s, e = min(max(s, 0), dim), min(max(e, 0), dim)
        else:
            s, e = min(max(s, -1), dim - 1), min(max(e, -1), dim - 1)
        output_shape[ax] = max(0, len(range(s, e, st)))
        n_starts.append(s); n_ends.append(e); n_axes.append(ax)
    return tuple(output_shape), n_starts, n_ends, n_axes


def _broadcast_const_to_size(const: torch.Tensor, size: int, dtype: torch.dtype) -> torch.Tensor:
    """Broadcast a constant tensor to a flat variable count.

    Handles scalars (numel==1), exact-size vectors, and integer-multiple
    repetitions (shape (C,) → flat (C*spatial,)). Empty constants fall back
    to zeros — only sound for ADD/SUB; MUL/DIV callers must reject empty.
    """
    flat = const.reshape(-1)
    if flat.numel() == 0:
        return torch.zeros(size, dtype=dtype)
    if flat.numel() == 1:
        return flat.expand(size).clone().to(dtype)
    if flat.numel() == size:
        return flat.clone().to(dtype)
    if size % flat.numel() == 0:
        return flat.repeat(size // flat.numel()).to(dtype)
    if flat.numel() % size == 0:
        return flat[:size].clone().to(dtype)
    raise ValueError(
        f"Cannot broadcast constant of shape {tuple(const.shape)} to flat size {size}"
    )


@dataclass(frozen=True)
class _DynIndexInterval:
    """Interval for a Slice index expression.

    ``base`` is a provenance token for the dynamic quantity. For cctsdb's
    supported pattern, both start and end trace to the same casted input
    coordinate and differ only by ``offset``. We require matching bases
    before declaring a constant window size; interval arithmetic alone is
    not enough because two unrelated variables can have intervals shifted
    by a constant.
    """

    lb: float
    ub: float
    base: Optional[str] = None
    offset: float = 0.0


def _const_tensor_for_node(self, node_name: str) -> Optional[torch.Tensor]:
    tensor = self._resolve_constant_tensor(node_name)
    if tensor is None:
        tensor = self._evaluate_constant_subgraph(node_name)
    return tensor.detach().clone() if isinstance(tensor, torch.Tensor) else None


def _fx_node_by_name(self, node_name: str):
    if self.fx_graph is None:
        return None
    return next((n for n in self.fx_graph.nodes if n.name == node_name), None)


def _constant_layer_tensor(self, node_name: str) -> Optional[torch.Tensor]:
    layer_id = self.node_to_layer_id.get(node_name)
    if layer_id is None or layer_id < 0 or layer_id >= len(self.layers):
        return None
    layer = self.layers[layer_id]
    if layer.kind != LayerKind.CONSTANT.value:
        return None
    value = layer.params.get("value")
    shape = layer.params.get("output_shape", layer.params.get("input_shape"))
    if not isinstance(value, torch.Tensor) or shape is None:
        return None
    return value.detach().clone().reshape(tuple(int(x) for x in shape))


def _input_bounds_flat(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    bounds = getattr(self, "input_bounds", None)
    if bounds is None:
        return None
    lb, ub = bounds
    if not isinstance(lb, torch.Tensor) or not isinstance(ub, torch.Tensor):
        return None
    if lb.numel() != ub.numel():
        return None
    return lb.detach().reshape(-1).to(torch.float64), ub.detach().reshape(-1).to(torch.float64)


def _constant_intervals_from_tensor(tensor: torch.Tensor) -> List[_DynIndexInterval]:
    vals = tensor.detach().cpu().reshape(-1).tolist()
    return [_DynIndexInterval(float(v), float(v), None, float(v)) for v in vals]


def _resolve_dyn_index_intervals(self, node_name: str) -> Optional[List[_DynIndexInterval]]:
    """Resolve a Slice starts/ends FX subgraph to conservative intervals.

    Supported deliberately-small grammar:
      constant
      Gather(input, const_idx)              -> input-box interval
      Cast(expr)                            -> conservative integer envelope
      Unsqueeze(expr)                       -> shape-only, same flat intervals
      Add/Sub(expr, constant)               -> offset shift

    Anything else returns None so the caller fails closed.
    """
    const = _const_tensor_for_node(self, node_name)
    if const is not None:
        return _constant_intervals_from_tensor(const)

    node = _fx_node_by_name(self, node_name)
    if node is None or node.op != "call_module":
        return None
    mod = self.modules.get(str(node.target))
    if mod is None:
        return None
    args = [a for a in node.args if hasattr(a, "name")]
    cls_name = type(mod).__name__

    if "Gather" in cls_name:
        if len(args) < 2:
            return None
        data_node, idx_node = args[0], args[1]
        if getattr(data_node, "op", None) != "placeholder":
            return None
        axis = int(getattr(mod, "_axis", getattr(mod, "axis", 0)))
        if axis != 0:
            return None
        idx_t = _const_tensor_for_node(self, idx_node.name)
        bounds = _input_bounds_flat(self)
        if idx_t is None or bounds is None:
            return None
        lb_flat, ub_flat = bounds
        out: List[_DynIndexInterval] = []
        for raw_idx in idx_t.detach().cpu().reshape(-1).tolist():
            idx = int(raw_idx)
            if idx < 0:
                idx += int(lb_flat.numel())
            if idx < 0 or idx >= int(lb_flat.numel()):
                return None
            out.append(
                _DynIndexInterval(
                    float(lb_flat[idx].item()),
                    float(ub_flat[idx].item()),
                    base=f"input:{idx}",
                    offset=0.0,
                )
            )
        return out

    if "Cast" in cls_name:
        if len(args) < 1:
            return None
        inner = _resolve_dyn_index_intervals(self, args[0].name)
        if inner is None:
            return None
        out = []
        for iv in inner:
            # ONNX / torch Cast from float to integer truncates toward zero.
            # Truncation is monotone, so applying it to the interval endpoints
            # gives a sound and tighter integer envelope than floor/ceil.
            lo = math.trunc(min(iv.lb, iv.ub))
            hi = math.trunc(max(iv.lb, iv.ub))
            out.append(
                _DynIndexInterval(
                    min(lo, hi),
                    max(lo, hi),
                    base=f"cast:{node_name}",
                    offset=0.0,
                )
            )
        return out

    if "Unsqueeze" in cls_name:
        if len(args) < 1:
            return None
        return _resolve_dyn_index_intervals(self, args[0].name)

    if "Concat" in cls_name:
        axis = int(getattr(mod, "axis", getattr(mod, "_axis", 0)))
        if axis not in (0, -1):
            return None
        out: List[_DynIndexInterval] = []
        for arg in args:
            part = _resolve_dyn_index_intervals(self, arg.name)
            if part is None:
                return None
            out.extend(part)
        return out

    if "BinaryMathOperation" in cls_name:
        if len(args) < 2:
            return None
        left = _resolve_dyn_index_intervals(self, args[0].name)
        right = _resolve_dyn_index_intervals(self, args[1].name)
        op_func = getattr(mod, "_operator", None)
        if op_func is None:
            op_func = getattr(mod, "math_op_function", None)
        op_name = getattr(op_func, "__name__", "").lower()
        if left is None and right is None:
            return None

        def _is_const_interval(xs: Optional[List[_DynIndexInterval]]) -> bool:
            return xs is not None and all(x.base is None for x in xs)

        if left is not None and right is not None:
            if _is_const_interval(left) and not _is_const_interval(right):
                dyn = right
                const_vals = [x.offset for x in left]
                dyn_is_left = False
            elif _is_const_interval(right) and not _is_const_interval(left):
                dyn = left
                const_vals = [x.offset for x in right]
                dyn_is_left = True
            else:
                # Do not attempt var-var index arithmetic in parser-time
                # envelopes. cctsdb only needs var +/- constant.
                return None
        elif left is not None:
            dyn = left
            dyn_is_left = True
            const_t = _const_tensor_for_node(self, args[1].name)
            if const_t is None:
                return None
            const_vals = const_t.detach().cpu().reshape(-1).tolist()
        else:
            dyn = right
            dyn_is_left = False
            const_t = _const_tensor_for_node(self, args[0].name)
            if const_t is None:
                return None
            const_vals = const_t.detach().cpu().reshape(-1).tolist()

        if dyn is None:
            return None
        if len(const_vals) == 1:
            const_vals = const_vals * len(dyn)
        if len(const_vals) != len(dyn):
            return None
        out = []
        for iv, c_raw in zip(dyn, const_vals):
            c = float(c_raw)
            if "add" in op_name:
                shift = c
            elif "sub" in op_name and dyn_is_left:
                shift = -c
            else:
                return None
            out.append(
                _DynIndexInterval(
                    iv.lb + shift,
                    iv.ub + shift,
                    iv.base,
                    iv.offset + shift,
                )
            )
        return out

    return None


def _normalised_start_bounds(raw_lb: int, raw_ub: int, dim: int, step: int) -> Optional[Tuple[int, int]]:
    if step <= 0:
        return None
    if raw_lb > raw_ub:
        return None
    # Enumerate the integer index interval to correctly handle negative
    # ONNX indices before clamping. Dynamic cctsdb ranges are small; a
    # broad range would be too loose and too expensive, so fail closed.
    if raw_ub - raw_lb > 10000:
        return None
    vals = []
    for raw in range(raw_lb, raw_ub + 1):
        v = raw + dim if raw < 0 else raw
        vals.append(min(max(v, 0), dim))
    return min(vals), max(vals)


def _try_emit_dynamic_slice_lut_bounds(
    self,
    node: fx.Node,
    args: List[fx.Node],
    axes: List[int],
    steps: List[int],
) -> bool:
    self._last_dynamic_slice_lut_error = None

    def fail(reason: str) -> bool:
        self._last_dynamic_slice_lut_error = reason
        return False

    starts_iv = _resolve_dyn_index_intervals(self, args[1].name)
    ends_iv = _resolve_dyn_index_intervals(self, args[2].name)
    if starts_iv is None or ends_iv is None or len(starts_iv) != len(ends_iv):
        return fail("dynamic Slice starts/ends are not in the supported bounded-index grammar")
    if len(starts_iv) != len(axes) or len(axes) != len(steps):
        return fail("dynamic Slice starts/ends rank does not match axes/steps")

    T = _constant_layer_tensor(self, args[0].name)
    if T is None:
        # ``_evaluate_constant_subgraph`` intentionally refuses placeholder-
        # rooted values. That is correct; dynamic Slice over arbitrary runtime
        # activations is outside this bounded-LUT subset.
        T = _const_tensor_for_node(self, args[0].name)
    if T is None:
        return fail("dynamic Slice over runtime activation is outside LUT_BOUNDS subset")
    T = T.detach().clone().to(self.dtype)
    rank = len(tuple(T.shape))
    if rank == 0:
        return fail("dynamic Slice over scalar tensor is unsupported")

    full_starts_lb = [0] * rank
    full_starts_ub = [0] * rank
    full_window = [int(d) for d in T.shape]
    full_steps = [1] * rank
    norm_axes: List[int] = []

    for s_iv, e_iv, ax_raw, step_raw in zip(starts_iv, ends_iv, axes, steps):
        ax = int(ax_raw) + rank if int(ax_raw) < 0 else int(ax_raw)
        if ax < 0 or ax >= rank:
            return fail(f"dynamic Slice axis {ax_raw} is outside rank-{rank} tensor")
        step = int(step_raw)
        if step <= 0:
            return fail("dynamic Slice LUT_BOUNDS only supports positive steps")
        if s_iv.base != e_iv.base:
            return fail("dynamic Slice start/end do not share the same index provenance")
        window_f = e_iv.offset - s_iv.offset
        if abs(window_f - round(window_f)) > 1e-9:
            return fail("dynamic Slice window size is not an integer constant")
        window = int(round(window_f))
        if window < 1:
            return fail("dynamic Slice window size must be positive")
        raw_lb = math.floor(min(s_iv.lb, s_iv.ub))
        raw_ub = math.ceil(max(s_iv.lb, s_iv.ub))
        nb = _normalised_start_bounds(raw_lb, raw_ub, int(T.shape[ax]), step)
        if nb is None:
            return fail("dynamic Slice start range is too broad or has unsupported step semantics")
        s_lb, s_ub = nb
        # All windows must be in-bounds after normalisation.
        if s_lb < 0 or s_ub + window * step > int(T.shape[ax]):
            return fail(
                "dynamic Slice start range can produce out-of-bounds / "
                "variable-shape windows; fixed-shape LUT_BOUNDS would be unsound"
            )
        full_starts_lb[ax] = s_lb
        full_starts_ub[ax] = s_ub
        full_window[ax] = window
        full_steps[ax] = step
        norm_axes.append(ax)

    # Avoid huge parser-time envelopes; cctsdb's supported ranges are small.
    cand = 1
    for lo, hi in zip(full_starts_lb, full_starts_ub):
        cand *= int(hi - lo + 1)
        if cand > 10000:
            return fail("dynamic Slice candidate lattice exceeds LUT_BOUNDS safety cap")

    from act.back_end.interval_tf.tf_mlp import precompute_lut_envelope

    lb_t, ub_t = precompute_lut_envelope(
        T,
        window_size=tuple(full_window),
        starts_lb=tuple(full_starts_lb),
        starts_ub=tuple(full_starts_ub),
        steps=tuple(full_steps),
    )
    out_vars = self._alloc_ids(int(lb_t.numel()) or 1)
    layer_id = self._add_layer(
        LayerKind.LUT_BOUNDS.value,
        {
            "lb": lb_t.to(torch.float64),
            "ub": ub_t.to(torch.float64),
            "input_shape": tuple(int(d) for d in T.shape),
            "output_shape": tuple(int(d) for d in lb_t.shape),
            "source_initializer_name": args[0].name,
            "window_starts_lb": tuple(full_starts_lb),
            "window_starts_ub": tuple(full_starts_ub),
            "window_steps": tuple(full_steps),
        },
        [],
        out_vars,
    )
    self.prev_out = out_vars
    self.shape = tuple(int(d) for d in lb_t.shape)
    self._register_node(node.name, layer_id)
    return True


@dataclass
class PerformanceMetrics:  # pragma: no cover
    """Performance metrics for validation operations."""
    execution_time: float
    peak_memory_mb: float
    cpu_usage_percent: float
    gpu_memory_mb: Optional[float] = None


@dataclass
class ParallelResult:  # pragma: no cover
    """Result from parallel execution."""
    results: List[Any]
    failed_tasks: List[Tuple[int, Exception]]
    total_time: float
    metrics: PerformanceMetrics


class PerformanceProfiler:  # pragma: no cover
    """Performance profiling utilities for validation operations."""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.start_memory: Optional[float] = None
        self.peak_memory: float = 0
        self.cpu_usage_samples: List[float] = []
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
    
    def start(self) -> None:
        """Start performance monitoring."""
        self.start_time = time.time()
        self.start_memory = self._get_memory_usage()
        self.peak_memory = self.start_memory
        self.cpu_usage_samples = []
        self._stop_monitoring.clear()
        
        # Start monitoring thread
        self._monitoring_thread = threading.Thread(target=self._monitor_resources)
        self._monitoring_thread.daemon = True
        self._monitoring_thread.start()
        
        logger.debug("Performance profiling started")
    
    def stop(self) -> PerformanceMetrics:
        """Stop monitoring and return performance metrics."""
        if self.start_time is None:
            raise RuntimeError("Profiler not started")
        
        # Stop monitoring thread
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=1.0)
        
        execution_time = time.time() - self.start_time
        avg_cpu_usage = sum(self.cpu_usage_samples) / len(self.cpu_usage_samples) if self.cpu_usage_samples else 0
        
        metrics = PerformanceMetrics(
            execution_time=execution_time,
            peak_memory_mb=self.peak_memory,
            cpu_usage_percent=avg_cpu_usage,
            gpu_memory_mb=self._get_gpu_memory() if torch.cuda.is_available() else None
        )
        
        logger.debug(f"Performance metrics: {metrics}")
        return metrics
    
    def _monitor_resources(self) -> None:
        """Monitor resource usage in background thread."""
        while not self._stop_monitoring.wait(0.1):  # Sample every 100ms
            try:
                # Monitor memory
                current_memory = self._get_memory_usage()
                self.peak_memory = max(self.peak_memory, current_memory)
                
                # Monitor CPU
                cpu_usage = psutil.cpu_percent(interval=None)
                self.cpu_usage_samples.append(cpu_usage)
                
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _get_gpu_memory(self) -> Optional[float]:
        """Get GPU memory usage in MB."""
        if not torch.cuda.is_available():
            return None
        try:
            return torch.cuda.memory_allocated() / 1024 / 1024
        except Exception:
            return None


@contextmanager
def profile_performance():  # pragma: no cover
    """Context manager for performance profiling."""
    profiler = PerformanceProfiler()
    profiler.start()
    try:
        yield profiler
    finally:
        metrics = profiler.stop()
        yield metrics


class ParallelExecutor:  # pragma: no cover
    """Utilities for parallel execution of validation tasks."""
    
    def __init__(self, max_workers: Optional[int] = None, timeout: Optional[float] = None):
        """
        Initialize parallel executor.
        
        Args:
            max_workers: Maximum number of worker threads
            timeout: Timeout for individual tasks in seconds
        """
        self.max_workers = max_workers or min(32, (psutil.cpu_count() or 1) + 4)
        self.timeout = timeout
    
    def execute_parallel(self, 
                        tasks: List[Callable],
                        task_args: Optional[List[Tuple]] = None,
                        task_kwargs: Optional[List[Dict]] = None) -> ParallelResult:
        """
        Execute tasks in parallel.
        
        Args:
            tasks: List of callable tasks to execute
            task_args: List of argument tuples for each task
            task_kwargs: List of keyword argument dicts for each task
            
        Returns:
            ParallelResult with results and performance metrics
        """
        if task_args is None:
            task_args = [() for _ in tasks]
        if task_kwargs is None:
            task_kwargs = [{} for _ in tasks]
        
        if len(tasks) != len(task_args) or len(tasks) != len(task_kwargs):
            raise ValueError("tasks, task_args, and task_kwargs must have same length")
        
        results = []
        failed_tasks = []
        
        with profile_performance() as profiler:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_index = {
                    executor.submit(task, *args, **kwargs): i
                    for i, (task, args, kwargs) in enumerate(zip(tasks, task_args, task_kwargs))
                }
                
                # Collect results
                for future in as_completed(future_to_index, timeout=self.timeout):
                    task_index = future_to_index[future]
                    try:
                        result = future.result()
                        results.append((task_index, result))
                    except Exception as e:
                        failed_tasks.append((task_index, e))
                        logger.error(f"Task {task_index} failed: {e}")
        
        # Sort results by original task index
        results.sort(key=lambda x: x[0])
        sorted_results = [result for _, result in results]
        
        metrics = profiler.stop()
        
        return ParallelResult(
            results=sorted_results,
            failed_tasks=failed_tasks,
            total_time=metrics.execution_time,
            metrics=metrics
        )
    
    def map_parallel(self, func: Callable, items: List[Any]) -> ParallelResult:
        """
        Apply function to list of items in parallel.
        
        Args:
            func: Function to apply to each item
            items: List of items to process
            
        Returns:
            ParallelResult with mapped results
        """
        tasks = [func for _ in items]
        task_args = [(item,) for item in items]
        
        return self.execute_parallel(tasks, task_args)


def print_memory_usage(prefix: str = "") -> None:  # pragma: no cover
    """Print current memory usage information."""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    memory_mb = memory_info.rss / 1024 / 1024
    cpu_percent = process.cpu_percent()
    
    gpu_info = ""
    if torch.cuda.is_available():
        gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        gpu_max_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        gpu_info = f", GPU: {gpu_memory_mb:.1f}MB (max: {gpu_max_mb:.1f}MB)"
    
    logger.info(f"{prefix}Memory: {memory_mb:.1f}MB, CPU: {cpu_percent:.1f}%{gpu_info}")


def clear_torch_cache() -> None:  # pragma: no cover
    """Clear PyTorch GPU cache if available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("Cleared PyTorch GPU cache")


def setup_logging(level: str = "INFO", format_str: Optional[str] = None) -> None:  # pragma: no cover
    """
    Setup logging configuration for the pipeline.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        format_str: Custom format string for log messages
    """
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Ensure log file goes to pipeline/log directory
    import os
    from pathlib import Path
    pipeline_dir = Path(__file__).parent
    log_dir = pipeline_dir / "log"
    log_dir.mkdir(exist_ok=True)
    log_file_path = log_dir / "pipeline_tests.log"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file_path)
        ]
    )
    
    # Reduce noise from some libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):  # pragma: no cover
    """
    Decorator to retry function on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier for delay
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
            
            raise last_exception
        
        return wrapper
    return decorator


def timeout_handler(timeout_seconds: float):  # pragma: no cover
    """
    Decorator to add timeout to function execution.
    
    Args:
        timeout_seconds: Timeout in seconds
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_signal_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds} seconds")
            
            # Set up signal handler
            old_handler = signal.signal(signal.SIGALRM, timeout_signal_handler)
            signal.alarm(int(timeout_seconds))
            
            try:
                result = func(*args, **kwargs)
            finally:
                # Restore old signal handler
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            
            return result
        
        return wrapper
    return decorator


class ProgressTracker:  # pragma: no cover
    """Track progress of long-running operations."""
    
    def __init__(self, total_items: int, description: str = "Processing"):
        self.total_items = total_items
        self.description = description
        self.completed_items = 0
        self.start_time = time.time()
    
    def update(self, completed: int = 1) -> None:
        """Update progress by specified number of completed items."""
        self.completed_items += completed
        self._print_progress()
    
    def _print_progress(self) -> None:
        """Print current progress."""
        if self.total_items == 0:
            return
        
        percentage = (self.completed_items / self.total_items) * 100
        elapsed_time = time.time() - self.start_time
        
        if self.completed_items > 0:
            eta = (elapsed_time / self.completed_items) * (self.total_items - self.completed_items)
            eta_str = f", ETA: {eta:.1f}s"
        else:
            eta_str = ""
        
        logger.info(f"{self.description}: {self.completed_items}/{self.total_items} "
                   f"({percentage:.1f}%) - {elapsed_time:.1f}s elapsed{eta_str}")
    
    def finish(self) -> None:
        """Mark progress as complete."""
        self.completed_items = self.total_items
        elapsed_time = time.time() - self.start_time
        logger.info(f"{self.description} completed in {elapsed_time:.1f}s")


# -----------------------------------------------------------------------------
# ONNX -> ACT layer handlers (bound onto _LayerGraphBuilder via setattr in
# torch2act.py). Kept here only to keep that file manageable; ``self`` is always
# a _LayerGraphBuilder instance and these touch its private API directly.
# -----------------------------------------------------------------------------

from act.back_end.layer_schema import LayerKind

def _convert_OnnxNeg(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxNeg: y = -x. Emitted as SCALE with a = -1."""
    if not self._get_predecessor_state(node):
        raise ValueError(f"OnnxNeg: missing predecessor for {node.name}")
    size = len(self.prev_out)
    out_vars = self._same_size_forward()
    layer_id = self._add_layer(
        LayerKind.SCALE.value,
        {"a": torch.full((size,), -1.0, dtype=self.dtype),
         "input_shape": self.shape, "output_shape": self.shape},
        self.prev_out, out_vars,
    )
    self.prev_out = out_vars
    self._register_node(node.name, layer_id)

def _convert_OnnxTranspose(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxTranspose: y = x.permute(perm)."""
    if not self._get_predecessor_state(node):
        raise ValueError(f"OnnxTranspose: missing predecessor for {node.name}")
    perm = tuple(int(p) for p in (getattr(mod, 'perm', None) or list(range(len(self.shape)))[::-1]))
    if len(perm) != len(self.shape):
        raise ValueError(f"OnnxTranspose: perm rank {len(perm)} != input rank {len(self.shape)}")
    output_shape = tuple(self.shape[p] for p in perm)
    out_vars = self._same_size_forward()
    layer_id = self._add_layer(
        LayerKind.TRANSPOSE.value,
        {
            "perm": perm,
            "input_shape": tuple(self.shape),
            "output_shape": output_shape,
        },
        self.prev_out,
        out_vars,
    )
    self.prev_out = out_vars
    self.shape = output_shape
    self._register_node(node.name, layer_id)

def _convert_OnnxReshape(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxReshape with ONNX 0/-1 dim semantics (0 = keep input dim, -1 = infer).

    Target shape resolution falls back through three tiers:
    direct get_attr → upstream layer's stored value → constant subgraph
    evaluation (handles e.g. shape derived via Concat-of-Shape ops).
    """
    if not self._get_predecessor_state(node):
        raise ValueError(f"OnnxReshape: missing predecessor for {node.name}")
    args = [a for a in node.args if isinstance(a, fx.Node)]
    shape_tensor = self._resolve_constant_tensor(args[1].name) if len(args) >= 2 else None
    if shape_tensor is None and len(args) >= 2:
        shape_tensor = self._evaluate_constant_subgraph(args[1].name)
    if shape_tensor is None:
        raise ValueError(f"OnnxReshape: cannot resolve target shape at {node.name}")
    raw = [int(x) for x in shape_tensor.flatten().tolist()]
    resolved = [int(self.shape[i]) if d == 0 else d for i, d in enumerate(raw)]
    if -1 in resolved:
        known = _prod(tuple(d for d in resolved if d != -1)) or 1
        resolved[resolved.index(-1)] = _prod(self.shape) // known
    output_shape = tuple(resolved)
    out_vars = self._same_size_forward()
    layer_id = self._add_layer(
        LayerKind.RESHAPE.value, {"target_shape": output_shape}, self.prev_out, out_vars,
    )
    self.prev_out = out_vars
    self.shape = output_shape
    self._register_node(node.name, layer_id)

def _convert_OnnxConstant(self, mod: nn.Module, node: fx.Node) -> None:
    """Emit ACT CONSTANT layer for an onnx2torch OnnxConstant module.

    OnnxConstant wraps a fixed tensor that the model's forward returns at this
    position.  Materialising it as a CONSTANT layer lets registered-var
    consumers (Reshape's shape arg, Slice bounds, Pow exponent, MatMul second
    operand, etc.) resolve it via ``node_outputs[node.name]``.  Lazy consumers
    that walk the FX graph through ``_evaluate_constant_subgraph`` continue
    to work because ``mod.forward()`` is unchanged.

    Integer dtypes are preserved (shape constants must stay int); only
    floating-point values get cast to ``self.dtype``.
    """
    val = getattr(mod, "value", None)
    if val is None:
        val = next(iter(mod.buffers()), None)
    if val is None:
        raise NotImplementedError(
            f"OnnxConstant at {node.name} has no .value attribute or buffer"
        )
    if not isinstance(val, torch.Tensor):
        val = torch.tensor(val)
    val = val.detach().clone()
    if val.is_floating_point():
        val = val.to(self.dtype)
    flat = val.reshape(-1)
    shape = tuple(int(d) for d in val.shape) or (1,)
    out_vars = self._alloc_ids(int(flat.numel()) or 1)
    layer_id = self._add_layer(
        LayerKind.CONSTANT.value,
        {"value": flat, "input_shape": shape, "output_shape": shape},
        [], out_vars,
    )
    self.node_outputs[node.name] = out_vars
    self.node_shapes[node.name] = shape
    self.node_to_layer_id[node.name] = layer_id


def _convert_OnnxConcat(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxConcat: y = cat(*input_tensors, axis).

    Constant-initializer args (e.g. ViT's [CLS] token) are materialised
    on-demand via ``_ensure_constant_vars`` before the CONCAT layer is
    emitted, so each input has registered vars in ``node_outputs``.
    """
    axis = int(getattr(mod, 'axis', 0))
    args = [a for a in node.args if isinstance(a, fx.Node)]
    if not args:
        raise ValueError(f"OnnxConcat: no inputs at {node.name}")
    all_vars: List[int] = []
    shapes: List[Tuple[int, ...]] = []
    for arg in args:
        if arg.name not in self.node_outputs and not self._ensure_constant_vars(arg.name):
            raise ValueError(f"OnnxConcat at {node.name}: input '{arg.name}' is neither a registered variable nor a resolvable constant")
        all_vars.extend(self.node_outputs[arg.name])
        shapes.append(self.node_shapes[arg.name])
    norm_axis = axis if axis >= 0 else axis + len(shapes[0])
    out_shape = list(shapes[0])
    out_shape[norm_axis] = sum(int(s[norm_axis]) for s in shapes)
    output_shape = tuple(out_shape)
    out_vars = self._alloc_ids(len(all_vars))
    layer_id = self._add_layer(
        LayerKind.CONCAT.value, {"concat_dim": axis}, all_vars, out_vars,
    )
    self.prev_out = out_vars
    self.shape = output_shape
    self._register_node(node.name, layer_id)

def _convert_OnnxReduceStaticAxes(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxReduceStaticAxes (ReduceMean / ReduceMax / ReduceMin / ReduceSum etc.).

    onnx2torch routes several reductions through this single class and
    distinguishes by ``math_op_function``. Supported mappings:
      * mean → LayerKind.MEAN
      * sum  → LayerKind.REDUCE_SUM (nn4sys pensieve uses this form for L2/Lp
                                     normalization preludes)
    Max/Min and L2-norm need their own LayerKind mapping (future enhancement)."""
    if not self._get_predecessor_state(node):
        raise ValueError(f"OnnxReduceStaticAxes: missing predecessor for {node.name}")
    op_func = getattr(mod, 'math_op_function', None)
    op_name = getattr(op_func, '__name__', '').lower() if op_func is not None else ''
    # OnnxReduceStaticAxes uses public ``axes`` / ``keepdims`` (different from
    # OnnxReduceSumStaticAxes which uses private ``_axes`` / ``_keepdims``).
    axes_attr = getattr(mod, 'axes', None) or list(range(len(self.shape)))
    keepdims = bool(getattr(mod, 'keepdims', True))
    norm_axes = _normalize_axes(axes_attr, len(self.shape))
    output_shape = _reduce_output_shape(self.shape, norm_axes, keepdims)
    out_vars = self._alloc_ids(_prod(output_shape) or 1)

    # LayerKind schemas differ between MEAN and REDUCE_SUM (dim/keepdim vs
    # axes/keepdims); route each kind to the right param names.
    if 'mean' in op_name:
        params = {"dim": list(norm_axes), "keepdim": int(keepdims),
                  "input_shape": self.shape, "output_shape": output_shape}
        layer_kind = LayerKind.MEAN.value
    elif 'sum' in op_name:
        params = {"axes": list(norm_axes), "keepdims": int(keepdims),
                  "input_shape": self.shape, "output_shape": output_shape}
        layer_kind = LayerKind.REDUCE_SUM.value
    else:
        raise NotImplementedError(
            f"OnnxReduceStaticAxes at {node.name}: only ReduceMean / ReduceSum "
            f"supported (got '{op_name}'; future enhancement)"
        )
    layer_id = self._add_layer(
        layer_kind,
        params,
        self.prev_out, out_vars,
    )
    self.prev_out = out_vars
    self.shape = output_shape
    self._register_node(node.name, layer_id)

def _convert_OnnxReduceSumStaticAxes(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxReduceSumStaticAxes: y = sum(x, axes, keepdim)."""
    if not self._get_predecessor_state(node):
        raise ValueError(f"OnnxReduceSumStaticAxes: missing predecessor for {node.name}")
    axes = getattr(mod, '_axes', None) or list(range(len(self.shape)))
    keepdims = bool(int(getattr(mod, '_keepdims', 1)))
    norm_axes = _normalize_axes(axes, len(self.shape))
    output_shape = _reduce_output_shape(self.shape, norm_axes, keepdims)
    out_vars = self._alloc_ids(_prod(output_shape) or 1)
    layer_id = self._add_layer(
        LayerKind.REDUCE_SUM.value,
        {"axes": list(norm_axes), "keepdims": int(keepdims),
         "input_shape": self.shape, "output_shape": output_shape},
        self.prev_out, out_vars,
    )
    self.prev_out = out_vars
    self.shape = output_shape
    self._register_node(node.name, layer_id)

def _convert_OnnxGather(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxGather: numpy.take(x, indices, axis=_axis).

    Indices may be a direct get_attr initializer OR the output of a
    Constant op (a call_module producing a constant tensor). Both are
    valid for ACT — indices must just be statically resolvable. nn4sys
    pensieve emits the latter, so the bare _resolve_constant_tensor
    misses it; fall through to _evaluate_constant_subgraph which walks
    upstream Constant chains."""
    if not self._get_predecessor_state(node):
        raise ValueError(f"OnnxGather: missing predecessor for {node.name}")
    axis = int(getattr(mod, '_axis', 0))
    args = [a for a in node.args if isinstance(a, fx.Node)]
    idx = self._resolve_constant_tensor(args[1].name) if len(args) >= 2 else None
    if idx is None and len(args) >= 2:
        idx = self._evaluate_constant_subgraph(args[1].name)
    if idx is None:
        raise ValueError(f"OnnxGather: cannot resolve indices at {node.name}")
    indices = idx.detach().clone().to(torch.int64)
    norm_axis = axis if axis >= 0 else axis + len(self.shape)
    if indices.dim() == 0:
        output_shape = tuple(self.shape[:norm_axis] + self.shape[norm_axis + 1:]) or (1,)
    else:
        output_shape = (*self.shape[:norm_axis], *indices.shape, *self.shape[norm_axis + 1:])
    out_vars = self._alloc_ids(_prod(output_shape) or 1)
    layer_id = self._add_layer(
        LayerKind.GATHER.value,
        {"indices": indices, "axis": axis,
         "input_shape": self.shape, "output_shape": output_shape},
        self.prev_out, out_vars,
    )
    self.prev_out = out_vars
    self.shape = output_shape
    self._register_node(node.name, layer_id)

def _convert_OnnxMatMul(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxMatMul: dispatches three cases by operand kind.

    - var × const-2D weight  → DENSE (Linear-equivalent, W.T as weight)
    - var × const-1D weight  → SCALE + REDUCE_SUM (matrix-vector dot)
    - var × var              → MATMUL layer (bilinear; TF support deferred)
    """
    args = [a for a in node.args if isinstance(a, fx.Node)]
    x_node, w_node = args[0], args[1]
    x_var = x_node.name in self.node_outputs
    w_var = w_node.name in self.node_outputs

    if x_var and w_var:
        xv = self.node_outputs[x_node.name]
        yv = self.node_outputs[w_node.name]
        xs = self.node_shapes[x_node.name]
        ys = self.node_shapes[w_node.name]
        try:
            output_shape = tuple(int(d) for d in torch.matmul(
                torch.zeros(xs, dtype=self.dtype),
                torch.zeros(ys, dtype=self.dtype),
            ).shape)
        except RuntimeError as e:
            raise ValueError(
                f"OnnxMatMul at {node.name}: incompatible var-var shapes {xs} @ {ys} ({e})"
            )
        out_vars = self._alloc_ids(_prod(output_shape) or 1)
        layer_id = self._add_layer(
            LayerKind.MATMUL.value,
            {"x_vars": xv, "y_vars": yv,
             "x_shape": xs, "y_shape": ys,
             "input_shape": xs, "output_shape": output_shape},
            xv + yv, out_vars,
        )
        self.prev_out = out_vars
        self.shape = output_shape
        self._register_node(node.name, layer_id)
        return

    if not x_var:
        # const × var: materialise the constant via _ensure_constant_vars
        # then re-enter the var × var branch above. Note matmul is
        # non-commutative — we keep operand order intact.
        if not self._ensure_constant_vars(x_node.name):
            raise ValueError(
                f"OnnxMatMul at {node.name}: cannot resolve constant first operand"
            )
        xv = self.node_outputs[x_node.name]
        yv = self.node_outputs[w_node.name]
        xs = self.node_shapes[x_node.name]
        ys = self.node_shapes[w_node.name]
        try:
            output_shape = tuple(int(d) for d in torch.matmul(
                torch.zeros(xs, dtype=self.dtype),
                torch.zeros(ys, dtype=self.dtype),
            ).shape)
        except RuntimeError as e:
            raise ValueError(
                f"OnnxMatMul at {node.name}: incompatible const-var shapes {xs} @ {ys} ({e})"
            )
        out_vars = self._alloc_ids(_prod(output_shape) or 1)
        layer_id = self._add_layer(
            LayerKind.MATMUL.value,
            {"x_vars": xv, "y_vars": yv,
             "x_shape": xs, "y_shape": ys,
             "input_shape": xs, "output_shape": output_shape},
            xv + yv, out_vars,
        )
        self.prev_out = out_vars
        self.shape = output_shape
        self._register_node(node.name, layer_id)
        return
    W = self._resolve_constant_tensor(w_node.name)
    if W is None:
        raise ValueError(f"OnnxMatMul at {node.name}: cannot resolve constant weight")
    self.prev_out = self.node_outputs[x_node.name].copy()
    self.shape = self.node_shapes[x_node.name]

    if W.dim() == 1:
        # PyTorch matrix-vector: (..., K) @ (K,) -> (...) — sum-product along last dim.
        # Realised as element-wise SCALE (broadcast W over var's leading dims) then REDUCE_SUM
        # along the last axis with keepdims=0.
        K = int(W.shape[0])
        if not self.shape or int(self.shape[-1]) != K:
            raise ValueError(
                f"OnnxMatMul at {node.name}: var last dim {self.shape[-1] if self.shape else None} != W len {K}"
            )
        scale_a = W.expand(*self.shape).contiguous().to(self.dtype).flatten()
        scale_out = self._same_size_forward()
        self._add_layer(
            LayerKind.SCALE.value,
            {"a": scale_a, "input_shape": self.shape, "output_shape": self.shape},
            self.prev_out, scale_out,
        )
        self.prev_out = scale_out
        output_shape = tuple(self.shape[:-1]) or (1,)
        out_vars = self._alloc_ids(_prod(output_shape) or 1)
        layer_id = self._add_layer(
            LayerKind.REDUCE_SUM.value,
            {"axes": [len(self.shape) - 1], "keepdims": 0,
             "input_shape": self.shape, "output_shape": output_shape},
            self.prev_out, out_vars,
        )
        self.prev_out = out_vars
        self.shape = output_shape
        self._register_node(node.name, layer_id)
        return

    if W.dim() != 2:
        raise NotImplementedError(
            f"OnnxMatMul at {node.name}: only 1D / 2D constant weights supported (got {tuple(W.shape)})"
        )
    in_features, out_features = int(W.shape[0]), int(W.shape[1])
    # Batched matmul: var shape (..., M, K) @ const W (K, N) -> (..., M, N).
    # The right consistency check is on the LAST dim of the var shape, not
    # the flattened length (which counts batch * M * K, not just K).
    if not self.shape or int(self.shape[-1]) != in_features:
        raise ValueError(
            f"OnnxMatMul at {node.name}: var last dim "
            f"{self.shape[-1] if self.shape else None} != weight in_features {in_features}"
        )
    output_shape = tuple(self.shape[:-1]) + (out_features,)
    out_vars = self._alloc_ids(_prod(output_shape) or out_features)
    layer_id = self._add_layer(
        LayerKind.DENSE.value,
        {"weight": W.t().contiguous().detach().clone().to(self.dtype),
         "in_features": in_features, "out_features": out_features,
         "input_shape": self.shape, "output_shape": output_shape},
        self.prev_out, out_vars,
    )
    self.prev_out = out_vars
    self.shape = output_shape
    self._register_node(node.name, layer_id)

def _convert_OnnxShape(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxShape: pure compile-time. Stash value in side map; emit no layer.

    A runtime CONSTANT layer was unsafe (zero-indegree → never seeded into
    ``analyze()``'s worklist; DualTF treats unknown layers as identity).
    Leaving ``node_outputs`` empty also keeps ``_build_preds_succs`` and
    ``_get_predecessor_state`` from mistaking Shape for a runtime tensor.
    """
    args = [a for a in node.args if isinstance(a, fx.Node)]
    if not args:
        raise ValueError(f"OnnxShape at {node.name}: no inputs")
    src = args[0]
    if src.name in self.node_shapes:
        src_shape = self.node_shapes[src.name]
    elif src.op == 'placeholder':
        src_shape = self.input_shape
    else:
        raise ValueError(f"OnnxShape at {node.name}: cannot resolve input shape for '{src.name}'")
    start = int(getattr(mod, 'start', 0) or 0)
    end_attr = getattr(mod, 'end', None)
    end = int(end_attr) if end_attr is not None else len(src_shape)
    if start < 0:
        start += len(src_shape)
    if end < 0:
        end += len(src_shape)
    start = max(0, min(start, len(src_shape)))
    end = max(start, min(end, len(src_shape)))
    self._compile_time_values[node.name] = torch.tensor(
        src_shape[start:end], dtype=self._ONNX_SHAPE_DTYPE,
    )

def _convert_OnnxSlice(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxSlice: y = x[starts:ends:steps along axes].

    The input tensor can be a constant initializer (e.g. YOLO's anchor
    constants); materialise it via ``_ensure_constant_vars`` before
    running the slice arithmetic.
    """
    args = [a for a in node.args if isinstance(a, fx.Node)]
    if len(args) < 3:
        raise ValueError(f"OnnxSlice at {node.name}: need at least 3 args")
    if args[0].name in self.node_outputs:
        self.prev_out = self.node_outputs[args[0].name].copy()
        self.shape = self.node_shapes[args[0].name]
    elif self._ensure_constant_vars(args[0].name):
        self.prev_out = self.node_outputs[args[0].name].copy()
        self.shape = self.node_shapes[args[0].name]
    else:
        raise ValueError(f"OnnxSlice: missing predecessor for {node.name}")
    starts = self._resolve_slice_input_to_int_list(args[1].name)
    ends = self._resolve_slice_input_to_int_list(args[2].name)
    if starts is None or ends is None:
        axes_dyn = (self._resolve_slice_input_to_int_list(args[3].name)
                    if len(args) > 3 else None)
        steps_dyn = (self._resolve_slice_input_to_int_list(args[4].name)
                     if len(args) > 4 else None)
        if axes_dyn is not None:
            if steps_dyn is None:
                steps_dyn = [1] * len(axes_dyn)
            if _try_emit_dynamic_slice_lut_bounds(
                self, node, args, axes_dyn, steps_dyn,
            ):
                return
            reason = getattr(self, "_last_dynamic_slice_lut_error", None)
            if reason:
                raise ValueError(f"OnnxSlice at {node.name}: {reason}")
        raise ValueError(f"OnnxSlice at {node.name}: cannot resolve starts/ends")
    axes = (self._resolve_slice_input_to_int_list(args[3].name)
            if len(args) > 3 else None) or list(range(len(starts)))
    steps = (self._resolve_slice_input_to_int_list(args[4].name)
             if len(args) > 4 else None) or [1] * len(starts)
    try:
        out_shape, n_starts, n_ends, n_axes = _compute_slice_output_shape(
            self.shape, starts, ends, axes, steps,
        )
    except ValueError as e:
        raise ValueError(f"OnnxSlice at {node.name}: {e}")
    out_vars = self._alloc_ids(_prod(out_shape) or 1)
    layer_id = self._add_layer(
        LayerKind.SLICE.value,
        {"starts": n_starts, "ends": n_ends, "axes": n_axes,
         "input_shape": self.shape, "output_shape": out_shape},
        self.prev_out, out_vars,
    )
    self.prev_out = out_vars
    self.shape = out_shape
    self._register_node(node.name, layer_id)

def _convert_OnnxPow(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxPow with constant non-negative integer exponent.

    Supported via repeated MUL: x^k = x * x * ... * x (k-1 multiplications).
    nn4sys pensieve uses x^3 (L2-norm-style chain), so the prior
    exponent==2-only restriction blocked the whole benchmark family.

    Exponents 1, 2, 3, 4 are common in NN expressivity ops. Higher values
    still work but produce O(log k) deep chains via square-and-multiply;
    we keep the naive linear chain for clarity since k is small in practice.

    The constant exponent may come from a get_attr OR a Constant subgraph
    (pensieve_small_parallel ships it as a Constant op, not initializer).
    Non-integer or negative exponents remain unsupported."""
    if not self._get_predecessor_state(node):
        raise ValueError(f"OnnxPow: missing predecessor for {node.name}")
    args = [a for a in node.args if isinstance(a, fx.Node)]
    if len(args) < 2:
        raise ValueError(f"OnnxPow at {node.name}: expected 2 args")
    exp_t = self._resolve_constant_tensor(args[1].name)
    if exp_t is None:
        exp_t = self._evaluate_constant_subgraph(args[1].name)
    if exp_t is None:
        raise NotImplementedError(
            f"OnnxPow at {node.name}: dynamic exponent (future enhancement)"
        )
    exp_val = float(exp_t.flatten().tolist()[0])
    exp_int = int(round(exp_val))
    if abs(exp_val - exp_int) > 1e-9 or exp_int < 1:
        raise NotImplementedError(
            f"OnnxPow at {node.name}: only positive integer exponents supported "
            f"(got {exp_val}; future enhancement for non-integer/negative)"
        )

    var_vars = self.node_outputs[args[0].name]
    # x^1 is identity — wire through without a layer.
    if exp_int == 1:
        self.node_outputs[node.name] = var_vars
        self.node_shapes[node.name] = self.shape
        self.prev_out = var_vars
        return

    # x^k for k>=2: chain (k-1) MULs. accumulator starts as x; each step
    # multiplies by the original x. Each intermediate result gets fresh
    # variable ids; the final layer's out_vars is the chain output.
    #
    # Predecessor wiring: each chain step is a helper (no FX node), so
    # the FX-based pred walk falls back to ``preds[i-1]`` (the previous
    # layer in id order), which is WRONG when an unrelated layer sits
    # between us and our source (e.g. a CONSTANT for the exponent). We
    # therefore register explicit preds for every MUL step. ``tf_mul``
    # reads positional ``preds[0]`` and ``preds[1]``, so even Mul(x, x)
    # — both operands from the same source — needs the source id LISTED
    # TWICE in the preds list; ``_set_explicit_preds`` preserves
    # duplicates for exactly this case.
    var_x_lid = self.node_to_layer_id.get(args[0].name)
    if var_x_lid is None:
        var_x_lid = -1
    accumulator = var_vars
    accumulator_producer = var_x_lid
    last_layer_id = -1
    for _step in range(exp_int - 1):
        out_vars = self._alloc_ids(len(var_vars))
        last_layer_id = self._add_layer(
            LayerKind.MUL.value,
            {"x_vars": accumulator, "y_vars": var_vars,
             "input_shape": self.shape, "output_shape": self.shape},
            accumulator + var_vars, out_vars,
        )
        if accumulator_producer >= 0 or var_x_lid >= 0:
            self._set_explicit_preds(last_layer_id,
                                     [accumulator_producer, var_x_lid])
        accumulator = out_vars
        accumulator_producer = last_layer_id
    self.prev_out = accumulator
    self._register_node(node.name, last_layer_id)

def _convert_OnnxSplit13(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxSplit / OnnxSplit13: split input along an axis into chunks.

    Two paths:
      (a) sizes-input form (opset 13+, OnnxSplit13): split sizes come from
          a constant input tensor (args[1]).
      (b) equal-axis form (older OnnxSplit, or opset 13+ when the optional
          sizes input is omitted): split into N equal chunks where N is
          inferred from the number of downstream ``getitem(split, i)`` fx
          children. nn4sys pensieve_big_parallel uses this form (axis=1,
          no sizes input).

    Both decompose to N SLICE layers; each downstream ``getitem(split, i)``
    fx node is pre-registered to point at the i-th SLICE's outputs;
    ``_process_getitem_operation`` honours the pre-registered state via
    its early-return guard.
    """
    if not self._get_predecessor_state(node):
        raise ValueError(f"OnnxSplit: missing predecessor for {node.name}")
    args = [a for a in node.args if isinstance(a, fx.Node)]
    axis_attr = getattr(mod, '_axis', None)
    if axis_attr is None:
        axis_attr = getattr(mod, 'axis', 0)
    rank = len(self.shape)
    norm_axis = int(axis_attr) + rank if int(axis_attr) < 0 else int(axis_attr)

    # opset-10 Split carries split sizes as an *attribute* (mod.split=[6,1]),
    # which onnx2torch exposes on the module. opset-13 Split moved the sizes
    # to an input tensor (args[1]). Both forms need to honour non-equal splits;
    # falling through to the equal-axis path on opset 10 produces the
    # spurious "axis dim (7) divisible by 2" error seen on nn4sys mscn_128d.
    attr_split = getattr(mod, 'split', None)
    if attr_split is not None and isinstance(attr_split, (list, tuple)) and len(attr_split) > 0:
        split_sizes = [int(x) for x in attr_split]
    elif len(args) >= 2:
        split_t = self._resolve_constant_tensor(args[1].name)
        if split_t is None:
            split_t = self._evaluate_constant_subgraph(args[1].name)
        if split_t is None:
            raise ValueError(f"OnnxSplit at {node.name}: cannot resolve split sizes")
        split_sizes = [int(x) for x in split_t.flatten().tolist()]
    else:
        # Equal-axis split: count downstream getitem children to find N.
        num_splits = 0
        if self.fx_graph is not None:
            child_indices = set()
            for n in self.fx_graph.nodes:
                if n.op == 'call_function' and 'getitem' in str(n.target).lower() and n.args:
                    if isinstance(n.args[0], fx.Node) and n.args[0].name == node.name and len(n.args) > 1:
                        idx_arg = n.args[1]
                        if isinstance(idx_arg, int):
                            child_indices.add(idx_arg)
            if child_indices:
                num_splits = max(child_indices) + 1
        if num_splits <= 0:
            raise ValueError(
                f"OnnxSplit at {node.name}: equal-axis split needs at least one "
                f"downstream getitem child to infer split count; found none"
            )
        axis_size = int(self.shape[norm_axis])
        if axis_size % num_splits != 0:
            raise ValueError(
                f"OnnxSplit at {node.name}: equal-axis split requires axis dim "
                f"({axis_size}) divisible by num_splits ({num_splits})"
            )
        chunk = axis_size // num_splits
        split_sizes = [chunk] * num_splits

    getitem_children: Dict[int, fx.Node] = {}
    if self.fx_graph is not None:
        for n in self.fx_graph.nodes:
            if n.op == 'call_function' and 'getitem' in str(n.target).lower() and n.args:
                if isinstance(n.args[0], fx.Node) and n.args[0].name == node.name and len(n.args) > 1:
                    idx_arg = n.args[1]
                    if isinstance(idx_arg, int):
                        getitem_children[idx_arg] = n

    var_vars = list(self.prev_out)
    var_shape = self.shape
    last_chunk_vars: List[int] = var_vars
    last_chunk_shape: Tuple[int, ...] = var_shape
    last_layer_id = -1
    offset = 0
    for i, size in enumerate(split_sizes):
        chunk_shape = list(var_shape)
        chunk_shape[norm_axis] = size
        chunk_shape_t = tuple(chunk_shape)
        chunk_vars = self._alloc_ids(_prod(chunk_shape_t) or 1)
        layer_id = self._add_layer(
            LayerKind.SLICE.value,
            {"starts": [offset], "ends": [offset + size], "axes": [norm_axis],
             "input_shape": var_shape, "output_shape": chunk_shape_t},
            var_vars, chunk_vars,
        )
        if i in getitem_children:
            git_node = getitem_children[i]
            self.node_outputs[git_node.name] = chunk_vars
            self.node_shapes[git_node.name] = chunk_shape_t
            self.node_to_layer_id[git_node.name] = layer_id
        last_chunk_vars = chunk_vars
        last_chunk_shape = chunk_shape_t
        last_layer_id = layer_id
        offset += size

    # The Split node itself isn't directly consumed (only via getitem), but
    # downstream code that propagates from it should see at least *some*
    # valid state -- use the last chunk as the canonical output.
    self.prev_out = last_chunk_vars
    self.shape = last_chunk_shape
    self._register_node(node.name, last_layer_id)

def _convert_OnnxResize(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxResize: spatial up/downsampling (nearest / linear / bilinear etc.).

    ONNX Resize takes (input, roi, scales, sizes) where any of roi/scales/sizes
    may be empty. We resolve a float scales tensor (preferred) or fall back to
    an int sizes tensor, then compute the output shape and emit UPSAMPLE.
    """
    if not self._get_predecessor_state(node):
        raise ValueError(f"OnnxResize: missing predecessor for {node.name}")
    args = [a for a in node.args if isinstance(a, fx.Node)]
    # ONNX Resize positional args: (input, roi, scales, sizes). Any of
    # roi/scales/sizes may be empty/skipped. Match by SHAPE not by type:
    # the real `scales` (or `sizes`) tensor has ``numel == len(input.shape)``
    # (one entry per dim, e.g. 4 for NCHW). ``roi`` typically has 8 entries
    # (start+end per spatial dim) and was previously matched as a stray
    # float candidate, causing Fix #8 failures on cgan_2023 iids 18/19/20
    # (`cannot resolve scales or sizes`).
    scales_t: Optional[torch.Tensor] = None
    sizes_t: Optional[torch.Tensor] = None
    expected_numel = len(self.shape)
    for a in args[1:]:
        t = self._resolve_constant_tensor(a.name)
        if t is None or t.numel() == 0:
            continue
        if t.numel() != expected_numel:
            # Likely the ``roi`` tensor (numel = 2 * spatial_rank). Skip.
            continue
        if t.dtype.is_floating_point and scales_t is None:
            scales_t = t
        elif not t.dtype.is_floating_point and sizes_t is None:
            sizes_t = t
    if scales_t is not None and scales_t.numel() == len(self.shape):
        output_shape = tuple(int(round(s * sc))
                             for s, sc in zip(self.shape, scales_t.tolist()))
        scale_factor = tuple(float(x) for x in scales_t.tolist())
        size_param = None
    elif sizes_t is not None and sizes_t.numel() == len(self.shape):
        output_shape = tuple(int(x) for x in sizes_t.tolist())
        scale_factor = None
        size_param = tuple(int(x) for x in sizes_t.tolist())
    else:
        raise ValueError(f"OnnxResize at {node.name}: cannot resolve scales or sizes")
    params: Dict[str, Any] = {
        "mode": str(getattr(mod, 'onnx_mode', 'nearest')),
        "input_shape": self.shape,
        "output_shape": output_shape,
    }
    if getattr(mod, 'align_corners', None) is not None:
        params["align_corners"] = bool(mod.align_corners)
    if scale_factor is not None:
        params["scale_factor"] = scale_factor
    if size_param is not None:
        params["size"] = size_param
    out_vars = self._alloc_ids(_prod(output_shape) or 1)
    layer_id = self._add_layer(
        LayerKind.UPSAMPLE.value, params, self.prev_out, out_vars,
    )
    self.prev_out = out_vars
    self.shape = output_shape
    self._register_node(node.name, layer_id)

def _convert_OnnxBinaryMathOperation(self, mod: nn.Module, node: fx.Node) -> None:
    """Add/Sub/Mul/Div: var-var → ADD/SUB/MUL/DIV; var-const → BIAS/SCALE (or SCALE+BIAS)."""
    op_raw = getattr(getattr(mod, 'math_op_function', None), '__name__', '').lower()
    op = {'add': 'add', 'sub': 'sub', 'mul': 'mul',
          '_onnx_div': 'div', 'div': 'div'}.get(op_raw)
    if op is None:
        raise NotImplementedError(f"OnnxBinaryMathOperation: unrecognised op '{op_raw}' at {node.name}")
    args = [a for a in node.args if isinstance(a, fx.Node)]
    x, y = args[0], args[1]
    x_var = x.name in self.node_outputs
    y_var = y.name in self.node_outputs

    if x_var and y_var:
        xv, yv = self.node_outputs[x.name], self.node_outputs[y.name]
        xs, ys = self.node_shapes[x.name], self.node_shapes[y.name]
        # Original producers BEFORE any broadcast helper is inserted —
        # consulted when registering the consumer's explicit preds below.
        x_src_lid = self.node_to_layer_id.get(x.name, -1)
        y_src_lid = self.node_to_layer_id.get(y.name, -1)
        if x_src_lid is None:
            x_src_lid = -1
        if y_src_lid is None:
            y_src_lid = -1

        if len(xv) != len(yv):
            # Broadcast mismatch — common in nn4sys mscn (e.g. Mul_39:
            # (3,128) × (3,1) → (3,128)) and pensieve L2-norm (scalar × tensor).
            #
            # Insert EXPAND helper(s) so both operands match a common
            # broadcast target shape. We compute the target via standard
            # numpy/PyTorch broadcasting rules from xs and ys, then expand
            # each side that isn't already at target.
            try:
                broadcast_shape = tuple(
                    int(d) for d in torch.broadcast_shapes(
                        torch.Size(xs), torch.Size(ys),
                    )
                )
            except RuntimeError as bc_err:
                raise NotImplementedError(
                    f"Var-var '{op}' shape mismatch (xs={xs}, ys={ys}) at "
                    f"{node.name}: not broadcast-compatible ({bc_err})"
                )
            target_n = _prod(broadcast_shape)
            if len(xv) != target_n:
                exp_vars = self._alloc_ids(target_n)
                exp_lid = self._add_layer(
                    LayerKind.EXPAND.value,
                    {"shape": list(broadcast_shape),
                     "input_shape": xs,
                     "output_shape": broadcast_shape},
                    xv, exp_vars,
                )
                self._set_explicit_preds(exp_lid, [x_src_lid])
                xv, xs = exp_vars, broadcast_shape
                x_src_lid = exp_lid
            if len(yv) != target_n:
                exp_vars = self._alloc_ids(target_n)
                exp_lid = self._add_layer(
                    LayerKind.EXPAND.value,
                    {"shape": list(broadcast_shape),
                     "input_shape": ys,
                     "output_shape": broadcast_shape},
                    yv, exp_vars,
                )
                self._set_explicit_preds(exp_lid, [y_src_lid])
                yv, ys = exp_vars, broadcast_shape
                y_src_lid = exp_lid
        kind = {'add': LayerKind.ADD, 'sub': LayerKind.SUB,
                'mul': LayerKind.MUL, 'div': LayerKind.DIV}[op]
        out_shape = xs if _prod(xs) >= _prod(ys) else ys
        out_vars = self._alloc_ids(len(xv))
        layer_id = self._add_layer(
            kind.value,
            {"x_vars": xv, "y_vars": yv,
             "input_shape": xs, "output_shape": out_shape},
            xv + yv, out_vars,
        )
        # If a broadcast helper was inserted above, x_src_lid / y_src_lid
        # now point at that helper rather than the original FX-tracked
        # producer. Either way, register the post-broadcast producers as
        # this layer's preds so the analyze() worklist visits them in the
        # right order and Bin is computed from helper output, not stale.
        if x_src_lid >= 0 or y_src_lid >= 0:
            self._set_explicit_preds(layer_id, [x_src_lid, y_src_lid])
        self.prev_out = out_vars
        self.shape = out_shape
        self._register_node(node.name, layer_id)
        return

    if x_var:
        var_node, const_node, var_first = x, y, True
    else:
        var_node, const_node, var_first = y, x, False
    const = self._resolve_constant_tensor(const_node.name)
    if const is None:
        raise ValueError(f"OnnxBinaryMathOperation: cannot resolve constant at {node.name}")
    self.prev_out = self.node_outputs[var_node.name].copy()
    self.shape = self.node_shapes[var_node.name]
    current_src_lid = self.node_to_layer_id.get(var_node.name, -1)
    if current_src_lid is None:
        current_src_lid = -1

    # PyTorch broadcasting may yield an output shape *larger* than either
    # operand (outer-product case, e.g. (1,226,1) op (54,) -> (1,226,54)).
    # Detect this and prepend an EXPAND layer that replicates the variable
    # to the broadcast shape before applying BIAS/SCALE; the constant gets
    # pre-broadcasted offline since it's known at conversion time.
    try:
        broadcast_shape = tuple(int(d) for d in torch.broadcast_shapes(self.shape, tuple(const.shape)))
    except RuntimeError as e:
        raise ValueError(
            f"OnnxBinaryMathOperation at {node.name}: shapes {self.shape} and "
            f"{tuple(const.shape)} are not broadcast-compatible ({e})"
        )
    if broadcast_shape != self.shape:
        expanded_size = _prod(broadcast_shape) or 1
        expanded_vars = self._alloc_ids(expanded_size)
        exp_lid = self._add_layer(
            LayerKind.EXPAND.value,
            {"shape": broadcast_shape, "input_shape": self.shape,
             "output_shape": broadcast_shape},
            self.prev_out, expanded_vars,
        )
        self._set_explicit_preds(exp_lid, [current_src_lid])
        self.prev_out = expanded_vars
        self.shape = broadcast_shape
        current_src_lid = exp_lid

    size = len(self.prev_out)
    const_b = const.expand(*broadcast_shape).contiguous() if tuple(const.shape) != broadcast_shape else const
    c = _broadcast_const_to_size(const_b, size, self.dtype)

    def emit(kind: LayerKind, key: str, t: torch.Tensor, register: bool) -> None:
        nonlocal current_src_lid
        out = self._same_size_forward()
        lid = self._add_layer(
            kind.value,
            {key: t, "input_shape": self.shape, "output_shape": self.shape},
            self.prev_out, out,
        )
        self._set_explicit_preds(lid, [current_src_lid])
        self.prev_out = out
        current_src_lid = lid
        if register:
            self._register_node(node.name, lid)

    if op == 'add':
        emit(LayerKind.BIAS, "c", c, register=True)
    elif op == 'sub':
        if var_first:
            emit(LayerKind.BIAS, "c", (-c).contiguous(), register=True)
        else:
            emit(LayerKind.SCALE, "a", torch.full((size,), -1.0, dtype=self.dtype), register=False)
            emit(LayerKind.BIAS, "c", c.contiguous(), register=True)
    elif op == 'mul':
        emit(LayerKind.SCALE, "a", c, register=True)
    else:  # 'div'
        if not var_first:
            raise NotImplementedError(f"const/var Div at {node.name} (future enhancement)")
        emit(LayerKind.SCALE, "a", (1.0 / c).to(self.dtype), register=True)

def _convert_OnnxRound(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxRound: dispatches Floor / Ceil / Round (single onnx2torch
    class with ``round_function`` selecting the variant).

    Sound interval transfer:
      * Floor / Ceil: monotone, ``[op(lb), op(ub)]``.
      * Round (banker's): ``[floor(lb), ceil(ub)]`` — conservative across
        the half-integer discontinuity. Any tighter rule would need a
        per-element correctness proof we are not prepared to underwrite.

    ml4acopf_2024 surfaces this op via floor / round preludes to its
    trigonometric chain; the same handler covers all three because
    onnx2torch routes all three through the OnnxRound class."""
    if not self._get_predecessor_state(node):
        raise ValueError(f"OnnxRound: missing predecessor for {node.name}")
    fn = getattr(mod, "round_function", None)
    fn_name = getattr(fn, "__name__", "") if fn is not None else ""
    fn_name = fn_name.lower()
    if "floor" in fn_name:
        kind = LayerKind.FLOOR.value
    elif "ceil" in fn_name:
        kind = LayerKind.CEIL.value
    elif "round" in fn_name:
        kind = LayerKind.ROUND.value
    else:
        raise NotImplementedError(
            f"OnnxRound at {node.name}: unrecognised round_function "
            f"{fn_name!r}; expected one of floor/ceil/round"
        )
    out_vars = self._same_size_forward()
    layer_id = self._add_layer(
        kind,
        {"input_shape": self.shape, "output_shape": self.shape},
        self.prev_out, out_vars,
    )
    self.prev_out = out_vars
    self._register_node(node.name, layer_id)


def _convert_OnnxConstantOfShape(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxConstantOfShape: y = full(shape, value).

    Produces a tensor of a given shape filled with a scalar value
    (stored on the module as the ``value`` buffer). Materialized as
    an ACT ``CONSTANT`` layer when the shape input is statically
    resolvable (initializer, constant subgraph, or a Shape op on the
    model input). cctsdb_yolo_2023 hits this pattern in its YOLO
    decoder where ``constant_of_shape_*`` is fed a static shape.

    The shape input may be: a get_attr initializer, the output of a
    Constant op, or a Shape op walked back through ``_evaluate_constant_subgraph``.
    All three are handled uniformly via the existing resolution chain.
    """
    args = [a for a in node.args if isinstance(a, fx.Node)]
    if not args:
        raise ValueError(
            f"OnnxConstantOfShape at {node.name}: expected at least 1 arg (shape)"
        )
    shape_t = self._resolve_constant_tensor(args[0].name)
    if shape_t is None:
        shape_t = self._evaluate_constant_subgraph(args[0].name)
    if shape_t is None:
        raise ValueError(
            f"OnnxConstantOfShape at {node.name}: cannot resolve target shape "
            f"(dynamic shape from variable activation; not supported in static ACT)"
        )
    out_shape = tuple(int(x) for x in shape_t.flatten().tolist())
    fill = mod.value
    if not isinstance(fill, torch.Tensor):
        fill = torch.tensor(fill, dtype=self.dtype)
    fill_scalar = float(fill.flatten()[0].item())
    n_out = _prod(out_shape) or 1
    flat = torch.full((n_out,), fill_scalar, dtype=self.dtype)
    out_vars = self._alloc_ids(n_out)
    layer_id = self._add_layer(
        LayerKind.CONSTANT.value,
        {"value": flat, "input_shape": out_shape, "output_shape": out_shape},
        [], out_vars,
    )
    self.prev_out = out_vars
    self.shape = out_shape
    self._register_node(node.name, layer_id)


def _convert_OnnxExpand(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxExpand: y = x.expand(shape) — broadcast to the given shape.

    The first arg can be a constant initializer (cctsdb_yolo); fall back
    to ``_ensure_constant_vars`` before reading state.
    """
    args = [a for a in node.args if isinstance(a, fx.Node)]
    if len(args) < 2:
        raise ValueError(f"OnnxExpand at {node.name}: expected 2 args")
    if args[0].name in self.node_outputs:
        self.prev_out = self.node_outputs[args[0].name].copy()
        self.shape = self.node_shapes[args[0].name]
    elif self._ensure_constant_vars(args[0].name):
        self.prev_out = self.node_outputs[args[0].name].copy()
        self.shape = self.node_shapes[args[0].name]
    else:
        raise ValueError(f"OnnxExpand: missing predecessor for {node.name}")
    shape_t = self._resolve_constant_tensor(args[1].name)
    if shape_t is None:
        shape_t = self._evaluate_constant_subgraph(args[1].name)
    if shape_t is None:
        raise ValueError(f"OnnxExpand at {node.name}: cannot resolve target shape")
    target_shape = tuple(int(x) for x in shape_t.flatten().tolist())
    try:
        broadcast_shape = tuple(int(d) for d in torch.broadcast_shapes(self.shape, target_shape))
    except RuntimeError as e:
        raise ValueError(
            f"OnnxExpand at {node.name}: cannot broadcast {self.shape} → {target_shape} ({e})"
        )
    out_vars = self._alloc_ids(_prod(broadcast_shape) or 1)
    layer_id = self._add_layer(
        LayerKind.EXPAND.value,
        {"shape": broadcast_shape,
         "input_shape": self.shape, "output_shape": broadcast_shape},
        self.prev_out, out_vars,
    )
    self.prev_out = out_vars
    self.shape = broadcast_shape
    self._register_node(node.name, layer_id)

def _convert_OnnxFlatten(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxFlatten: flatten dims [axis:] into one trailing dim, keep [:axis] intact."""
    if not self._get_predecessor_state(node):
        raise ValueError(f"OnnxFlatten: missing predecessor for {node.name}")
    axis = int(getattr(mod, '_axis', getattr(mod, 'axis', 1)))
    if axis < 0:
        axis += len(self.shape)
    a = _prod(self.shape[:axis]) or 1
    b = _prod(self.shape[axis:]) or 1
    output_shape = (a, b)
    out_vars = self._same_size_forward()
    layer_id = self._add_layer(
        LayerKind.FLATTEN.value,
        {"start_dim": axis, "end_dim": -1,
         "input_shape": self.shape, "output_shape": output_shape},
        self.prev_out, out_vars,
    )
    self.prev_out = out_vars
    self.shape = output_shape
    self._register_node(node.name, layer_id)

def _convert_OnnxMinMax(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxMinMax: element-wise Min / Max of two operands -> LayerKind.MIN / MAX."""
    op_func = getattr(mod, '_operator', None)
    op_name = getattr(op_func, '__name__', '').lower() if op_func is not None else ''
    kind = LayerKind.MIN if 'min' in op_name else LayerKind.MAX
    args = [a for a in node.args if isinstance(a, fx.Node)]
    if len(args) < 2:
        raise ValueError(f"OnnxMinMax at {node.name}: expected 2 args")
    for n in args[:2]:
        if n.name not in self.node_outputs and not self._ensure_constant_vars(n.name):
            raise ValueError(f"OnnxMinMax at {node.name}: '{n.name}' not registered")
    xv, yv = self.node_outputs[args[0].name], self.node_outputs[args[1].name]
    xs, ys = self.node_shapes[args[0].name], self.node_shapes[args[1].name]
    try:
        output_shape = tuple(int(d) for d in torch.broadcast_shapes(xs, ys))
    except RuntimeError:
        output_shape = xs if _prod(xs) >= _prod(ys) else ys
    out_vars = self._alloc_ids(_prod(output_shape) or 1)
    layer_id = self._add_layer(
        kind.value,
        {"x_vars": xv, "y_vars": yv,
         "input_shape": xs, "output_shape": output_shape},
        xv + yv, out_vars,
    )
    self.prev_out = out_vars
    self.shape = output_shape
    self._register_node(node.name, layer_id)

def _convert_OnnxScatterND(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxScatterND: y = data; y[indices] = updates (writes ``updates`` into ``data``).

    For static-shape conversion the output shape equals the data input's
    shape; we only emit a SCATTER_ND layer recording the three input
    var-streams. Soundness depends on the verifier's TF (deferred).
    """
    args = [a for a in node.args if isinstance(a, fx.Node)]
    if len(args) < 3:
        raise ValueError(f"OnnxScatterND at {node.name}: expected 3 args")
    for n in args[:3]:
        if n.name not in self.node_outputs and not self._ensure_constant_vars(n.name):
            raise ValueError(f"OnnxScatterND at {node.name}: '{n.name}' not registered")
    data_vars = self.node_outputs[args[0].name]
    idx_vars = self.node_outputs[args[1].name]
    upd_vars = self.node_outputs[args[2].name]
    data_shape = self.node_shapes[args[0].name]
    out_vars = self._alloc_ids(_prod(data_shape) or 1)
    layer_id = self._add_layer(
        LayerKind.SCATTER_ND.value,
        {"data_vars": data_vars, "indices_vars": idx_vars, "updates_vars": upd_vars,
         "input_shape": data_shape, "output_shape": data_shape},
        data_vars + idx_vars + upd_vars, out_vars,
    )
    self.prev_out = out_vars
    self.shape = data_shape
    self._register_node(node.name, layer_id)

def _convert_OnnxSqueezeDynamicAxes(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxSqueezeDynamicAxes: drop size-1 dims at axes given by the second arg."""
    if not self._get_predecessor_state(node):
        raise ValueError(f"OnnxSqueezeDynamicAxes: missing predecessor for {node.name}")
    args = [a for a in node.args if isinstance(a, fx.Node)]
    axes_t = self._resolve_constant_tensor(args[1].name) if len(args) >= 2 else None
    if axes_t is None and len(args) >= 2:
        axes_t = self._evaluate_constant_subgraph(args[1].name)
    rank = len(self.shape)
    if axes_t is not None:
        axes = sorted({(int(a) + rank) if int(a) < 0 else int(a) for a in axes_t.flatten().tolist()})
    else:
        axes = [i for i, d in enumerate(self.shape) if int(d) == 1]
    output_shape = tuple(int(d) for i, d in enumerate(self.shape) if i not in axes) or (1,)
    out_vars = self._same_size_forward()
    layer_id = self._add_layer(
        LayerKind.SQUEEZE.value,
        {"dims": list(axes)},
        self.prev_out, out_vars,
    )
    self.prev_out = out_vars
    self.shape = output_shape
    self._register_node(node.name, layer_id)

def _convert_OnnxWhere(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxWhere: y = where(cond, x, y_else) — pointwise conditional select."""
    args = [a for a in node.args if isinstance(a, fx.Node)]
    if len(args) < 3:
        raise ValueError(f"OnnxWhere at {node.name}: expected 3 args")
    for n in args[:3]:
        if n.name not in self.node_outputs and not self._ensure_constant_vars(n.name):
            raise ValueError(f"OnnxWhere at {node.name}: '{n.name}' not registered")
    cv = self.node_outputs[args[0].name]
    xv = self.node_outputs[args[1].name]
    yv = self.node_outputs[args[2].name]
    try:
        output_shape = tuple(int(d) for d in torch.broadcast_shapes(
            self.node_shapes[args[0].name],
            self.node_shapes[args[1].name],
            self.node_shapes[args[2].name],
        ))
    except RuntimeError:
        output_shape = self.node_shapes[args[1].name]
    out_vars = self._alloc_ids(_prod(output_shape) or 1)
    layer_id = self._add_layer(
        LayerKind.WHERE.value,
        {"cond_vars": cv, "x_vars": xv, "y_vars": yv,
         "input_shape": self.node_shapes[args[1].name], "output_shape": output_shape},
        cv + xv + yv, out_vars,
    )
    self.prev_out = out_vars
    self.shape = output_shape
    self._register_node(node.name, layer_id)

def _convert_OnnxFunction(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxFunction: dispatch by inner-function name."""
    func_name = getattr(getattr(mod, 'function', None), '__name__', '').lower()
    kind = {'sign': LayerKind.SIGN, 'abs': LayerKind.ABS,
            'tanh': LayerKind.TANH, 'sin': LayerKind.SIN,
            'cos': LayerKind.COS}.get(func_name)
    if kind is None:
        raise NotImplementedError(f"OnnxFunction({func_name}) at {node.name} (future enhancement)")
    if not self._get_predecessor_state(node):
        raise ValueError(f"OnnxFunction: missing predecessor for {node.name}")
    out_vars = self._same_size_forward()
    layer_id = self._add_layer(
        kind.value,
        {"input_shape": self.shape, "output_shape": self.shape},
        self.prev_out, out_vars,
    )
    self.prev_out = out_vars
    self._register_node(node.name, layer_id)

def _convert_OnnxCast(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxCast: dtype-only conversion; ACT tracks values, not dtype, so passthrough."""
    args = [a for a in node.args if isinstance(a, fx.Node)]
    if not args:
        raise ValueError(f"OnnxCast at {node.name}: no inputs")
    self._propagate_node_state(node.name, args[0].name)

def _convert_OnnxArgExtremum(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxArgExtremum: argmax / argmin along an axis."""
    if not self._get_predecessor_state(node):
        raise ValueError(f"OnnxArgExtremum: missing predecessor for {node.name}")
    op_func = getattr(mod, 'extremum_function', None)
    op_name = getattr(op_func, '__name__', '').lower() if op_func is not None else ''
    op = 'argmax' if 'max' in op_name else 'argmin'
    axis = int(getattr(mod, 'axis', 0))
    keepdims = bool(getattr(mod, 'keepdims', True))
    if axis < 0:
        axis += len(self.shape)
    if keepdims:
        output_shape = tuple(1 if i == axis else int(d) for i, d in enumerate(self.shape))
    else:
        output_shape = tuple(int(d) for i, d in enumerate(self.shape) if i != axis) or (1,)
    out_vars = self._alloc_ids(_prod(output_shape) or 1)
    layer_id = self._add_layer(
        LayerKind.ARG_EXTREMUM.value,
        {"op": op, "axis": axis, "keepdims": int(keepdims),
         "input_shape": self.shape, "output_shape": output_shape},
        self.prev_out, out_vars,
    )
    self.prev_out = out_vars
    self.shape = output_shape
    self._register_node(node.name, layer_id)

def _convert_OnnxCompare(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxCompare: element-wise comparison (eq/ne/lt/le/gt/ge) producing bool vars."""
    op_func = getattr(mod, 'compare_function', None)
    op_raw = getattr(op_func, '__name__', '').lower() if op_func is not None else ''
    op_map = {'equal': 'eq', 'eq': 'eq', 'less': 'lt', 'lt': 'lt',
              'greater': 'gt', 'gt': 'gt', 'less_equal': 'le', 'le': 'le',
              'greater_equal': 'ge', 'ge': 'ge', 'not_equal': 'ne', 'ne': 'ne'}
    op = op_map.get(op_raw)
    if op is None:
        raise NotImplementedError(f"OnnxCompare at {node.name}: unrecognised op '{op_raw}'")
    args = [a for a in node.args if isinstance(a, fx.Node)]
    x_node, y_node = args[0], args[1]
    for n in (x_node, y_node):
        if n.name not in self.node_outputs and not self._ensure_constant_vars(n.name):
            raise ValueError(f"OnnxCompare at {node.name}: '{n.name}' not registered")
    xv, yv = self.node_outputs[x_node.name], self.node_outputs[y_node.name]
    xs, ys = self.node_shapes[x_node.name], self.node_shapes[y_node.name]
    try:
        output_shape = tuple(int(d) for d in torch.broadcast_shapes(xs, ys))
    except RuntimeError:
        output_shape = xs if _prod(xs) >= _prod(ys) else ys
    out_vars = self._alloc_ids(_prod(output_shape) or 1)
    layer_id = self._add_layer(
        LayerKind.COMPARE.value,
        {"op": op, "x_vars": xv, "y_vars": yv,
         "input_shape": xs, "output_shape": output_shape},
        xv + yv, out_vars,
    )
    self.prev_out = out_vars
    self.shape = output_shape
    self._register_node(node.name, layer_id)

def _convert_OnnxDropoutDynamic(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxDropoutDynamic in eval mode: identity passthrough (no layer emitted)."""
    args = [a for a in node.args if isinstance(a, fx.Node)]
    if not args:
        raise ValueError(f"OnnxDropoutDynamic at {node.name}: no inputs")
    self._propagate_node_state(node.name, args[0].name)

def _convert_OnnxUnsqueezeStaticAxes(self, mod: nn.Module, node: fx.Node) -> None:
    """OnnxUnsqueezeStaticAxes: insert size-1 dims at static ``_axes``."""
    if not self._get_predecessor_state(node):
        raise ValueError(f"OnnxUnsqueezeStaticAxes: missing predecessor for {node.name}")
    axes = list(getattr(mod, '_axes', None) or [])
    if not axes:
        raise ValueError(f"OnnxUnsqueezeStaticAxes at {node.name}: missing _axes")
    rank_after = len(self.shape) + len(axes)
    norm_axes = sorted({a + rank_after if a < 0 else a for a in (int(x) for x in axes)})
    output_shape = list(self.shape)
    for ax in norm_axes:
        output_shape.insert(ax, 1)
    output_shape = tuple(output_shape)
    out_vars = self._same_size_forward()
    layer_id = self._add_layer(
        LayerKind.UNSQUEEZE.value,
        {"dims": list(norm_axes)},
        self.prev_out, out_vars,
    )
    self.prev_out = out_vars
    self.shape = output_shape
    self._register_node(node.name, layer_id)

ONNX_HANDLERS = {
    'OnnxArgExtremum': _convert_OnnxArgExtremum,
    'OnnxBinaryMathOperation': _convert_OnnxBinaryMathOperation,
    'OnnxCast': _convert_OnnxCast,
    'OnnxCompare': _convert_OnnxCompare,
    'OnnxConcat': _convert_OnnxConcat,
    'OnnxConstant': _convert_OnnxConstant,
    'OnnxConstantOfShape': _convert_OnnxConstantOfShape,
    'OnnxRound': _convert_OnnxRound,
    'OnnxDropoutDynamic': _convert_OnnxDropoutDynamic,
    'OnnxExpand': _convert_OnnxExpand,
    'OnnxFlatten': _convert_OnnxFlatten,
    'OnnxFunction': _convert_OnnxFunction,
    'OnnxGather': _convert_OnnxGather,
    'OnnxMatMul': _convert_OnnxMatMul,
    'OnnxMinMax': _convert_OnnxMinMax,
    'OnnxNeg': _convert_OnnxNeg,
    'OnnxPow': _convert_OnnxPow,
    'OnnxReduceStaticAxes': _convert_OnnxReduceStaticAxes,
    'OnnxReduceSumStaticAxes': _convert_OnnxReduceSumStaticAxes,
    'OnnxReshape': _convert_OnnxReshape,
    'OnnxResize': _convert_OnnxResize,
    'OnnxScatterND': _convert_OnnxScatterND,
    'OnnxShape': _convert_OnnxShape,
    'OnnxSlice': _convert_OnnxSlice,
    'OnnxSplit': _convert_OnnxSplit13,
    'OnnxSplit13': _convert_OnnxSplit13,
    'OnnxSqueezeDynamicAxes': _convert_OnnxSqueezeDynamicAxes,
    'OnnxTranspose': _convert_OnnxTranspose,
    'OnnxUnsqueezeStaticAxes': _convert_OnnxUnsqueezeStaticAxes,
    'OnnxWhere': _convert_OnnxWhere,
}
