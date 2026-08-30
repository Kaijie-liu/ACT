"""Request-local CUDA program for the exact-ReLU phase-projection path.

This module contains arithmetic plumbing only.  It does not select a property,
choose phases, solve an LP, inspect a dual, or authorize a verifier result.
Every tensor owned here is created for one verifier request; there is no
module-global cache and no alternate runtime backend.

The candidate and terminal interfaces are deliberately separated.  A
``CandidateProgram`` may contain selected CSR rows and phase schedules.  A
``TerminalProgram`` contains only stored graph data and an optional deadline,
and :func:`terminal_interval_forward` accepts only a decoded binary64 input and
that terminal program.  Candidate outputs, phases, margins, LP values, and
error bands therefore cannot enter the terminal through its API.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from act.back_end.hybridz_tf import (
    forward_exact_relu_live_row_stream_candidate as _live,
)
from act.back_end.hybridz_tf import operator_hz as _oh


_SUPPORTED = frozenset(
    {
        "INPUT",
        "INPUT_SPEC",
        "CONV2D",
        "DENSE",
        "SCALE",
        "BIAS",
        "ADD",
        "RELU",
        "FLATTEN",
        "ASSERT",
    }
)
_MAX_HOST_OUTPUT_BYTES = 2_000_000_000


class PhaseProjectionDeviceProgramError(RuntimeError):
    """Fail-closed request-local device-program error."""


@dataclass(frozen=True)
class LayerStep:
    layer_id: int
    kind: str
    predecessors: Tuple[int, ...]
    width: int


@dataclass(frozen=True)
class AffineTopology:
    """Owned scalar-only convolution geometry."""

    input_shape: Tuple[int, int, int, int]
    output_shape: Tuple[int, int, int, int]
    stride: Tuple[int, int]
    padding: Tuple[int, int]
    dilation: Tuple[int, int]
    groups: int


@dataclass(frozen=True)
class CandidateAffine:
    """The stored affine data visible to candidate center propagation."""

    kind: str
    weight: torch.Tensor
    bias: np.ndarray
    output_size: int
    topology: Optional[AffineTopology]
    deadline: Optional[float] = None


@dataclass(frozen=True)
class TerminalAffine:
    """Stored affine data needed by the independent outward terminal."""

    kind: str
    weight: torch.Tensor
    absolute_weight: torch.Tensor
    bias: np.ndarray
    output_size: int
    topology: Optional[AffineTopology]
    fanin: np.ndarray
    gamma: np.ndarray
    absolute_bias: np.ndarray


@dataclass(frozen=True)
class StoredBinary64Input:
    """One immutable owned input shared by raw-BOX and terminal authority.

    The bytes-backed array cannot be made writeable again.  Production callers
    must construct this object once, read :attr:`values` for the raw BOX, and
    pass that same object to :func:`terminal_interval_forward`.
    """

    values: np.ndarray

    def __init__(self, values: Any):
        raw = np.asarray(values)
        if raw.dtype != np.dtype(np.float64):
            raise PhaseProjectionDeviceProgramError(
                "terminal input is not stored binary64"
            )
        sealed = _readonly_array(
            raw, dtype=np.float64, name="stored_binary64_input"
        ).reshape(-1)
        object.__setattr__(self, "values", sealed)


@dataclass(frozen=True)
class InputBatch:
    rows: torch.Tensor
    columns: torch.Tensor
    radii: torch.Tensor
    start: int
    stop: int


@dataclass(frozen=True)
class CandidateProgram:
    """Immutable request-local resources used only by candidate arithmetic."""

    steps: Tuple[LayerStep, ...]
    affines: Mapping[int, CandidateAffine]
    pointwise_device: Mapping[int, torch.Tensor]
    matrices: Mapping[int, _live._DeviceCSR]
    live_rows: Mapping[int, np.ndarray]
    device_rows: Mapping[int, torch.Tensor]
    input_batches: Tuple[InputBatch, ...]
    successor_uses: Mapping[int, int]
    n_cont: int
    assert_layer_id: int
    assert_width: int
    device: torch.device
    deadline: Optional[float]
    build_seconds: float
    cuda_bytes: int


@dataclass(frozen=True)
class TerminalProgram:
    """Candidate-blind stored graph program for zero-width outward replay."""

    steps: Tuple[LayerStep, ...]
    affines: Mapping[int, TerminalAffine]
    pointwise: Mapping[int, np.ndarray]
    successor_uses: Mapping[int, int]
    output_layer_id: int
    device: torch.device
    deadline: Optional[float]


@dataclass(frozen=True)
class RequestLocalPrograms:
    candidate: CandidateProgram
    terminal: TerminalProgram


@dataclass(frozen=True)
class FixedPhaseSchedule:
    stream_rows: Mapping[int, torch.Tensor]
    active_rows: Mapping[int, torch.Tensor]


@dataclass(frozen=True)
class DeltaSchedule:
    exact_rows: Mapping[int, torch.Tensor]
    active_rows: Mapping[int, torch.Tensor]
    changed_rows: Mapping[int, Tuple[Tuple[int, int], ...]]
    width: int


def _mapping(values: Mapping[int, Any]) -> Mapping[int, Any]:
    return MappingProxyType(dict(values))


def _readonly_array(values: Any, *, dtype: Any, name: str) -> np.ndarray:
    try:
        copied = np.array(values, dtype=dtype, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PhaseProjectionDeviceProgramError(
            f"{name} could not be sealed"
        ) from exc
    if np.issubdtype(copied.dtype, np.floating) and not np.all(
        np.isfinite(copied)
    ):
        raise PhaseProjectionDeviceProgramError(f"{name} is not finite")
    # ``setflags(write=False)`` alone is reversible when the array owns its
    # storage.  Rebuild on an immutable bytes owner so neither a retained input
    # alias nor a later ``setflags(write=True)`` can change sealed authority.
    immutable = copied.tobytes(order="C")
    result = np.frombuffer(immutable, dtype=copied.dtype).reshape(copied.shape)
    result.setflags(write=False)
    return result


def seal_terminal_input(decoded: Any) -> StoredBinary64Input:
    """Return the single owned binary64 object used by BOX and terminal replay."""

    return StoredBinary64Input(decoded)


def _seal_pair(values: Any, *, name: str) -> Tuple[int, int]:
    try:
        result = tuple(int(value) for value in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PhaseProjectionDeviceProgramError(f"{name} is malformed") from exc
    if len(result) != 2:
        raise PhaseProjectionDeviceProgramError(f"{name} is malformed")
    return result


def _seal_shape(values: Any, *, name: str) -> Tuple[int, int, int, int]:
    try:
        result = tuple(int(value) for value in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PhaseProjectionDeviceProgramError(f"{name} is malformed") from exc
    if len(result) != 4 or any(value <= 0 for value in result):
        raise PhaseProjectionDeviceProgramError(f"{name} is malformed")
    return result


def _seal_topology(snapshot: Any, *, name: str) -> Optional[AffineTopology]:
    try:
        kind = str(snapshot.kind)
        topology = snapshot.topology
    except AttributeError as exc:
        raise PhaseProjectionDeviceProgramError(f"{name} is malformed") from exc
    if kind == "DENSE":
        if topology is not None:
            raise PhaseProjectionDeviceProgramError(f"{name} is malformed")
        return None
    if kind != "CONV2D" or topology is None:
        raise PhaseProjectionDeviceProgramError(f"{name} is malformed")
    input_shape = _seal_shape(topology.input_shape, name=f"{name}.input_shape")
    output_shape = _seal_shape(
        topology.output_shape, name=f"{name}.output_shape"
    )
    stride = _seal_pair(topology.stride, name=f"{name}.stride")
    padding = _seal_pair(topology.padding, name=f"{name}.padding")
    dilation = _seal_pair(topology.dilation, name=f"{name}.dilation")
    try:
        groups = int(topology.groups)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PhaseProjectionDeviceProgramError(
            f"{name}.groups is malformed"
        ) from exc
    if (
        groups <= 0
        or any(value <= 0 for value in stride)
        or any(value < 0 for value in padding)
        or any(value <= 0 for value in dilation)
    ):
        raise PhaseProjectionDeviceProgramError(f"{name} is malformed")
    return AffineTopology(
        input_shape=input_shape,
        output_shape=output_shape,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def _checked_host_output_bytes(
    shapes: Sequence[Tuple[int, int]], *, name: str
) -> int:
    """Check all binary64 result arrays before the first allocation."""

    total_elements = 0
    limit_elements = _MAX_HOST_OUTPUT_BYTES // np.dtype(np.float64).itemsize
    for shape in shapes:
        if (
            type(shape) is not tuple
            or len(shape) != 2
            or any(type(value) is not int or value < 0 for value in shape)
        ):
            raise PhaseProjectionDeviceProgramError(
                f"{name} output geometry is malformed"
            )
        rows, columns = shape
        if rows and columns > limit_elements // rows:
            raise PhaseProjectionDeviceProgramError(
                f"{name} outputs exceed the checked byte cap"
            )
        elements = rows * columns
        if elements > limit_elements - total_elements:
            raise PhaseProjectionDeviceProgramError(
                f"{name} outputs exceed the checked byte cap"
            )
        total_elements += elements
    return total_elements * np.dtype(np.float64).itemsize


def _deadline(deadline: Optional[float], stage: str) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise PhaseProjectionDeviceProgramError(
            f"phase-projection deadline expired at {stage}"
        )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _strip_traceback_frames(primary: BaseException) -> None:
    """Drop failed implementation frames without replacing the primary error."""

    pending = [primary]
    seen = set()
    while pending:
        error = pending.pop()
        identity = id(error)
        if identity in seen:
            continue
        seen.add(identity)
        error.__traceback__ = None
        if error.__cause__ is not None:
            pending.append(error.__cause__)
        if error.__context__ is not None:
            pending.append(error.__context__)


def seal_layer_steps(net: Any, order: Sequence[Any]) -> Tuple[LayerStep, ...]:
    """Seal the already-topological graph into a request-local read-only form."""

    seen = set()
    result = []
    for layer in order:
        try:
            layer_id = int(layer.id)
            kind = _oh._kind(layer.kind)
            width = int(len(layer.out_vars))
            predecessors = tuple(
                int(value) for value in net.preds.get(layer_id, [])
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise PhaseProjectionDeviceProgramError(
                "graph schedule is malformed"
            ) from exc
        if (
            layer_id in seen
            or kind not in _SUPPORTED
            or width < 0
            or any(parent not in seen for parent in predecessors)
        ):
            raise PhaseProjectionDeviceProgramError(
                "graph schedule is unsupported or not topological"
            )
        seen.add(layer_id)
        result.append(LayerStep(layer_id, kind, predecessors, width))
    if not result:
        raise PhaseProjectionDeviceProgramError("graph schedule is empty")
    return tuple(result)


def _successor_uses(steps: Sequence[LayerStep]) -> Mapping[int, int]:
    uses = {step.layer_id: 0 for step in steps}
    for step in steps:
        for predecessor in step.predecessors:
            try:
                uses[predecessor] += 1
            except KeyError as exc:
                raise PhaseProjectionDeviceProgramError(
                    "graph predecessor is absent from the schedule"
                ) from exc
    return _mapping(uses)


def _validate_deadline(deadline: Optional[float]) -> Optional[float]:
    if deadline is None:
        return None
    if type(deadline) not in {int, float} or not math.isfinite(float(deadline)):
        raise PhaseProjectionDeviceProgramError("deadline is malformed")
    return float(deadline)


def _seal_affine_pair(
    snapshot: Any,
    step: LayerStep,
    *,
    device: torch.device,
    deadline: Optional[float],
) -> Tuple[CandidateAffine, TerminalAffine]:
    """Seal disjoint candidate/terminal affine state from one source snapshot."""

    layer_id = step.layer_id
    try:
        kind = str(snapshot.kind)
        output_size = int(snapshot.output_size)
        stored_weight = _readonly_array(
            snapshot.weight,
            dtype=np.float64,
            name=f"weight[{layer_id}]",
        )
        stored_bias = _readonly_array(
            snapshot.bias,
            dtype=np.float64,
            name=f"bias[{layer_id}]",
        ).reshape(-1)
        topology = _seal_topology(snapshot, name=f"topology[{layer_id}]")
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise PhaseProjectionDeviceProgramError(
            f"affine snapshot {layer_id} is malformed"
        ) from exc
    if (
        kind != step.kind
        or output_size != step.width
        or stored_bias.size != output_size
        or np.any(stored_weight == 0.0)
        or (
            kind == "DENSE"
            and (
                stored_weight.ndim != 2
                or stored_weight.shape[0] != output_size
            )
        )
        or (
            kind == "CONV2D"
            and (
                stored_weight.ndim != 4
                or topology is None
                or math.prod(topology.output_shape) != output_size
            )
        )
    ):
        raise PhaseProjectionDeviceProgramError(
            "device program requires finite all-nonzero affine weights"
        )
    candidate_weight = torch.tensor(
        stored_weight, dtype=torch.float64, device=device
    )
    terminal_weight = torch.tensor(
        stored_weight, dtype=torch.float64, device=device
    )
    absolute_weight = torch.abs(terminal_weight)
    fanin = _readonly_array(
        _live._affine_fanin(snapshot),
        dtype=np.float64,
        name=f"fanin[{layer_id}]",
    ).reshape(-1)
    gamma = _readonly_array(
        _oh._gamma_ops(
            2.0 * fanin + 2.0,
            name=f"phase_projection.program_gamma[{layer_id}]",
        ),
        dtype=np.float64,
        name=f"gamma[{layer_id}]",
    ).reshape(-1)
    absolute_bias = _readonly_array(
        np.abs(stored_bias),
        dtype=np.float64,
        name=f"absolute_bias[{layer_id}]",
    ).reshape(-1)
    candidate_bias = _readonly_array(
        stored_bias,
        dtype=np.float64,
        name=f"candidate_bias[{layer_id}]",
    ).reshape(-1)
    terminal_bias = _readonly_array(
        stored_bias,
        dtype=np.float64,
        name=f"terminal_bias[{layer_id}]",
    ).reshape(-1)
    candidate = CandidateAffine(
        kind=kind,
        weight=candidate_weight,
        bias=candidate_bias,
        output_size=output_size,
        topology=topology,
        deadline=deadline,
    )
    terminal = TerminalAffine(
        kind=kind,
        weight=terminal_weight,
        absolute_weight=absolute_weight,
        bias=terminal_bias,
        output_size=output_size,
        topology=topology,
        fanin=fanin,
        gamma=gamma,
        absolute_bias=absolute_bias,
    )
    return candidate, terminal


def _build_request_local_programs_impl(
    net: Any,
    order: Sequence[Any],
    affines: Mapping[int, Any],
    pointwise: Mapping[int, np.ndarray],
    matrices: Mapping[int, _live._DeviceCSR],
    live_rows: Mapping[int, np.ndarray],
    *,
    input_rows: np.ndarray,
    input_radius: np.ndarray,
    assert_layer_id: int,
    output_layer_id: int,
    deadline: Optional[float],
) -> RequestLocalPrograms:
    """Build one pair of request-local programs without a global cache.

    Candidate and terminal affines have disjoint CUDA storage and own all host
    metadata.  In particular, the terminal has no selected matrix, live-row,
    input-factor, phase-schedule, or source-snapshot field.
    """

    deadline = _validate_deadline(deadline)
    if not torch.cuda.is_available():
        raise PhaseProjectionDeviceProgramError(
            "request-local device program requires CUDA"
        )
    device = torch.device("cuda")
    _deadline(deadline, "device program build")
    _sync(device)
    _deadline(deadline, "device program initial sync")
    entry_bytes = int(torch.cuda.memory_allocated(device))
    started = time.monotonic()
    _deadline(deadline, "device graph schedule")
    steps = seal_layer_steps(net, order)
    _deadline(deadline, "device graph schedule complete")
    by_id = {step.layer_id: step for step in steps}
    if (
        type(assert_layer_id) is not int
        or type(output_layer_id) is not int
        or assert_layer_id not in by_id
        or output_layer_id not in by_id
        or by_id[assert_layer_id].kind != "ASSERT"
    ):
        raise PhaseProjectionDeviceProgramError(
            "terminal or ASSERT layer id is malformed"
        )

    terminal_affines: Dict[int, TerminalAffine] = {}
    candidate_affines: Dict[int, CandidateAffine] = {}
    for layer_id, snapshot in affines.items():
        _deadline(deadline, f"device affine {layer_id}")
        if layer_id not in by_id or by_id[layer_id].kind not in {
            "CONV2D",
            "DENSE",
        }:
            raise PhaseProjectionDeviceProgramError(
                "affine snapshot does not match the graph schedule"
            )
        candidate_affine, terminal_affine = _seal_affine_pair(
            snapshot,
            by_id[layer_id],
            device=device,
            deadline=deadline,
        )
        candidate_affines[int(layer_id)] = candidate_affine
        terminal_affines[int(layer_id)] = terminal_affine
        _deadline(deadline, f"device affine {layer_id} complete")

    expected_affines = {
        step.layer_id for step in steps if step.kind in {"CONV2D", "DENSE"}
    }
    if set(terminal_affines) != expected_affines:
        raise PhaseProjectionDeviceProgramError(
            "affine program does not cover the graph"
        )

    host_pointwise: Dict[int, np.ndarray] = {}
    device_pointwise: Dict[int, torch.Tensor] = {}
    expected_pointwise = {
        step.layer_id for step in steps if step.kind in {"SCALE", "BIAS"}
    }
    for layer_id, values in pointwise.items():
        _deadline(deadline, f"device pointwise {layer_id}")
        sealed = _readonly_array(
            values, dtype=np.float64, name=f"pointwise[{layer_id}]"
        ).reshape(-1)
        if layer_id not in by_id or sealed.size != by_id[layer_id].width:
            raise PhaseProjectionDeviceProgramError(
                "pointwise program does not match the graph"
        )
        host_pointwise[int(layer_id)] = sealed
        device_pointwise[int(layer_id)] = torch.tensor(
            sealed, dtype=torch.float64, device=device
        )
        _deadline(deadline, f"device pointwise {layer_id} complete")
    if set(host_pointwise) != expected_pointwise:
        raise PhaseProjectionDeviceProgramError(
            "pointwise program does not cover the graph"
        )

    expected_matrices = expected_affines
    if set(matrices) != expected_matrices:
        raise PhaseProjectionDeviceProgramError(
            "selected CSR program does not cover the graph"
        )
    sealed_matrices: Dict[int, _live._DeviceCSR] = {}
    sealed_live_rows: Dict[int, np.ndarray] = {}
    device_rows: Dict[int, torch.Tensor] = {}
    expected_live_rows = {step.layer_id for step in steps}
    if set(live_rows) != expected_live_rows:
        raise PhaseProjectionDeviceProgramError(
            "live-row program does not cover the graph"
        )
    for layer_id, values in live_rows.items():
        _deadline(deadline, f"device live rows {layer_id}")
        sealed_live_rows[layer_id] = _readonly_array(
            values,
            dtype=np.int64,
            name=f"live_rows[{layer_id}]",
        ).reshape(-1)
        _deadline(deadline, f"device live rows {layer_id} complete")
    for layer_id in expected_matrices:
        _deadline(deadline, f"device selected CSR {layer_id}")
        matrix = matrices[layer_id]
        if not all(
            isinstance(value, torch.Tensor)
            and value.device.type == "cuda"
            for value in (matrix.indptr, matrix.indices, matrix.data)
        ):
            raise PhaseProjectionDeviceProgramError(
                "selected CSR is not request-local CUDA data"
            )
        rows = sealed_live_rows[layer_id]
        if rows.size != int(matrix.rows):
            raise PhaseProjectionDeviceProgramError(
                "selected CSR row ids drifted"
        )
        sealed_matrices[layer_id] = matrix
        device_rows[layer_id] = torch.tensor(
            rows, dtype=torch.int64, device=device
        )
        _deadline(deadline, f"device selected CSR {layer_id} complete")

    input_rows = _readonly_array(
        input_rows, dtype=np.int64, name="input_rows"
    ).reshape(-1)
    input_radius = _readonly_array(
        input_radius, dtype=np.float64, name="input_radius"
    ).reshape(-1)
    if (
        not input_rows.size
        or np.any(input_rows < 0)
        or np.any(input_rows >= input_radius.size)
        or np.any(input_rows[1:] <= input_rows[:-1])
        or np.any(input_radius[input_rows] <= 0.0)
    ):
        raise PhaseProjectionDeviceProgramError(
            "input-factor schedule is malformed"
        )
    input_batches = []
    n_cont = int(input_rows.size)
    for start in range(0, n_cont, _live._FACTOR_BATCH):
        _deadline(deadline, f"device input batch {start}")
        stop = min(n_cont, start + _live._FACTOR_BATCH)
        selected_rows = input_rows[start:stop]
        input_batches.append(
            InputBatch(
                rows=torch.tensor(
                    selected_rows, dtype=torch.int64, device=device
                ),
                columns=torch.arange(
                    stop - start, dtype=torch.int64, device=device
                ),
                radii=torch.tensor(
                    input_radius[selected_rows],
                    dtype=torch.float64,
                    device=device,
                ),
                start=start,
                stop=stop,
            )
        )
        _deadline(deadline, f"device input batch {start} complete")

    uses = _successor_uses(steps)
    _sync(device)
    _deadline(deadline, "device program build sync")
    build_seconds = time.monotonic() - started
    cuda_bytes = max(
        0, int(torch.cuda.memory_allocated(device)) - entry_bytes
    )
    candidate = CandidateProgram(
        steps=steps,
        affines=_mapping(candidate_affines),
        pointwise_device=_mapping(device_pointwise),
        matrices=_mapping(sealed_matrices),
        live_rows=_mapping(sealed_live_rows),
        device_rows=_mapping(device_rows),
        input_batches=tuple(input_batches),
        successor_uses=uses,
        n_cont=n_cont,
        assert_layer_id=assert_layer_id,
        assert_width=by_id[assert_layer_id].width,
        device=device,
        deadline=deadline,
        build_seconds=build_seconds,
        cuda_bytes=cuda_bytes,
    )
    terminal = TerminalProgram(
        steps=steps,
        affines=_mapping(terminal_affines),
        pointwise=_mapping(host_pointwise),
        successor_uses=uses,
        output_layer_id=output_layer_id,
        device=device,
        deadline=deadline,
    )
    result = RequestLocalPrograms(candidate=candidate, terminal=terminal)
    _deadline(deadline, "device program build return")
    return result


def build_request_local_programs(
    net: Any,
    order: Sequence[Any],
    affines: Mapping[int, Any],
    pointwise: Mapping[int, np.ndarray],
    matrices: Mapping[int, _live._DeviceCSR],
    live_rows: Mapping[int, np.ndarray],
    *,
    input_rows: np.ndarray,
    input_radius: np.ndarray,
    assert_layer_id: int,
    output_layer_id: int,
    deadline: Optional[float],
) -> RequestLocalPrograms:
    """Build programs and release the implementation frame on every failure."""

    try:
        return _build_request_local_programs_impl(
            net,
            order,
            affines,
            pointwise,
            matrices,
            live_rows,
            input_rows=input_rows,
            input_radius=input_radius,
            assert_layer_id=assert_layer_id,
            output_layer_id=output_layer_id,
            deadline=deadline,
        )
    except BaseException as primary:
        # CUDA tensors held only by implementation locals would otherwise stay
        # alive through the traceback.  Strip that frame before propagating the
        # exact same non-Exception primary or a fail-closed ordinary error.
        _strip_traceback_frames(primary)
        if isinstance(primary, PhaseProjectionDeviceProgramError):
            raise primary
        if isinstance(primary, Exception):
            raise PhaseProjectionDeviceProgramError(
                "request-local device program build failed"
            ) from primary
        raise primary


def _candidate_affine_center_impl(
    source: _live._Shadow,
    affine: CandidateAffine,
    *,
    layer_id: int,
) -> _live._Shadow:
    """Run the frozen center operation with a presealed device weight.

    The matmul/conv, host copy, bias addition, and finite check retain the
    original order.  Only repeated CPU-to-CUDA weight construction is removed.
    """

    _deadline(affine.deadline, f"device center {layer_id}")
    if affine.kind == "DENSE":
        value = torch.matmul(
            affine.weight,
            torch.tensor(
                source.center, dtype=torch.float64, device=affine.weight.device
            ),
        )
    else:
        topology = affine.topology
        if affine.kind != "CONV2D" or topology is None:
            raise PhaseProjectionDeviceProgramError(
                "candidate affine topology is unavailable"
            )
        batch, channels, height, width = topology.input_shape
        value = torch.nn.functional.conv2d(
            torch.tensor(
                source.center.reshape(batch, channels, height, width),
                dtype=torch.float64,
                device=affine.weight.device,
            ),
            affine.weight,
            bias=None,
            stride=topology.stride,
            padding=topology.padding,
            dilation=topology.dilation,
            groups=topology.groups,
        )
    center = value.detach().cpu().numpy().reshape(-1) + affine.bias
    if not np.all(np.isfinite(center)):
        raise PhaseProjectionDeviceProgramError(
            f"candidate affine center overflowed at layer {layer_id}"
        )
    _sync(affine.weight.device)
    _deadline(affine.deadline, f"device center {layer_id} complete")
    zero = np.zeros(center.size, dtype=np.float64)
    result = _live._Shadow(center, zero, np.abs(center))
    _deadline(affine.deadline, f"device center {layer_id} return")
    return result


def candidate_affine_center(
    source: _live._Shadow,
    affine: CandidateAffine,
    *,
    layer_id: int,
) -> _live._Shadow:
    """Run one owned candidate affine and fail closed on ordinary errors."""

    try:
        return _candidate_affine_center_impl(
            source, affine, layer_id=layer_id
        )
    except BaseException as primary:
        _strip_traceback_frames(primary)
        if isinstance(primary, PhaseProjectionDeviceProgramError):
            raise primary
        if isinstance(primary, Exception):
            raise PhaseProjectionDeviceProgramError(
                f"candidate affine center failed at layer {layer_id}"
            ) from primary
        raise primary


def seal_fixed_phase_schedule(
    program: CandidateProgram,
    frames: Mapping[int, Any],
) -> FixedPhaseSchedule:
    """Seal one fixed-cell stream schedule; no triangle injection is allowed."""

    stream_rows: Dict[int, torch.Tensor] = {}
    active_rows: Dict[int, torch.Tensor] = {}
    relu_ids = {step.layer_id for step in program.steps if step.kind == "RELU"}
    if set(frames) != relu_ids:
        raise PhaseProjectionDeviceProgramError(
            "fixed phase frames do not cover the graph ReLUs"
        )
    for layer_id in sorted(relu_ids):
        _deadline(program.deadline, f"fixed phase schedule {layer_id}")
        frame = frames[layer_id]
        rows = np.asarray(frame.stream_rows, dtype=np.int64).reshape(-1)
        exact = np.asarray(frame.exact, dtype=np.int64).reshape(-1)
        columns = np.asarray(
            frame.stream_continuous_columns, dtype=np.int64
        ).reshape(-1)
        half_widths = np.asarray(
            frame.stream_half_widths, dtype=np.float64
        ).reshape(-1)
        if (
            exact.size
            or columns.size
            or half_widths.size
            or (rows.size and np.any(rows[1:] <= rows[:-1]))
        ):
            raise PhaseProjectionDeviceProgramError(
                "stream schedule is not a fixed phase cell"
            )
        try:
            live = program.live_rows[layer_id]
        except KeyError as exc:
            raise PhaseProjectionDeviceProgramError(
                "fixed phase schedule lacks live rows"
            ) from exc
        active = np.intersect1d(
            live, np.asarray(frame.active, dtype=np.int64), assume_unique=True
        )
        stream_rows[layer_id] = torch.tensor(
            rows, dtype=torch.int64, device=program.device
        )
        active_rows[layer_id] = torch.tensor(
            active, dtype=torch.int64, device=program.device
        )
        _deadline(program.deadline, f"fixed phase schedule {layer_id} complete")
    result = FixedPhaseSchedule(
        stream_rows=_mapping(stream_rows), active_rows=_mapping(active_rows)
    )
    _deadline(program.deadline, "fixed phase schedule return")
    return result


def _stream_fixed_cell_generators_impl(
    program: CandidateProgram,
    schedule: FixedPhaseSchedule,
) -> Tuple[Dict[int, np.ndarray], np.ndarray, float]:
    """Propagate the first stream with presealed per-batch index tensors.

    Full preactivation and output arrays live on the host, as required by the
    subsequent LP.  Only the current factor batch is resident on the device;
    this deliberately avoids a ``phase_rows x n_cont`` CUDA allocation on
    TinyImageNet.
    """

    _deadline(program.deadline, "device first stream allocation")
    output_shapes = tuple(
        (int(rows.numel()), program.n_cont)
        for rows in schedule.stream_rows.values()
    ) + ((program.assert_width, program.n_cont),)
    _checked_host_output_bytes(output_shapes, name="first stream")
    preactivation_dense = {
        layer_id: np.empty(
            (int(rows.numel()), program.n_cont), dtype=np.float64
        )
        for layer_id, rows in schedule.stream_rows.items()
    }
    output_dense = np.empty(
        (program.assert_width, program.n_cont), dtype=np.float64
    )
    _deadline(program.deadline, "device first stream allocation complete")
    finite = torch.ones((), dtype=torch.bool, device=program.device)
    started = time.monotonic()
    for batch in program.input_batches:
        _deadline(program.deadline, f"device first stream {batch.start}")
        width = batch.stop - batch.start
        values: Dict[int, torch.Tensor] = {}
        remaining_uses = dict(program.successor_uses)
        try:
            for step in program.steps:
                layer_id = step.layer_id
                _deadline(
                    program.deadline,
                    f"device first stream {batch.start} layer {layer_id}",
                )
                kind = step.kind
                predecessors = step.predecessors
                source = None
                selected = None
                value = None
                if kind == "INPUT":
                    value = torch.zeros(
                        (step.width, width),
                        dtype=torch.float64,
                        device=program.device,
                    )
                    value[batch.rows, batch.columns] = batch.radii
                    values[layer_id] = value
                elif kind in {"INPUT_SPEC", "FLATTEN", "ASSERT"}:
                    values[layer_id] = values[predecessors[0]]
                elif kind in {"CONV2D", "DENSE"}:
                    selected = _live._ordered_csr_dense(
                        program.matrices[layer_id], values[predecessors[0]]
                    )
                    value = torch.zeros(
                        (step.width, width),
                        dtype=torch.float64,
                        device=program.device,
                    )
                    rows = program.device_rows[layer_id]
                    if rows.numel():
                        value[rows] = selected
                    values[layer_id] = value
                    finite = finite & torch.isfinite(selected).all()
                elif kind in {"SCALE", "BIAS"}:
                    source = values[predecessors[0]]
                    if kind == "SCALE":
                        values[layer_id] = source * program.pointwise_device[
                            layer_id
                        ].reshape(-1, 1)
                        finite = finite & torch.isfinite(values[layer_id]).all()
                    else:
                        values[layer_id] = source
                elif kind == "ADD":
                    values[layer_id] = (
                        values[predecessors[0]] + values[predecessors[1]]
                    )
                    finite = finite & torch.isfinite(values[layer_id]).all()
                elif kind == "RELU":
                    source = values[predecessors[0]]
                    preactivation_dense[layer_id][
                        :, batch.start : batch.stop
                    ] = source[
                        schedule.stream_rows[layer_id]
                    ].detach().cpu().numpy()
                    value = torch.zeros_like(source)
                    active = schedule.active_rows[layer_id]
                    if active.numel():
                        value[active] = source[active]
                    values[layer_id] = value
                else:  # pragma: no cover - sealed at construction
                    raise PhaseProjectionDeviceProgramError(
                        f"unsupported stream kind {kind}"
                    )
                for predecessor in predecessors:
                    remaining_uses[predecessor] -= 1
                    if remaining_uses[predecessor] == 0:
                        del values[predecessor]
                source = None
                selected = None
                value = None
                _deadline(
                    program.deadline,
                    f"device first stream {batch.start} layer {layer_id} complete",
                )
            output_dense[:, batch.start : batch.stop] = values[
                program.assert_layer_id
            ].detach().cpu().numpy()
            _deadline(
                program.deadline,
                f"device first stream {batch.start} complete",
            )
        finally:
            values.clear()
    if not bool(finite.item()):
        raise PhaseProjectionDeviceProgramError(
            "device generator stream overflowed"
        )
    _sync(program.device)
    _deadline(program.deadline, "device first stream sync")
    result = preactivation_dense, output_dense, time.monotonic() - started
    _deadline(program.deadline, "device first stream return")
    return result


def stream_fixed_cell_generators(
    program: CandidateProgram,
    schedule: FixedPhaseSchedule,
) -> Tuple[Dict[int, np.ndarray], np.ndarray, float]:
    """Propagate one fixed cell, releasing local CUDA data on all failures."""

    try:
        return _stream_fixed_cell_generators_impl(program, schedule)
    except BaseException as primary:
        _strip_traceback_frames(primary)
        if isinstance(primary, PhaseProjectionDeviceProgramError):
            raise primary
        if isinstance(primary, Exception):
            raise PhaseProjectionDeviceProgramError(
                "device generator stream failed"
            ) from primary
        raise primary


def seal_delta_schedule(
    program: CandidateProgram,
    original_frames: Mapping[int, Any],
    target_frames: Mapping[int, Any],
    changes: Sequence[Tuple[int, int, bool, bool]],
) -> DeltaSchedule:
    """Seal the single deterministic phase-delta schedule once per request."""

    relu_steps = tuple(step for step in program.steps if step.kind == "RELU")
    relu_ids = {step.layer_id for step in relu_steps}
    if set(original_frames) != relu_ids or set(target_frames) != relu_ids:
        raise PhaseProjectionDeviceProgramError(
            "delta phase frames do not cover the graph ReLUs"
        )
    _deadline(program.deadline, "delta schedule")
    expected_changes = []
    sealed_frames: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for step in relu_steps:
        layer_id = step.layer_id
        try:
            rows = np.asarray(
                original_frames[layer_id].stream_rows, dtype=np.int64
            ).reshape(-1)
            target_rows = np.asarray(
                target_frames[layer_id].stream_rows, dtype=np.int64
            ).reshape(-1)
            base_active_rows = np.asarray(
                original_frames[layer_id].active, dtype=np.int64
            ).reshape(-1)
            target_active_rows = np.asarray(
                target_frames[layer_id].active, dtype=np.int64
            ).reshape(-1)
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            raise PhaseProjectionDeviceProgramError(
                "delta phase frame is malformed"
            ) from exc
        if (
            (rows.size and np.any(rows[1:] <= rows[:-1]))
            or not np.array_equal(rows, target_rows)
            or (base_active_rows.size and np.any(
                base_active_rows[1:] <= base_active_rows[:-1]
            ))
            or (target_active_rows.size and np.any(
                target_active_rows[1:] <= target_active_rows[:-1]
            ))
            or np.any(base_active_rows < 0)
            or np.any(base_active_rows >= step.width)
            or np.any(target_active_rows < 0)
            or np.any(target_active_rows >= step.width)
        ):
            raise PhaseProjectionDeviceProgramError(
                "delta phase frame is malformed"
            )
        base_set = set(int(value) for value in base_active_rows)
        target_set = set(int(value) for value in target_active_rows)
        for row in sorted(base_set.symmetric_difference(target_set)):
            expected_changes.append(
                (layer_id, row, row in base_set, row in target_set)
            )
        sealed_frames[layer_id] = (rows, base_active_rows, target_active_rows)

    normalized_changes = []
    try:
        for change in changes:
            if len(change) != 4:
                raise PhaseProjectionDeviceProgramError(
                    "phase change is malformed"
                )
            layer_id, row, base_active, target_active = change
            if type(base_active) is not bool or type(target_active) is not bool:
                raise PhaseProjectionDeviceProgramError(
                    "phase change is malformed"
                )
            normalized_changes.append(
                (int(layer_id), int(row), base_active, target_active)
            )
    except (TypeError, ValueError, OverflowError) as exc:
        raise PhaseProjectionDeviceProgramError(
            "phase change is malformed"
        ) from exc
    if tuple(normalized_changes) != tuple(expected_changes):
        raise PhaseProjectionDeviceProgramError(
            "phase changes are not the exact topological active-set difference"
        )
    change_index = {
        (layer_id, row): index
        for index, (layer_id, row, _base, _target) in enumerate(
            normalized_changes
        )
    }
    exact_rows: Dict[int, torch.Tensor] = {}
    active_rows: Dict[int, torch.Tensor] = {}
    changed_rows: Dict[int, Tuple[Tuple[int, int], ...]] = {}
    for step in relu_steps:
        layer_id = step.layer_id
        _deadline(program.deadline, f"delta schedule {layer_id}")
        rows, _base_active_rows, target_active_rows = sealed_frames[layer_id]
        row_set = set(int(value) for value in rows)
        local = tuple(
            (int(row), index)
            for (local_layer, row), index in change_index.items()
            if local_layer == layer_id
        )
        if any(row not in row_set for row, _index in local):
            raise PhaseProjectionDeviceProgramError(
                "phase change is outside the exact-row schedule"
            )
        live = program.live_rows[layer_id]
        active = np.intersect1d(
            live,
            target_active_rows,
            assume_unique=True,
        )
        exact_rows[layer_id] = torch.tensor(
            rows, dtype=torch.int64, device=program.device
        )
        active_rows[layer_id] = torch.tensor(
            active, dtype=torch.int64, device=program.device
        )
        changed_rows[layer_id] = local
        _deadline(program.deadline, f"delta schedule {layer_id} complete")
    _deadline(program.deadline, "delta schedule return")
    return DeltaSchedule(
        exact_rows=_mapping(exact_rows),
        active_rows=_mapping(active_rows),
        changed_rows=_mapping(changed_rows),
        width=len(normalized_changes),
    )


def _stream_phase_deltas_impl(
    program: CandidateProgram,
    schedule: DeltaSchedule,
) -> Tuple[Dict[int, np.ndarray], np.ndarray, float]:
    """Propagate one phase-delta program using the presealed request schedule."""

    _deadline(program.deadline, "device phase delta allocation")
    output_shapes = tuple(
        (int(rows.numel()), schedule.width)
        for rows in schedule.exact_rows.values()
    ) + ((program.assert_width, schedule.width),)
    _checked_host_output_bytes(output_shapes, name="phase delta")
    delta_pre = {
        layer_id: np.empty(
            (int(rows.numel()), schedule.width), dtype=np.float64
        )
        for layer_id, rows in schedule.exact_rows.items()
    }
    delta_output = np.empty(
        (program.assert_width, schedule.width), dtype=np.float64
    )
    _deadline(program.deadline, "device phase delta allocation complete")
    started = time.monotonic()
    finite = torch.ones((), dtype=torch.bool, device=program.device)
    for start in range(0, schedule.width, _live._FACTOR_BATCH):
        _deadline(program.deadline, f"device phase delta {start}")
        stop = min(schedule.width, start + _live._FACTOR_BATCH)
        width = stop - start
        values: Dict[int, torch.Tensor] = {}
        remaining_uses = dict(program.successor_uses)
        try:
            for step in program.steps:
                layer_id = step.layer_id
                _deadline(
                    program.deadline,
                    f"device phase delta {start} layer {layer_id}",
                )
                kind = step.kind
                predecessors = step.predecessors
                source = None
                selected = None
                value = None
                if kind == "INPUT":
                    values[layer_id] = torch.zeros(
                        (step.width, width),
                        dtype=torch.float64,
                        device=program.device,
                    )
                elif kind in {"INPUT_SPEC", "FLATTEN", "ASSERT"}:
                    values[layer_id] = values[predecessors[0]]
                elif kind in {"CONV2D", "DENSE"}:
                    selected = _live._ordered_csr_dense(
                        program.matrices[layer_id], values[predecessors[0]]
                    )
                    value = torch.zeros(
                        (step.width, width),
                        dtype=torch.float64,
                        device=program.device,
                    )
                    rows = program.device_rows[layer_id]
                    if rows.numel():
                        value[rows] = selected
                    values[layer_id] = value
                    finite = finite & torch.isfinite(selected).all()
                elif kind in {"SCALE", "BIAS"}:
                    source = values[predecessors[0]]
                    if kind == "SCALE":
                        values[layer_id] = source * program.pointwise_device[
                            layer_id
                        ].reshape(-1, 1)
                        finite = finite & torch.isfinite(values[layer_id]).all()
                    else:
                        values[layer_id] = source
                elif kind == "ADD":
                    values[layer_id] = (
                        values[predecessors[0]] + values[predecessors[1]]
                    )
                    finite = finite & torch.isfinite(values[layer_id]).all()
                elif kind == "RELU":
                    source = values[predecessors[0]]
                    delta_pre[layer_id][:, start:stop] = source[
                        schedule.exact_rows[layer_id]
                    ].detach().cpu().numpy()
                    value = torch.zeros_like(source)
                    active = schedule.active_rows[layer_id]
                    if active.numel():
                        value[active] = source[active]
                    # Preserve the frozen row-by-row identity-injection order.
                    for row, column in schedule.changed_rows[layer_id]:
                        value[row] = 0.0
                        if start <= column < stop:
                            value[row, column - start] = 1.0
                    values[layer_id] = value
                else:  # pragma: no cover - sealed at construction
                    raise PhaseProjectionDeviceProgramError(
                        f"unsupported delta kind {kind}"
                    )
                for predecessor in predecessors:
                    remaining_uses[predecessor] -= 1
                    if remaining_uses[predecessor] == 0:
                        del values[predecessor]
                source = None
                selected = None
                value = None
                _deadline(
                    program.deadline,
                    f"device phase delta {start} layer {layer_id} complete",
                )
            delta_output[:, start:stop] = values[
                program.assert_layer_id
            ].detach().cpu().numpy()
            _deadline(
                program.deadline,
                f"device phase delta {start} complete",
            )
        finally:
            values.clear()
    if not bool(finite.item()):
        raise PhaseProjectionDeviceProgramError("device phase delta overflowed")
    _sync(program.device)
    _deadline(program.deadline, "device phase delta sync")
    result = delta_pre, delta_output, time.monotonic() - started
    _deadline(program.deadline, "device phase delta return")
    return result


def stream_phase_deltas(
    program: CandidateProgram,
    schedule: DeltaSchedule,
) -> Tuple[Dict[int, np.ndarray], np.ndarray, float]:
    """Propagate phase deltas, releasing local CUDA data on all failures."""

    try:
        return _stream_phase_deltas_impl(program, schedule)
    except BaseException as primary:
        _strip_traceback_frames(primary)
        if isinstance(primary, PhaseProjectionDeviceProgramError):
            raise primary
        if isinstance(primary, Exception):
            raise PhaseProjectionDeviceProgramError(
                "device phase delta failed"
            ) from primary
        raise primary


def _terminal_support(
    affine: TerminalAffine,
    mass_mask: np.ndarray,
    error_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute exact support booleans using the already-stored abs weight.

    The admitted affine domain has no zero weights.  Every convolution term is
    therefore nonnegative and at least one term is strictly positive exactly
    when structural support exists; no extra all-ones kernel is retained.
    """

    if affine.kind == "DENSE":
        return (
            np.full(affine.output_size, bool(np.any(mass_mask)), dtype=bool),
            np.full(affine.output_size, bool(np.any(error_mask)), dtype=bool),
        )
    topology = affine.topology
    if affine.kind != "CONV2D" or topology is None:
        raise PhaseProjectionDeviceProgramError(
            "terminal convolution topology is unavailable"
        )
    batch, channels, height, width = topology.input_shape
    source = np.stack((mass_mask, error_mask), axis=0)
    device_source = torch.tensor(
        source.reshape(2 * batch, channels, height, width),
        dtype=torch.float64,
        device=affine.absolute_weight.device,
    )
    counts = torch.nn.functional.conv2d(
        device_source,
        affine.absolute_weight,
        bias=None,
        stride=topology.stride,
        padding=topology.padding,
        dilation=topology.dilation,
        groups=topology.groups,
    )
    masks = (counts > 0.0).detach().cpu().numpy().reshape(
        2, affine.output_size
    )
    return masks[0], masks[1]


def _terminal_affine_shadow(
    source: _live._Shadow,
    affine: TerminalAffine,
    *,
    layer_id: int,
) -> _live._Shadow:
    if affine.kind == "DENSE":
        source_center = torch.tensor(
            source.center.reshape(-1, 1),
            dtype=torch.float64,
            device=affine.weight.device,
        )
        source_nonnegative = torch.tensor(
            np.column_stack((source.mass_upper, source.error)),
            dtype=torch.float64,
            device=affine.weight.device,
        )
        raw_center_tensor = torch.matmul(affine.weight, source_center)
        raw_nonnegative_tensor = torch.matmul(
            affine.absolute_weight, source_nonnegative
        )
    else:
        topology = affine.topology
        if affine.kind != "CONV2D" or topology is None:
            raise PhaseProjectionDeviceProgramError(
                "terminal convolution topology is unavailable"
            )
        batch, channels, height, width = topology.input_shape
        source_center = torch.tensor(
            source.center.reshape(batch, channels, height, width),
            dtype=torch.float64,
            device=affine.weight.device,
        )
        raw_center_tensor = torch.nn.functional.conv2d(
            source_center,
            affine.weight,
            bias=None,
            stride=topology.stride,
            padding=topology.padding,
            dilation=topology.dilation,
            groups=topology.groups,
        )
        nonnegative = np.stack((source.mass_upper, source.error), axis=0)
        source_nonnegative = torch.tensor(
            nonnegative.reshape(2 * batch, channels, height, width),
            dtype=torch.float64,
            device=affine.weight.device,
        )
        raw_nonnegative_tensor = torch.nn.functional.conv2d(
            source_nonnegative,
            affine.absolute_weight,
            bias=None,
            stride=topology.stride,
            padding=topology.padding,
            dilation=topology.dilation,
            groups=topology.groups,
        ).reshape(2, affine.output_size).transpose(0, 1)
    raw_center = raw_center_tensor.detach().cpu().numpy().reshape(-1)
    raw_nonnegative = raw_nonnegative_tensor.detach().cpu().numpy().reshape(
        affine.output_size, 2
    )
    center = raw_center + affine.bias
    mass_support, error_support = _terminal_support(
        affine, source.mass_upper > 0.0, source.error > 0.0
    )
    try:
        transformed = _live._positive_gpu_result_upper(
            raw_nonnegative[:, 0],
            affine.fanin,
            mass_support,
            name=f"phase_projection.program_mass[{layer_id}]",
        )
        arithmetic_mass = _oh._nonnegative_sum_upper(
            transformed,
            affine.absolute_bias,
            name=f"phase_projection.program_arithmetic_mass[{layer_id}]",
        )
        propagated = _live._positive_gpu_result_upper(
            raw_nonnegative[:, 1],
            affine.fanin,
            error_support,
            name=f"phase_projection.program_propagated_error[{layer_id}]",
        )
        arithmetic_error = _oh._inflate_nonnegative(
            affine.gamma * arithmetic_mass,
            4,
            active=arithmetic_mass > 0.0,
            name=f"phase_projection.program_arithmetic_error[{layer_id}]",
        )
        error = _oh._nonnegative_sum_upper(
            propagated,
            arithmetic_error,
            name=f"phase_projection.program_total_error[{layer_id}]",
        )
        mass = _oh._nonnegative_sum_upper(
            transformed,
            affine.absolute_bias,
            arithmetic_error,
            name=f"phase_projection.program_output_mass[{layer_id}]",
        )
    except Exception as exc:
        raise PhaseProjectionDeviceProgramError(
            f"terminal affine envelope failed at layer {layer_id}"
        ) from exc
    if not np.all(np.isfinite(center)):
        raise PhaseProjectionDeviceProgramError(
            f"terminal affine center overflowed at layer {layer_id}"
        )
    return _live._Shadow(center, error, mass)


def _terminal_interval_forward_impl(
    decoded: StoredBinary64Input,
    program: TerminalProgram,
) -> Tuple[np.ndarray, np.ndarray]:
    """Replay one decoded binary64 input; no candidate object is accepted."""

    if type(decoded) is not StoredBinary64Input:
        raise PhaseProjectionDeviceProgramError(
            "terminal requires one sealed stored-binary64 input"
        )
    flat = decoded.values
    owner: Any = flat
    while isinstance(getattr(owner, "base", None), np.ndarray):
        owner = owner.base
    if (
        type(flat) is not np.ndarray
        or flat.dtype != np.dtype(np.float64)
        or flat.ndim != 1
        or not flat.flags.c_contiguous
        or flat.flags.writeable
        or not isinstance(getattr(owner, "base", None), bytes)
        or not np.all(np.isfinite(flat))
    ):
        raise PhaseProjectionDeviceProgramError(
            "terminal input is not immutable owned stored binary64"
        )
    _deadline(program.deadline, "device terminal input")
    shadows: Dict[int, _live._Shadow] = {}
    try:
        remaining_uses = dict(program.successor_uses)
        terminal_output: Optional[_live._Shadow] = None
        input_mass = np.abs(flat)
        for step in program.steps:
            layer_id = step.layer_id
            _deadline(program.deadline, f"device terminal layer {layer_id}")
            kind = step.kind
            predecessors = step.predecessors
            source = None
            if kind == "INPUT":
                if flat.size != step.width:
                    raise PhaseProjectionDeviceProgramError(
                        "terminal input width drifted"
                    )
                shadows[layer_id] = _live._Shadow(
                    flat, np.zeros(flat.size, dtype=np.float64), input_mass
                )
            elif kind in {"INPUT_SPEC", "FLATTEN", "ASSERT"}:
                shadows[layer_id] = shadows[predecessors[0]]
            elif kind in {"CONV2D", "DENSE"}:
                shadows[layer_id] = _terminal_affine_shadow(
                    shadows[predecessors[0]],
                    program.affines[layer_id],
                    layer_id=layer_id,
                )
            elif kind in {"SCALE", "BIAS"}:
                source = shadows[predecessors[0]]
                lower = np.nextafter(source.center - source.error, -np.inf)
                upper = np.nextafter(source.center + source.error, np.inf)
                parameter = program.pointwise[layer_id]
                if kind == "SCALE":
                    first = lower * parameter
                    second = upper * parameter
                    lower = np.nextafter(np.minimum(first, second), -np.inf)
                    upper = np.nextafter(np.maximum(first, second), np.inf)
                else:
                    lower = np.nextafter(lower + parameter, -np.inf)
                    upper = np.nextafter(upper + parameter, np.inf)
                if not (
                    np.all(np.isfinite(lower))
                    and np.all(np.isfinite(upper))
                    and np.all(lower <= upper)
                ):
                    raise PhaseProjectionDeviceProgramError(
                        f"terminal {kind} interval overflowed"
                    )
                center, error = _oh._enclosing_center_radius(
                    lower,
                    upper,
                    name=f"phase_projection.program_{kind.lower()}[{layer_id}]",
                )
                mass = _oh._nonnegative_sum_upper(
                    np.abs(center),
                    error,
                    name=(
                        f"phase_projection.program_{kind.lower()}_mass[{layer_id}]"
                    ),
                )
                shadows[layer_id] = _live._Shadow(center, error, mass)
            elif kind == "ADD":
                shadows[layer_id] = _live._add_shadow(
                    shadows[predecessors[0]],
                    shadows[predecessors[1]],
                    layer_id=layer_id,
                )
            elif kind == "RELU":
                source = shadows[predecessors[0]]
                lower = np.nextafter(source.center - source.error, -np.inf)
                upper = np.nextafter(source.center + source.error, np.inf)
                if not (
                    np.all(np.isfinite(lower))
                    and np.all(np.isfinite(upper))
                    and np.all(lower <= upper)
                ):
                    raise PhaseProjectionDeviceProgramError(
                        "terminal ReLU input interval overflowed"
                    )
                relu_lower = np.maximum(lower, 0.0)
                relu_upper = np.maximum(upper, 0.0)
                center, error = _oh._enclosing_center_radius(
                    relu_lower,
                    relu_upper,
                    name=f"phase_projection.program_relu[{layer_id}]",
                )
                mass = _oh._nonnegative_sum_upper(
                    np.abs(center),
                    error,
                    name=f"phase_projection.program_relu_mass[{layer_id}]",
                )
                shadows[layer_id] = _live._Shadow(center, error, mass)
            else:  # pragma: no cover - sealed at construction
                raise PhaseProjectionDeviceProgramError(
                    f"unsupported terminal kind {kind}"
                )
            if layer_id == program.output_layer_id:
                terminal_output = shadows[layer_id]
            source = None
            for predecessor in predecessors:
                remaining_uses[predecessor] -= 1
                if remaining_uses[predecessor] == 0:
                    del shadows[predecessor]
            _deadline(
                program.deadline,
                f"device terminal layer {layer_id} complete",
            )
        if terminal_output is None:
            raise PhaseProjectionDeviceProgramError(
                "terminal did not reach the graph output"
            )
        lower = np.nextafter(
            terminal_output.center - terminal_output.error, -np.inf
        )
        upper = np.nextafter(
            terminal_output.center + terminal_output.error, np.inf
        )
        if not (
            np.all(np.isfinite(lower))
            and np.all(np.isfinite(upper))
            and np.all(lower <= upper)
        ):
            raise PhaseProjectionDeviceProgramError(
                "terminal output interval is malformed"
            )
        _sync(program.device)
        _deadline(program.deadline, "device terminal sync")
        sealed_lower = _readonly_array(
            lower, dtype=np.float64, name="terminal_lower"
        )
        sealed_upper = _readonly_array(
            upper, dtype=np.float64, name="terminal_upper"
        )
        _deadline(program.deadline, "device terminal return")
        return sealed_lower, sealed_upper
    finally:
        shadows.clear()


def terminal_interval_forward(
    decoded: StoredBinary64Input,
    program: TerminalProgram,
) -> Tuple[np.ndarray, np.ndarray]:
    """Replay one owned input and release all local shadows on every failure."""

    try:
        return _terminal_interval_forward_impl(decoded, program)
    except BaseException as primary:
        _strip_traceback_frames(primary)
        if isinstance(primary, PhaseProjectionDeviceProgramError):
            raise primary
        if isinstance(primary, Exception):
            raise PhaseProjectionDeviceProgramError(
                "terminal device program failed"
            ) from primary
        raise primary


__all__ = [
    "AffineTopology",
    "CandidateAffine",
    "CandidateProgram",
    "DeltaSchedule",
    "FixedPhaseSchedule",
    "InputBatch",
    "LayerStep",
    "PhaseProjectionDeviceProgramError",
    "RequestLocalPrograms",
    "StoredBinary64Input",
    "TerminalAffine",
    "TerminalProgram",
    "build_request_local_programs",
    "candidate_affine_center",
    "seal_delta_schedule",
    "seal_fixed_phase_schedule",
    "seal_layer_steps",
    "seal_terminal_input",
    "stream_fixed_cell_generators",
    "stream_phase_deltas",
    "terminal_interval_forward",
]
