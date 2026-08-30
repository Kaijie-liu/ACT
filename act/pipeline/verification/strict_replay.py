"""Independent ONNX Runtime + raw-VNNLIB counterexample replay.

The callable in this module is intentionally separate from ACT's converted
model and canonical output rows.  A candidate is accepted only after the
original ONNX model runs on CPU and the original VNNLIB 1.0/2.0 assertion tree
holds at literal zero tolerance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from act.front_end.vnnlib_loader.vnnlib_parser import (
    evaluate_vnnlib_concrete,
    extract_vnnlib_concrete_layout,
)


_CPU_PROVIDER = "CPUExecutionProvider"
_ORT_DTYPE_MAP = {
    "tensor(float16)": np.dtype(np.float16),
    "tensor(float)": np.dtype(np.float32),
    "tensor(double)": np.dtype(np.float64),
    "tensor(int8)": np.dtype(np.int8),
    "tensor(uint8)": np.dtype(np.uint8),
    "tensor(int16)": np.dtype(np.int16),
    "tensor(uint16)": np.dtype(np.uint16),
    "tensor(int32)": np.dtype(np.int32),
    "tensor(uint32)": np.dtype(np.uint32),
    "tensor(int64)": np.dtype(np.int64),
    "tensor(uint64)": np.dtype(np.uint64),
    "tensor(bool)": np.dtype(np.bool_),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(tuple(int(dim) for dim in array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _as_numeric_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "numpy") and callable(value.numpy):
        value = value.numpy()
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise TypeError(f"candidate dtype {array.dtype} is not real numeric")
    if array.size == 0:
        raise ValueError("candidate input is empty")
    if not bool(np.all(np.isfinite(array))):
        raise ValueError("candidate input contains NaN or infinity")
    return array


def _safe_cast(value: np.ndarray, target: np.dtype) -> np.ndarray:
    if target == np.dtype(np.bool_):
        if not bool(np.all((value == 0) | (value == 1))):
            raise ValueError("boolean ONNX input requires only 0/1 candidate values")
    elif np.issubdtype(target, np.integer):
        if not bool(np.all(value == np.trunc(value))):
            raise ValueError("integer ONNX input requires integral candidate values")
        limits = np.iinfo(target)
        if not bool(np.all((value >= limits.min) & (value <= limits.max))):
            raise ValueError(f"candidate value is outside {target} range")
    with np.errstate(over="ignore", invalid="ignore"):
        cast = value.astype(target, copy=False)
    if np.issubdtype(target, np.floating) and not bool(np.all(np.isfinite(cast))):
        raise ValueError(f"candidate is non-finite after cast to {target}")
    return np.ascontiguousarray(cast)


def _positive_fixed_dim(dim: Any) -> Optional[int]:
    if isinstance(dim, bool) or dim is None:
        return None
    if isinstance(dim, (int, np.integer)):
        value = int(dim)
        return value if value > 0 else None
    try:
        value = int(dim)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _shape_matches_metadata(shape: Sequence[int], metadata: Sequence[Any]) -> bool:
    if len(shape) != len(metadata):
        return False
    return all(
        fixed is None or int(actual) == fixed
        for actual, raw in zip(shape, metadata)
        for fixed in [_positive_fixed_dim(raw)]
    )


def _resolve_model_input_shape(
    metadata_shape: Sequence[Any],
    declared_shape: Sequence[int],
    candidate_shape: Sequence[int],
    numel: int,
) -> Tuple[int, ...]:
    dynamic_positions = [
        index
        for index, dim in enumerate(metadata_shape)
        if _positive_fixed_dim(dim) is None
    ]
    if any(index != 0 for index in dynamic_positions):
        raise ValueError(
            "strict replay only permits a dynamic leading batch dimension; "
            f"got ONNX input shape {list(metadata_shape)!r}"
        )

    declared = tuple(int(dim) for dim in declared_shape)
    candidate = tuple(int(dim) for dim in candidate_shape)
    candidates = [declared, candidate]
    if len(metadata_shape) == len(declared) + 1:
        candidates.append((1, *declared))
    if declared and declared[0] == 1 and len(metadata_shape) + 1 == len(declared):
        candidates.append(declared[1:])

    seen = set()
    for shape in candidates:
        if shape in seen:
            continue
        seen.add(shape)
        if int(np.prod(shape, dtype=np.int64)) != int(numel):
            continue
        if _shape_matches_metadata(shape, metadata_shape):
            if dynamic_positions and shape[0] != 1:
                continue
            return shape

    fixed = [_positive_fixed_dim(dim) for dim in metadata_shape]
    dynamic = [index for index, dim in enumerate(fixed) if dim is None]
    if not dynamic:
        shape = tuple(int(dim) for dim in fixed if dim is not None)
        if int(np.prod(shape, dtype=np.int64)) == int(numel):
            return shape
    elif dynamic == [0]:
        known = 1
        for dim in fixed:
            if dim is not None:
                known *= dim
        if known > 0 and numel % known == 0:
            inferred = numel // known
            shape_list = [
                inferred if dim is None else int(dim)
                for dim in fixed
            ]
            shape = tuple(shape_list)
            if inferred == 1 and _shape_matches_metadata(shape, metadata_shape):
                return shape

    raise ValueError(
        f"candidate with {numel} elements cannot match model input shape "
        f"{list(metadata_shape)!r} and VNNLIB shape {list(declared_shape)!r}"
    )


def _validate_model_output_shape(
    actual_shape: Sequence[int],
    metadata_shape: Sequence[Any],
    declared_shape: Sequence[int],
) -> None:
    """Validate one-output shape, allowing only an explicit batch of one."""

    actual = tuple(int(dim) for dim in actual_shape)
    declared = tuple(int(dim) for dim in declared_shape)
    dynamic_positions = [
        index
        for index, dim in enumerate(metadata_shape)
        if _positive_fixed_dim(dim) is None
    ]
    if any(index != 0 for index in dynamic_positions):
        raise ValueError(
            "strict replay only permits a dynamic leading output batch; "
            f"got ONNX output shape {list(metadata_shape)!r}"
        )
    if not _shape_matches_metadata(actual, metadata_shape):
        raise ValueError(
            f"ORT output shape {actual} does not match ONNX metadata "
            f"{list(metadata_shape)!r}"
        )
    if dynamic_positions and (not actual or actual[0] != 1):
        raise ValueError(
            f"strict replay requires output batch size 1, got {actual}"
        )

    admissible = {declared}
    if declared:
        admissible.add((1, *declared))
        if declared[0] == 1:
            admissible.add(declared[1:])
    if actual not in admissible:
        raise ValueError(
            f"ORT output shape {actual} is incompatible with VNNLIB output "
            f"shape {declared}"
        )


def _json_fingerprint(value: Dict[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class StrictReplay:
    """Lazy, serialized CPU ONNX Runtime authority for one model/spec pair."""

    def __init__(self, onnx_path: str | Path, vnnlib_path: str | Path):
        self.onnx_path = Path(onnx_path).expanduser().resolve()
        self.vnnlib_path = Path(vnnlib_path).expanduser().resolve()
        self._session = None
        self._ort_version: Optional[str] = None
        self._session_config: Optional[Dict[str, Any]] = None
        self._input_meta = None
        self._output_meta = None
        self._declared_input_shape: Optional[Tuple[int, ...]] = None
        self._declared_output_shape: Optional[Tuple[int, ...]] = None
        self._dialect: Optional[str] = None
        self._vnnlib_version: Optional[str] = None
        self._model_sha256: Optional[str] = None
        self._vnnlib_sha256: Optional[str] = None
        self._lock = threading.Lock()

    def _ensure_hashes(self) -> None:
        if not self.onnx_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")
        model_sha256 = _sha256_file(self.onnx_path)
        if not self.vnnlib_path.is_file():
            raise FileNotFoundError(f"VNNLIB spec not found: {self.vnnlib_path}")
        vnnlib_sha256 = _sha256_file(self.vnnlib_path)
        if (
            self._session is not None
            and self._model_sha256 is not None
            and model_sha256 != self._model_sha256
        ):
            raise RuntimeError(
                "ONNX model changed after strict replay session creation"
            )
        if (
            self._session is not None
            and self._vnnlib_sha256 is not None
            and vnnlib_sha256 != self._vnnlib_sha256
        ):
            raise RuntimeError(
                "VNNLIB spec changed after strict replay session creation"
            )
        self._model_sha256 = model_sha256
        self._vnnlib_sha256 = vnnlib_sha256

    def _ensure_session(self) -> None:
        if self._session is not None:
            return

        import onnxruntime as ort

        self._ensure_hashes()
        layout = extract_vnnlib_concrete_layout(self.vnnlib_path)

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if not hasattr(options, "use_deterministic_compute"):
            raise RuntimeError(
                "installed ONNX Runtime lacks deterministic-compute control"
            )
        options.use_deterministic_compute = True
        session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=options,
            providers=[_CPU_PROVIDER],
        )
        if hasattr(session, "disable_fallback"):
            session.disable_fallback()
        providers = list(session.get_providers())
        if providers != [_CPU_PROVIDER]:
            raise RuntimeError(
                f"strict replay requires CPU-only ORT, got providers={providers!r}"
            )
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                "strict replay requires exactly one ONNX input and one ONNX output"
            )
        if inputs[0].type not in _ORT_DTYPE_MAP:
            raise ValueError(f"unsupported ONNX input dtype {inputs[0].type!r}")
        if outputs[0].type not in _ORT_DTYPE_MAP:
            raise ValueError(f"unsupported ONNX output dtype {outputs[0].type!r}")

        session_config = {
            "providers": providers,
            "intra_op_num_threads": 1,
            "inter_op_num_threads": 1,
            "execution_mode": "ORT_SEQUENTIAL",
            "graph_optimization_level": "ORT_ENABLE_ALL",
            "use_deterministic_compute": True,
        }
        self._ort_version = str(ort.__version__)
        self._session_config = session_config
        self._input_meta = inputs[0]
        self._output_meta = outputs[0]
        self._declared_input_shape = tuple(
            int(dim) for dim in layout["input_shape"]
        )
        self._declared_output_shape = tuple(
            int(dim) for dim in layout["output_shape"]
        )
        self._dialect = str(layout["dialect"])
        self._vnnlib_version = str(layout["vnnlib_version"])
        self._session = session

    def _invalid_receipt(
        self,
        reason: str,
        *,
        started_at: float,
        error: Optional[BaseException] = None,
        replay_state: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        state = {
            "replay_completed": False,
            "ort_executed": False,
            "raw_spec_evaluated": False,
            "zero_tolerance_holds": False,
        }
        if replay_state is not None:
            state.update({
                key: bool(value)
                for key, value in replay_state.items()
                if key in state
            })
        receipt: Dict[str, Any] = {
            "schema_version": 1,
            "authority": "onnxruntime_cpu_raw_vnnlib_zero_tolerance",
            "valid_counterexample": False,
            "reason": reason,
            "tolerance": 0.0,
            "model_path": str(self.onnx_path),
            "vnnlib_path": str(self.vnnlib_path),
            "model_sha256": self._model_sha256,
            "vnnlib_sha256": self._vnnlib_sha256,
            "vnnlib_dialect": self._dialect,
            "vnnlib_version": self._vnnlib_version,
            "onnxruntime_version": self._ort_version,
            "session_config": self._session_config,
            "session_config_sha256": (
                _json_fingerprint(self._session_config)
                if self._session_config is not None else None
            ),
            "property_evaluated": state["raw_spec_evaluated"],
            "property_holds": state["zero_tolerance_holds"],
            **state,
            "property": None,
            "input": None,
            "output": None,
            "elapsed_seconds": float(time.perf_counter() - started_at),
            "error": None,
        }
        if error is not None:
            receipt["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        return receipt

    def __call__(self, x: Any) -> Dict[str, Any]:
        """Replay one candidate; every error returns an invalid receipt."""

        started_at = time.perf_counter()
        replay_state = {
            "replay_completed": False,
            "ort_executed": False,
            "raw_spec_evaluated": False,
            "zero_tolerance_holds": False,
        }
        with self._lock:
            try:
                session_was_cached = self._session is not None
                self._ensure_hashes()
                self._ensure_session()
                assert self._session is not None
                assert self._input_meta is not None
                assert self._output_meta is not None
                assert self._declared_input_shape is not None
                assert self._declared_output_shape is not None
                assert self._session_config is not None
                assert self._dialect is not None
                assert self._vnnlib_version is not None

                candidate = _as_numeric_array(x)
                candidate_sha256 = _sha256_array(candidate)
                expected_numel = int(
                    np.prod(self._declared_input_shape, dtype=np.int64)
                )
                if int(candidate.size) != expected_numel:
                    raise ValueError(
                        f"candidate has {candidate.size} elements but VNNLIB input "
                        f"shape {self._declared_input_shape} expects {expected_numel}"
                    )
                model_shape = _resolve_model_input_shape(
                    self._input_meta.shape,
                    self._declared_input_shape,
                    candidate.shape,
                    int(candidate.size),
                )
                target_dtype = _ORT_DTYPE_MAP[self._input_meta.type]
                cast_from = str(candidate.dtype)
                model_input = _safe_cast(
                    candidate.reshape(model_shape), target_dtype
                )
                if not bool(np.all(np.isfinite(model_input))):
                    raise ValueError("model input contains NaN or infinity")

                outputs = self._session.run(
                    [self._output_meta.name],
                    {self._input_meta.name: model_input},
                )
                replay_state["ort_executed"] = True
                if len(outputs) != 1:
                    raise RuntimeError(
                        f"ORT returned {len(outputs)} outputs, expected one"
                    )
                model_output = np.asarray(outputs[0])
                if model_output.dtype.kind not in "biuf":
                    raise TypeError(
                        f"ORT output dtype {model_output.dtype} is not real numeric"
                    )
                if not bool(np.all(np.isfinite(model_output))):
                    raise ValueError("ORT output contains NaN or infinity")
                output_dtype = _ORT_DTYPE_MAP[self._output_meta.type]
                if model_output.dtype != output_dtype:
                    raise TypeError(
                        f"ORT output dtype {model_output.dtype} does not match "
                        f"ONNX metadata {self._output_meta.type}"
                    )
                _validate_model_output_shape(
                    model_output.shape,
                    self._output_meta.shape,
                    self._declared_output_shape,
                )
                expected_outputs = int(np.prod(
                    self._declared_output_shape, dtype=np.int64
                ))
                if model_output.size != expected_outputs:
                    raise ValueError(
                        f"ORT output has {model_output.size} elements but VNNLIB "
                        f"output shape {self._declared_output_shape} expects "
                        f"{expected_outputs}"
                    )

                property_result = evaluate_vnnlib_concrete(
                    self.vnnlib_path,
                    model_input,
                    model_output,
                    tol=0.0,
                )
                self._ensure_hashes()
                property_evaluated = bool(
                    property_result.get("evaluated", False)
                )
                property_holds = bool(property_result.get("holds", False))
                if property_evaluated and (
                    property_result.get("dialect") != self._dialect
                    or str(property_result.get("vnnlib_version"))
                    != self._vnnlib_version
                ):
                    raise RuntimeError(
                        "raw evaluator dialect/version disagrees with replay "
                        "session layout"
                    )
                replay_state["raw_spec_evaluated"] = property_evaluated
                replay_state["zero_tolerance_holds"] = (
                    property_evaluated and property_holds
                )
                replay_state["replay_completed"] = (
                    replay_state["ort_executed"] and property_evaluated
                )
                valid = bool(
                    replay_state["replay_completed"]
                    and replay_state["ort_executed"]
                    and replay_state["raw_spec_evaluated"]
                    and replay_state["zero_tolerance_holds"]
                )
                reason = (
                    "raw_vnnlib_holds_zero_tolerance"
                    if valid
                    else "raw_vnnlib_rejected"
                    if property_evaluated
                    else "raw_vnnlib_evaluation_failed"
                )

                receipt = {
                    "schema_version": 1,
                    "authority": "onnxruntime_cpu_raw_vnnlib_zero_tolerance",
                    "valid_counterexample": bool(valid),
                    "reason": reason,
                    "tolerance": 0.0,
                    "model_path": str(self.onnx_path),
                    "vnnlib_path": str(self.vnnlib_path),
                    "model_sha256": self._model_sha256,
                    "vnnlib_sha256": self._vnnlib_sha256,
                    "vnnlib_dialect": self._dialect,
                    "vnnlib_version": self._vnnlib_version,
                    "onnxruntime_version": self._ort_version,
                    "session_config": self._session_config,
                    "session_config_sha256": _json_fingerprint(
                        self._session_config
                    ),
                    "session_was_cached": bool(session_was_cached),
                    "property_evaluated": property_evaluated,
                    "property_holds": property_holds,
                    **replay_state,
                    "property": property_result,
                    "input": {
                        "candidate_shape": list(candidate.shape),
                        "candidate_dtype": str(candidate.dtype),
                        "candidate_sha256": candidate_sha256,
                        "cast_from_dtype": cast_from,
                        "cast_to_dtype": str(target_dtype),
                        "cast_performed": bool(
                            candidate.dtype != target_dtype
                        ),
                        "model_name": self._input_meta.name,
                        "model_declared_shape": [
                            str(dim) for dim in self._input_meta.shape
                        ],
                        "model_concrete_shape": list(model_input.shape),
                        "model_dtype": self._input_meta.type,
                        "actual_sha256": _sha256_array(model_input),
                        "actual_values": model_input.reshape(-1).tolist(),
                    },
                    "output": {
                        "model_name": self._output_meta.name,
                        "model_declared_shape": [
                            str(dim) for dim in self._output_meta.shape
                        ],
                        "actual_shape": list(model_output.shape),
                        "actual_dtype": str(model_output.dtype),
                        "actual_sha256": _sha256_array(model_output),
                        "actual_values": model_output.reshape(-1).tolist(),
                    },
                    "elapsed_seconds": float(time.perf_counter() - started_at),
                    "error": (
                        None
                        if property_evaluated
                        else property_result.get("error")
                    ),
                }
                json.dumps(receipt, allow_nan=False)
                return receipt
            except Exception as exc:
                receipt = self._invalid_receipt(
                    "strict_replay_failed",
                    started_at=started_at,
                    error=exc,
                    replay_state=replay_state,
                )
                json.dumps(receipt, allow_nan=False)
                return receipt


def make_strict_replay(
    onnx_path: str | Path,
    vnnlib_path: str | Path,
) -> StrictReplay:
    """Build a lazy callable suitable for ``counterexample_replay_fn``."""

    return StrictReplay(onnx_path, vnnlib_path)


def make_strict_replay_callable(
    onnx_path: str | Path,
    vnnlib_path: str | Path,
) -> StrictReplay:
    """Explicitly named alias for callers that prefer the callable terminology."""

    return make_strict_replay(onnx_path, vnnlib_path)


__all__ = [
    "StrictReplay",
    "make_strict_replay",
    "make_strict_replay_callable",
]
