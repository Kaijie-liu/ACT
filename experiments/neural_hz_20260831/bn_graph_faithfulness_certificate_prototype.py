"""Isolated graph-event/variable-producer faithfulness certificate.

This module is deliberately production-independent.  It snapshots the small
surface of ACT ``Layer`` objects needed to prove that graph predecessor edges
describe the same dataflow as ``in_vars``/``out_vars``.  In particular, a
BatchNorm decomposition must be represented by both events in order::

    upstream -> SCALE -> BIAS -> downstream

The companion repair planner recognizes only the deterministic historical
failure shape where SCALE and its paired BIAS were wired as siblings.  It
returns an immutable, CAS-bound plan and can build a repaired *clone*; it never
mutates a Net, Layer, predecessor dictionary, or successor dictionary.

This is experiment-only infrastructure.  It is default-off and creates no
verification result or score.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch


FORMAL_BASELINE = "1870/2413"
NO_CLAIMS = (
    "experiment_only_default_off",
    "no_production_source_is_modified",
    "no_hz_gain_or_formal_score_is_claimed",
    "full_13_family_2413_replay_is_required_before_enablement",
)

_MULTI_OPERAND_KINDS = frozenset({"ADD", "SUB", "MUL", "MATMUL"})
_SINGLE_PRODUCER_KINDS = frozenset(
    {
        "INPUT_SPEC",
        "CONV2D",
        "SCALE",
        "BIAS",
        "DENSE",
        "RELU",
    }
)
_PATH_LINEAR_EVENT_SEMANTICS = {
    "CONV2D": "implicit_conv2d",
    "SCALE": "diagonal_scale",
    "DENSE": "dense_linear",
}
_PATH_PASSIVE_EVENT_KINDS = frozenset(
    {"INPUT", "INPUT_SPEC", "ASSERT", "BIAS", "ADD", "RELU"}
)


class GraphFaithfulnessReject(ValueError):
    """Fail-closed rejection of malformed or stale graph evidence."""


@dataclass(frozen=True)
class LayerSnapshot:
    layer_id: int
    kind: str
    in_vars: tuple[int, ...]
    out_vars: tuple[int, ...]
    is_batchnorm_decomposition: bool
    paired_with_scale: bool
    x_vars: tuple[int, ...]
    y_vars: tuple[int, ...]
    numeric_payload_key: str
    numeric_payload_size: int
    numeric_payload_dtype: str
    numeric_payload_sha256: str


@dataclass(frozen=True)
class GraphIssue:
    code: str
    layer_id: Optional[int]
    expected: tuple[int, ...] = ()
    observed: tuple[int, ...] = ()
    variable_count: int = 0
    variable_sha256: str = ""
    variable_sample: tuple[int, ...] = ()
    related_layer_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class BatchNormPair:
    scale_layer_id: int
    bias_layer_id: int
    upstream_layer_ids: tuple[int, ...]
    width: int
    graph_edge_present: bool


@dataclass(frozen=True)
class GraphFaithfulnessCertificate:
    accepted: bool
    graph_sha256: str
    layer_count: int
    edge_count: int
    checked_input_variables: int
    batchnorm_pairs: tuple[BatchNormPair, ...]
    issues: tuple[GraphIssue, ...]
    formal_baseline: str = FORMAL_BASELINE
    gain: int = 0
    default_enabled: bool = False


@dataclass(frozen=True)
class _ExpectedPredecessorEvidence:
    expected: tuple[int, ...]
    missing_variables: tuple[int, ...]
    operand_producer_counts: tuple[int, ...] = ()
    ambiguous_variables: tuple[int, ...] = ()
    ambiguous_producers: tuple[int, ...] = ()


@dataclass(frozen=True)
class PathOperatorObservation:
    """One immutable HZ-lineage observation tied to one ACT graph event."""

    occurrence_layer_id: int
    semantic_kind: str


@dataclass(frozen=True)
class PathLineageIssue:
    code: str
    occurrence_layer_id: Optional[int]
    expected_semantic_kind: str = ""
    observed_semantic_kind: str = ""


@dataclass(frozen=True)
class PathOperatorCertificate:
    accepted: bool
    graph_sha256: str
    path_layer_ids: tuple[int, ...]
    expected_operators: tuple[PathOperatorObservation, ...]
    observed_operators: tuple[PathOperatorObservation, ...]
    bias_transition_layer_ids: tuple[int, ...]
    issues: tuple[PathLineageIssue, ...]
    gain: int = 0
    default_enabled: bool = False


@dataclass(frozen=True)
class EdgeReplacement:
    layer_id: int
    old_predecessors: tuple[int, ...]
    new_predecessors: tuple[int, ...]
    reason: str = "batchnorm_scale_bias_sibling_to_chain"


@dataclass(frozen=True)
class BatchNormRepairPlan:
    authorized: bool
    reason: str
    source_graph_sha256: str
    candidate_graph_sha256: str
    replacements: tuple[EdgeReplacement, ...]
    candidate_certificate: Optional[GraphFaithfulnessCertificate]
    gain: int = 0
    default_enabled: bool = False


@dataclass(frozen=True)
class ImmutableGraphClone:
    preds: tuple[tuple[int, tuple[int, ...]], ...]
    succs: tuple[tuple[int, tuple[int, ...]], ...]
    graph_sha256: str

    def predecessor_dict(self) -> dict[int, list[int]]:
        return {key: list(values) for key, values in self.preds}

    def successor_dict(self) -> dict[int, list[int]]:
        return {key: list(values) for key, values in self.succs}


@dataclass(frozen=True)
class RepairTransactionResult:
    committed: bool
    reason: str
    plan: Optional[BatchNormRepairPlan]
    clone: Optional[ImmutableGraphClone]
    certificate: Optional[GraphFaithfulnessCertificate]


def _exact_nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise GraphFaithfulnessReject(f"{label} must be an exact nonnegative int")
    return value


def _exact_bool_marker(params: Mapping[str, Any], key: str) -> bool:
    value = params.get(key, False)
    if type(value) is not bool:
        raise GraphFaithfulnessReject(f"{key} must be an exact bool")
    return value


def _var_tuple(values: Any, label: str) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)):
        raise GraphFaithfulnessReject(f"{label} must be a list or tuple")
    result = tuple(_exact_nonnegative_int(value, label) for value in values)
    return result


def _numeric_vector_snapshot(
    value: Any,
    *,
    key: str,
    label: str,
    expected_width: int,
) -> tuple[str, int, str, str]:
    """Freeze one exact finite BN vector into the graph/CAS digest."""

    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.layout != torch.strided:
            raise GraphFaithfulnessReject(
                f"{label} must be a CPU strided tensor"
            )
        try:
            raw = value.detach().numpy()
        except Exception as exc:
            raise GraphFaithfulnessReject(
                f"{label} tensor snapshot failed"
            ) from exc
    elif type(value) is np.ndarray or isinstance(value, (list, tuple)):
        try:
            raw = np.asarray(value)
        except Exception as exc:
            raise GraphFaithfulnessReject(
                f"{label} numeric snapshot failed"
            ) from exc
    else:
        raise GraphFaithfulnessReject(
            f"{label} must be an ndarray, tensor, list or tuple"
        )
    if raw.ndim != 1:
        raise GraphFaithfulnessReject(f"{label} must be one-dimensional")
    if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise GraphFaithfulnessReject(f"{label} must be real numeric")
    if int(raw.size) != expected_width or expected_width <= 0:
        raise GraphFaithfulnessReject(
            f"{label} width must equal layer variable width"
        )
    try:
        canonical = np.array(raw, dtype="<f8", order="C", copy=True)
    except Exception as exc:
        raise GraphFaithfulnessReject(
            f"{label} float64 snapshot failed"
        ) from exc
    if not np.all(np.isfinite(canonical)):
        raise GraphFaithfulnessReject(f"{label} must be finite")
    dtype = raw.dtype.str
    digest = hashlib.sha256()
    digest.update(b"bn-graph-numeric-vector-v2\0")
    for payload in (
        key.encode("ascii"),
        dtype.encode("ascii"),
        str(expected_width).encode("ascii"),
        canonical.tobytes(order="C"),
    ):
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return key, expected_width, dtype, digest.hexdigest()


def _freeze_layers(layers: Sequence[Any]) -> tuple[LayerSnapshot, ...]:
    if not isinstance(layers, (list, tuple)):
        raise GraphFaithfulnessReject("layers must be a list or tuple")
    snapshots: list[LayerSnapshot] = []
    for position, layer in enumerate(layers):
        layer_id = _exact_nonnegative_int(getattr(layer, "id"), "layer.id")
        if layer_id != position:
            raise GraphFaithfulnessReject(
                "layer ids must be contiguous and equal to their list position"
            )
        kind = getattr(layer, "kind")
        if type(kind) is not str or not kind:
            raise GraphFaithfulnessReject("layer.kind must be a nonempty exact str")
        params = getattr(layer, "params")
        if not isinstance(params, Mapping):
            raise GraphFaithfulnessReject("layer.params must be a mapping")
        in_vars = _var_tuple(getattr(layer, "in_vars"), "layer.in_vars")
        out_vars = _var_tuple(getattr(layer, "out_vars"), "layer.out_vars")
        if len(set(out_vars)) != len(out_vars):
            raise GraphFaithfulnessReject("one layer may not emit a variable twice")
        is_bn = _exact_bool_marker(params, "is_batchnorm_decomposition")
        paired = _exact_bool_marker(params, "paired_with_scale")

        x_vars: tuple[int, ...] = ()
        y_vars: tuple[int, ...] = ()
        if kind in _MULTI_OPERAND_KINDS:
            if "x_vars" not in params or "y_vars" not in params:
                raise GraphFaithfulnessReject(
                    f"layer {layer_id} must provide both x_vars and y_vars"
                )
            x_vars = _var_tuple(params["x_vars"], "params.x_vars")
            y_vars = _var_tuple(params["y_vars"], "params.y_vars")
            if not x_vars or not y_vars or in_vars != x_vars + y_vars:
                raise GraphFaithfulnessReject(
                    f"layer {layer_id} operand vars do not equal in_vars"
                )

        numeric_payload_key = ""
        numeric_payload_size = 0
        numeric_payload_dtype = ""
        numeric_payload_sha256 = ""
        if is_bn and kind in {"SCALE", "BIAS"}:
            if not in_vars or len(in_vars) != len(out_vars):
                raise GraphFaithfulnessReject(
                    f"layer {layer_id} BN in/out widths must be equal and nonzero"
                )
            payload_key = "a" if kind == "SCALE" else "c"
            if payload_key not in params:
                raise GraphFaithfulnessReject(
                    f"layer {layer_id} BN {kind} missing {payload_key}"
                )
            (
                numeric_payload_key,
                numeric_payload_size,
                numeric_payload_dtype,
                numeric_payload_sha256,
            ) = _numeric_vector_snapshot(
                params[payload_key],
                key=payload_key,
                label=f"layer {layer_id} BN {kind}.{payload_key}",
                expected_width=len(out_vars),
            )

        snapshots.append(
            LayerSnapshot(
                layer_id=layer_id,
                kind=kind,
                in_vars=in_vars,
                out_vars=out_vars,
                is_batchnorm_decomposition=is_bn,
                paired_with_scale=paired,
                x_vars=x_vars,
                y_vars=y_vars,
                numeric_payload_key=numeric_payload_key,
                numeric_payload_size=numeric_payload_size,
                numeric_payload_dtype=numeric_payload_dtype,
                numeric_payload_sha256=numeric_payload_sha256,
            )
        )
    if not snapshots:
        raise GraphFaithfulnessReject("empty layer sequence")
    return tuple(snapshots)


def _freeze_edge_map(
    values: Mapping[int, Sequence[int]],
    layer_count: int,
    label: str,
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    if not isinstance(values, Mapping):
        raise GraphFaithfulnessReject(f"{label} must be a mapping")
    expected_keys = set(range(layer_count))
    actual_keys: set[int] = set()
    frozen: list[tuple[int, tuple[int, ...]]] = []
    for raw_key, raw_neighbors in values.items():
        key = _exact_nonnegative_int(raw_key, f"{label} key")
        actual_keys.add(key)
        neighbors = _var_tuple(raw_neighbors, f"{label}[{key}]")
        for neighbor in neighbors:
            if neighbor >= layer_count:
                raise GraphFaithfulnessReject(
                    f"{label}[{key}] references unknown layer {neighbor}"
                )
        frozen.append((key, neighbors))
    if actual_keys != expected_keys:
        raise GraphFaithfulnessReject(
            f"{label} keys must be exactly 0..{layer_count - 1}"
        )
    return tuple(sorted(frozen))


def _edge_dict(
    frozen: tuple[tuple[int, tuple[int, ...]], ...]
) -> dict[int, tuple[int, ...]]:
    return dict(frozen)


def _graph_payload(
    snapshots: tuple[LayerSnapshot, ...],
    frozen_preds: tuple[tuple[int, tuple[int, ...]], ...],
    frozen_succs: tuple[tuple[int, tuple[int, ...]], ...],
) -> dict[str, Any]:
    return {
        "layers": [
            {
                "id": layer.layer_id,
                "kind": layer.kind,
                "in_vars": list(layer.in_vars),
                "out_vars": list(layer.out_vars),
                "is_batchnorm_decomposition": layer.is_batchnorm_decomposition,
                "paired_with_scale": layer.paired_with_scale,
                "x_vars": list(layer.x_vars),
                "y_vars": list(layer.y_vars),
                "numeric_payload_key": layer.numeric_payload_key,
                "numeric_payload_size": layer.numeric_payload_size,
                "numeric_payload_dtype": layer.numeric_payload_dtype,
                "numeric_payload_sha256": layer.numeric_payload_sha256,
            }
            for layer in snapshots
        ],
        "preds": [[key, list(values)] for key, values in frozen_preds],
        "succs": [[key, list(values)] for key, values in frozen_succs],
    }


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _snapshot_graph(
    layers: Sequence[Any],
    preds: Mapping[int, Sequence[int]],
    succs: Mapping[int, Sequence[int]],
) -> tuple[
    tuple[LayerSnapshot, ...],
    tuple[tuple[int, tuple[int, ...]], ...],
    tuple[tuple[int, tuple[int, ...]], ...],
    str,
]:
    snapshots = _freeze_layers(layers)
    frozen_preds = _freeze_edge_map(preds, len(snapshots), "preds")
    frozen_succs = _freeze_edge_map(succs, len(snapshots), "succs")
    digest = _payload_sha256(_graph_payload(snapshots, frozen_preds, frozen_succs))
    return snapshots, frozen_preds, frozen_succs, digest


def _audit_with_frozen_layers(
    snapshots: tuple[LayerSnapshot, ...],
    preds: Mapping[int, Sequence[int]],
    succs: Mapping[int, Sequence[int]],
) -> GraphFaithfulnessCertificate:
    frozen_preds = _freeze_edge_map(preds, len(snapshots), "preds")
    frozen_succs = _freeze_edge_map(succs, len(snapshots), "succs")
    digest = _payload_sha256(_graph_payload(snapshots, frozen_preds, frozen_succs))
    return _audit_snapshot(snapshots, frozen_preds, frozen_succs, digest)


def _expected_predecessors(
    layer: LayerSnapshot,
    prior_producer: Mapping[int, int],
) -> _ExpectedPredecessorEvidence:
    missing: list[int] = []
    if layer.x_vars and layer.y_vars:
        ordered: list[int] = []
        producer_counts: list[int] = []
        ambiguous_variables: list[int] = []
        ambiguous_producers: list[int] = []
        for operand in (layer.x_vars, layer.y_vars):
            operand_producers = {
                prior_producer[value]
                for value in operand
                if value in prior_producer
            }
            producer_counts.append(len(operand_producers))
            absent = [value for value in operand if value not in prior_producer]
            missing.extend(absent)
            if len(operand_producers) != 1:
                ambiguous_variables.extend(operand)
                ambiguous_producers.extend(sorted(operand_producers))
                ordered.extend(sorted(operand_producers))
            else:
                ordered.append(next(iter(operand_producers)))
        return _ExpectedPredecessorEvidence(
            expected=tuple(ordered),
            missing_variables=tuple(dict.fromkeys(missing)),
            operand_producer_counts=tuple(producer_counts),
            ambiguous_variables=tuple(dict.fromkeys(ambiguous_variables)),
            ambiguous_producers=tuple(dict.fromkeys(ambiguous_producers)),
        )

    ordered_unique: list[int] = []
    for variable in layer.in_vars:
        producer = prior_producer.get(variable)
        if producer is None:
            missing.append(variable)
        elif producer not in ordered_unique:
            ordered_unique.append(producer)
    return _ExpectedPredecessorEvidence(
        expected=tuple(ordered_unique),
        missing_variables=tuple(dict.fromkeys(missing)),
    )


def _variable_evidence(values: Sequence[int]) -> dict[str, Any]:
    """Return bounded, address-free evidence for an arbitrary-width var tuple."""

    frozen = tuple(values)
    digest = hashlib.sha256()
    for value in frozen:
        digest.update(str(value).encode("ascii"))
        digest.update(b"\0")
    if len(frozen) <= 8:
        sample = frozen
    else:
        sample = frozen[:4] + frozen[-4:]
    return {
        "variable_count": len(frozen),
        "variable_sha256": digest.hexdigest(),
        "variable_sample": sample,
    }


def _inverse_successors(
    predecessors: Mapping[int, Sequence[int]], layer_count: int
) -> dict[int, tuple[int, ...]]:
    successors: dict[int, list[int]] = {layer_id: [] for layer_id in range(layer_count)}
    for layer_id in range(layer_count):
        for predecessor in predecessors[layer_id]:
            if layer_id not in successors[predecessor]:
                successors[predecessor].append(layer_id)
    return {key: tuple(values) for key, values in successors.items()}


def _audit_snapshot(
    snapshots: tuple[LayerSnapshot, ...],
    frozen_preds: tuple[tuple[int, tuple[int, ...]], ...],
    frozen_succs: tuple[tuple[int, tuple[int, ...]], ...],
    digest: str,
) -> GraphFaithfulnessCertificate:
    preds = _edge_dict(frozen_preds)
    succs = _edge_dict(frozen_succs)
    issues: list[GraphIssue] = []
    expected_by_layer: dict[int, tuple[int, ...]] = {}
    prior_producer: dict[int, int] = {}
    checked_variables = 0

    for layer in snapshots:
        checked_variables += len(layer.in_vars)
        predecessor_evidence = _expected_predecessors(layer, prior_producer)
        expected = predecessor_evidence.expected
        missing = predecessor_evidence.missing_variables
        expected_by_layer[layer.layer_id] = expected
        if missing:
            issues.append(
                GraphIssue(
                    "input_variable_without_prior_producer",
                    layer.layer_id,
                    **_variable_evidence(missing),
                )
            )
        if predecessor_evidence.ambiguous_variables:
            issues.append(
                GraphIssue(
                    "multi_operand_producer_ambiguity",
                    layer.layer_id,
                    expected=(1,) * len(predecessor_evidence.operand_producer_counts),
                    observed=predecessor_evidence.operand_producer_counts,
                    related_layer_ids=predecessor_evidence.ambiguous_producers,
                    **_variable_evidence(predecessor_evidence.ambiguous_variables),
                )
            )
        if (
            layer.kind in _SINGLE_PRODUCER_KINDS
            and not missing
            and len(expected) != 1
        ):
            issues.append(
                GraphIssue(
                    "single_operand_producer_ambiguity",
                    layer.layer_id,
                    expected=(1,),
                    observed=(len(expected),),
                    related_layer_ids=expected,
                    **_variable_evidence(layer.in_vars),
                )
            )
        observed = preds[layer.layer_id]
        if (
            not missing
            and not predecessor_evidence.ambiguous_variables
            and not (
                layer.kind in _SINGLE_PRODUCER_KINDS
                and len(expected) != 1
            )
            and observed != expected
        ):
            issues.append(
                GraphIssue(
                    "predecessor_producer_mismatch",
                    layer.layer_id,
                    expected=expected,
                    observed=observed,
                    **_variable_evidence(layer.in_vars),
                )
            )
        duplicate_outputs = tuple(
            variable for variable in layer.out_vars if variable in prior_producer
        )
        exact_wrapper_alias = (
            layer.kind in {"INPUT_SPEC", "ASSERT"}
            and layer.out_vars == layer.in_vars
            and bool(layer.out_vars)
        )
        if duplicate_outputs and not exact_wrapper_alias:
            issues.append(
                GraphIssue(
                    "duplicate_output_variable_definition",
                    layer.layer_id,
                    related_layer_ids=tuple(
                        dict.fromkeys(
                            prior_producer[value]
                            for value in duplicate_outputs
                        )
                    ),
                    **_variable_evidence(duplicate_outputs),
                )
            )
        for variable in layer.out_vars:
            prior_producer[variable] = layer.layer_id

    expected_succs = _inverse_successors(preds, len(snapshots))
    for layer in snapshots:
        observed = succs[layer.layer_id]
        expected = expected_succs[layer.layer_id]
        if observed != expected:
            issues.append(
                GraphIssue(
                    "predecessor_successor_asymmetry",
                    layer.layer_id,
                    expected=expected,
                    observed=observed,
                )
            )

    pairs: list[BatchNormPair] = []
    paired_bias_ids: set[int] = set()
    for scale in snapshots:
        if not scale.is_batchnorm_decomposition:
            if scale.paired_with_scale:
                issues.append(GraphIssue("paired_marker_without_bn_marker", scale.layer_id))
            continue
        if scale.kind not in {"SCALE", "BIAS"}:
            issues.append(GraphIssue("bn_marker_on_unsupported_kind", scale.layer_id))
            continue
        if scale.kind == "BIAS":
            if not scale.paired_with_scale:
                issues.append(GraphIssue("bn_bias_missing_pair_marker", scale.layer_id))
            continue
        if scale.paired_with_scale:
            issues.append(GraphIssue("bn_scale_has_bias_only_pair_marker", scale.layer_id))

        candidates = tuple(
            layer.layer_id
            for layer in snapshots
            if layer.kind == "BIAS"
            and layer.is_batchnorm_decomposition
            and layer.paired_with_scale
            and layer.in_vars == scale.out_vars
        )
        if len(candidates) != 1:
            issues.append(
                GraphIssue(
                    "bn_scale_pair_count_mismatch",
                    scale.layer_id,
                    expected=(1,),
                    observed=(len(candidates),),
                    related_layer_ids=candidates,
                )
            )
            continue
        bias_id = candidates[0]
        paired_bias_ids.add(bias_id)
        bias = snapshots[bias_id]
        if bias_id != scale.layer_id + 1:
            issues.append(
                GraphIssue(
                    "bn_pair_not_adjacent_in_variable_program",
                    bias_id,
                    expected=(scale.layer_id + 1,),
                    observed=(bias_id,),
                    related_layer_ids=(scale.layer_id,),
                )
            )
        edge_present = (
            preds[bias_id] == (scale.layer_id,)
            and bias_id in succs[scale.layer_id]
        )
        if not edge_present:
            issues.append(
                GraphIssue(
                    "bn_scale_bias_graph_event_missing",
                    bias_id,
                    expected=(scale.layer_id,),
                    observed=preds[bias_id],
                    **_variable_evidence(bias.in_vars),
                    related_layer_ids=(scale.layer_id,),
                )
            )
        pairs.append(
            BatchNormPair(
                scale_layer_id=scale.layer_id,
                bias_layer_id=bias_id,
                upstream_layer_ids=expected_by_layer[scale.layer_id],
                width=len(scale.out_vars),
                graph_edge_present=edge_present,
            )
        )

    for bias in snapshots:
        if (
            bias.kind == "BIAS"
            and bias.is_batchnorm_decomposition
            and bias.paired_with_scale
            and bias.layer_id not in paired_bias_ids
        ):
            issues.append(GraphIssue("bn_bias_without_unique_scale", bias.layer_id))

    issues.sort(
        key=lambda issue: (
            -1 if issue.layer_id is None else issue.layer_id,
            issue.code,
            issue.expected,
            issue.observed,
        )
    )
    return GraphFaithfulnessCertificate(
        accepted=not issues,
        graph_sha256=digest,
        layer_count=len(snapshots),
        edge_count=sum(len(values) for values in preds.values()),
        checked_input_variables=checked_variables,
        batchnorm_pairs=tuple(pairs),
        issues=tuple(issues),
    )


def audit_graph_faithfulness(
    layers: Sequence[Any],
    preds: Mapping[int, Sequence[int]],
    succs: Mapping[int, Sequence[int]],
) -> GraphFaithfulnessCertificate:
    """Return a deterministic fail-closed producer/graph certificate.

    Ordinary malformed-input exceptions are converted to one rejected
    certificate.  ``KeyboardInterrupt``, ``SystemExit`` and other
    ``BaseException`` subclasses intentionally propagate.
    """

    try:
        snapshots, frozen_preds, frozen_succs, digest = _snapshot_graph(
            layers, preds, succs
        )
        return _audit_snapshot(snapshots, frozen_preds, frozen_succs, digest)
    except Exception as exc:
        return GraphFaithfulnessCertificate(
            accepted=False,
            graph_sha256="",
            layer_count=0,
            edge_count=0,
            checked_input_variables=0,
            batchnorm_pairs=(),
            issues=(
                GraphIssue(
                    code=f"malformed_input:{type(exc).__name__}",
                    layer_id=None,
                ),
            ),
        )


def audit_path_operator_lineage(
    layers: Sequence[Any],
    preds: Mapping[int, Sequence[int]],
    succs: Mapping[int, Sequence[int]],
    path_layer_ids: Sequence[int],
    observations: Sequence[PathOperatorObservation],
) -> PathOperatorCertificate:
    """Prove ordered graph-linear-event to HZ-operator occurrence coverage.

    The graph itself must first pass :func:`audit_graph_faithfulness`.  Along
    the supplied edge-contiguous path, each CONV2D/SCALE/DENSE event requires
    exactly one matching HZ operator occurrence in the same order.  BIAS is
    recorded as a bias transition and must not masquerade as a multiplicative
    operator.  Nonlinear and residual-boundary events remain graph transitions.

    This does not inspect production caches.  A caller must construct the
    observations from immutable runtime lineage; hand-authored observations
    are only unit-test or diagnostic evidence.
    """

    graph_digest = ""
    try:
        snapshots, frozen_preds, frozen_succs, digest = _snapshot_graph(
            layers, preds, succs
        )
        graph_digest = digest
        graph_certificate = _audit_snapshot(
            snapshots, frozen_preds, frozen_succs, digest
        )
        if not graph_certificate.accepted:
            return PathOperatorCertificate(
                accepted=False,
                graph_sha256=graph_certificate.graph_sha256,
                path_layer_ids=(),
                expected_operators=(),
                observed_operators=(),
                bias_transition_layer_ids=(),
                issues=(PathLineageIssue("source_graph_not_faithful", None),),
            )
        if not isinstance(path_layer_ids, (list, tuple)) or not path_layer_ids:
            raise GraphFaithfulnessReject("path must be a nonempty list or tuple")
        path = tuple(
            _exact_nonnegative_int(value, "path layer id")
            for value in path_layer_ids
        )
        if len(set(path)) != len(path):
            raise GraphFaithfulnessReject("path may not repeat a graph event")
        if any(value >= len(snapshots) for value in path):
            raise GraphFaithfulnessReject("path references an unknown layer")
        pred_view = _edge_dict(frozen_preds)
        path_issues: list[PathLineageIssue] = []
        for previous, current in zip(path, path[1:]):
            if previous not in pred_view[current]:
                path_issues.append(
                    PathLineageIssue(
                        "path_skips_graph_event", current
                    )
                )
        for layer_id in path:
            kind = snapshots[layer_id].kind
            if (
                kind not in _PATH_LINEAR_EVENT_SEMANTICS
                and kind not in _PATH_PASSIVE_EVENT_KINDS
            ):
                path_issues.append(
                    PathLineageIssue("unsupported_path_event_kind", layer_id)
                )

        if not isinstance(observations, (list, tuple)):
            raise GraphFaithfulnessReject(
                "operator observations must be a list or tuple"
            )
        frozen_observations: list[PathOperatorObservation] = []
        for observation in observations:
            if type(observation) is not PathOperatorObservation:
                raise GraphFaithfulnessReject(
                    "each observation must be an exact PathOperatorObservation"
                )
            occurrence = _exact_nonnegative_int(
                observation.occurrence_layer_id,
                "operator occurrence layer id",
            )
            semantic_kind = observation.semantic_kind
            if type(semantic_kind) is not str or not semantic_kind:
                raise GraphFaithfulnessReject(
                    "operator semantic kind must be a nonempty exact str"
                )
            frozen_observations.append(
                PathOperatorObservation(occurrence, semantic_kind)
            )
        observed = tuple(frozen_observations)

        expected = tuple(
            PathOperatorObservation(
                layer_id,
                _PATH_LINEAR_EVENT_SEMANTICS[snapshots[layer_id].kind],
            )
            for layer_id in path
            if snapshots[layer_id].kind in _PATH_LINEAR_EVENT_SEMANTICS
        )
        bias_transitions = tuple(
            layer_id for layer_id in path if snapshots[layer_id].kind == "BIAS"
        )
        expected_by_occurrence = {
            item.occurrence_layer_id: item for item in expected
        }
        observed_counts: dict[int, int] = {}
        for item in observed:
            observed_counts[item.occurrence_layer_id] = (
                observed_counts.get(item.occurrence_layer_id, 0) + 1
            )
        for occurrence, count in sorted(observed_counts.items()):
            if count > 1:
                path_issues.append(
                    PathLineageIssue(
                        "duplicate_lineage_operator_occurrence", occurrence
                    )
                )

        first_observed: dict[int, PathOperatorObservation] = {}
        for item in observed:
            first_observed.setdefault(item.occurrence_layer_id, item)
        for expected_item in expected:
            observed_item = first_observed.get(expected_item.occurrence_layer_id)
            if observed_item is None:
                path_issues.append(
                    PathLineageIssue(
                        "unaccounted_linear_event",
                        expected_item.occurrence_layer_id,
                        expected_semantic_kind=expected_item.semantic_kind,
                    )
                )
            elif observed_item.semantic_kind != expected_item.semantic_kind:
                path_issues.append(
                    PathLineageIssue(
                        "operator_semantic_kind_mismatch",
                        expected_item.occurrence_layer_id,
                        expected_semantic_kind=expected_item.semantic_kind,
                        observed_semantic_kind=observed_item.semantic_kind,
                    )
                )
        for observed_item in observed:
            if observed_item.occurrence_layer_id not in expected_by_occurrence:
                path_issues.append(
                    PathLineageIssue(
                        "extra_lineage_operator",
                        observed_item.occurrence_layer_id,
                        observed_semantic_kind=observed_item.semantic_kind,
                    )
                )

        expected_order = tuple(
            item.occurrence_layer_id
            for item in expected
            if item.occurrence_layer_id in first_observed
        )
        observed_order = tuple(
            item.occurrence_layer_id
            for item in observed
            if item.occurrence_layer_id in expected_by_occurrence
        )
        if (
            len(observed_order) == len(set(observed_order))
            and observed_order != expected_order
        ):
            path_issues.append(
                PathLineageIssue("lineage_operator_order_mismatch", None)
            )

        return PathOperatorCertificate(
            accepted=not path_issues,
            graph_sha256=digest,
            path_layer_ids=path,
            expected_operators=expected,
            observed_operators=observed,
            bias_transition_layer_ids=bias_transitions,
            issues=tuple(path_issues),
        )
    except Exception as exc:
        return PathOperatorCertificate(
            accepted=False,
            graph_sha256=graph_digest,
            path_layer_ids=(),
            expected_operators=(),
            observed_operators=(),
            bias_transition_layer_ids=(),
            issues=(
                PathLineageIssue(
                    f"malformed_path_lineage_input:{type(exc).__name__}", None
                ),
            ),
        )


def _candidate_graph_from_replacements(
    source_preds: Mapping[int, Sequence[int]],
    layer_count: int,
    replacements: Iterable[EdgeReplacement],
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    candidate_preds = {
        layer_id: list(source_preds[layer_id]) for layer_id in range(layer_count)
    }
    for replacement in replacements:
        if tuple(candidate_preds[replacement.layer_id]) != replacement.old_predecessors:
            raise GraphFaithfulnessReject("repair predecessor CAS mismatch")
        candidate_preds[replacement.layer_id] = list(replacement.new_predecessors)
    inverse = _inverse_successors(candidate_preds, layer_count)
    candidate_succs = {key: list(values) for key, values in inverse.items()}
    return candidate_preds, candidate_succs


def plan_batchnorm_graph_repair(
    layers: Sequence[Any],
    preds: Mapping[int, Sequence[int]],
    succs: Mapping[int, Sequence[int]],
) -> BatchNormRepairPlan:
    """Plan only the exact SCALE/BIAS sibling-to-chain correction.

    Any unrelated producer mismatch, asymmetric source graph, malformed BN
    markers, missing variables, or non-sibling wiring makes the plan
    unauthorized.  The function never changes its arguments.
    """

    try:
        snapshots, frozen_preds, frozen_succs, digest = _snapshot_graph(
            layers, preds, succs
        )
        source = _audit_snapshot(
            snapshots, frozen_preds, frozen_succs, digest
        )
    except Exception:
        return BatchNormRepairPlan(
            False, "malformed_source_graph", "", "", (), None
        )
    if source.accepted:
        return BatchNormRepairPlan(
            False,
            "source_graph_already_faithful",
            source.graph_sha256,
            source.graph_sha256,
            (),
            source,
        )
    if any(issue.code == "predecessor_successor_asymmetry" for issue in source.issues):
        return BatchNormRepairPlan(
            False, "source_graph_edges_are_asymmetric", source.graph_sha256, "", (), None
        )

    try:
        pred_view = _edge_dict(frozen_preds)
        snapshot_by_id = {layer.layer_id: layer for layer in snapshots}
        replacements: list[EdgeReplacement] = []
        for pair in source.batchnorm_pairs:
            if pair.graph_edge_present:
                continue
            scale = snapshot_by_id[pair.scale_layer_id]
            bias = snapshot_by_id[pair.bias_layer_id]
            if pair.bias_layer_id != pair.scale_layer_id + 1:
                raise GraphFaithfulnessReject("non-adjacent BN pair")
            if bias.in_vars != scale.out_vars or not bias.in_vars:
                raise GraphFaithfulnessReject("BN variable chain mismatch")
            scale_preds = pred_view[scale.layer_id]
            bias_preds = pred_view[bias.layer_id]
            if not scale_preds or bias_preds != scale_preds:
                raise GraphFaithfulnessReject("BN mismatch is not the sibling failure shape")
            replacements.append(
                EdgeReplacement(
                    layer_id=bias.layer_id,
                    old_predecessors=bias_preds,
                    new_predecessors=(scale.layer_id,),
                )
            )
        if not replacements:
            raise GraphFaithfulnessReject("no eligible BN sibling edge")
        candidate_preds, candidate_succs = _candidate_graph_from_replacements(
            pred_view, len(snapshots), replacements
        )
        candidate = _audit_with_frozen_layers(
            snapshots, candidate_preds, candidate_succs
        )
        if not candidate.accepted:
            raise GraphFaithfulnessReject("BN-only clone remains unfaithful")
        return BatchNormRepairPlan(
            authorized=True,
            reason="authorized_bn_sibling_to_chain_clone_only",
            source_graph_sha256=digest,
            candidate_graph_sha256=candidate.graph_sha256,
            replacements=tuple(replacements),
            candidate_certificate=candidate,
        )
    except Exception as exc:
        return BatchNormRepairPlan(
            authorized=False,
            reason=f"repair_rejected:{type(exc).__name__}",
            source_graph_sha256=source.graph_sha256,
            candidate_graph_sha256="",
            replacements=(),
            candidate_certificate=None,
        )


def apply_repair_plan_to_clone(
    layers: Sequence[Any],
    preds: Mapping[int, Sequence[int]],
    succs: Mapping[int, Sequence[int]],
    plan: BatchNormRepairPlan,
) -> tuple[ImmutableGraphClone, GraphFaithfulnessCertificate]:
    """Apply an authorized plan to private containers and return an immutable clone."""

    if type(plan) is not BatchNormRepairPlan or not plan.authorized:
        raise GraphFaithfulnessReject("an authorized exact repair plan is required")
    derived_plan = plan_batchnorm_graph_repair(layers, preds, succs)
    if not derived_plan.authorized or plan != derived_plan:
        raise GraphFaithfulnessReject(
            "repair plan provenance or source digest CAS mismatch"
        )
    # Consume only the independently re-derived plan.  Public construction or
    # object.__setattr__ mutation of a frozen dataclass is not authority.
    plan = derived_plan
    snapshots, frozen_preds, _frozen_succs, source_digest = _snapshot_graph(
        layers, preds, succs
    )
    if source_digest != plan.source_graph_sha256:
        raise GraphFaithfulnessReject("source graph digest CAS mismatch")
    candidate_preds, candidate_succs = _candidate_graph_from_replacements(
        _edge_dict(frozen_preds), len(snapshots), plan.replacements
    )
    certificate = _audit_with_frozen_layers(
        snapshots, candidate_preds, candidate_succs
    )
    if not certificate.accepted:
        raise GraphFaithfulnessReject("candidate clone failed its certificate")
    if certificate.graph_sha256 != plan.candidate_graph_sha256:
        raise GraphFaithfulnessReject("candidate graph digest CAS mismatch")
    clone = ImmutableGraphClone(
        preds=tuple(
            (key, tuple(candidate_preds[key])) for key in range(len(snapshots))
        ),
        succs=tuple(
            (key, tuple(candidate_succs[key])) for key in range(len(snapshots))
        ),
        graph_sha256=certificate.graph_sha256,
    )
    return clone, certificate


def execute_clone_only_repair_transaction(
    layers: Sequence[Any],
    preds: Mapping[int, Sequence[int]],
    succs: Mapping[int, Sequence[int]],
    *,
    candidate_validator: Optional[
        Callable[[ImmutableGraphClone, GraphFaithfulnessCertificate], bool]
    ] = None,
) -> RepairTransactionResult:
    """Stage and validate a repair clone with an explicit exception boundary.

    Ordinary ``Exception`` rejects and publishes no clone.  ``BaseException``
    subclasses outside ``Exception`` propagate.  The optional callback sees
    only immutable/private clone data, never caller-owned graph containers.
    """

    try:
        plan = plan_batchnorm_graph_repair(layers, preds, succs)
        if not plan.authorized:
            return RepairTransactionResult(
                False, plan.reason, plan, None, plan.candidate_certificate
            )
        clone, certificate = apply_repair_plan_to_clone(
            layers, preds, succs, plan
        )
        if candidate_validator is not None:
            accepted = candidate_validator(clone, certificate)
            fresh_plan = plan_batchnorm_graph_repair(layers, preds, succs)
            if not fresh_plan.authorized or fresh_plan != plan:
                return RepairTransactionResult(
                    False,
                    "candidate_validator_changed_source_graph",
                    None,
                    None,
                    None,
                )
            fresh_clone, fresh_certificate = apply_repair_plan_to_clone(
                layers, preds, succs, fresh_plan
            )
            if clone != fresh_clone or certificate != fresh_certificate:
                return RepairTransactionResult(
                    False,
                    "candidate_validator_changed_private_evidence",
                    fresh_plan,
                    None,
                    fresh_certificate,
                )
            if accepted is not True:
                return RepairTransactionResult(
                    False,
                    "candidate_validator_rejected",
                    fresh_plan,
                    None,
                    fresh_certificate,
                )
            plan = fresh_plan
            clone = fresh_clone
            certificate = fresh_certificate
        return RepairTransactionResult(
            True, "clone_committed", plan, clone, certificate
        )
    except Exception as exc:
        return RepairTransactionResult(
            False,
            f"transaction_rejected:{type(exc).__name__}",
            None,
            None,
            None,
        )


def certificate_to_jsonable(
    certificate: GraphFaithfulnessCertificate,
) -> dict[str, Any]:
    return {
        "accepted": certificate.accepted,
        "graph_sha256": certificate.graph_sha256,
        "layer_count": certificate.layer_count,
        "edge_count": certificate.edge_count,
        "checked_input_variables": certificate.checked_input_variables,
        "batchnorm_pairs": [
            {
                "scale_layer_id": pair.scale_layer_id,
                "bias_layer_id": pair.bias_layer_id,
                "upstream_layer_ids": list(pair.upstream_layer_ids),
                "width": pair.width,
                "graph_edge_present": pair.graph_edge_present,
            }
            for pair in certificate.batchnorm_pairs
        ],
        "issues": [
            {
                "code": issue.code,
                "layer_id": issue.layer_id,
                "expected": list(issue.expected),
                "observed": list(issue.observed),
                "variable_count": issue.variable_count,
                "variable_sha256": issue.variable_sha256,
                "variable_sample": list(issue.variable_sample),
                "related_layer_ids": list(issue.related_layer_ids),
            }
            for issue in certificate.issues
        ],
        "formal_baseline": certificate.formal_baseline,
        "gain": certificate.gain,
        "default_enabled": certificate.default_enabled,
        "no_claims": list(NO_CLAIMS),
    }


def repair_plan_to_jsonable(plan: BatchNormRepairPlan) -> dict[str, Any]:
    return {
        "authorized": plan.authorized,
        "reason": plan.reason,
        "source_graph_sha256": plan.source_graph_sha256,
        "candidate_graph_sha256": plan.candidate_graph_sha256,
        "replacements": [
            {
                "layer_id": replacement.layer_id,
                "old_predecessors": list(replacement.old_predecessors),
                "new_predecessors": list(replacement.new_predecessors),
                "reason": replacement.reason,
            }
            for replacement in plan.replacements
        ],
        "candidate_accepted": (
            None
            if plan.candidate_certificate is None
            else plan.candidate_certificate.accepted
        ),
        "gain": plan.gain,
        "default_enabled": plan.default_enabled,
        "no_claims": list(NO_CLAIMS),
    }


def path_operator_certificate_to_jsonable(
    certificate: PathOperatorCertificate,
) -> dict[str, Any]:
    def observation(item: PathOperatorObservation) -> dict[str, Any]:
        return {
            "occurrence_layer_id": item.occurrence_layer_id,
            "semantic_kind": item.semantic_kind,
        }

    return {
        "accepted": certificate.accepted,
        "graph_sha256": certificate.graph_sha256,
        "path_layer_ids": list(certificate.path_layer_ids),
        "expected_operators": [
            observation(item) for item in certificate.expected_operators
        ],
        "observed_operators": [
            observation(item) for item in certificate.observed_operators
        ],
        "bias_transition_layer_ids": list(
            certificate.bias_transition_layer_ids
        ),
        "issues": [
            {
                "code": issue.code,
                "occurrence_layer_id": issue.occurrence_layer_id,
                "expected_semantic_kind": issue.expected_semantic_kind,
                "observed_semantic_kind": issue.observed_semantic_kind,
            }
            for issue in certificate.issues
        ],
        "gain": certificate.gain,
        "default_enabled": certificate.default_enabled,
        "runtime_lineage_adapter_proven": False,
        "no_claims": list(NO_CLAIMS),
    }
