"""Repair decomposed BatchNorm edges while the loader privately owns the graph.

Only the marked SCALE/BIAS sibling shape is eligible. Numeric payloads and
variable identities are preserved. Invalid or unrelated dataflow fails the
opt-in conversion instead of publishing a partially repaired graph.
"""

from __future__ import annotations

import torch


def _inverse(preds):
    succs = {lid: [] for lid in preds}
    for lid, parents in preds.items():
        for parent in parents:
            if type(parent) is not int or parent not in succs or parent >= lid:
                raise ValueError("BN graph repair requires a topologically ordered DAG")
            if lid not in succs[parent]:
                succs[parent].append(lid)
    return succs


def _marker(params, key):
    value = params.get(key, False)
    if type(value) is not bool:
        raise ValueError(f"invalid BatchNorm marker: {key}")
    return value


def _check_vector(layer, key):
    vector = layer.params.get(key)
    width = len(layer.out_vars)
    if (
        not isinstance(vector, torch.Tensor)
        or vector.ndim != 1
        or not vector.is_floating_point()
        or vector.numel() != width
        or width == 0
        or len(layer.in_vars) != width
        or not bool(torch.isfinite(vector).all())
    ):
        raise ValueError(f"invalid BatchNorm {key} vector at layer {layer.id}")


def _validate_variable_flow(layers, preds):
    producers = {}
    for layer in layers:
        if any(type(v) is not int or v < 0 for v in (*layer.in_vars, *layer.out_vars)):
            raise ValueError("invalid variable id")
        if len(set(layer.out_vars)) != len(layer.out_vars):
            raise ValueError("duplicate output variable in layer")
        try:
            if layer.kind in {"ADD", "SUB", "MUL", "MATMUL"}:
                operands = [layer.params.get(name) for name in ("x_vars", "y_vars")]
                if any(not isinstance(v, (tuple, list)) or not v for v in operands):
                    raise ValueError("missing ordered binary operands")
                if [*operands[0], *operands[1]] != list(layer.in_vars):
                    raise ValueError("binary operands disagree with input variables")
                expected = []
                for operand in operands:
                    owners = {producers[v] for v in operand}
                    if len(owners) != 1:
                        raise ValueError("ambiguous binary operand producer")
                    expected.append(next(iter(owners)))
            else:
                expected = list(dict.fromkeys(producers[v] for v in layer.in_vars))
        except KeyError as exc:
            raise ValueError("input variable has no prior producer") from exc
        if preds[layer.id] != expected:
            raise ValueError(f"unrelated variable/edge mismatch at layer {layer.id}")
        if layer.kind in {"INPUT_SPEC", "CONV2D", "SCALE", "BIAS", "DENSE", "RELU"} and len(expected) != 1:
            raise ValueError("ambiguous single operand producer")
        aliases = (
            layer.kind in {"INPUT_SPEC", "ASSERT"}
            and bool(layer.out_vars)
            and list(layer.out_vars) == list(layer.in_vars)
        )
        if not aliases and any(v in producers for v in layer.out_vars):
            raise ValueError("duplicate output variable producer")
        producers.update((v, layer.id) for v in layer.out_vars)


def repair_batchnorm_producer_graph(layers, preds, succs):
    """Return a fully checked private edge clone, or the identical correct maps.

    Called synchronously by TorchToACT before it exposes a Net. No callback,
    caller-provided repair plan or numerical rewrite is accepted. The caller
    retains its original maps on every exception.
    """
    marked = [
        layer for layer in layers
        if _marker(layer.params, "is_batchnorm_decomposition")
        or _marker(layer.params, "paired_with_scale")
    ]
    if not marked:
        return preds, succs
    if [layer.id for layer in layers] != list(range(len(layers))):
        raise ValueError("noncanonical layer ids")
    keys = set(range(len(layers)))
    if set(preds) != keys or set(succs) != keys:
        raise ValueError("incomplete graph maps")
    source = {lid: list(preds[lid]) for lid in range(len(layers))}
    if _inverse(source) != {lid: list(succs[lid]) for lid in range(len(layers))}:
        raise ValueError("asymmetric source graph")
    candidate = {lid: list(parents) for lid, parents in source.items()}
    paired = set()
    for scale in marked:
        if not _marker(scale.params, "is_batchnorm_decomposition"):
            raise ValueError("pair marker without BatchNorm marker")
        if scale.kind == "BIAS":
            if not _marker(scale.params, "paired_with_scale"):
                raise ValueError("BatchNorm bias missing pair marker")
            continue
        if scale.kind != "SCALE" or _marker(scale.params, "paired_with_scale"):
            raise ValueError("invalid BatchNorm scale marker")
        bid = scale.id + 1
        if bid >= len(layers):
            raise ValueError("BatchNorm scale missing bias")
        bias = layers[bid]
        if (
            bias.kind != "BIAS"
            or not _marker(bias.params, "is_batchnorm_decomposition")
            or not _marker(bias.params, "paired_with_scale")
            or list(bias.in_vars) != list(scale.out_vars)
        ):
            raise ValueError("incomplete adjacent BatchNorm pair")
        _check_vector(scale, "a")
        _check_vector(bias, "c")
        paired.add(bid)
        if source[bid] == [scale.id]:
            continue
        if len(source[scale.id]) != 1 or source[bid] != source[scale.id]:
            raise ValueError("BatchNorm edge fault is not the sibling shape")
        candidate[bid] = [scale.id]
    if paired != {layer.id for layer in marked if layer.kind == "BIAS"}:
        raise ValueError("BatchNorm bias has no unique scale")
    _validate_variable_flow(layers, candidate)
    candidate_succs = _inverse(candidate)
    if candidate == source:
        return preds, succs
    return candidate, candidate_succs
