"""Exclusive, read-only C3 necessary-condition and BN forward preflight.

No abstract propagation, solver, witness search, or production mutation occurs.
The path expansion is a structural rejection screen, never runtime authority.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.neural_hz_20260831 import (
    bn_graph_faithfulness_certificate_prototype as cert,
)
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import (
    _atomic_exclusive_json,
    _sha256,
)

EXPERIMENT = ROOT / "experiments/neural_hz_20260831"
ANCHOR = EXPERIMENT / "evidence/tiny_iid143_bn_graph_faithfulness_audit_v2.json"
ANCHOR_SHA256 = "31ad2f6b0d5a30cff226340d554ff9fa92883c894a3b9f1c0e02dcfd4f8f2c2a"
SOURCE_KINDS = frozenset({"INPUT", "INPUT_SPEC", "RELU"})
AFFINE_KINDS = frozenset({"CONV2D", "SCALE", "BIAS", "ADD"})


@dataclass(frozen=True)
class AffinePath:
    source: int
    events: tuple[int, ...]


def affine_source_paths(layers, preds, terminal, *, max_paths=4096):
    """Expand ALL operand occurrences, stopping only at allowed C3 sources.

    Repeated operand occurrences remain repeated. A path limit is an error,
    never a partial negative or positive result. No payload is simplified.
    """
    by_id = {layer.id: layer for layer in layers}
    if len(by_id) != len(layers):
        raise ValueError("duplicate layer id")
    cache = {}
    active = set()

    def visit(lid):
        if lid in active:
            raise ValueError("cycle in graph")
        if lid in cache:
            return cache[lid]
        layer = by_id[lid]
        if layer.kind in SOURCE_KINDS:
            answer = (AffinePath(lid, ()),)
        else:
            if layer.kind not in AFFINE_KINDS:
                raise ValueError(f"unsupported affine boundary: {layer.kind}")
            parents = preds[lid]
            if len(parents) != (2 if layer.kind == "ADD" else 1):
                raise ValueError("wrong operand count")
            active.add(lid)
            answer = tuple(
                AffinePath(path.source, (*path.events, lid))
                for parent in parents for path in visit(parent)
            )
            active.remove(lid)
            if len(answer) > max_paths:
                raise ValueError("complete path budget exceeded")
        cache[lid] = answer
        return answer

    return visit(terminal)


def c3_necessary_condition(layers, preds, terminal):
    """Only reject impossible event shapes; passing does NOT authorize C3."""
    by_id = {layer.id: layer for layer in layers}
    paths = affine_source_paths(layers, preds, terminal)
    classified = []
    common_add = None
    for path in paths:
        kinds = tuple(by_id[lid].kind for lid in path.events)
        adds = tuple(lid for lid in path.events if by_id[lid].kind == "ADD")
        reason = "necessary_shape_only"
        if len(adds) != 1:
            reason = "nested_add" if len(adds) > 1 else "missing_add"
        else:
            add = adds[0]
            if common_add is None:
                common_add = add
            elif add != common_add:
                reason = "different_add_occurrence"
            cut = path.events.index(add)
            before = tuple(k for k in kinds[:cut] if k != "BIAS")
            after = tuple(k for k in kinds[cut + 1:] if k != "BIAS")
            if before and not (
                before[0] == "CONV2D" and all(k == "SCALE" for k in before[1:])
            ):
                reason = "incomplete_inner_branch"
            if after.count("CONV2D") != 1 or any(
                k not in {"SCALE", "CONV2D"} for k in after
            ):
                reason = "incomplete_outer_branch"
        classified.append({
            "source": path.source,
            "source_kind": by_id[path.source].kind,
            "events": list(path.events),
            "kinds": list(kinds),
            "add_occurrences": list(adds),
            "reason": reason,
        })
    return {
        "scope": "necessary_graph_shape_under_frozen_source_boundaries",
        "allowed_source_kinds": sorted(SOURCE_KINDS),
        "terminal": terminal,
        "complete_path_count": len(paths),
        "paths": classified,
        "reason_counts": dict(Counter(row["reason"] for row in classified)),
        "structurally_impossible": any(
            row["reason"] != "necessary_shape_only" for row in classified
        ),
        "runtime_lineage_proved": False,
        "positive_authorization": False,
    }


def load_registered_graph():
    if _sha256(ANCHOR) != ANCHOR_SHA256:
        raise ValueError("frozen V2 anchor changed")
    anchor = json.loads(ANCHOR.read_text())
    target = anchor["target"]
    for name in ("model", "converted_spec"):
        if _sha256(Path(target[name])) != target[f"{name}_sha256"]:
            raise ValueError(f"frozen {name} changed")
    from act.front_end.spec_creator_base import LabeledInputTensor
    from act.front_end.verifiable_model import (
        InputLayer, InputSpecLayer, OutputSpecLayer, VerifiableModel,
    )
    from act.front_end.vnnlib_loader.onnx_converter import convert_onnx_to_pytorch
    from act.front_end.vnnlib_loader.vnnlib_parser import parse_vnnlib_queries
    from act.pipeline.verification.torch2act import TorchToACT

    model = convert_onnx_to_pytorch(Path(target["model"])).eval()
    dtype = next(model.parameters()).dtype
    shape = tuple(target["input_shape"])
    labeled = LabeledInputTensor(torch.zeros(shape, dtype=dtype), torch.tensor([0]))
    queries = parse_vnnlib_queries(Path(target["converted_spec"]), labeled_tensor=labeled)
    if len(queries) != 1:
        raise ValueError("registered query arity changed")
    input_spec, output_spec = queries[0]
    input_spec.lb = input_spec.lb.to(dtype=dtype)
    input_spec.ub = input_spec.ub.to(dtype=dtype)
    wrapped = VerifiableModel(
        input_layer=InputLayer(labeled_input=labeled, shape=shape, dtype=dtype),
        input_spec=InputSpecLayer(input_spec), model=model,
        output_spec=OutputSpecLayer(output_spec),
    )
    net = TorchToACT(wrapped).run()
    plan = cert.plan_batchnorm_graph_repair(net.layers, net.preds, net.succs)
    if not plan.authorized:
        raise ValueError(plan.reason)
    clone, candidate = cert.apply_repair_plan_to_clone(
        net.layers, net.preds, net.succs, plan,
    )
    if (
        plan.source_graph_sha256 != anchor["source_certificate"]["graph_sha256"]
        or candidate.graph_sha256 != anchor["candidate_certificate"]["graph_sha256"]
        or not candidate.accepted
    ):
        raise ValueError("rebuild differs from pinned source/candidate graph")
    center = ((input_spec.lb + input_spec.ub) * 0.5).reshape(shape)
    return anchor, model, net, clone, center


def _apply_layer(layer, inputs):
    p, kind = layer.params, layer.kind
    if kind == "ADD":
        if len(inputs) != 2:
            raise ValueError("ADD needs two operands")
        return inputs[0] + inputs[1]
    if len(inputs) != 1:
        raise ValueError(f"{kind} needs one operand")
    x = inputs[0]
    if kind in {"INPUT_SPEC", "ASSERT", "FLATTEN", "RESHAPE"}:
        return x.reshape(-1)
    if kind == "CONV2D":
        return F.conv2d(
            x.reshape(p["input_shape"]), p["weight"], p.get("bias"),
            stride=p.get("stride", 1), padding=p.get("padding", 0),
            dilation=p.get("dilation", 1), groups=p.get("groups", 1),
        ).reshape(-1)
    if kind == "SCALE":
        return x * p["a"].reshape(-1)
    if kind == "BIAS":
        return x + p["c"].reshape(-1)
    if kind == "RELU":
        return torch.relu(x)
    if kind == "DENSE":
        return F.linear(x, p["weight"], p.get("bias"))
    raise ValueError(f"unsupported concrete layer: {kind}")


def concrete_forward(layers, preds, value, *, variable_program=False):
    """Two independent dataflow readings, with the same numeric primitives."""
    values = {}
    producers = {}
    for layer in layers:
        if layer.kind == "INPUT":
            out = value.reshape(-1)
        else:
            if variable_program:
                if layer.kind == "ADD":
                    inputs = []
                    for name in ("x_vars", "y_vars"):
                        token = tuple(layer.params[name])
                        owner_ids = {producers[v][0] for v in token}
                        if len(owner_ids) != 1:
                            raise ValueError("ambiguous variable operand")
                        owner = next(iter(owner_ids))
                        indices = [producers[v][1] for v in token]
                        inputs.append(values[owner][indices])
                else:
                    token = tuple(layer.in_vars)
                    owner_ids = {producers[v][0] for v in token}
                    if len(owner_ids) != 1:
                        raise ValueError("ambiguous variable input")
                    owner = next(iter(owner_ids))
                    inputs = [values[owner][[producers[v][1] for v in token]]]
            else:
                inputs = [values[parent] for parent in preds[layer.id]]
            out = _apply_layer(layer, inputs)
        out = out.reshape(-1)
        if out.numel() != len(layer.out_vars) or not torch.isfinite(out).all():
            raise ValueError(f"invalid output at layer {layer.id}")
        values[layer.id] = out
        for index, variable in enumerate(layer.out_vars):
            producers[variable] = (layer.id, index)
    return values


def build_record():
    started = time.monotonic()
    torch.set_num_threads(1)
    anchor, model, net, clone, center = load_registered_graph()
    repaired_preds = clone.predecessor_dict()
    with torch.inference_mode():
        variable = concrete_forward(net.layers, net.preds, center, variable_program=True)
        repaired = concrete_forward(net.layers, repaired_preds, center)
        original = concrete_forward(net.layers, net.preds, center)
        expected = model(center)
        while isinstance(expected, (tuple, list)) and len(expected) == 1:
            expected = expected[0]
        if not isinstance(expected, torch.Tensor):
            raise ValueError("unexpected model output")
        expected = expected.reshape(-1)
    last = net.layers[-1].id
    mismatches = [lid for lid in repaired if not torch.equal(repaired[lid], variable[lid])]
    output_difference = float(torch.max(torch.abs(repaired[last] - expected)))
    output_tolerance = 1e-10 + 1e-10 * float(torch.max(torch.abs(expected)))
    original_differences = {
        str(lid): float(torch.max(torch.abs(original[lid] - variable[lid])))
        for lid in variable if not torch.equal(original[lid], variable[lid])
    }
    if mismatches or output_difference > output_tolerance:
        raise ValueError("BN corrected graph failed concrete semantic preflight")
    # The registered frontier is fixed by the experiment, not by a rule selector.
    frontier = 36
    parents = repaired_preds[frontier]
    if net.layers[frontier].kind != "RELU" or len(parents) != 1:
        raise ValueError("registered frontier changed")
    shape = c3_necessary_condition(net.layers, repaired_preds, parents[0])
    sources = [
        Path(__file__),
        Path(cert.__file__),
        ROOT / "act/pipeline/verification/torch2act.py",
        ROOT / "act/back_end/hybridz_tf/tf_cnn.py",
        EXPERIMENT / "s0_c3_runtime_lineage_adapter_prototype.py",
        EXPERIMENT / "test_s0_c3_runtime_lineage_adapter_prototype.py",
        EXPERIMENT / "S0_C3_RUNTIME_LINEAGE_ADAPTER_PREREG_V1.md",
        EXPERIMENT / "S0_C3_IDENTITY_MIDDLE_PREREG_V1.md",
    ]
    return {
        "schema": "s0_c3_graph_preflight_v1", "date": "2026-09-05",
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "formal_baseline": "1870/2413", "external_E0": "61/400", "gain": 0,
        "default_enabled": False, "verifier_run": False,
        "anchor_sha256": ANCHOR_SHA256, "target": anchor["target"],
        "code_sha256": {str(p.relative_to(ROOT)): _sha256(p) for p in sources},
        "source_graph_sha256": anchor["source_certificate"]["graph_sha256"],
        "repaired_graph_sha256": clone.graph_sha256,
        "forward": {
            "input": "one fixed property-box center; diagnostic only",
            "input_sha256": hashlib.sha256(center.numpy().tobytes()).hexdigest(),
            "checked_layers": len(variable),
            "repaired_vs_variable_program_mismatching_layers": mismatches,
            "original_vs_variable_program_max_abs_by_layer": original_differences,
            "repaired_vs_pytorch_output_max_abs": output_difference,
            "pytorch_output_tolerance": output_tolerance,
            "universal_numeric_equivalence_proved": False,
        },
        "frontier": frontier, "c3_necessary_condition": shape,
        "second_consumers": {
            str(lid): clone.successor_dict()[lid]
            for lid in sorted({event for row in shape["paths"] for event in row["add_occurrences"]})
        },
        "seconds": time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    if output.parent.resolve() != (EXPERIMENT / "evidence").resolve():
        raise ValueError("output must be a fresh file in isolated evidence directory")
    if os.path.lexists(output):
        raise FileExistsError(output)
    record = build_record()
    _atomic_exclusive_json(output, record)
    print(json.dumps({"output": str(output), "sha256": _sha256(output),
                      "shape": record["c3_necessary_condition"]["reason_counts"],
                      "forward": record["forward"]}, indent=2))


if __name__ == "__main__":
    main()
