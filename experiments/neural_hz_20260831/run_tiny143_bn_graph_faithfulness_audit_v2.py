#!/usr/bin/env python3
"""Rebuild the Tiny143 ACT graph and publish one exclusive V2 audit record.

This runner converts the model, parses the already converted VNNLIB, builds
the ACT graph and exercises only the isolated graph-faithfulness prototype.
It never invokes propagation, a transfer function, a solver or a verifier.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
from typing import Any


EXPECTED_OUTPUT_NAME = "tiny_iid143_bn_graph_faithfulness_audit_v2.json"
EXPECTED_IID = 143


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _git(act_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=act_root, text=True
    ).strip()


def _run_frozen_tests(act_root: Path) -> dict[str, Any]:
    relative_tests = (
        "experiments/neural_hz_20260831/"
        "test_bn_graph_faithfulness_certificate_prototype.py",
        "experiments/neural_hz_20260831/"
        "test_bn_graph_faithfulness_certificate_adversarial.py",
    )
    command = [sys.executable, "-m", "pytest", "-q", *relative_tests]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        command,
        cwd=act_root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = completed.stdout.strip()
    if completed.returncode != 0 or "78 passed" not in output:
        raise RuntimeError(
            "frozen BN graph tests did not produce the required 78/78 pass"
        )
    return {
        "command": [sys.executable, "-m", "pytest", "-q", *relative_tests],
        "returncode": completed.returncode,
        "passed": 78,
        "total": 78,
        "output_tail": output.splitlines()[-1],
    }


def _atomic_exclusive_json(output: Path, payload: dict[str, Any]) -> None:
    """Publish complete JSON by an exclusive hard-link in the target directory."""

    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite audit output: {output}")
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary = output.with_name(
        f".{output.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
    )
    descriptor: int | None = None
    published = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("zero-byte audit write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.link(temporary, output)
        published = True
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink(missing_ok=True)
        except Exception:
            if not published:
                raise


def _numeric_summary(layer: Any, snapshot: Any) -> dict[str, Any]:
    import numpy as np
    import torch

    raw = layer.params[snapshot.numeric_payload_key]
    if isinstance(raw, torch.Tensor):
        array = raw.detach().cpu().numpy()
    else:
        array = np.asarray(raw)
    canonical = np.asarray(array, dtype=np.float64).reshape(-1)
    summary: dict[str, Any] = {
        "layer_id": snapshot.layer_id,
        "kind": snapshot.kind,
        "payload_key": snapshot.numeric_payload_key,
        "width": snapshot.numeric_payload_size,
        "source_dtype": snapshot.numeric_payload_dtype,
        "payload_sha256": snapshot.numeric_payload_sha256,
        "finite": bool(np.all(np.isfinite(canonical))),
        "minimum": float(canonical.min()),
        "maximum": float(canonical.max()),
    }
    if snapshot.kind == "SCALE":
        summary["not_equal_one"] = int(np.count_nonzero(canonical != 1.0))
    else:
        summary["nonzero"] = int(np.count_nonzero(canonical))
    return summary


def _validate_trial8(trial8_path: Path) -> dict[str, Any]:
    data = json.loads(trial8_path.read_text())
    if data.get("bench") != "tinyimagenet_2024" or data.get("iid") != 143:
        raise ValueError("Trial8 is not the registered TinyImageNet iid143 record")
    structures = data.get("layer_hz_structure")
    if not isinstance(structures, list) or len(structures) != 81:
        raise ValueError("Trial8 layer_hz_structure must contain 81 entries")
    by_id = {row.get("id"): row for row in structures if isinstance(row, dict)}
    expected_kinds = {
        29: "CONV2D", 30: "SCALE", 31: "BIAS", 32: "ADD",
        33: "CONV2D", 34: "SCALE", 35: "BIAS", 36: "RELU",
    }
    if any(by_id.get(layer_id, {}).get("kind") != kind
           for layer_id, kind in expected_kinds.items()):
        raise ValueError("Trial8 registered ReLU36 neighborhood changed")
    return {
        "sha256": _sha256(trial8_path),
        "schema": data.get("schema"),
        "arm": data.get("arm"),
        "verdict": data.get("verdict"),
        "stop_after_layer": data.get("stop_after_layer"),
        "layer36_lazy_terms": by_id[36].get("lazy_terms"),
        "layer36_lazy_operator_entries": by_id[36].get(
            "lazy_operator_entries"
        ),
    }


def _build_record(args: argparse.Namespace) -> dict[str, Any]:
    act_root = args.act_root.resolve(strict=True)
    benchmark_root = args.benchmark_root.resolve(strict=True)
    converted_spec = args.converted_spec.resolve(strict=True)
    trial8_path = args.trial8.resolve(strict=True)

    if benchmark_root.name != "tinyimagenet_2024" or args.iid != EXPECTED_IID:
        raise ValueError("this V2 runner is frozen to tinyimagenet_2024 iid143")
    rows = [
        line.split(",")
        for line in (benchmark_root / "instances.csv").read_text().splitlines()
        if line.strip()
    ]
    if args.iid >= len(rows):
        raise IndexError("iid is outside instances.csv")
    model_path = (benchmark_root / rows[args.iid][0].removeprefix("./")).resolve(
        strict=True
    )

    if str(act_root) not in sys.path:
        sys.path.insert(0, str(act_root))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch
    from act.front_end.spec_creator_base import LabeledInputTensor
    from act.front_end.verifiable_model import (
        InputLayer,
        InputSpecLayer,
        OutputSpecLayer,
        VerifiableModel,
    )
    from act.front_end.vnnlib_loader.onnx_converter import (
        convert_onnx_to_pytorch,
        get_onnx_input_shape,
    )
    from act.front_end.vnnlib_loader.vnnlib_parser import parse_vnnlib_queries
    from act.pipeline.verification.torch2act import TorchToACT
    from experiments.neural_hz_20260831 import (
        bn_graph_faithfulness_certificate_prototype as graph_cert,
    )

    input_shape = tuple(get_onnx_input_shape(model_path))
    model = convert_onnx_to_pytorch(model_path).eval()
    model_dtype = next(model.parameters()).dtype
    labeled = LabeledInputTensor(
        tensor=torch.zeros(input_shape, dtype=model_dtype),
        label=torch.tensor([0]),
    )
    queries = parse_vnnlib_queries(converted_spec, labeled_tensor=labeled)
    if len(queries) != 1:
        raise ValueError("Tiny143 converted spec must contain exactly one query")
    input_spec, output_spec = queries[0]
    input_spec.lb = input_spec.lb.to(dtype=model_dtype)
    input_spec.ub = input_spec.ub.to(dtype=model_dtype)
    wrapped = VerifiableModel(
        input_layer=InputLayer(
            labeled_input=labeled,
            shape=input_shape,
            dtype=model_dtype,
        ),
        input_spec=InputSpecLayer(input_spec),
        model=model,
        output_spec=OutputSpecLayer(output_spec),
    )
    net = TorchToACT(wrapped).run()

    source = graph_cert.audit_graph_faithfulness(
        net.layers, net.preds, net.succs
    )
    plan = graph_cert.plan_batchnorm_graph_repair(
        net.layers, net.preds, net.succs
    )
    if not plan.authorized:
        raise RuntimeError(f"registered BN repair plan rejected: {plan.reason}")
    clone, candidate = graph_cert.apply_repair_plan_to_clone(
        net.layers, net.preds, net.succs, plan
    )
    trial8 = _validate_trial8(trial8_path)
    observations = [
        graph_cert.PathOperatorObservation(29, "implicit_conv2d"),
        graph_cert.PathOperatorObservation(33, "implicit_conv2d"),
    ]
    path_certificate = graph_cert.audit_path_operator_lineage(
        net.layers,
        clone.predecessor_dict(),
        clone.successor_dict(),
        [29, 30, 31, 32, 33, 34, 35, 36],
        observations,
    )

    issue_counts = Counter(issue.code for issue in source.issues)
    expected_issue_counts = {
        "bn_scale_bias_graph_event_missing": 19,
        "predecessor_producer_mismatch": 19,
    }
    path_issues = [
        (issue.code, issue.occurrence_layer_id)
        for issue in path_certificate.issues
    ]
    if (
        source.accepted
        or source.layer_count != 81
        or len(source.batchnorm_pairs) != 19
        or dict(issue_counts) != expected_issue_counts
        or len(plan.replacements) != 19
        or not candidate.accepted
        or candidate.issues
        or path_certificate.accepted
        or path_issues
        != [
            ("unaccounted_linear_event", 30),
            ("unaccounted_linear_event", 34),
        ]
    ):
        raise RuntimeError("actual Tiny143 V2 audit did not match frozen findings")

    snapshots = graph_cert._freeze_layers(net.layers)
    payloads = [
        _numeric_summary(net.layers[item.layer_id], item)
        for item in snapshots
        if item.numeric_payload_key
    ]
    prototype_path = act_root / (
        "experiments/neural_hz_20260831/"
        "bn_graph_faithfulness_certificate_prototype.py"
    )
    official_test = act_root / (
        "experiments/neural_hz_20260831/"
        "test_bn_graph_faithfulness_certificate_prototype.py"
    )
    adversarial_test = act_root / (
        "experiments/neural_hz_20260831/"
        "test_bn_graph_faithfulness_certificate_adversarial.py"
    )
    v1_evidence = act_root / (
        "experiments/neural_hz_20260831/evidence/"
        "tiny_iid143_bn_graph_faithfulness_audit_v1.json"
    )
    tests = _run_frozen_tests(act_root)

    return {
        "format_version": "tiny_iid143_bn_graph_faithfulness_audit_v2",
        "recorded_date": "2026-08-31",
        "branch": _git(act_root, "branch", "--show-current"),
        "commit": _git(act_root, "rev-parse", "HEAD"),
        "formal_baseline": "1870/2413",
        "family_non_regression_constraint": 13,
        "gain": 0,
        "default_enabled": False,
        "verifier_run": False,
        "v1_superseded": {
            "relative_path": str(v1_evidence.relative_to(act_root)),
            "sha256": _sha256(v1_evidence),
            "reason": [
                "V1 graph digest did not bind BN SCALE.a and BIAS.c numeric payloads",
                "V1 preceded strict SSA/single-producer and multi-operand arity hardening",
                "V1 preceded repair-plan re-derivation and callback post-check dual CAS",
                "V1 preceded the independent adversarial suite",
            ],
        },
        "code_provenance": {
            "prototype_sha256": _sha256(prototype_path),
            "official_test_sha256": _sha256(official_test),
            "adversarial_test_sha256": _sha256(adversarial_test),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "torch2act_sha256": _sha256(
                act_root / "act/pipeline/verification/torch2act.py"
            ),
            "tf_cnn_sha256": _sha256(
                act_root / "act/back_end/hybridz_tf/tf_cnn.py"
            ),
        },
        "tests": tests,
        "target": {
            "family": "tinyimagenet_2024",
            "iid": args.iid,
            "model": str(model_path),
            "model_sha256": _sha256(model_path),
            "converted_spec": str(converted_spec),
            "converted_spec_sha256": _sha256(converted_spec),
            "input_shape": list(input_shape),
            "dtype": str(model_dtype),
        },
        "strict_parser": {
            "accepted_real_layer_types_without_coercion": True,
            "layer_count": source.layer_count,
            "checked_input_variables": source.checked_input_variables,
            "edge_count": source.edge_count,
            "batchnorm_pair_count": len(source.batchnorm_pairs),
            "malformed_input": source.graph_sha256 == "",
        },
        "source_certificate": {
            "accepted": source.accepted,
            "graph_sha256": source.graph_sha256,
            "issue_count": len(source.issues),
            "issue_counts": dict(sorted(issue_counts.items())),
            "issue_layer_ids": {
                code: [
                    issue.layer_id
                    for issue in source.issues
                    if issue.code == code
                ]
                for code in sorted(issue_counts)
            },
        },
        "bn_numeric_payloads": payloads,
        "repair_plan": {
            "authorized": plan.authorized,
            "reason": plan.reason,
            "source_graph_sha256": plan.source_graph_sha256,
            "candidate_graph_sha256": plan.candidate_graph_sha256,
            "replacements": [
                {
                    "layer_id": item.layer_id,
                    "old_predecessors": list(item.old_predecessors),
                    "new_predecessors": list(item.new_predecessors),
                }
                for item in plan.replacements
            ],
            "rederived_before_clone": True,
        },
        "candidate_certificate": {
            "accepted": candidate.accepted,
            "graph_sha256": candidate.graph_sha256,
            "layer_count": candidate.layer_count,
            "edge_count": candidate.edge_count,
            "batchnorm_pair_count": len(candidate.batchnorm_pairs),
            "issue_count": len(candidate.issues),
        },
        "relu36_path_diagnostic": {
            "accepted": path_certificate.accepted,
            "path_layer_ids": list(path_certificate.path_layer_ids),
            "expected_operators": [
                [item.occurrence_layer_id, item.semantic_kind]
                for item in path_certificate.expected_operators
            ],
            "observed_operators": [
                [item.occurrence_layer_id, item.semantic_kind]
                for item in path_certificate.observed_operators
            ],
            "bias_transition_layer_ids": list(
                path_certificate.bias_transition_layer_ids
            ),
            "issues": [
                [issue.code, issue.occurrence_layer_id]
                for issue in path_certificate.issues
            ],
            "observation_source": (
                "content-addressed Trial8 lazy-core interpretation from the "
                "prior C2 audit; not a production runtime-lineage adapter"
            ),
        },
        "trial8": trial8,
        "claims": {
            "production_patch": False,
            "runtime_lineage_adapter_proven": False,
            "numeric_path_observation_binding_proven": False,
            "c3_target_authorized": False,
            "full_2413_replay": False,
            "all_13_families_replayed": False,
        },
        "advancement_gate": (
            "A production loader repair and runtime lineage adapter must pass "
            "forward/HZ equivalence and a complete 2413 replay retaining all "
            "1870 solved rows and every one of 13 family counts."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--act-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--iid", type=int, required=True)
    parser.add_argument("--converted-spec", type=Path, required=True)
    parser.add_argument("--trial8", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    act_root = args.act_root.resolve(strict=True)
    evidence_root = (
        act_root / "experiments/neural_hz_20260831/evidence"
    ).resolve(strict=True)
    output = args.output.resolve(strict=False)
    if output.parent != evidence_root or output.name != EXPECTED_OUTPUT_NAME:
        raise ValueError(
            f"output must be the exclusive V2 evidence path under {evidence_root}"
        )
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite audit output: {output}")
    payload = _build_record(args)
    _atomic_exclusive_json(output, payload)
    print(json.dumps({"output": str(output), "sha256": _sha256(output)}))


if __name__ == "__main__":
    main()
