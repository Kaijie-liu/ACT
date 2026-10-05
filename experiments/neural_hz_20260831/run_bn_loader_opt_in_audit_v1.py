"""Check the default-off production loader against the pinned Tiny143 clone."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from act.pipeline.verification.torch2act import TorchToACT
from act.front_end.spec_creator_base import LabeledInputTensor
from act.front_end.verifiable_model import (
    InputLayer, InputSpecLayer, OutputSpecLayer, VerifiableModel,
)
from act.front_end.vnnlib_loader.vnnlib_parser import parse_vnnlib_queries
from experiments.neural_hz_20260831 import bn_graph_faithfulness_certificate_prototype as cert
from experiments.neural_hz_20260831.run_s0_c3_graph_preflight_v1 import (
    EXPERIMENT, ANCHOR_SHA256, load_registered_graph, concrete_forward,
)
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import (
    _atomic_exclusive_json, _sha256,
)


def build_record():
    started = time.monotonic()
    torch.set_num_threads(1)
    anchor, model, old, clone, center = load_registered_graph()
    shape, dtype = tuple(center.shape), center.dtype
    labeled = LabeledInputTensor(torch.zeros_like(center), torch.tensor([0]))
    queries = parse_vnnlib_queries(Path(anchor["target"]["converted_spec"]), labeled_tensor=labeled)
    if len(queries) != 1:
        raise ValueError("query count changed")
    input_spec, output_spec = queries[0]
    input_spec.lb = input_spec.lb.to(dtype=dtype)
    input_spec.ub = input_spec.ub.to(dtype=dtype)
    wrapped = VerifiableModel(
        input_layer=InputLayer(labeled_input=labeled, shape=shape, dtype=dtype),
        input_spec=InputSpecLayer(input_spec), model=model,
        output_spec=OutputSpecLayer(output_spec),
    )
    candidate = TorchToACT(wrapped, repair_batchnorm_producer_graph=True).run()
    off = TorchToACT(wrapped, repair_batchnorm_producer_graph=False).run()
    certificate = cert.audit_graph_faithfulness(candidate.layers, candidate.preds, candidate.succs)
    if not certificate.accepted or certificate.graph_sha256 != clone.graph_sha256:
        raise ValueError("production flag does not match pinned independent clone")
    if old.preds != off.preds or old.succs != off.succs:
        raise ValueError("default/explicit-off topology changed")
    with torch.inference_mode():
        expected = concrete_forward(old.layers, clone.predecessor_dict(), center)
        actual = concrete_forward(candidate.layers, candidate.preds, center)
    if set(expected) != set(actual) or any(
        expected[k].numpy().tobytes() != actual[k].numpy().tobytes() for k in actual
    ):
        raise ValueError("production corrected values differ from independent clone")
    paths = [
        Path(__file__),
        ROOT / "act/pipeline/verification/torch2act.py",
        ROOT / "act/pipeline/verification/batchnorm_graph.py",
        EXPERIMENT / "test_bn_loader_opt_in_v1.py",
        EXPERIMENT / "run_s0_c3_graph_preflight_v1.py",
        Path(cert.__file__),
    ]
    return {
        "schema": "bn_loader_opt_in_audit_v1", "date": "2026-09-05",
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "anchor_sha256": ANCHOR_SHA256, "target": anchor["target"],
        "formal_baseline": "1870/2413", "external_E0": "61/400", "gain": 0,
        "flag": "repair_batchnorm_producer_graph", "default_enabled": False,
        "verifier_run": False, "family_retention_replay_performed": False,
        "full_2413_replay_performed": False,
        "default_equals_explicit_off": True,
        "source_graph_sha256": anchor["source_certificate"]["graph_sha256"],
        "production_candidate_graph_sha256": certificate.graph_sha256,
        "checked_layers": len(actual), "bitwise_equal_to_clone_all_layers": True,
        "changed_predecessor_entries": [
            {"layer": lid, "before": old.preds[lid], "after": candidate.preds[lid]}
            for lid in old.preds if old.preds[lid] != candidate.preds[lid]
        ],
        "code_sha256": {str(p.relative_to(ROOT)): _sha256(p) for p in paths},
        "seconds": time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    if output.parent.resolve() != (EXPERIMENT / "evidence").resolve():
        raise ValueError("output must be a fresh isolated evidence file")
    if os.path.lexists(output):
        raise FileExistsError(output)
    record = build_record()
    _atomic_exclusive_json(output, record)
    print(json.dumps({"output": str(output), "sha256": _sha256(output),
                      "changed_edges": len(record["changed_predecessor_entries"]),
                      "checked_layers": record["checked_layers"], "gain": 0}))


if __name__ == "__main__":
    main()
