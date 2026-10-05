"""Observe fixed ReLU36, never expose the evaluation id to C5 dispatch."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from act.back_end.hybridz_tf import tf_cnn as cnn
from experiments.neural_hz_20260831 import c5_live_transaction_worker_v1 as qualifier
from experiments.neural_hz_20260831 import c5_functional_transaction_v1 as guarded
from experiments.neural_hz_20260831 import c5_corrected_prefix_worker_v2 as prefix
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect
from experiments.neural_hz_20260831.c5_runtime_materializer_v2 import installed

EXPERIMENT = Path(__file__).resolve().parent
OUTPUT = EXPERIMENT / "evidence/c5_relu36_live_20260905_v1.json"


def main():
    directory = Path(sys.argv[1]).resolve()
    if directory.parent != EXPERIMENT / "results" or OUTPUT.exists():
        raise ValueError("unsafe/occupied ReLU36 qualification output")
    evidence = EXPERIMENT / "evidence/c5_live_transaction_20260905_v2.json"
    if qualifier._sha256(evidence) != "5b2af877318515dd7daa06eb0c79181d13c3dd71a351c6cb4b2cbeddc91fde94":
        raise ValueError("earlier live qualification drift")
    old_collect, old_guarded = qualifier.collect, guarded.collect
    old_output, old_write = qualifier.OUTPUT, qualifier._atomic_exclusive_json

    def annotated_write(path, payload):
        if path == OUTPUT:
            stages = payload.get("stages", [])
            payload = dict(payload, schema="c5_relu36_live_v1", root_adapter="live_roots_v2",
                wrapper_sha256=qualifier._sha256(Path(__file__)),
                numeric_equivalence_passed=len(stages) == 2 and all(
                    bool(stage.get("bitwise_checks")) and all(stage["bitwise_checks"].values()) for stage in stages),
                measurement_point_only=36, prior_failures_preserved=True)
        return old_write(path, payload)

    qualifier.collect = guarded.collect = collect
    qualifier.OUTPUT, qualifier._atomic_exclusive_json = OUTPUT, annotated_write
    try:
        with (directory / "c5_events.jsonl").open("x") as stream:
            def emit(event):
                stream.write(json.dumps(event) + "\n")
                stream.flush()
            with installed(enabled=True, emit=emit):
                old_phase, old_deferred = cnn._try_phase_selective_exact_relu, cnn._try_deferred_expr_conv_relu
                context = {}

                def deferred(layer, expr, result, tf):
                    previous = dict(context)
                    context.update(producer_fact=result, incoming_expr=expr)
                    try:
                        return old_deferred(layer, expr, result, tf)
                    finally:
                        context.clear()
                        context.update(previous)

                def phase(expr, bounds, tf, relu):
                    if relu.id != 36:
                        return old_phase(expr, bounds, tf, relu)
                    if not qualifier.matches(expr):
                        raise ValueError("registered ReLU36 structure missing")
                    return qualifier.qualify(expr, bounds, tf, relu, dict(context), directory)

                cnn._try_phase_selective_exact_relu, cnn._try_deferred_expr_conv_relu = phase, deferred
                try:
                    prefix.main()
                except qualifier.LiveQualificationStop:
                    print(json.dumps({"status": "qualification_stopped_before_relu36", "formal_gain": 0}), flush=True)
                finally:
                    cnn._try_phase_selective_exact_relu, cnn._try_deferred_expr_conv_relu = old_phase, old_deferred
    finally:
        qualifier.collect, guarded.collect = old_collect, old_guarded
        qualifier.OUTPUT, qualifier._atomic_exclusive_json = old_output, old_write


if __name__ == "__main__":
    main()
