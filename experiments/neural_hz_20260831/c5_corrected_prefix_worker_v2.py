"""Incremental observation around the unchanged V1 corrected prefix worker."""

import json
import os
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from act.back_end.hybridz_tf.hybridz_tf import HybridzTF
from experiments.neural_hz_20260831 import c5_corrected_prefix_worker_v1 as v1
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256


def main():
    directory = Path(sys.argv[1]).resolve()
    if directory.parent != ROOT / "experiments/neural_hz_20260831/results":
        raise ValueError("snapshot outside isolated run")
    original = HybridzTF.apply
    started = time.monotonic()
    with (directory / "events.jsonl").open("x") as events:
        def event(layer, stage):
            events.write(json.dumps({"layer": layer.id, "kind": layer.kind, "stage": stage,
                                     "elapsed_s": time.monotonic() - started}) + "\n")
            events.flush()

        def observed(self, layer, *args, **kwargs):
            event(layer, "start")
            result = original(self, layer, *args, **kwargs)
            event(layer, "transfer_complete")
            if layer.kind in {"RELU", "ADD"}:
                after = args[3] if len(args) > 3 else kwargs["after"]
                payload = {"schema": "c5_incremental_snapshot_v2", "layer": layer.id,
                           "net": self._net, "hz_cache": self._sparse_hz_cache,
                           "expr_cache": self._sparse_affine_expr_cache,
                           "bounds": {lid: fact.bounds for lid, fact in after.items()},
                           "terminal_bounds": result.bounds,
                           "precomputed_relu": self._sparse_precomputed_relu,
                           "provenance": v1.worker._provenance(ROOT), "formal_gain": 0}
                temporary = directory / f"layer{layer.id:02d}.pickle.partial"
                target = directory / f"layer{layer.id:02d}.pickle"
                with temporary.open("xb") as stream:
                    pickle.dump(payload, stream, protocol=5)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(temporary, target)
                _atomic_exclusive_json(directory / f"layer{layer.id:02d}.snapshot.json", {
                    "schema": payload["schema"], "layer": layer.id, "formal_gain": 0,
                    "pickle_sha256": _sha256(target), "pickle_bytes": target.stat().st_size,
                    "worker_sha256": _sha256(Path(__file__)), "provenance": payload["provenance"],
                    "hz_layers": sorted(self._sparse_hz_cache), "expr_layers": sorted(self._sparse_affine_expr_cache)})
                event(layer, "snapshot_sealed")
            return result

        HybridzTF.apply = observed
        try:
            v1.main()
        finally:
            HybridzTF.apply = original


if __name__ == "__main__":
    main()
