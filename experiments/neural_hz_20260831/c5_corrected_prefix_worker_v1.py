"""Capture one corrected, otherwise unchanged HZ prefix; no candidate execution."""

import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from act.pipeline.verification.torch2act import TorchToACT
from act.back_end.hybridz_tf.hybridz_tf import HybridzTF
from experiments.neural_hz_20260831 import shadow_worker_dtype_v2 as worker
from experiments.neural_hz_20260831 import bn_graph_faithfulness_certificate_prototype as cert
from experiments.neural_hz_20260831.run_s0_c3_graph_preflight_v1 import ANCHOR, ANCHOR_SHA256
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256


def main():
    directory = Path(sys.argv[1]).resolve()
    if directory.parent != ROOT / "experiments/neural_hz_20260831/results":
        raise ValueError("snapshot outside isolated run")
    if _sha256(ANCHOR) != ANCHOR_SHA256:
        raise ValueError("anchor drift")
    anchor = json.loads(ANCHOR.read_text())
    original_init, original_run, original_apply = TorchToACT.__init__, TorchToACT.run, HybridzTF.apply

    def repaired_init(self, *args, **kwargs):
        kwargs["repair_batchnorm_producer_graph"] = True
        original_init(self, *args, **kwargs)

    def checked_run(self, *args, **kwargs):
        net = original_run(self, *args, **kwargs)
        certificate = cert.audit_graph_faithfulness(net.layers, net.preds, net.succs)
        if not certificate.accepted or certificate.graph_sha256 != anchor["candidate_certificate"]["graph_sha256"]:
            raise ValueError("prefix graph differs from pinned corrected graph")
        return net

    def capture_apply(self, layer, *args, **kwargs):
        result = original_apply(self, layer, *args, **kwargs)
        if layer.id == 32:
            # These are the real HZ objects, not reconstructions from intervals.
            after = args[3] if len(args) > 3 else kwargs["after"]
            payload = {"schema": "c5_corrected_prefix_snapshot_v1", "layer": 32,
                       "net": self._net, "hz_cache": self._sparse_hz_cache,
                       "expr_cache": self._sparse_affine_expr_cache,
                       "bounds": {lid: fact.bounds for lid, fact in after.items()},
                       "terminal_bounds": result.bounds,
                       "precomputed_relu": self._sparse_precomputed_relu,
                       "provenance": worker._provenance(ROOT), "formal_gain": 0}
            temporary, target = directory / "prefix32.pickle.partial", directory / "prefix32.pickle"
            with temporary.open("xb") as stream:
                pickle.dump(payload, stream, protocol=5)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, target)
            # Keep the linked staging name as an explicit recoverable artifact.
            _atomic_exclusive_json(directory / "snapshot.json", {
                "schema": payload["schema"], "formal_gain": 0, "pickle_sha256": _sha256(target),
                "bytes": target.stat().st_size, "worker_sha256": _sha256(Path(__file__)),
                "provenance": payload["provenance"], "corrected_graph_sha256": anchor["candidate_certificate"]["graph_sha256"],
                "hz_layers": sorted(self._sparse_hz_cache), "expr_layers": sorted(self._sparse_affine_expr_cache)})
        return result

    TorchToACT.__init__, TorchToACT.run, HybridzTF.apply = repaired_init, checked_run, capture_apply
    sys.argv = [worker.__file__, *sys.argv[2:]]
    try:
        worker.main()
    finally:
        TorchToACT.__init__, TorchToACT.run, HybridzTF.apply = original_init, original_run, original_apply


if __name__ == "__main__":
    main()
