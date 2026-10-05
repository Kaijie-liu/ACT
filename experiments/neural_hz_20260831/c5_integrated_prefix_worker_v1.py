"""Fresh prefix with opt-in C5 and unchanged native ReLU/cache publication."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.neural_hz_20260831 import c5_corrected_prefix_worker_v2 as prefix
from experiments.neural_hz_20260831.c5_runtime_materializer_v1 import installed
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256


def main():
    directory = Path(sys.argv[1]).resolve()
    evidence = ROOT / "experiments/neural_hz_20260831/evidence/c5_live_transaction_20260905_v2.json"
    if _sha256(evidence) != "5b2af877318515dd7daa06eb0c79181d13c3dd71a351c6cb4b2cbeddc91fde94":
        raise ValueError("live qualification drift")
    qualification = json.loads(evidence.read_text())
    if not qualification.get("live_registered_gates_passed") or not qualification.get("input_roots_unchanged"):
        raise ValueError("live qualification did not pass")
    with (directory / "c5_events.jsonl").open("x") as stream:
        def emit(event):
            stream.write(json.dumps(event) + "\n")
            stream.flush()
        with installed(enabled=True, emit=emit):
            prefix.main()


if __name__ == "__main__":
    main()
