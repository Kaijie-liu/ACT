"""Same live transaction, complete OrderedDict state registration."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.neural_hz_20260831 import c5_live_transaction_worker_v1 as prior
from experiments.neural_hz_20260831 import c5_functional_transaction_v1 as guarded
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect


def main():
    old_collect, old_guarded = prior.collect, guarded.collect
    old_output, old_write = prior.OUTPUT, prior._atomic_exclusive_json
    output = prior.EXPERIMENT / "evidence/c5_live_transaction_20260905_v2.json"

    def annotated_write(path, payload):
        if path == output:
            payload = dict(payload, schema="c5_live_transaction_v2", root_adapter="live_roots_v2",
                           wrapper_sha256=prior._sha256(Path(__file__)), prior_schema_failure_preserved=True)
        return old_write(path, payload)

    prior.collect = guarded.collect = collect
    prior.OUTPUT, prior._atomic_exclusive_json = output, annotated_write
    try:
        prior.main()
    finally:
        prior.collect, guarded.collect = old_collect, old_guarded
        prior.OUTPUT, prior._atomic_exclusive_json = old_output, old_write


if __name__ == "__main__":
    main()
