"""Identical numerical diagnostic, new explicit partial-owner measurement."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.neural_hz_20260831 import run_c5_full_native_transaction_v1 as prior
from experiments.neural_hz_20260831.c5_partial_csr_owner_ledger_v3 import snapshot_partial_csr_owners


def main():
    output = prior.EXPERIMENT / "evidence/c5_full_native_owner_ledger_20260905_v2.json"
    old_output, old_snapshot, old_write = prior.OUTPUT, prior.snapshot_known_buffers, prior._atomic_exclusive_json

    def annotated_write(path, payload):
        if path != output:
            raise ValueError("unexpected measurement output")
        record = dict(payload, schema="c5_full_native_owner_ledger_v2",
                      accounting_adapter="explicit_schema_v2_plus_csr_data_interval_union_v3",
                      accounting_only_repetition=True,
                      prior_diagnostic_sha256="d25080c9318c3ffadc7c0735ebb29f3939f21769503be53e392e8e75d0a6ba0d",
                      adapter_wrapper_sha256=prior._sha256(Path(__file__)))
        return old_write(path, record)

    prior.OUTPUT = output
    prior.snapshot_known_buffers = snapshot_partial_csr_owners
    prior._atomic_exclusive_json = annotated_write
    try:
        prior.main()
    finally:
        prior.OUTPUT, prior.snapshot_known_buffers, prior._atomic_exclusive_json = old_output, old_snapshot, old_write


if __name__ == "__main__":
    main()
