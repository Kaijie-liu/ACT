#!/usr/bin/env python3
"""CPU-safe envelope tests for the controlled V3 CUDA toy audit."""

from __future__ import annotations

import json
import inspect
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from act.pipeline.verification import query_dual_v3_cuda_toy_audit as audit


class QueryDualV3CudaToyAuditTests(unittest.TestCase):
    def test_source_manifest_includes_audit_and_is_sha256(self) -> None:
        hashes = audit._source_hashes()
        self.assertIn(
            "act/pipeline/verification/query_dual_v3_cuda_toy_audit.py",
            hashes,
        )
        self.assertTrue(hashes)
        self.assertTrue(
            all(
                len(value) == 64
                and set(value) <= set("0123456789abcdef")
                for value in hashes.values()
            )
        )
        self.assertIn("act/util/device_manager.py", hashes)
        self.assertIn(
            "act/back_end/hybridz_tf/"
            "test_query_dual_operator_integration.py",
            hashes,
        )

    def test_transaction_uses_production_selector_and_solver_defaults(self):
        source = inspect.getsource(audit._run_transaction)
        self.assertNotIn("solver_factory", source)
        self.assertNotIn("selector=", source)
        self.assertIn('candidate_device="cuda"', source)
        self.assertIn(
            "validate_verified_query_dual_feedback", source
        )
        self.assertIn("steps=1", source)
        self.assertIn("block_size=1", source)
        self.assertIn("replay_chunk_size=16", source)
        self.assertEqual(audit._TARGETS, (7,))
        self.assertEqual(audit._QUOTAS, (1,))
        toy = audit._residual_two_relu_toy()
        self.assertIn(
            "ADD",
            {
                str(getattr(layer.kind, "value", layer.kind)).upper()
                for layer in toy.net.layers
            },
        )

    def test_atomic_output_does_not_clobber_without_opt_in(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "audit.json"
            audit._atomic_json(output, {"first": True}, overwrite=False)
            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                audit._atomic_json(output, {"second": True}, overwrite=False)
            self.assertEqual(json.loads(output.read_text()), {"first": True})
            audit._atomic_json(output, {"second": True}, overwrite=True)
            self.assertEqual(json.loads(output.read_text()), {"second": True})

    def test_invalid_repeat_config_returns_bound_failure_receipt(self) -> None:
        result = audit.run_audit(warmups=-1, repetitions=1)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["error"]["type"], "ValueError")
        self.assertIs(result["source_integrity_stable"], True)
        body = dict(result)
        claimed = body.pop("receipt_sha256")
        self.assertEqual(claimed, audit._canonical_sha256(body))

        noninteger = audit.run_audit(warmups=0.5, repetitions=1)
        self.assertEqual(noninteger["status"], "fail")
        self.assertEqual(noninteger["error"]["type"], "ValueError")


if __name__ == "__main__":
    unittest.main()
