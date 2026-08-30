#!/usr/bin/env python3
"""CPU-safe envelope tests for the controlled V3 toy audit."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from act.pipeline.verification import query_dual_v3_toy_audit as audit


class QueryDualV3ToyAuditTests(unittest.TestCase):
    def test_source_manifest_includes_audit_and_envelope_test(self) -> None:
        hashes = audit._source_hashes()
        self.assertIn(
            "act/pipeline/verification/query_dual_v3_toy_audit.py",
            hashes,
        )
        self.assertIn(
            "act/pipeline/verification/test_query_dual_v3_toy_audit.py",
            hashes,
        )
        self.assertTrue(
            all(
                len(value) == 64
                and set(value) <= set("0123456789abcdef")
                for value in hashes.values()
            )
        )

    def test_benchmark_exception_returns_canonical_failure_receipt(self) -> None:
        with mock.patch.object(
            audit,
            "_run_soundness_suite",
            return_value={"pass": True, "tests_run": 1},
        ), mock.patch.object(
            audit,
            "_static_prepare_benchmark",
            side_effect=RuntimeError("controlled-benchmark-failure"),
        ), mock.patch.object(
            audit, "_transaction_benchmark"
        ) as transaction:
            result = audit.run_audit()

        self.assertEqual(result["status"], "fail")
        self.assertIs(result["proof_authority"], False)
        self.assertEqual(result["error"]["type"], "RuntimeError")
        self.assertIn(
            "controlled-benchmark-failure", result["error"]["message"]
        )
        self.assertIs(result["source_integrity_stable"], True)
        self.assertEqual(result["soundness_tightness"]["tests_run"], 1)
        transaction.assert_not_called()
        body = dict(result)
        claimed = body.pop("receipt_sha256")
        self.assertEqual(claimed, audit._canonical_sha256(body))

    def test_failure_receipt_is_atomically_publishable(self) -> None:
        with mock.patch.object(
            audit,
            "_run_soundness_suite",
            side_effect=RuntimeError("publishable-failure"),
        ):
            result = audit.run_audit()
        with TemporaryDirectory() as directory:
            output = Path(directory) / "audit.json"
            audit._atomic_json(output, result, overwrite=False)
            self.assertEqual(json.loads(output.read_text()), result)
            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                audit._atomic_json(
                    output, {"must_not": "clobber"}, overwrite=False
                )
            self.assertEqual(json.loads(output.read_text()), result)
            self.assertFalse(
                list(output.parent.glob(f".{output.name}.*.tmp"))
            )


if __name__ == "__main__":
    unittest.main()
