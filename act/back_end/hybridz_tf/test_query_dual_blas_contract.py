"""Controlled tests for the non-authoritative V5.1 BLAS contract."""

from __future__ import annotations

from dataclasses import replace
import os
import time
import unittest
from unittest import mock

from threadpoolctl import threadpool_limits

from act.back_end.hybridz_tf import query_dual_blas_contract as contract


class QueryDualBlasContractTests(unittest.TestCase):
    def _probe(self):
        with mock.patch.dict(
            os.environ,
            {"MKL_DYNAMIC": "FALSE", "OMP_DYNAMIC": "FALSE"},
            clear=False,
        ), threadpool_limits(limits=4):
            return contract.probe_query_dual_blas_contract(
                required_threads=4,
                deadline=time.monotonic() + 30.0,
            )

    def test_nontrivial_fraction_kernel_and_live_recheck(self):
        value = self._probe()
        self.assertFalse(value.proof_authority)
        self.assertTrue(
            contract.validate_query_dual_blas_contract(value)
        )
        matrix = value.receipt["matrix_kernel"]
        self.assertEqual(matrix["fraction_cells_checked"], 17 * 13)
        self.assertEqual(matrix["cancellation_exact"], "1")
        self.assertNotEqual(matrix["positive_subnormal_hex"], 0.0.hex())
        self.assertEqual(
            value.receipt["blas"]["selected"]["internal_api"], "mkl"
        )
        with mock.patch.dict(
            os.environ,
            {"MKL_DYNAMIC": "FALSE", "OMP_DYNAMIC": "FALSE"},
            clear=False,
        ), threadpool_limits(limits=4):
            self.assertTrue(
                contract.validate_query_dual_blas_contract(
                    value,
                    recheck_current_platform=True,
                    deadline=time.monotonic() + 30.0,
                )
            )

    def test_thread_and_dynamic_environment_fail_closed(self):
        with mock.patch.dict(
            os.environ,
            {"MKL_DYNAMIC": "FALSE", "OMP_DYNAMIC": "FALSE"},
            clear=False,
        ), threadpool_limits(limits=4):
            with self.assertRaises(
                contract.QueryDualBlasContractError
            ) as caught:
                contract.probe_query_dual_blas_contract(
                    required_threads=3
                )
        self.assertEqual(caught.exception.code, "THREAD_MISMATCH")

        with mock.patch.dict(
            os.environ,
            {"MKL_DYNAMIC": "TRUE", "OMP_DYNAMIC": "FALSE"},
            clear=False,
        ), threadpool_limits(limits=4):
            with self.assertRaises(
                contract.QueryDualBlasContractError
            ) as caught:
                contract.probe_query_dual_blas_contract(
                    required_threads=4
                )
        self.assertEqual(caught.exception.code, "DYNAMIC_THREADS")

    def test_backend_deadline_and_receipt_tamper_fail_closed(self):
        with mock.patch.dict(
            os.environ,
            {"MKL_DYNAMIC": "FALSE", "OMP_DYNAMIC": "FALSE"},
            clear=False,
        ), threadpool_limits(limits=4), mock.patch.object(
            contract, "threadpool_info", return_value=[]
        ):
            with self.assertRaises(
                contract.QueryDualBlasContractError
            ) as caught:
                contract.probe_query_dual_blas_contract(
                    required_threads=4
                )
        self.assertEqual(caught.exception.code, "UNSUPPORTED_BLAS")

        with self.assertRaises(
            contract.QueryDualBlasContractError
        ) as caught:
            contract.probe_query_dual_blas_contract(
                required_threads=4,
                deadline=time.monotonic() - 1.0,
            )
        self.assertEqual(caught.exception.code, "DEADLINE_EXPIRED")

        value = self._probe()
        forged = replace(value, content_sha256="0" * 64)
        self.assertFalse(
            contract.validate_query_dual_blas_contract(forged)
        )
        receipt = dict(value.receipt)
        receipt["required_threads"] = 1
        forged_receipt = replace(value, receipt=receipt)
        self.assertFalse(
            contract.validate_query_dual_blas_contract(
                forged_receipt
            )
        )


if __name__ == "__main__":
    unittest.main()
