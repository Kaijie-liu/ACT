#!/usr/bin/env python3
"""Verifier-side anti-spoof tests for property-bundled PairHull rows."""

from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
import json
import unittest
from unittest import mock

import torch

from act.back_end.config import BackendConfig, HybridZConfig
from act.back_end.core import Layer, Net
from act.back_end.hybridz_tf.property_pairhull import (
    DEFAULT_PAIRHULL_DIRECTIONS,
    build_pairhull_projection,
    exact_pairhull_beta,
)
from act.back_end.transfer_functions import (
    set_solver_mode,
    set_transfer_function_mode,
)
from act.back_end.verifier import (
    _validate_property_tail_pairhull_receipt,
    verify_once,
)
from act.front_end.specs import OutKind, OutputSpec
from act.util.stats import VerifyStatus


def _sha(value) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _finalize(mapping, field: str = "receipt_sha256"):
    result = dict(mapping)
    result.pop(field, None)
    result[field] = _sha(result)
    return result


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _disabled_receipt():
    inner = _finalize(
        {
            "schema": "act.property_pairhull.candidates.v1",
            "candidate_only": True,
            "proof_authority": (
                "stored_binary64_fraction_q+singleton_endpoints+"
                "exact_pairhull_four_phase+outward_intercept"
            ),
            "pair_selector_proof_authority": False,
            "foundation_rows_must_remain_retained": True,
            "requested_pair_budget": 0,
            "time_limit_seconds": 0.0,
            "status": "disabled",
            "whole_batch_complete": True,
            "selected_candidates": 0,
            "global_pair_count": 0,
            "candidate_records": [],
        }
    )
    return _finalize(
        {
            "schema": "operator_hz_property_tail_pairhull_v1",
            "enabled": False,
            "status": "disabled",
            "safe_only": True,
            "proof_authority": False,
            "selection_candidate_only": True,
            "selection_proof_authority": False,
            "exact_search_complete": False,
            "error_included": True,
            "compact_sparse_projection": True,
            "baseline_fallback_retained_per_rival": True,
            "foundation_slopes_reused": True,
            "foundation_intercept_outward_slack_inherited": True,
            "full_row_outward_affine": False,
            "prunes_prefix_frame": False,
            "budget_semantics": "global_unique_disjoint_pairs_v1",
            "max_rows_per_rival": 1,
            "pair_budget": 0,
            "time_limit_seconds": 0.0,
            "selected_rivals": 0,
            "selected_rival_ids": [],
            "global_pair_count": 0,
            "candidate_rows_sha256": _digest("empty rows"),
            "candidate_intercepts_sha256": _digest("empty intercepts"),
            "candidate_receipt": inner,
        }
    )


def _applied_receipt():
    projection = build_pairhull_projection(
        center=(Fraction(0), Fraction(1, 5)),
        generators=((Fraction(1), Fraction(0)), (Fraction(1), Fraction(1, 100))),
        error=(Fraction(0), Fraction(0)),
        directions=DEFAULT_PAIRHULL_DIRECTIONS,
    )
    exact = exact_pairhull_beta(
        projection,
        q=(Fraction(1), Fraction(-1)),
        candidate_slope=(Fraction(0), Fraction(0)),
    )
    record = {
        "rival_id": 1,
        "foundation_index": 0,
        "pair": [0, 1],
        "candidate_selection_proof_authority": False,
        "outward_intercept_validated": True,
        "candidate_plane_sha256": _digest("candidate plane"),
        "source_affine_sha256": projection.source_affine_sha256,
        "constraints_sha256": projection.constraints_sha256,
        "exact_pairhull_receipt": exact.receipt,
    }
    record = _finalize(record, "record_sha256")
    records = [record]
    inner = _finalize(
        {
            "schema": "act.property_pairhull.candidates.v1",
            "candidate_only": True,
            "proof_authority": (
                "stored_binary64_fraction_q+singleton_endpoints+"
                "exact_pairhull_four_phase+outward_intercept"
            ),
            "pair_selector_proof_authority": False,
            "foundation_rows_must_remain_retained": True,
            "requested_pair_budget": 2,
            "time_limit_seconds": 1.0,
            "status": "generated",
            "whole_batch_complete": True,
            "selected_candidates": 1,
            "global_pair_count": 1,
            "at_most_one_candidate_per_rival": True,
            "foundation_rows_retained_by_caller": True,
            "candidate_records": records,
            "candidate_records_sha256": _sha(records),
        }
    )
    return _finalize(
        {
            "schema": "operator_hz_property_tail_pairhull_v1",
            "enabled": True,
            "status": "applied",
            "safe_only": True,
            "proof_authority": True,
            "selection_candidate_only": True,
            "selection_proof_authority": False,
            "exact_search_complete": True,
            "error_included": True,
            "compact_sparse_projection": True,
            "baseline_fallback_retained_per_rival": True,
            "foundation_slopes_reused": True,
            "foundation_intercept_outward_slack_inherited": True,
            "full_row_outward_affine": True,
            "prunes_prefix_frame": False,
            "budget_semantics": "global_unique_disjoint_pairs_v1",
            "max_rows_per_rival": 1,
            "pair_budget": 2,
            "time_limit_seconds": 1.0,
            "selected_rivals": 1,
            "selected_rival_ids": [1],
            "selected_foundation_indices": [0],
            "selected_pair_indices": [[0, 1]],
            "global_pair_count": 1,
            "candidate_rows_sha256": _digest("exported candidate rows"),
            "candidate_intercepts_sha256": _digest(
                "exported candidate intercepts"
            ),
            "candidate_receipt": inner,
        }
    )


def _verified_pairhull_toy() -> Net:
    """Two correlated ReLUs whose separate relaxation leaves a 0.405 gap."""

    dtype = torch.float64
    input_vars = [0, 1]
    preactivation_vars = [2, 3]
    relu_vars = [4, 5]
    output_vars = [6]
    assertion = OutputSpec(
        kind=OutKind.LINEAR_LE,
        c=torch.tensor([1.0], dtype=dtype),
        d=torch.tensor([0.1], dtype=dtype),
    ).encode_linear(
        B=1,
        n_out=1,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    layers = [
        Layer(
            id=0,
            kind="INPUT",
            params={"shape": (1, 2), "dtype": "torch.float64"},
            in_vars=[],
            out_vars=input_vars,
        ),
        Layer(
            id=1,
            kind="INPUT_SPEC",
            params={
                "kind": "BOX",
                "lb": torch.tensor([[-1.0, -1.0]], dtype=dtype),
                "ub": torch.tensor([[1.0, 1.0]], dtype=dtype),
            },
            in_vars=input_vars,
            out_vars=input_vars,
        ),
        Layer(
            id=2,
            kind="DENSE",
            params={
                "weight": torch.tensor(
                    [[1.0, 0.0], [1.0, 0.01]], dtype=dtype
                ),
                "bias": torch.tensor([0.0, 0.2], dtype=dtype),
                "in_features": 2,
                "out_features": 2,
            },
            in_vars=input_vars,
            out_vars=preactivation_vars,
        ),
        Layer(
            id=3,
            kind="RELU",
            params={},
            in_vars=preactivation_vars,
            out_vars=relu_vars,
        ),
        Layer(
            id=4,
            kind="DENSE",
            params={
                "weight": torch.tensor([[1.0, -1.0]], dtype=dtype),
                "bias": torch.zeros(1, dtype=dtype),
                "in_features": 2,
                "out_features": 1,
            },
            in_vars=relu_vars,
            out_vars=output_vars,
        ),
        Layer(
            id=5,
            kind="ASSERT",
            params=assertion,
            in_vars=output_vars,
            out_vars=output_vars,
        ),
    ]
    return Net(
        layers=layers,
        preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]},
        succs={0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: []},
    )


def _pairhull_backend_config() -> BackendConfig:
    return BackendConfig(
        solver="hybridz",
        device="cpu",
        dtype="float64",
        hybridz=HybridZConfig(
            timeout=3.0,
            engine="operator_hz_objbound",
            property_tail_upper=True,
            property_tail_alpha_steps=16,
            property_tail_alpha_time_limit=1.0,
            property_tail_alpha_device="cpu",
            property_tail_pairhull_budget=1,
            property_tail_pairhull_time_limit=1.0,
            lp_prefilter_fraction=0.0,
            lp_prefilter_max_seconds=0.0,
        ),
    )


class PropertyPairHullVerifierReceiptTests(unittest.TestCase):
    def _valid(
        self,
        receipt,
        *,
        budget=2,
        time_limit=1.0,
        rivals=None,
        kinds=None,
    ):
        if rivals is None:
            rivals = [0, 1, 2]
        if kinds is None:
            kinds = [
                "negative_alpha_materialized",
                "pairhull_joint_materialized",
                "add_source_alpha0",
            ]
        return _validate_property_tail_pairhull_receipt(
            receipt,
            requested_budget=budget,
            requested_time_limit=time_limit,
            alternative_rivals=rivals,
            alternative_kinds=kinds,
            rival_count=3,
        )

    def test_disabled_receipt_is_bound_to_disabled_config(self) -> None:
        receipt = _disabled_receipt()
        self.assertTrue(
            self._valid(
                receipt,
                budget=0,
                time_limit=0.0,
                rivals=[],
                kinds=[],
            )
        )
        lightweight_disabled = dict(receipt)
        lightweight_disabled.pop("candidate_receipt")
        lightweight_disabled = _finalize(lightweight_disabled)
        self.assertTrue(
            self._valid(
                lightweight_disabled,
                budget=0,
                time_limit=0.0,
                rivals=[],
                kinds=[],
            )
        )
        self.assertFalse(
            self._valid(
                receipt,
                budget=1,
                time_limit=1.0,
                rivals=[],
                kinds=[],
            )
        )

    def test_applied_receipt_matches_exactly_one_exported_pair_row(self) -> None:
        self.assertTrue(self._valid(_applied_receipt()))

    def test_missing_tampered_and_rehashed_spoofs_fail_closed(self) -> None:
        self.assertFalse(self._valid({}))

        outer_tamper = _applied_receipt()
        outer_tamper["selected_rivals"] = 2
        self.assertFalse(self._valid(outer_tamper))

        pair_tamper = _applied_receipt()
        pair_tamper["selected_pair_indices"] = [[0, 2]]
        pair_tamper = _finalize(pair_tamper)
        self.assertFalse(self._valid(pair_tamper))

        nested_tamper = _applied_receipt()
        nested_tamper["candidate_receipt"]["candidate_records"][0][
            "exact_pairhull_receipt"
        ]["beta_exact"] = "999"
        nested_tamper = _finalize(nested_tamper)
        self.assertFalse(self._valid(nested_tamper))

        # Recomputing every ordinary checksum is not enough to forge the
        # required exact-enumeration semantics.
        semantic_spoof = _applied_receipt()
        exact = semantic_spoof["candidate_receipt"]["candidate_records"][0][
            "exact_pairhull_receipt"
        ]
        exact["float_lp_proof_authority"] = True
        exact = _finalize(exact)
        record = semantic_spoof["candidate_receipt"]["candidate_records"][0]
        record["exact_pairhull_receipt"] = exact
        record = _finalize(record, "record_sha256")
        inner = semantic_spoof["candidate_receipt"]
        inner["candidate_records"] = [record]
        inner["candidate_records_sha256"] = _sha([record])
        inner = _finalize(inner)
        semantic_spoof["candidate_receipt"] = inner
        semantic_spoof = _finalize(semantic_spoof)
        self.assertFalse(self._valid(semantic_spoof))

    def test_config_mapping_and_duplicate_row_mismatches_are_rejected(self) -> None:
        receipt = _applied_receipt()
        self.assertFalse(self._valid(receipt, budget=1))
        self.assertFalse(self._valid(receipt, time_limit=0.5))
        self.assertFalse(
            self._valid(
                receipt,
                rivals=[0, 2, 1],
            )
        )
        self.assertFalse(
            self._valid(
                receipt,
                rivals=[1, 1],
                kinds=[
                    "pairhull_joint_materialized",
                    "pairhull_joint_materialized",
                ],
            )
        )

        pair_row_on_fallback = copy.deepcopy(receipt)
        pair_row_on_fallback.update(
            {
                "status": "no_strict_guarded_cube_improvement",
                "proof_authority": False,
                "selected_rivals": 0,
                "selected_rival_ids": [],
            }
        )
        pair_row_on_fallback = _finalize(pair_row_on_fallback)
        self.assertFalse(self._valid(pair_row_on_fallback))

    def test_verify_once_accepts_applied_receipt_and_rejects_nested_tamper(
        self,
    ) -> None:
        from act.back_end.hybridz_tf import operator_hz as operator_module

        set_solver_mode("hybridz")
        set_transfer_function_mode("hybridz")
        try:
            result = verify_once(
                _verified_pairhull_toy(),
                backend_cfg=_pairhull_backend_config(),
            )[0]
            self.assertEqual(result.status, VerifyStatus.CERTIFIED)
            self.assertEqual(result.metadata["hz_verdict"], "SAFE")
            tail = result.metadata["operator_hz"]["property_tail_upper"]
            self.assertEqual(
                tail["alternative_plane_kinds"],
                [
                    "negative_alpha_materialized",
                    "pairhull_joint_materialized",
                ],
            )
            self.assertEqual(
                tail["pairhull_candidates"]["status"], "applied"
            )
            self.assertEqual(
                result.metadata["cfg_property_tail_pairhull_budget"], 1
            )
            self.assertEqual(
                result.metadata["cfg_property_tail_pairhull_time_limit"], 1.0
            )

            original_build = operator_module.build_operator_hz

            def tampered_build(*args, **kwargs):
                build = original_build(*args, **kwargs)
                pairhull = build.metadata["property_tail_upper"][
                    "pairhull_candidates"
                ]
                pairhull["candidate_receipt"]["candidate_records"][0][
                    "exact_pairhull_receipt"
                ]["beta_exact"] = "forged"
                return build

            with mock.patch(
                "act.back_end.hybridz_tf.operator_hz.build_operator_hz",
                side_effect=tampered_build,
            ):
                tampered = verify_once(
                    _verified_pairhull_toy(),
                    backend_cfg=_pairhull_backend_config(),
                )[0]
            self.assertEqual(tampered.status, VerifyStatus.UNKNOWN)
            self.assertEqual(
                tampered.metadata["reason"],
                "hybridz_operator_build_failed",
            )
            self.assertIn(
                "grouped upper-plane receipt",
                tampered.metadata["operator_error"],
            )
        finally:
            set_solver_mode(None)
            set_transfer_function_mode("interval")


if __name__ == "__main__":
    unittest.main(verbosity=2)
