"""Permanent unit audits for the strict large-classification gate runner."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import yaml

from act.back_end.config import BackendConfig, HybridZConfig
from act.pipeline.verification.hybridz_largecls_gate import (
    DEFAULT_CONFIG,
    CUDA_PEAK_MEMORY_SCHEMA,
    GateConfigError,
    SCHEMA_VERSION,
    _RUNTIME_BUILTIN_INTEGER_FIELDS,
    _build_parser,
    _capture_cuda_peak_memory,
    _canonical_json,
    _classify_result,
    _cuda_peak_memory_policy,
    _cuda_peak_memory_receipt_valid,
    _cuda_peak_memory_unavailable,
    _engine_connected,
    _experiment_fingerprint,
    _fixed_worker_environment,
    _json_safe,
    _query_dual_effective_by_family,
    _query_dual_candidate_policy,
    _query_dual_family_snapshot,
    _query_dual_worker_payload,
    _runtime_from_args,
    _sha256_bytes,
    _start_cuda_peak_memory,
    _summarize_cuda_peak_memory,
    _validate_property_micro_rlt_parent_only_selection,
    _validate_property_micro_rlt_settings,
    _validate_operator_phase_clique_selection,
    _validate_runtime,
    _validate_worker_feature_receipts,
    run_gate,
    validate_promotion,
)


ENGINE = "operator_hz_objbound"
FAMILIES = [
    "cifar100_medium",
    "cifar100_large",
    "tinyimagenet_medium",
]


def _runtime(**updates):
    value = {
        "default_gate": 6,
        "wall_timeout_seconds": 100.0,
        "device": "cuda",
        "gpu_index": 0,
        "dtype": "float64",
        "engine": ENGINE,
        "operator_exact_budget": 0,
        "operator_phase_clique_time_limit": 0.0,
        "operator_materialize_add": True,
        "query_dual_feedback_steps": 0,
        "query_dual_feedback_time_limit": 0.0,
        "query_dual_feedback_block_size": 1024,
        "query_dual_feedback_device": "cuda",
        "preactivation_lp_budget": 0,
        "preactivation_lp_time_limit": 0.0,
        "property_residual_budget": 0,
        "property_residual_time_limit": 0.0,
        "property_residual_max_adjoint_cells": 30_000_000,
        "property_residual_pool_per_rival": 8,
        "property_tail_upper": False,
        "property_micro_rlt_product_cap": 0,
        "property_micro_rlt_packet_mode": "both",
        "property_micro_rlt_parent_prefilter_seconds": 0.0,
        "property_micro_rlt_parent_only_diagnostic": False,
        "property_tail_add_source_planes": False,
        "property_tail_alpha_steps": 0,
        "property_tail_alpha_time_limit": 0.0,
        "property_tail_alpha_learning_rate": 0.08,
        "property_tail_alpha_max_cells": 50_000_000,
        "property_tail_alpha_device": "cuda",
        "property_tail_mixture_grid_bits": 0,
        "property_tail_pairhull_budget": 0,
        "property_tail_pairhull_time_limit": 0.0,
        "property_tail_suffix_blocks": 0,
        "property_tail_suffix_alpha_steps": 0,
        "property_tail_suffix_alpha_time_limit": 0.0,
        "property_tail_suffix_alpha_device": "cuda",
        "gpu_dual_steps": 0,
        "gpu_dual_time_limit": 0.0,
        "gpu_dual_row_topk": 0,
        "gpu_dual_learning_rate": 0.08,
        "lp_prefilter_fraction": 0.2,
        "lp_prefilter_max_seconds": 8.0,
        "row_workers": 4,
        "total_solver_threads": 20,
        "max_inconclusive_per_family": 1,
    }
    value.update(updates)
    return value


def _unknown(metadata=None):
    base_metadata = {
        "solver": "hybridz",
        "engine": ENGINE,
        "hz_verdict": "UNKNOWN",
        "hz_has_witness": False,
    }
    if metadata:
        base_metadata.update(metadata)
    return {
        "worker_state": "completed",
        "status": "unknown",
        "expected_engine": ENGINE,
        "metadata": base_metadata,
        "has_counterexample": False,
    }


def _query_receipt(body):
    value = dict(body)
    value.pop("receipt_sha256", None)
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    value["receipt_sha256"] = hashlib.sha256(encoded).hexdigest()
    return value


def _phase_clique_disabled_receipt():
    return _query_receipt(
        {
            "schema": "act.operator_phase_clique_pipeline.v1",
            "enabled": False,
            "status": "no_op_disabled",
            "candidate_attempted": False,
            "candidate_only": True,
            "proof_authority": False,
            "identity_preserved": True,
            "materialized": False,
            "materialization_receipt_sha256": None,
            "verdict_path": "hz_objbound_decide_only",
            "candidate_budget_fraction": 0.40,
            "materializer_reserve_fraction": 0.60,
            "timings": {"total_seconds": 0.0},
        }
    )


def _phase_clique_fallback_receipt():
    return _query_receipt(
        {
            "schema": "act.operator_phase_clique_pipeline.v1",
            "enabled": True,
            "status": "baseline_fallback_no_k4_clique",
            "candidate_attempted": True,
            "candidate_only": True,
            "proof_authority": False,
            "identity_preserved": True,
            "materialized": False,
            "fallback_reason": "no_complete_k4_clique",
            "failed_stage": "exact_k4_candidate",
            "error_type": "_PipelineFallback",
            "full_rival_count": 99,
            "focused_encoded_row": 3,
            "hardness_vector_digest": "a" * 64,
            "focused_subset_digest": "b" * 64,
            "selection_digest": "c" * 64,
            "subset_binding_digest": None,
            "source_parent_semantic_digest": "1" * 64,
            "source_frame_digest": "2" * 64,
            "solver_handoff_status": "issued",
            "solver_handoff_one_use": True,
            "solver_handoff_owner_bound": True,
            "solver_handoff_pid_bound": True,
            "solver_handoff_private_core_readonly": True,
            "materialization_receipt_sha256": None,
            "verdict_path": "hz_objbound_decide_only",
            "candidate_budget_fraction": 0.40,
            "materializer_reserve_fraction": 0.60,
            "timings": {
                "raw_top1_seconds": 0.05,
                "hardness_seconds": 0.04,
                "focus_and_replay_seconds": 0.03,
                "literal_selection_seconds": 0.02,
                "total_seconds": 0.25,
            },
        }
    )


def _phase_clique_success_receipt(*, source_rows=8):
    fresh_rows = source_rows + 1
    clique_id = "c" * 64
    nested = _query_receipt(
        {
            "schema": (
                "act.operator_exact_relu_phase_clique_"
                "materialization.v2"
            ),
            "status": "fresh_verified_clique_cuts_materialized",
            "candidate_only": True,
            "proof_authority": False,
            "hardened_exact_result_verifier_passed": True,
            "one_use_snapshot_consumed": True,
            "verdict_path": "hz_objbound_decide_only",
            "verified_snapshot_digest": "d" * 64,
            "verified_result_digest": "e" * 64,
            "parent_semantic_digest": "1" * 64,
            "verified_cut_semantic_digest": "2" * 64,
            "fresh_semantic_digest": "2" * 64,
            "ordered_source_frame_sha256": "f" * 64,
            "source_frame_digest": "0" * 64,
            "fresh_frame_digest": "a" * 64,
            "selection_digest": "3" * 64,
            "focused_property_digest": "4" * 64,
            "subset_binding_digest": "5" * 64,
            "clique_ids": [clique_id],
            "cut_row_tags": [
                "operator_exact_relu_phase_clique_cut:v1:0:"
                f"{clique_id}"
            ],
            "cut_row_count": 1,
            "source_upper_rows": source_rows,
            "fresh_upper_rows": fresh_rows,
            "copied_parent_attributes": [
                "full_col_ids",
                "operator_input_center",
                "operator_input_radius",
                "_solver_continuous_column_layer_ids",
            ],
            "row_prefix_frames": "fresh_empty",
            "incompatible_receipts": "rejected_not_copied",
            "constructive_nonempty_reissued": True,
            "constructive_nonempty_scope": (
                "private_solver_handoff_only"
            ),
            "public_constructive_nonempty_token": "absent",
            "solver_caches_stats_safe_tokens": "not_copied",
            "solver_handoff_one_use": True,
            "solver_handoff_owner_bound": True,
            "solver_handoff_pid_bound": True,
            "solver_handoff_private_core_readonly": True,
            "constructive_nonempty_reason": (
                "operator_hz_redundant_exact_integer_phase_clique_cuts_v1"
            ),
            "constructive_rule": (
                "full_parent_exact_pair_conflicts_imply_redundant_"
                "integer_clique_rows"
            ),
            "caps": {
                "max_parent_variables": 2_000_000,
                "max_parent_rows": 2_000_000,
                "max_parent_nonzeros": 50_000_000,
                "max_parent_buffer_items": 120_000_000,
                "max_top_literals": 4,
                "max_total_pairs": 6,
                "max_cliques": 1,
                "max_clique_search_nodes": 100_000,
                "max_source_terms": 128,
                "max_multiplier_bits": 256,
                "max_exact_bits": 4096,
                "max_exact_nonzeros": 200_000,
            },
        }
    )
    return _query_receipt(
        {
            "schema": "act.operator_phase_clique_pipeline.v1",
            "enabled": True,
            "status": "fresh_verified_k4_clique_materialized",
            "candidate_attempted": True,
            "candidate_only": True,
            "proof_authority": False,
            "identity_preserved": False,
            "materialized": True,
            "full_rival_count": 99,
            "focus_count": 1,
            "focused_encoded_row": 3,
            "ranked_literal_count": 4,
            "pair_count": 6,
            "certified_edge_count": 6,
            "clique_count": 1,
            "cut_row_count": 1,
            "source_upper_rows": source_rows,
            "fresh_upper_rows": fresh_rows,
            "source_parent_semantic_digest": "1" * 64,
            "full_batch_sha256": "6" * 64,
            "full_live_assert_sha256": "7" * 64,
            "full_property_digest": "8" * 64,
            "interval_frame_sha256": "9" * 64,
            "hardness_vector_digest": "a" * 64,
            "focused_subset_digest": "b" * 64,
            "selection_digest": "3" * 64,
            "focused_property_digest": "4" * 64,
            "subset_binding_digest": "5" * 64,
            "fresh_semantic_digest": "2" * 64,
            "materialization_receipt_sha256": (
                nested["receipt_sha256"]
            ),
            "materialization_receipt": nested,
            "solver_handoff_status": "issued",
            "solver_handoff_one_use": True,
            "solver_handoff_owner_bound": True,
            "solver_handoff_pid_bound": True,
            "solver_handoff_private_core_readonly": True,
            "verdict_path": "hz_objbound_decide_only",
            "candidate_budget_fraction": 0.40,
            "materializer_reserve_fraction": 0.60,
            "initial_budget_seconds": 20.0,
            "candidate_budget_seconds": 8.0,
            "minimum_materializer_reserve_seconds": 12.0,
            "candidate_elapsed_seconds": 1.0,
            "timings": {
                "raw_top1_seconds": 0.10,
                "hardness_seconds": 0.10,
                "focus_and_replay_seconds": 0.10,
                "literal_selection_seconds": 0.10,
                "k4_candidate_seconds": 0.10,
                "materializer_and_recheck_seconds": 0.25,
                "terminal_seal_seconds": 0.10,
                "total_seconds": 2.0,
            },
        }
    )


def _phase_clique_handoff_receipt(pipeline_receipt):
    materialized = pipeline_receipt.get("materialized") is True
    semantic_digest = pipeline_receipt.get(
        "fresh_semantic_digest"
        if materialized
        else "source_parent_semantic_digest"
    )
    return _query_receipt(
        {
            "schema": (
                "verifier_operator_phase_clique_solver_handoff_v1"
            ),
            "status": "consumed_private",
            "proof_authority": False,
            "one_use_consumed": True,
            "owner_bound": True,
            "pid_bound": True,
            "private_core_readonly": True,
            "solver_hz_is_public_result_hz": False,
            "materialized": materialized,
            "pipeline_receipt_sha256": pipeline_receipt[
                "receipt_sha256"
            ],
            "semantic_digest": semantic_digest,
            "verdict_path": "hz_objbound_decide_only",
        }
    )


def _set_phase_clique_receipt(metadata, pipeline_receipt):
    metadata["operator_phase_clique_materialization"] = pipeline_receipt
    if pipeline_receipt.get("enabled") is True:
        metadata["operator_phase_clique_solver_handoff"] = (
            _phase_clique_handoff_receipt(pipeline_receipt)
        )
    else:
        metadata.pop("operator_phase_clique_solver_handoff", None)


def _set_phase_clique_nested_receipt(metadata, nested_receipt):
    top = dict(metadata["operator_phase_clique_materialization"])
    nested = _query_receipt(nested_receipt)
    top["materialization_receipt"] = nested
    top["materialization_receipt_sha256"] = nested["receipt_sha256"]
    _set_phase_clique_receipt(metadata, _query_receipt(top))


def _property_upper_sha(values):
    digest = hashlib.sha256()
    digest.update(
        json.dumps([len(values)], separators=(",", ":")).encode("ascii")
    )
    digest.update(b"\0<f8\0")
    for value in values:
        digest.update(struct.pack("<d", value))
    return digest.hexdigest()


def _query_transaction_fixture(
    *,
    targets,
    steps,
    time_limit,
    block_size,
    device,
):
    common_hash = "a" * 64
    candidate_policy = _query_dual_candidate_policy()
    if steps == 0:
        return (
            {
                "schema": "verifier_query_dual_feedback_transaction_v1",
                "status": "disabled",
                "proof_authority": False,
                "source": "configuration",
                "targets": list(targets),
                "steps": 0,
                "time_limit": 0.0,
                "block_size": block_size,
                "device": device,
                "reason": "steps_zero",
            },
            None,
        )
    target_receipts = []
    for index, target in enumerate(targets):
        target_receipts.append(
            _query_receipt(
                {
                    "schema": candidate_policy["target_stage_schema"],
                    "status": "verified",
                    "proof_authority": True,
                    "stage_index": index,
                    "target_relu_lid": target,
                    "commit": "atomic_whole_stage",
                    "parent_boxes_sha256": common_hash,
                    "result_boxes_sha256": common_hash,
                    "candidate_bounds_sha256": common_hash,
                    "candidate_receipt_sha256": common_hash,
                    "candidate_schema": candidate_policy[
                        "candidate_schema"
                    ],
                    "candidate_protocol": candidate_policy[
                        "candidate_protocol"
                    ],
                    "candidate_status": candidate_policy[
                        "candidate_success_status"
                    ],
                    "candidate_descriptor_coverage_sha256": common_hash,
                    "target_bounds_sha256": common_hash,
                    "block_receipt_sha256": [common_hash],
                    "strict_improvements": index + 1,
                }
            )
        )
    property_values = [0.25, -0.5]
    property_upper_sha = _property_upper_sha(property_values)
    property_receipt = _query_receipt(
        {
            "schema": candidate_policy["property_stage_schema"],
            "status": "verified",
            "proof_authority": True,
            "direction": "UPPER",
            "quantity": "C_y_minus_threshold",
            "objective": "-C",
            "replay_query_bias": "+threshold",
            "upper_reconstruction": "-LB(-C_y+threshold)",
            "coverage_complete": True,
            "parent_boxes_sha256": common_hash,
            "candidate_bounds_sha256": common_hash,
            "candidate_receipt_sha256": common_hash,
            "candidate_schema": candidate_policy["candidate_schema"],
            "candidate_protocol": candidate_policy[
                "candidate_protocol"
            ],
            "candidate_status": candidate_policy[
                "candidate_success_status"
            ],
            "candidate_descriptor_coverage_sha256": common_hash,
            "block_receipt_sha256": [common_hash],
            "property_rows": len(property_values),
            "property_spec_sha256": common_hash,
            "property_upper_sha256": property_upper_sha,
        }
    )
    pipeline = _query_receipt(
        {
            "schema": candidate_policy["pipeline_schema"],
            "status": "verified",
            "proof_authority": True,
            "transaction": "all_or_nothing",
            "ordinary_interval_facts_consumed": False,
            "target_relu_ids": list(targets),
            "steps": steps,
            "block_size": block_size,
            "replay_chunk_size": block_size,
            "candidate_device": device,
            "candidate_device_fallback": False,
            "completed_before_deadline": True,
            "root_boxes_sha256": common_hash,
            "final_boxes_sha256": common_hash,
            "property_spec_sha256": common_hash,
            "property_upper_sha256": property_upper_sha,
            "candidate_schema": candidate_policy["candidate_schema"],
            "candidate_protocol": candidate_policy[
                "candidate_protocol"
            ],
            "non_authoritative_audit_fields": candidate_policy[
                "pipeline_non_authoritative_audit_fields"
            ],
            "target_candidate_receipt_sha256": [
                common_hash for _ in targets
            ],
            "target_candidate_descriptor_coverage_sha256": [
                common_hash for _ in targets
            ],
            "property_candidate_receipt_sha256": common_hash,
            "property_candidate_descriptor_coverage_sha256": common_hash,
            "stage_receipt_sha256": [
                receipt["receipt_sha256"] for receipt in target_receipts
            ],
            "property_receipt_sha256": property_receipt["receipt_sha256"],
        }
    )
    operator = {
        "schema": "operator_hz_verified_query_dual_feedback_v1",
        "proof_authority": True,
        "process_local_validation": True,
        "receipt_rehydration_authority": False,
        "target_relu_ids": list(targets),
        "transaction_receipt_sha256": pipeline["receipt_sha256"],
        "root_boxes_sha256": common_hash,
        "final_boxes_sha256": common_hash,
        "property_spec_sha256": common_hash,
        "property_upper_sha256": property_upper_sha,
    }
    transaction = {
        "schema": "verifier_query_dual_feedback_transaction_v1",
        "status": "applied",
        "proof_authority": True,
        "source": "built_in_verify_once",
        "targets": list(targets),
        "steps": steps,
        "time_limit": time_limit,
        "block_size": block_size,
        "device": device,
        "elapsed_seconds": 0.5,
        "pipeline_schema": candidate_policy["pipeline_schema"],
        "target_stage_schema": candidate_policy["target_stage_schema"],
        "property_stage_schema": candidate_policy[
            "property_stage_schema"
        ],
        "candidate_schema": candidate_policy["candidate_schema"],
        "candidate_protocol": candidate_policy["candidate_protocol"],
        "candidate_non_authoritative_audit_fields": candidate_policy[
            "candidate_non_authoritative_audit_fields"
        ],
        "pipeline_non_authoritative_audit_fields": candidate_policy[
            "pipeline_non_authoritative_audit_fields"
        ],
        "replay_chunk_size": block_size,
        "pipeline_receipt": pipeline,
        "target_stage_receipts": target_receipts,
        "property_stage_receipt": property_receipt,
        "root_bounds_count": 5,
        "target_stage_count": len(targets),
        "target_block_count": len(targets),
        "property_block_count": 1,
        "strict_improvements_total": sum(
            range(1, len(targets) + 1)
        ),
        "property_rows": len(property_values),
        "property_upper_sha256": property_upper_sha,
        "property_upper_hex": [value.hex() for value in property_values],
        "operator_transaction_receipt_sha256": pipeline["receipt_sha256"],
    }
    return transaction, operator


def _rehash_query_export_chain(metadata):
    """Repair every checksum edge without repairing semantic mismatches."""

    transaction = metadata["query_dual_feedback_transaction"]
    target_receipts = [
        _query_receipt(receipt)
        for receipt in transaction["target_stage_receipts"]
    ]
    property_receipt = _query_receipt(
        transaction["property_stage_receipt"]
    )
    pipeline_body = dict(transaction["pipeline_receipt"])
    pipeline_body["stage_receipt_sha256"] = [
        receipt["receipt_sha256"] for receipt in target_receipts
    ]
    pipeline_body["property_receipt_sha256"] = property_receipt[
        "receipt_sha256"
    ]
    pipeline = _query_receipt(pipeline_body)
    transaction["target_stage_receipts"] = target_receipts
    transaction["property_stage_receipt"] = property_receipt
    transaction["pipeline_receipt"] = pipeline
    transaction["operator_transaction_receipt_sha256"] = pipeline[
        "receipt_sha256"
    ]
    operator = metadata["operator_hz"][
        "verified_query_dual_feedback"
    ]
    operator["transaction_receipt_sha256"] = pipeline["receipt_sha256"]


def _pass_record(gate: int, *, chain):
    from act.pipeline.verification.hybridz_largecls_gate import (
        _canonical_json,
        _sha256_bytes,
    )

    delta = 6 if gate == 6 else 8
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "run_end",
        "run_id": f"gate-{gate}-run",
        "status": "PASS",
        "gate": gate,
        "delta_count": delta,
        "cumulative_count": gate,
        "selected_families": list(FAMILIES),
        "manifest_sha256": "m" * 64,
        "experiment_sha256": "x" * 64,
        "source_sha256": "s" * 64,
        "artifact_sha256": "a" * 64,
        "environment_sha256": "e" * 64,
        "all_expected_completed": True,
        "all_results_conclusive": True,
        "expected_instance_count": delta,
        "completed_instance_count": delta,
        "global_failure_class": None,
        "global_failure_reason": None,
        "run_end_integrity": {"passed": True},
        "unpromoted_diagnostic": False,
        "partial_family_diagnostic": False,
        "property_micro_rlt_parent_only_diagnostic": False,
        "property_micro_rlt_packet_mode": "both",
        "diagnostic_only": False,
        "promotion_eligible": True,
        "promotion_chain": chain,
        "promotion_chain_sha256": _sha256_bytes(
            _canonical_json(chain)
        ),
    }


class _FakeCuda:
    def __init__(
        self,
        *,
        available=True,
        device_index=2,
        total=16_000,
        allocated=4_000,
        reserved=8_000,
        capture_error=None,
        reset_error=None,
    ):
        self.available = available
        self.device_index = device_index
        self.total = total
        self.allocated = allocated
        self.reserved = reserved
        self.capture_error = capture_error
        self.reset_error = reset_error
        self.reset_calls = []
        self.synchronize_calls = []

    def is_available(self):
        return self.available

    def current_device(self):
        return self.device_index

    def get_device_properties(self, device):
        if device != self.device_index:
            raise AssertionError("wrong fake CUDA device")
        return SimpleNamespace(total_memory=self.total)

    def reset_peak_memory_stats(self, device):
        self.reset_calls.append(device)
        if self.reset_error is not None:
            raise self.reset_error

    def synchronize(self, device):
        self.synchronize_calls.append(device)

    def max_memory_allocated(self, device):
        if self.capture_error is not None:
            raise self.capture_error
        return self.allocated

    def max_memory_reserved(self, device):
        return self.reserved


class CudaPeakMemoryAudit(unittest.TestCase):
    def test_cpu_unavailable_is_explicit_and_never_fabricated(self):
        cuda = _FakeCuda(available=False)
        observation = _start_cuda_peak_memory(
            SimpleNamespace(cuda=cuda)
        )
        self.assertEqual(observation["schema"], CUDA_PEAK_MEMORY_SCHEMA)
        self.assertEqual(observation["observation_status"], "unavailable")
        self.assertFalse(observation["available"])
        self.assertFalse(observation["reset_performed"])
        for key in (
            "logical_device_index",
            "device_total_bytes",
            "max_memory_allocated_bytes",
            "max_memory_reserved_bytes",
        ):
            self.assertIsNone(observation[key])
        self.assertEqual(cuda.reset_calls, [])
        self.assertTrue(
            _cuda_peak_memory_receipt_valid(
                observation,
                require_captured=False,
            )
        )
        self.assertFalse(
            _cuda_peak_memory_receipt_valid(
                observation,
                require_captured=True,
            )
        )

    def test_cuda_reset_and_capture_are_device_exact(self):
        cuda = _FakeCuda()
        torch_module = SimpleNamespace(cuda=cuda)
        tracking = _start_cuda_peak_memory(torch_module)
        self.assertEqual(tracking["observation_status"], "tracking")
        self.assertEqual(tracking["logical_device_index"], 2)
        self.assertEqual(tracking["device_total_bytes"], 16_000)
        self.assertEqual(cuda.reset_calls, [2])

        captured = _capture_cuda_peak_memory(torch_module, tracking)
        self.assertEqual(captured["observation_status"], "captured")
        self.assertEqual(captured["max_memory_allocated_bytes"], 4_000)
        self.assertEqual(captured["max_memory_reserved_bytes"], 8_000)
        self.assertEqual(cuda.synchronize_calls, [2])
        self.assertTrue(
            _cuda_peak_memory_receipt_valid(
                captured,
                require_captured=True,
            )
        )

        for key, value in (
            ("schema", "forged"),
            ("available", False),
            ("reset_performed", False),
            ("logical_device_index", -1),
            ("device_total_bytes", 0),
            ("max_memory_allocated_bytes", 9_000),
            ("max_memory_reserved_bytes", 17_000),
        ):
            with self.subTest(key=key):
                forged = dict(captured)
                forged[key] = value
                self.assertFalse(
                    _cuda_peak_memory_receipt_valid(
                        forged,
                        require_captured=True,
                    )
                )

    def test_reset_and_capture_errors_are_explicit(self):
        reset_cuda = _FakeCuda(reset_error=RuntimeError("reset failed"))
        reset = _start_cuda_peak_memory(
            SimpleNamespace(cuda=reset_cuda)
        )
        self.assertEqual(reset["observation_status"], "reset_error")
        self.assertTrue(reset["available"])
        self.assertFalse(reset["reset_performed"])
        self.assertIn("reset failed", reset["capture_error"])

        capture_cuda = _FakeCuda(
            capture_error=RuntimeError("capture failed")
        )
        torch_module = SimpleNamespace(cuda=capture_cuda)
        capture = _capture_cuda_peak_memory(
            torch_module,
            _start_cuda_peak_memory(torch_module),
        )
        self.assertEqual(capture["observation_status"], "capture_error")
        self.assertIsNone(capture["max_memory_allocated_bytes"])
        self.assertIsNone(capture["max_memory_reserved_bytes"])
        self.assertIn("capture failed", capture["capture_error"])

    def test_summary_aggregates_only_valid_captured_receipts(self):
        first_cuda = _FakeCuda(allocated=4_000, reserved=8_000)
        first_module = SimpleNamespace(cuda=first_cuda)
        first = _capture_cuda_peak_memory(
            first_module,
            _start_cuda_peak_memory(first_module),
        )
        second_cuda = _FakeCuda(allocated=6_000, reserved=9_000)
        second_module = SimpleNamespace(cuda=second_cuda)
        second = _capture_cuda_peak_memory(
            second_module,
            _start_cuda_peak_memory(second_module),
        )
        unavailable = _cuda_peak_memory_unavailable(
            "worker_hard_killed_before_receipt"
        )
        summary = _summarize_cuda_peak_memory(
            [
                {"cuda_peak_memory": first, "result": {}},
                {"result": {"cuda_peak_memory": second}},
                {"result": {"cuda_peak_memory": unavailable}},
                {"result": {}},
            ]
        )
        self.assertEqual(summary["captured_count"], 2)
        self.assertEqual(summary["unavailable_count"], 1)
        self.assertEqual(summary["missing_count"], 1)
        self.assertEqual(summary["error_count"], 0)
        self.assertFalse(summary["all_instance_results_captured"])
        self.assertEqual(summary["max_memory_allocated_bytes"], 6_000)
        self.assertEqual(summary["max_memory_reserved_bytes"], 9_000)
        self.assertEqual(summary["device_total_bytes"], 16_000)

    def test_observation_policy_is_bound_into_experiment_fingerprint(self):
        runtime = _runtime()
        common = {
            "provenance": {"manifest_sha256": "m" * 64},
            "source_sha256": "s" * 64,
            "artifact_sha256": "a" * 64,
            "environment_sha256": "e" * 64,
            "engine": ENGINE,
            "runtime": runtime,
        }
        baseline = _experiment_fingerprint(**common)
        policy = _cuda_peak_memory_policy()
        policy["capture_scope"] = "tampered"
        with mock.patch(
            "act.pipeline.verification.hybridz_largecls_gate."
            "_cuda_peak_memory_policy",
            return_value=policy,
        ):
            tampered = _experiment_fingerprint(**common)
        self.assertNotEqual(baseline, tampered)


class ClassificationAudit(unittest.TestCase):
    def test_false_conflict_keys_do_not_become_p0(self):
        result = _unknown(
            {
                "lp_base_feasibility_conflict": False,
                "nested": {
                    "p0_latched": False,
                    "soundness_conflict": False,
                    "comment": "the token p0 here is not an assertion",
                },
            }
        )
        classification = _classify_result(result)
        self.assertIsNone(classification.failure_class)
        self.assertFalse(classification.conclusive)

    def test_true_conflict_flag_is_p0(self):
        classification = _classify_result(
            _unknown({"lp_base_feasibility_conflict": True})
        )
        self.assertEqual(classification.failure_class, "FAIL_P0")
        self.assertIn("lp_base_feasibility_conflict", classification.reason)

    def test_operator_build_failure_exact_reason_is_p0(self):
        classification = _classify_result(
            _unknown(
                {
                    "hz_verdict": "",
                    "reason": "hybridz_operator_build_failed",
                }
            )
        )
        self.assertEqual(classification.failure_class, "FAIL_P0")
        self.assertIn("hybridz_operator_build_failed", classification.reason)

    def test_unknown_requires_an_explicit_hz_verdict(self):
        classification = _classify_result(_unknown({"hz_verdict": ""}))
        self.assertEqual(classification.failure_class, "FAIL_P0")
        self.assertEqual(
            classification.reason,
            "unknown_metadata_inconsistent",
        )

    def test_phantom_unsafe_replay_rejection_is_stoploss(self):
        classification = _classify_result(
            _unknown(
                {
                    "hz_verdict": "UNSAFE",
                    "hz_has_witness": True,
                    "hz_independent_replay": "independent_replay_rejected",
                    "reason": "hybridz_unsafe_candidate_not_replayed",
                }
            )
        )
        self.assertIsNone(classification.failure_class)
        self.assertFalse(classification.conclusive)

    def test_valid_strict_falsified_receipt_is_conclusive(self):
        onnx_sha = "a" * 64
        vnnlib_sha = "v" * 64
        classification = _classify_result(
            {
                "worker_state": "completed",
                "status": "falsified",
                "expected_engine": ENGINE,
                "has_counterexample": True,
                "input_integrity": {
                    "expected": {
                        "onnx": onnx_sha,
                        "vnnlib": vnnlib_sha,
                    }
                },
                "metadata": {
                    "solver": "hybridz",
                    "engine": ENGINE,
                    "hz_verdict": "UNSAFE",
                    "hz_has_witness": True,
                    "hz_independent_replay": "independent_replay_accepted",
                    "hz_replay_receipt": {
                        "authority": (
                            "onnxruntime_cpu_raw_vnnlib_zero_tolerance"
                        ),
                        "tolerance": 0.0,
                        "valid_counterexample": True,
                        "property_evaluated": True,
                        "property_holds": True,
                        "ort_executed": True,
                        "raw_spec_evaluated": True,
                        "zero_tolerance_holds": True,
                        "replay_completed": True,
                        "model_sha256": onnx_sha,
                        "vnnlib_sha256": vnnlib_sha,
                    },
                },
            }
        )
        self.assertIsNone(classification.failure_class)
        self.assertTrue(classification.conclusive)

    def test_falsified_without_receipt_is_p0(self):
        result = {
            "worker_state": "completed",
            "status": "falsified",
            "expected_engine": ENGINE,
            "has_counterexample": True,
            "metadata": {
                "solver": "hybridz",
                "engine": ENGINE,
                "hz_verdict": "UNSAFE",
                "hz_has_witness": True,
                "hz_independent_replay": "independent_replay_accepted",
            },
        }
        self.assertEqual(
            _classify_result(result).failure_class,
            "FAIL_P0",
        )

    def test_worker_failures_are_split(self):
        resource = _classify_result(
            {
                "worker_state": "error",
                "status": "error",
                "error": {
                    "type": "OutOfMemoryError",
                    "message": "CUDA out of memory",
                },
            }
        )
        ordinary = _classify_result(
            {
                "worker_state": "error",
                "status": "error",
                "error": {"type": "ImportError", "message": "missing module"},
            }
        )
        engine = _classify_result(
            {
                "worker_state": "error",
                "status": "error",
                "error": {
                    "type": "GateConfigError",
                    "message": "HybridZ engine is not connected",
                },
            }
        )
        self.assertEqual(resource.failure_class, "BLOCKED_RESOURCE")
        self.assertEqual(ordinary.failure_class, "FAIL_ERROR")
        self.assertEqual(engine.failure_class, "BLOCKED_ENGINE")


class WorkerFeatureReceiptAudit(unittest.TestCase):
    def test_json_safe_never_truncates_a_checksum_covered_receipt(
        self,
    ) -> None:
        receipt = _query_receipt(
            {
                "schema": "test.large_checksum_receipt.v1",
                "generated_upper_row_tags": [
                    f"row:{index}" for index in range(8208)
                ],
            }
        )
        sanitized = _json_safe(receipt)
        self.assertEqual(
            sanitized["generated_upper_row_tags"],
            receipt["generated_upper_row_tags"],
        )
        payload = dict(sanitized)
        expected = payload.pop("receipt_sha256")
        self.assertEqual(
            _sha256_bytes(_canonical_json(payload)), expected
        )
        ordinary = _json_safe(
            {"rows": list(range(8208))}
        )
        self.assertEqual(len(ordinary["rows"]), 513)
        self.assertEqual(
            ordinary["rows"][-1], "<7696 items omitted>"
        )

    @staticmethod
    def _payload(
        *,
        source_planes=False,
        mixture_bits=0,
        pairhull_budget=0,
        pairhull_time_limit=0.0,
        micro_rlt_cap=0,
        micro_rlt_packet_mode="both",
        micro_rlt_seconds=0.0,
        micro_rlt_parent_only=False,
        query_targets=None,
        query_steps=0,
        query_time_limit=0.0,
        query_block_size=1024,
        query_device="cuda",
        phase_clique_time_limit=0.0,
    ):
        return {
            "operator_phase_clique_time_limit": (
                phase_clique_time_limit
            ),
            "query_dual_feedback_targets": list(query_targets or []),
            "query_dual_feedback_steps": query_steps,
            "query_dual_feedback_time_limit": query_time_limit,
            "query_dual_feedback_block_size": query_block_size,
            "query_dual_feedback_device": query_device,
            "property_tail_add_source_planes": source_planes,
            "property_micro_rlt_product_cap": micro_rlt_cap,
            "property_micro_rlt_packet_mode": micro_rlt_packet_mode,
            "property_micro_rlt_parent_prefilter_seconds": (
                micro_rlt_seconds
            ),
            "property_micro_rlt_parent_only_diagnostic": (
                micro_rlt_parent_only
            ),
            "property_tail_mixture_grid_bits": mixture_bits,
            "property_tail_pairhull_budget": pairhull_budget,
            "property_tail_pairhull_time_limit": pairhull_time_limit,
        }

    @staticmethod
    def _metadata(
        *,
        source_planes=False,
        mixture_bits=0,
        pairhull_budget=0,
        pairhull_time_limit=0.0,
        micro_rlt_cap=0,
        micro_rlt_packet_mode="both",
        micro_rlt_seconds=0.0,
        micro_rlt_parent_only=False,
        query_targets=None,
        query_steps=0,
        query_time_limit=0.0,
        query_block_size=1024,
        query_device="cuda",
        phase_clique_time_limit=0.0,
    ):
        transaction, operator = _query_transaction_fixture(
            targets=list(query_targets or []),
            steps=query_steps,
            time_limit=query_time_limit,
            block_size=query_block_size,
            device=query_device,
        )
        phase_clique_receipt = (
            _phase_clique_fallback_receipt()
            if phase_clique_time_limit > 0.0
            else _phase_clique_disabled_receipt()
        )
        metadata = {
            "cfg_operator_phase_clique_time_limit": (
                phase_clique_time_limit
            ),
            "operator_phase_clique_materialization": phase_clique_receipt,
            "cfg_query_dual_feedback_targets": list(
                query_targets or []
            ),
            "cfg_query_dual_feedback_steps": query_steps,
            "cfg_query_dual_feedback_time_limit": query_time_limit,
            "cfg_query_dual_feedback_block_size": query_block_size,
            "cfg_query_dual_feedback_device": query_device,
            "cfg_property_tail_add_source_planes": source_planes,
            "cfg_property_micro_rlt_product_cap": micro_rlt_cap,
            "cfg_property_micro_rlt_packet_mode": micro_rlt_packet_mode,
            "cfg_property_micro_rlt_parent_prefilter_seconds": (
                micro_rlt_seconds
            ),
            "cfg_property_micro_rlt_parent_only_diagnostic": (
                micro_rlt_parent_only
            ),
            "cfg_property_tail_mixture_grid_bits": mixture_bits,
            "cfg_property_tail_pairhull_budget": pairhull_budget,
            "cfg_property_tail_pairhull_time_limit": pairhull_time_limit,
            "query_dual_feedback_transaction": transaction,
        }
        if phase_clique_time_limit > 0.0:
            metadata["operator_phase_clique_solver_handoff"] = (
                _phase_clique_handoff_receipt(phase_clique_receipt)
            )
        if operator is not None:
            metadata["operator_hz"] = {
                "verified_query_dual_feedback": operator
            }
        if micro_rlt_parent_only:
            operator_micro_rlt = _query_receipt(
                {
                    "schema": "operator_hz_property_micro_rlt_v1",
                    "enabled": True,
                    "requested_product_factor_cap": micro_rlt_cap,
                    "status": "no_op_cap_exceeded",
                    "live_result_validation_passed": False,
                }
            )
            metadata.setdefault("operator_hz", {})[
                "property_micro_rlt"
            ] = operator_micro_rlt
            parent_status = (
                "operator_receipt_ineligible_diagnostic_stop"
            )
            metadata.update(
                {
                    "hz_verdict": "UNKNOWN",
                    "hz_has_witness": False,
                    "reason": (
                        "property_micro_rlt_parent_only_diagnostic"
                    ),
                    "property_micro_rlt_parent_prefilter": {
                        "proof_authority": False,
                        "parent_call_count": 0,
                        "status": parent_status,
                    },
                    "property_phase_split": {
                        "proof_authority": False,
                        "diagnostic_only": True,
                        "actual_child_count": 0,
                        "phase_enumeration_skipped": True,
                        "children": [],
                    },
                }
            )
            metadata[
                "property_micro_rlt_parent_only_diagnostic"
            ] = _query_receipt(
                {
                    "schema": (
                        "verifier_property_micro_rlt_"
                        "parent_only_diagnostic_v1"
                    ),
                    "enabled": True,
                    "diagnostic_only": True,
                    "proof_authority": False,
                    "verdict_forced_unknown": True,
                    "operator_receipt_status": (
                        operator_micro_rlt["status"]
                    ),
                    "operator_receipt_sha256": (
                        operator_micro_rlt["receipt_sha256"]
                    ),
                    "operator_live_validation_passed": False,
                    "parent_prefilter_status": parent_status,
                    "parent_call_count": 0,
                    "parent_solver_verdict": None,
                    "parent_safe_contract_observed": False,
                    "shared_deadline_expired": False,
                    "phase_cover_attempted": False,
                    "phase_children_created": 0,
                    "baseline_solver_attempted": False,
                    "stop_reason": parent_status,
                }
            )
        return metadata

    @staticmethod
    def _mixture_receipt(*, grid_bits=8, status="no_strict_proxy_improvement"):
        return {
            "schema": "hz_safe_group_dyadic_mixture_v1",
            "enabled": True,
            "status": status,
            "candidate_only": True,
            "proof_authority": False,
            "grid_bits": grid_bits,
        }

    def _phase_clique_success_case(self):
        payload = self._payload(phase_clique_time_limit=20.0)
        metadata = self._metadata(phase_clique_time_limit=20.0)
        metadata.update(
            {
                "operator_source_n_ub": 8,
                "operator_n_ub": 9,
            }
        )
        _set_phase_clique_receipt(
            metadata,
            _phase_clique_success_receipt(source_rows=8),
        )
        return metadata, payload

    def test_operator_phase_clique_execution_receipt_is_strict(self):
        disabled_payload = self._payload()
        disabled = self._metadata()
        _validate_worker_feature_receipts(
            disabled, disabled_payload
        )

        missing = deepcopy(disabled)
        del missing["operator_phase_clique_materialization"]
        with self.assertRaisesRegex(
            GateConfigError, "phase-clique receipt"
        ):
            _validate_worker_feature_receipts(
                missing, disabled_payload
            )

        stale = deepcopy(disabled)
        stale_receipt = dict(
            stale["operator_phase_clique_materialization"]
        )
        stale_receipt["materialization_receipt"] = {"stale": True}
        stale["operator_phase_clique_materialization"] = (
            _query_receipt(stale_receipt)
        )
        with self.assertRaisesRegex(
            GateConfigError, "stale or noncanonical"
        ):
            _validate_worker_feature_receipts(
                stale, disabled_payload
            )

        enabled_payload = self._payload(
            phase_clique_time_limit=20.0
        )
        fallback = self._metadata(
            phase_clique_time_limit=20.0
        )
        _validate_worker_feature_receipts(
            fallback, enabled_payload
        )

        not_executed = deepcopy(fallback)
        not_executed["operator_phase_clique_materialization"] = (
            _phase_clique_disabled_receipt()
        )
        with self.assertRaisesRegex(
            GateConfigError, "did not execute"
        ):
            _validate_worker_feature_receipts(
                not_executed, enabled_payload
            )

    def test_operator_phase_clique_success_receipt_is_bound(self):
        payload = self._payload(
            phase_clique_time_limit=20.0
        )
        metadata = self._metadata(
            phase_clique_time_limit=20.0
        )
        metadata.update(
            {
                "operator_source_n_ub": 8,
                "operator_n_ub": 9,
            }
        )
        _set_phase_clique_receipt(
            metadata,
            _phase_clique_success_receipt(source_rows=8),
        )
        _validate_worker_feature_receipts(metadata, payload)

        wrong_rows = deepcopy(metadata)
        wrong_rows["operator_n_ub"] = 10
        with self.assertRaisesRegex(
            GateConfigError, "not bound to the final"
        ):
            _validate_worker_feature_receipts(
                wrong_rows, payload
            )

        wrong_count = deepcopy(metadata)
        top = dict(
            wrong_count["operator_phase_clique_materialization"]
        )
        top["certified_edge_count"] = 5
        _set_phase_clique_receipt(
            wrong_count,
            _query_receipt(top),
        )
        with self.assertRaisesRegex(
            GateConfigError, "malformed"
        ):
            _validate_worker_feature_receipts(
                wrong_count, payload
            )

        nested_mismatch = deepcopy(metadata)
        top = dict(
            nested_mismatch[
                "operator_phase_clique_materialization"
            ]
        )
        nested = dict(top["materialization_receipt"])
        nested["selection_digest"] = "f" * 64
        nested = _query_receipt(nested)
        top["materialization_receipt"] = nested
        top["materialization_receipt_sha256"] = nested[
            "receipt_sha256"
        ]
        _set_phase_clique_receipt(
            nested_mismatch,
            _query_receipt(top),
        )
        with self.assertRaisesRegex(
            GateConfigError, "does not match"
        ):
            _validate_worker_feature_receipts(
                nested_mismatch, payload
            )

        tiny_budget_payload = self._payload(
            phase_clique_time_limit=1.0e-6
        )
        tiny_budget_metadata = self._metadata(
            phase_clique_time_limit=1.0e-6
        )
        tiny_budget_metadata.update(
            {
                "operator_source_n_ub": 8,
                "operator_n_ub": 9,
            }
        )
        _set_phase_clique_receipt(
            tiny_budget_metadata,
            _phase_clique_success_receipt(source_rows=8),
        )
        with self.assertRaisesRegex(
            GateConfigError, "exceeds its configured"
        ):
            _validate_worker_feature_receipts(
                tiny_budget_metadata, tiny_budget_payload
            )

        over_budget_fallback = self._metadata(
            phase_clique_time_limit=0.1
        )
        with self.assertRaisesRegex(
            GateConfigError, "exceeds its configured"
        ):
            _validate_worker_feature_receipts(
                over_budget_fallback,
                self._payload(phase_clique_time_limit=0.1),
            )

    def test_operator_phase_clique_counts_are_builtin_ints(self):
        expected_top_counts = {
            "full_rival_count": 99,
            "focus_count": 1,
            "focused_encoded_row": 3,
            "ranked_literal_count": 4,
            "pair_count": 6,
            "certified_edge_count": 6,
            "clique_count": 1,
            "cut_row_count": 1,
            "source_upper_rows": 8,
            "fresh_upper_rows": 9,
        }
        for name, expected in expected_top_counts.items():
            for value in (True, float(expected)):
                with self.subTest(scope="top", name=name, value=value):
                    metadata, payload = self._phase_clique_success_case()
                    top = dict(
                        metadata[
                            "operator_phase_clique_materialization"
                        ]
                    )
                    top[name] = value
                    _set_phase_clique_receipt(
                        metadata,
                        _query_receipt(top),
                    )
                    with self.assertRaisesRegex(
                        GateConfigError,
                        "malformed|not bound",
                    ):
                        _validate_worker_feature_receipts(
                            metadata, payload
                        )

        expected_nested_counts = {
            "cut_row_count": 1,
            "source_upper_rows": 8,
            "fresh_upper_rows": 9,
        }
        for name, expected in expected_nested_counts.items():
            for value in (True, float(expected)):
                with self.subTest(scope="nested", name=name, value=value):
                    metadata, payload = self._phase_clique_success_case()
                    nested = deepcopy(
                        metadata[
                            "operator_phase_clique_materialization"
                        ]["materialization_receipt"]
                    )
                    nested[name] = value
                    _set_phase_clique_nested_receipt(metadata, nested)
                    with self.assertRaisesRegex(
                        GateConfigError,
                        "nested materialization receipt",
                    ):
                        _validate_worker_feature_receipts(
                            metadata, payload
                        )

    def test_operator_phase_clique_budgets_and_timings_are_strict(self):
        budget_fields = (
            "initial_budget_seconds",
            "candidate_budget_seconds",
            "minimum_materializer_reserve_seconds",
            "candidate_elapsed_seconds",
        )
        for name in budget_fields:
            with self.subTest(negative_zero=name):
                metadata, payload = self._phase_clique_success_case()
                top = dict(
                    metadata["operator_phase_clique_materialization"]
                )
                top[name] = -0.0
                _set_phase_clique_receipt(metadata, _query_receipt(top))
                with self.assertRaisesRegex(GateConfigError, "malformed"):
                    _validate_worker_feature_receipts(metadata, payload)

        metadata, payload = self._phase_clique_success_case()
        top = dict(metadata["operator_phase_clique_materialization"])
        for name in budget_fields:
            top[name] = 0.0
        top["timings"] = {
            name: 0.0
            for name in top["timings"]
        }
        _set_phase_clique_receipt(metadata, _query_receipt(top))
        with self.assertRaisesRegex(GateConfigError, "malformed"):
            _validate_worker_feature_receipts(metadata, payload)

        metadata, payload = self._phase_clique_success_case()
        top = dict(metadata["operator_phase_clique_materialization"])
        timings = dict(top["timings"])
        timings["raw_top1_seconds"] = -0.0
        top["timings"] = timings
        _set_phase_clique_receipt(metadata, _query_receipt(top))
        with self.assertRaisesRegex(
            GateConfigError,
            "common transaction contract",
        ):
            _validate_worker_feature_receipts(metadata, payload)

        metadata, payload = self._phase_clique_success_case()
        top = dict(metadata["operator_phase_clique_materialization"])
        top["candidate_elapsed_seconds"] = 2.0
        top["timings"] = {
            name: (2.0 if name == "total_seconds" else 0.4)
            for name in top["timings"]
        }
        _set_phase_clique_receipt(metadata, _query_receipt(top))
        with self.assertRaisesRegex(
            GateConfigError,
            "impossible segmented timings",
        ):
            _validate_worker_feature_receipts(metadata, payload)

    def test_operator_phase_clique_nested_v2_critical_bindings(self):
        def remove(name):
            return lambda receipt: receipt.pop(name)

        def assign(name, value):
            return lambda receipt: receipt.__setitem__(name, value)

        mutations = (
            ("snapshot_missing", remove("verified_snapshot_digest")),
            (
                "verified_result_sha",
                assign("verified_result_digest", "not-a-sha"),
            ),
            (
                "verified_cut_binding",
                assign("verified_cut_semantic_digest", "b" * 64),
            ),
            ("source_frame", assign("source_frame_digest", "bad")),
            ("fresh_frame_missing", remove("fresh_frame_digest")),
            ("clique_id", assign("clique_ids", ["clique:0"])),
            ("cut_tag", assign("cut_row_tags", ["wrong"])),
            (
                "handoff_one_use",
                assign("solver_handoff_one_use", False),
            ),
            (
                "constructive_scope",
                assign("constructive_nonempty_scope", "public"),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                metadata, payload = self._phase_clique_success_case()
                nested = deepcopy(
                    metadata["operator_phase_clique_materialization"][
                        "materialization_receipt"
                    ]
                )
                mutate(nested)
                _set_phase_clique_nested_receipt(metadata, nested)
                with self.assertRaisesRegex(
                    GateConfigError,
                    "nested materialization receipt",
                ):
                    _validate_worker_feature_receipts(metadata, payload)

        cap_mutations = (
            ("bool", "max_top_literals", True),
            ("float", "max_total_pairs", 6.0),
            ("value", "max_cliques", 2),
            ("extra", "unexpected_cap", 1),
        )
        for label, name, value in cap_mutations:
            with self.subTest(caps=label):
                metadata, payload = self._phase_clique_success_case()
                nested = deepcopy(
                    metadata["operator_phase_clique_materialization"][
                        "materialization_receipt"
                    ]
                )
                nested["caps"][name] = value
                _set_phase_clique_nested_receipt(metadata, nested)
                with self.assertRaisesRegex(
                    GateConfigError,
                    "nested materialization receipt",
                ):
                    _validate_worker_feature_receipts(metadata, payload)

    def test_operator_phase_clique_solver_handoff_is_strictly_bound(self):
        metadata, payload = self._phase_clique_success_case()
        missing = deepcopy(metadata)
        del missing["operator_phase_clique_solver_handoff"]
        with self.assertRaisesRegex(GateConfigError, "solver handoff"):
            _validate_worker_feature_receipts(missing, payload)

        for name, value in (
            ("semantic_digest", "f" * 64),
            ("pipeline_receipt_sha256", "e" * 64),
            ("materialized", False),
            ("one_use_consumed", False),
            ("solver_hz_is_public_result_hz", True),
        ):
            with self.subTest(name=name):
                mutated = deepcopy(metadata)
                handoff = dict(
                    mutated["operator_phase_clique_solver_handoff"]
                )
                handoff[name] = value
                mutated["operator_phase_clique_solver_handoff"] = (
                    _query_receipt(handoff)
                )
                with self.assertRaisesRegex(
                    GateConfigError,
                    "solver handoff",
                ):
                    _validate_worker_feature_receipts(mutated, payload)

        fallback = self._metadata(phase_clique_time_limit=20.0)
        fallback_handoff = dict(
            fallback["operator_phase_clique_solver_handoff"]
        )
        fallback_handoff["semantic_digest"] = "f" * 64
        fallback["operator_phase_clique_solver_handoff"] = (
            _query_receipt(fallback_handoff)
        )
        with self.assertRaisesRegex(GateConfigError, "solver handoff"):
            _validate_worker_feature_receipts(
                fallback,
                self._payload(phase_clique_time_limit=20.0),
            )

        disabled = self._metadata()
        disabled["operator_phase_clique_solver_handoff"] = (
            _phase_clique_handoff_receipt(
                _phase_clique_fallback_receipt()
            )
        )
        with self.assertRaisesRegex(GateConfigError, "stale.*handoff"):
            _validate_worker_feature_receipts(disabled, self._payload())

    def test_verifier_config_receipts_are_required_and_exact(self):
        payload = self._payload()
        missing_source = self._metadata()
        del missing_source["cfg_property_tail_add_source_planes"]
        with self.assertRaisesRegex(
            GateConfigError, "cfg_property_tail_add_source_planes"
        ):
            _validate_worker_feature_receipts(missing_source, payload)

        wrong_source = self._metadata(source_planes=True)
        with self.assertRaisesRegex(
            GateConfigError, "cfg_property_tail_add_source_planes"
        ):
            _validate_worker_feature_receipts(wrong_source, payload)

        missing_mixture = self._metadata()
        del missing_mixture["cfg_property_tail_mixture_grid_bits"]
        with self.assertRaisesRegex(
            GateConfigError, "cfg_property_tail_mixture_grid_bits"
        ):
            _validate_worker_feature_receipts(missing_mixture, payload)

        wrong_mixture_type = self._metadata()
        wrong_mixture_type["cfg_property_tail_mixture_grid_bits"] = False
        with self.assertRaisesRegex(
            GateConfigError, "cfg_property_tail_mixture_grid_bits"
        ):
            _validate_worker_feature_receipts(
                wrong_mixture_type, payload
            )

        wrong_mixture_value = self._metadata(mixture_bits=7)
        with self.assertRaisesRegex(
            GateConfigError, "cfg_property_tail_mixture_grid_bits"
        ):
            _validate_worker_feature_receipts(
                wrong_mixture_value, payload
            )

        missing_pairhull_budget = self._metadata()
        del missing_pairhull_budget["cfg_property_tail_pairhull_budget"]
        with self.assertRaisesRegex(
            GateConfigError, "cfg_property_tail_pairhull_budget"
        ):
            _validate_worker_feature_receipts(
                missing_pairhull_budget, payload
            )

        wrong_pairhull_budget = self._metadata(pairhull_budget=2)
        with self.assertRaisesRegex(
            GateConfigError, "cfg_property_tail_pairhull_budget"
        ):
            _validate_worker_feature_receipts(
                wrong_pairhull_budget, payload
            )

        boolean_pairhull_budget = self._metadata()
        boolean_pairhull_budget[
            "cfg_property_tail_pairhull_budget"
        ] = False
        with self.assertRaisesRegex(
            GateConfigError, "cfg_property_tail_pairhull_budget"
        ):
            _validate_worker_feature_receipts(
                boolean_pairhull_budget, payload
            )

        missing_pairhull_time = self._metadata()
        del missing_pairhull_time["cfg_property_tail_pairhull_time_limit"]
        with self.assertRaisesRegex(
            GateConfigError, "cfg_property_tail_pairhull_time_limit"
        ):
            _validate_worker_feature_receipts(
                missing_pairhull_time, payload
            )

        wrong_pairhull_time = self._metadata(pairhull_time_limit=1.0)
        with self.assertRaisesRegex(
            GateConfigError, "cfg_property_tail_pairhull_time_limit"
        ):
            _validate_worker_feature_receipts(
                wrong_pairhull_time, payload
            )

        for forged_time in (False, float("nan")):
            forged = self._metadata()
            forged["cfg_property_tail_pairhull_time_limit"] = forged_time
            with (
                self.subTest(forged_time=forged_time),
                self.assertRaisesRegex(
                    GateConfigError,
                    "cfg_property_tail_pairhull_time_limit",
                ),
            ):
                _validate_worker_feature_receipts(forged, payload)

    def test_micro_rlt_cfg_receipts_reject_payload_tampering(self):
        payload = self._payload(
            micro_rlt_cap=64,
            micro_rlt_packet_mode="first",
            micro_rlt_seconds=1.0,
            micro_rlt_parent_only=True,
        )
        metadata = self._metadata(
            micro_rlt_cap=64,
            micro_rlt_packet_mode="first",
            micro_rlt_seconds=1.0,
            micro_rlt_parent_only=True,
        )
        _validate_worker_feature_receipts(metadata, payload)

        mutations = (
            ("property_micro_rlt_product_cap", 63),
            ("property_micro_rlt_product_cap", True),
            ("property_micro_rlt_packet_mode", "second"),
            ("property_micro_rlt_packet_mode", 1),
            ("property_micro_rlt_packet_mode", "invalid"),
            ("property_micro_rlt_parent_prefilter_seconds", 0.5),
            ("property_micro_rlt_parent_prefilter_seconds", False),
            ("property_micro_rlt_parent_only_diagnostic", False),
            ("property_micro_rlt_parent_only_diagnostic", 1),
        )
        for key, forged_value in mutations:
            with self.subTest(payload_key=key, value=forged_value):
                forged_payload = dict(payload)
                forged_payload[key] = forged_value
                with self.assertRaisesRegex(
                    GateConfigError,
                    "cfg_property_micro_rlt",
                ):
                    _validate_worker_feature_receipts(
                        metadata,
                        forged_payload,
                    )

        for key, forged_value in (
            ("cfg_property_micro_rlt_product_cap", 63),
            ("cfg_property_micro_rlt_product_cap", True),
            ("cfg_property_micro_rlt_packet_mode", "second"),
            ("cfg_property_micro_rlt_packet_mode", 1),
            ("cfg_property_micro_rlt_packet_mode", "invalid"),
            (
                "cfg_property_micro_rlt_parent_prefilter_seconds",
                0.5,
            ),
            (
                "cfg_property_micro_rlt_parent_prefilter_seconds",
                float("nan"),
            ),
            (
                "cfg_property_micro_rlt_parent_only_diagnostic",
                False,
            ),
            (
                "cfg_property_micro_rlt_parent_only_diagnostic",
                1,
            ),
        ):
            with self.subTest(metadata_key=key, value=forged_value):
                forged_metadata = dict(metadata)
                forged_metadata[key] = forged_value
                with self.assertRaisesRegex(GateConfigError, key):
                    _validate_worker_feature_receipts(
                        forged_metadata,
                        payload,
                    )

        for mutation in (
            "diagnostic_child_count",
            "phase_child_count",
            "parent_proof_authority",
            "operator_binding",
        ):
            with self.subTest(parent_only_mutation=mutation):
                forged = deepcopy(metadata)
                if mutation == "diagnostic_child_count":
                    receipt = dict(
                        forged[
                            "property_micro_rlt_parent_only_diagnostic"
                        ]
                    )
                    receipt["phase_children_created"] = 1
                    forged[
                        "property_micro_rlt_parent_only_diagnostic"
                    ] = _query_receipt(receipt)
                elif mutation == "phase_child_count":
                    forged["property_phase_split"][
                        "actual_child_count"
                    ] = 1
                elif mutation == "parent_proof_authority":
                    forged[
                        "property_micro_rlt_parent_prefilter"
                    ]["proof_authority"] = True
                else:
                    operator_receipt = dict(
                        forged["operator_hz"]["property_micro_rlt"]
                    )
                    operator_receipt["status"] = "applied"
                    forged["operator_hz"]["property_micro_rlt"] = (
                        _query_receipt(operator_receipt)
                    )
                with self.assertRaisesRegex(
                    GateConfigError, "parent-only diagnostic"
                ):
                    _validate_worker_feature_receipts(
                        forged,
                        payload,
                    )

    def test_query_dual_config_receipts_reject_tampering(self):
        targets = [10, 14, 22, 40]
        payload = self._payload(
            query_targets=targets,
            query_steps=8,
            query_time_limit=12.0,
            query_block_size=1024,
            query_device="cuda",
        )
        metadata = self._metadata(
            query_targets=targets,
            query_steps=8,
            query_time_limit=12.0,
            query_block_size=1024,
            query_device="cuda",
        )
        _validate_worker_feature_receipts(metadata, payload)

        mutations = {
            "cfg_query_dual_feedback_targets": [10, 14, 22],
            "cfg_query_dual_feedback_steps": 7,
            "cfg_query_dual_feedback_time_limit": 11.0,
            "cfg_query_dual_feedback_block_size": 512,
            "cfg_query_dual_feedback_device": "cpu",
        }
        for key, value in mutations.items():
            with self.subTest(key=key):
                forged = dict(metadata)
                forged[key] = value
                with self.assertRaisesRegex(GateConfigError, key):
                    _validate_worker_feature_receipts(forged, payload)

        duplicate = dict(metadata)
        duplicate["cfg_query_dual_feedback_targets"] = [10, 10, 14, 22, 40]
        with self.assertRaisesRegex(
            GateConfigError, "cfg_query_dual_feedback_targets"
        ):
            _validate_worker_feature_receipts(duplicate, payload)

    def test_query_dual_authority_transaction_is_fail_closed(self):
        targets = [10, 14, 22, 40]
        payload = self._payload(
            query_targets=targets,
            query_steps=8,
            query_time_limit=12.0,
        )
        metadata = self._metadata(
            query_targets=targets,
            query_steps=8,
            query_time_limit=12.0,
        )
        _validate_worker_feature_receipts(metadata, payload)

        mutations = []
        missing = deepcopy(metadata)
        del missing["query_dual_feedback_transaction"]
        mutations.append(("missing", missing))

        fallback = deepcopy(metadata)
        fallback["query_dual_feedback_transaction"]["status"] = (
            "error_fallback_baseline"
        )
        fallback["query_dual_feedback_transaction"][
            "proof_authority"
        ] = False
        mutations.append(("fallback", fallback))

        stage_drop = deepcopy(metadata)
        stage_drop["query_dual_feedback_transaction"][
            "target_stage_receipts"
        ].pop()
        mutations.append(("coverage", stage_drop))

        pipeline_fallback = deepcopy(metadata)
        pipeline_fallback["query_dual_feedback_transaction"][
            "pipeline_receipt"
        ]["candidate_device_fallback"] = True
        mutations.append(("device_fallback", pipeline_fallback))

        property_tamper = deepcopy(metadata)
        property_tamper["query_dual_feedback_transaction"][
            "property_upper_hex"
        ][0] = (0.5).hex()
        mutations.append(("property_tamper", property_tamper))

        operator_tamper = deepcopy(metadata)
        operator_tamper["operator_hz"]["verified_query_dual_feedback"][
            "transaction_receipt_sha256"
        ] = "b" * 64
        mutations.append(("operator_tamper", operator_tamper))

        for label, forged in mutations:
            with self.subTest(label=label), self.assertRaises(
                GateConfigError
            ):
                _validate_worker_feature_receipts(forged, payload)

        disabled_payload = self._payload()
        disabled_metadata = self._metadata()
        disabled_metadata["query_dual_feedback_transaction"] = deepcopy(
            metadata["query_dual_feedback_transaction"]
        )
        with self.assertRaises(GateConfigError):
            _validate_worker_feature_receipts(
                disabled_metadata, disabled_payload
            )

    def test_query_dual_fallback_error_preserves_bounded_root_cause(self):
        targets = [10, 14, 22, 40]
        payload = self._payload(
            query_targets=targets,
            query_steps=8,
            query_time_limit=12.0,
        )
        metadata = self._metadata(
            query_targets=targets,
            query_steps=8,
            query_time_limit=12.0,
        )
        transaction = metadata["query_dual_feedback_transaction"]
        transaction.update(
            {
                "status": "error_fallback_baseline",
                "proof_authority": False,
                "elapsed_seconds": 1.25,
                "error_type": "QueryDualPipelineError",
                "error_code": "QUERY_COVERAGE_ERROR",
                "error": (
                    "pipeline root\n\x1b[31m"
                    + ("x" * 900)
                    + " final-cause"
                ),
                # Root cause must win even when the same fallback wrapper is
                # malformed in another field.
                "time_limit": 999.0,
            }
        )

        with self.assertRaises(GateConfigError) as caught:
            _validate_worker_feature_receipts(metadata, payload)

        message = str(caught.exception)
        self.assertIn("observed_status=error_fallback_baseline", message)
        self.assertIn("elapsed_seconds=1.25", message)
        self.assertIn("error_type=QueryDualPipelineError", message)
        self.assertIn("error_code=QUERY_COVERAGE_ERROR", message)
        self.assertIn("error=pipeline root\\n\\x1b[31m", message)
        self.assertIn("final-cause", message)
        self.assertIn("<truncated>", message)
        self.assertNotIn("\n", message)
        self.assertNotIn("\x1b", message)
        self.assertLess(len(message), 1_200)

    def test_query_dual_operator_fallback_uses_operator_error_aliases(self):
        targets = [10, 14, 22, 40]
        payload = self._payload(
            query_targets=targets,
            query_steps=8,
            query_time_limit=12.0,
        )
        metadata = self._metadata(
            query_targets=targets,
            query_steps=8,
            query_time_limit=12.0,
        )
        transaction = metadata["query_dual_feedback_transaction"]
        transaction.update(
            {
                "status": "operator_error_no_application",
                "proof_authority": False,
                "elapsed_seconds": 2.5,
                "operator_error_type": "OperatorHZBuildError",
                "operator_error": "operator transaction rejected",
            }
        )

        with self.assertRaises(GateConfigError) as caught:
            _validate_worker_feature_receipts(metadata, payload)

        message = str(caught.exception)
        self.assertIn("observed_status=operator_error_no_application", message)
        self.assertIn("elapsed_seconds=2.5", message)
        self.assertIn("error_type=OperatorHZBuildError", message)
        self.assertIn("error_code=<missing>", message)
        self.assertIn("error=operator transaction rejected", message)

    def test_query_dual_v2_rejects_fully_rehashed_legacy_and_mismatches(
        self,
    ):
        targets = [10, 14, 22, 40]
        payload = self._payload(
            query_targets=targets,
            query_steps=8,
            query_time_limit=12.0,
        )
        cases = (
            (
                "legacy_all_schemas",
                lambda tx: (
                    tx["pipeline_receipt"].__setitem__(
                        "schema", "act.verified_query_dual_feedback.v1"
                    ),
                    tx["target_stage_receipts"][0].__setitem__(
                        "schema", "act.verified_query_dual_stage.v1"
                    ),
                    tx["property_stage_receipt"].__setitem__(
                        "schema", "act.verified_query_dual_property.v1"
                    ),
                ),
            ),
            (
                "pipeline_candidate_schema",
                lambda tx: tx["pipeline_receipt"].__setitem__(
                    "candidate_schema", "act.query_dual_candidates.v1"
                ),
            ),
            (
                "pipeline_protocol",
                lambda tx: tx["pipeline_receipt"].__setitem__(
                    "candidate_protocol", "frozen_alpha_replay_v1"
                ),
            ),
            (
                "pipeline_replay_chunk_size",
                lambda tx: tx["pipeline_receipt"].__setitem__(
                    "replay_chunk_size", 2
                ),
            ),
            (
                "pipeline_audit_whitelist",
                lambda tx: tx["pipeline_receipt"].__setitem__(
                    "non_authoritative_audit_fields",
                    ["candidate_generator"],
                ),
            ),
            (
                "target_candidate_schema",
                lambda tx: tx["target_stage_receipts"][0].__setitem__(
                    "candidate_schema", "act.query_dual_candidates.v1"
                ),
            ),
            (
                "target_protocol",
                lambda tx: tx["target_stage_receipts"][0].__setitem__(
                    "candidate_protocol", "frozen_alpha_replay_v1"
                ),
            ),
            (
                "target_v1_status",
                lambda tx: tx["target_stage_receipts"][0].__setitem__(
                    "candidate_status", "generated"
                ),
            ),
            (
                "property_candidate_schema",
                lambda tx: tx["property_stage_receipt"].__setitem__(
                    "candidate_schema", "act.query_dual_candidates.v1"
                ),
            ),
            (
                "property_protocol",
                lambda tx: tx["property_stage_receipt"].__setitem__(
                    "candidate_protocol", "frozen_alpha_replay_v1"
                ),
            ),
            (
                "property_v1_status",
                lambda tx: tx["property_stage_receipt"].__setitem__(
                    "candidate_status", "generated"
                ),
            ),
            (
                "wrapper_pipeline_schema",
                lambda tx: tx.__setitem__(
                    "pipeline_schema",
                    "act.verified_query_dual_feedback.v1",
                ),
            ),
            (
                "wrapper_replay_chunk_size",
                lambda tx: tx.__setitem__("replay_chunk_size", 2),
            ),
            (
                "wrapper_candidate_audit_whitelist",
                lambda tx: tx.__setitem__(
                    "candidate_non_authoritative_audit_fields",
                    ["solver"],
                ),
            ),
            (
                "target_hash_two_way_mismatch",
                lambda tx: tx["pipeline_receipt"][
                    "target_candidate_receipt_sha256"
                ].__setitem__(0, "b" * 64),
            ),
            (
                "target_coverage_two_way_mismatch",
                lambda tx: tx["pipeline_receipt"][
                    "target_candidate_descriptor_coverage_sha256"
                ].__setitem__(0, "b" * 64),
            ),
            (
                "property_hash_two_way_mismatch",
                lambda tx: tx["pipeline_receipt"].__setitem__(
                    "property_candidate_receipt_sha256", "b" * 64
                ),
            ),
        )
        for label, mutate in cases:
            with self.subTest(label=label):
                metadata = self._metadata(
                    query_targets=targets,
                    query_steps=8,
                    query_time_limit=12.0,
                )
                transaction = metadata[
                    "query_dual_feedback_transaction"
                ]
                mutate(transaction)
                _rehash_query_export_chain(metadata)
                with self.assertRaises(GateConfigError):
                    _validate_worker_feature_receipts(metadata, payload)

    def test_query_dual_v2_no_query_target_status_is_exact(self):
        targets = [10, 14, 22, 40]
        payload = self._payload(
            query_targets=targets,
            query_steps=8,
            query_time_limit=12.0,
        )
        metadata = self._metadata(
            query_targets=targets,
            query_steps=8,
            query_time_limit=12.0,
        )
        transaction = metadata["query_dual_feedback_transaction"]
        first = transaction["target_stage_receipts"][0]
        first["status"] = "verified_no_improvement"
        first["candidate_status"] = "no_queries_fallback"
        first["block_receipt_sha256"] = []
        first["strict_improvements"] = 0
        transaction["target_block_count"] -= 1
        transaction["strict_improvements_total"] -= 1
        _rehash_query_export_chain(metadata)
        _validate_worker_feature_receipts(metadata, payload)

        legacy = deepcopy(metadata)
        legacy["query_dual_feedback_transaction"][
            "target_stage_receipts"
        ][0]["candidate_status"] = "no_improvement_fallback"
        _rehash_query_export_chain(legacy)
        with self.assertRaises(GateConfigError):
            _validate_worker_feature_receipts(legacy, payload)

    def test_source_plane_requires_operator_execution_receipt(self):
        payload = self._payload(source_planes=True)
        missing = self._metadata(source_planes=True)
        missing["operator_hz"] = {}
        with self.assertRaisesRegex(
            GateConfigError, "add_source_planes.enabled"
        ):
            _validate_worker_feature_receipts(missing, payload)

        disabled = self._metadata(source_planes=True)
        disabled["operator_hz"] = {
            "property_tail_upper": {
                "add_source_planes": {"enabled": False}
            }
        }
        with self.assertRaisesRegex(
            GateConfigError, "add_source_planes.enabled"
        ):
            _validate_worker_feature_receipts(disabled, payload)

        enabled = self._metadata(source_planes=True)
        enabled["operator_hz"] = {
            "property_tail_upper": {
                "add_source_planes": {"enabled": True}
            }
        }
        _validate_worker_feature_receipts(enabled, payload)

    def test_pairhull_requires_verified_operator_execution_receipt(self):
        from act.back_end.hybridz_tf.test_property_pairhull_verifier import (
            _applied_receipt,
        )

        payload = self._payload(
            pairhull_budget=2,
            pairhull_time_limit=1.0,
        )

        def metadata_with(receipt):
            metadata = self._metadata(
                pairhull_budget=2,
                pairhull_time_limit=1.0,
            )
            metadata["operator_hz"] = {
                "property_tail_upper": {
                    "baseline_plane_count": 3,
                    "alternative_plane_rival_ids": [0, 1, 2],
                    "alternative_plane_kinds": [
                        "negative_alpha_materialized",
                        "pairhull_joint_materialized",
                        "add_source_alpha0",
                    ],
                    "pairhull_candidates": receipt,
                }
            }
            return metadata

        applied_metadata = metadata_with(_applied_receipt())
        _validate_worker_feature_receipts(applied_metadata, payload)

        # Worker metadata is sanitized once before validation and once more
        # when wrapped in its persisted result record.  Deep exact PairHull
        # phase receipts must survive both passes and a JSON round trip without
        # checksum-covered fields being replaced by the recursion guard.
        sanitized = _json_safe(applied_metadata)
        _validate_worker_feature_receipts(sanitized, payload)
        persisted = json.loads(
            json.dumps(_json_safe({"metadata": sanitized}))
        )["metadata"]
        self.assertNotIn("<depth-limit>", json.dumps(persisted))
        _validate_worker_feature_receipts(persisted, payload)

        with self.assertRaisesRegex(
            GateConfigError, "pairhull_candidates"
        ):
            _validate_worker_feature_receipts(metadata_with(None), payload)

        forged = _applied_receipt()
        forged["selected_pair_indices"] = [[0, 2]]
        with self.assertRaisesRegex(
            GateConfigError, "pairhull_candidates"
        ):
            _validate_worker_feature_receipts(
                metadata_with(forged), payload
            )

    def test_mixture_requires_solver_execution_receipt(self):
        payload = self._payload(mixture_bits=8)
        missing = self._metadata(mixture_bits=8)
        missing["operator_hz"] = {}
        with self.assertRaisesRegex(
            GateConfigError, "safe_row_dyadic_mixture.enabled"
        ):
            _validate_worker_feature_receipts(missing, payload)

        disabled = self._metadata(mixture_bits=8)
        disabled["operator_hz"] = {}
        disabled["safe_row_dyadic_mixture"] = {"enabled": False}
        with self.assertRaisesRegex(
            GateConfigError, "safe_row_dyadic_mixture.enabled"
        ):
            _validate_worker_feature_receipts(disabled, payload)

        flattened = self._metadata(mixture_bits=8)
        flattened["operator_hz"] = {}
        flattened["safe_row_dyadic_mixture"] = self._mixture_receipt()
        _validate_worker_feature_receipts(flattened, payload)

        nested = self._metadata(mixture_bits=8)
        nested["operator_hz"] = {}
        nested["hz_objbound_stats"] = {
            "safe_row_dyadic_mixture": self._mixture_receipt()
        }
        _validate_worker_feature_receipts(nested, payload)

    def test_mixture_receipt_rejects_missing_or_forged_core_fields(self):
        payload = self._payload(mixture_bits=8)
        cases = {}

        missing_schema = self._mixture_receipt()
        missing_schema.pop("schema")
        cases["schema"] = missing_schema

        wrong_grid_bits = self._mixture_receipt(grid_bits=7)
        cases["grid_bits"] = wrong_grid_bits

        boolean_grid_bits = self._mixture_receipt(grid_bits=True)
        cases["grid_bits_type"] = boolean_grid_bits

        not_candidate_only = self._mixture_receipt()
        not_candidate_only["candidate_only"] = False
        cases["candidate_only"] = not_candidate_only

        forged_proof_authority = self._mixture_receipt()
        forged_proof_authority["proof_authority"] = True
        cases["proof_authority"] = forged_proof_authority

        pending = self._mixture_receipt(status="pending")
        cases["status"] = pending

        expected_error = {
            "schema": "schema",
            "grid_bits": "grid_bits",
            "grid_bits_type": "grid_bits",
            "candidate_only": "candidate_only",
            "proof_authority": "proof_authority",
            "status": "status",
        }
        for name, receipt in cases.items():
            metadata = self._metadata(mixture_bits=8)
            metadata["operator_hz"] = {}
            metadata["safe_row_dyadic_mixture"] = receipt
            with (
                self.subTest(case=name),
                self.assertRaisesRegex(
                    GateConfigError, expected_error[name]
                ),
            ):
                _validate_worker_feature_receipts(metadata, payload)

    def test_generated_mixture_requires_all_exact_audits(self):
        payload = self._payload(mixture_bits=8)
        required = (
            "stored_dyadic_weights_validated",
            "dyadic_convexity_validated",
            "exact_search_complete",
        )
        for missing_field in required:
            receipt = self._mixture_receipt(status="generated")
            for field in required:
                receipt[field] = field != missing_field
            metadata = self._metadata(mixture_bits=8)
            metadata["operator_hz"] = {}
            metadata["safe_row_dyadic_mixture"] = receipt
            with (
                self.subTest(missing_field=missing_field),
                self.assertRaisesRegex(
                    GateConfigError, missing_field
                ),
            ):
                _validate_worker_feature_receipts(metadata, payload)

        valid = self._mixture_receipt(status="generated")
        valid.update({field: True for field in required})
        metadata = self._metadata(mixture_bits=8)
        metadata["operator_hz"] = {}
        metadata["safe_row_dyadic_mixture"] = valid
        _validate_worker_feature_receipts(metadata, payload)

    def test_missing_operator_receipt_preserves_fail_closed_result(self):
        metadata = self._metadata(source_planes=True, mixture_bits=8)
        metadata["reason"] = "operator_failed_closed_before_metadata"
        _validate_worker_feature_receipts(
            metadata,
            self._payload(source_planes=True, mixture_bits=8),
        )

    def test_present_operator_receipt_must_be_a_mapping(self):
        metadata = self._metadata()
        metadata["operator_hz"] = None
        with self.assertRaisesRegex(GateConfigError, "must be a mapping"):
            _validate_worker_feature_receipts(
                metadata,
                self._payload(),
            )


class RuntimeAudit(unittest.TestCase):
    def test_current_private_phase_clique_hook_is_connected(self):
        connected, reason = _engine_connected(
            ENGINE,
            operator_phase_clique_time_limit=20.0,
        )
        self.assertTrue(connected, reason)
        self.assertEqual(
            reason,
            "operator_hz phase-clique dispatch found",
        )

    def test_runtime_integer_fields_require_builtin_int(self):
        for name in _RUNTIME_BUILTIN_INTEGER_FIELDS:
            with self.subTest(name=name, value=True):
                runtime = _runtime()
                runtime[name] = True
                with self.assertRaisesRegex(
                    GateConfigError,
                    rf"{name} must be an integer",
                ):
                    _validate_runtime(runtime)
        with self.assertRaisesRegex(
            GateConfigError,
            "default_gate must be an integer",
        ):
            _validate_runtime(_runtime(default_gate=6.0))

    def test_run_gate_revalidates_runtime_before_opening_receipt(self):
        provenance = {
            "query_dual_feedback_families": {
                "cifar100_medium": {
                    "targets": [],
                    "status": "gate1_candidate",
                }
            }
        }
        with mock.patch(
            "act.pipeline.verification.hybridz_largecls_gate."
            "AppendOnlyReceipt"
        ) as receipt_type:
            with self.assertRaisesRegex(
                GateConfigError,
                "row_workers must be an integer",
            ):
                run_gate(
                    gate=6,
                    sentinels=[],
                    stages={},
                    all_families=["cifar100_medium"],
                    selected_families=["cifar100_medium"],
                    runtime=_runtime(row_workers=True),
                    provenance=provenance,
                    source_sha256="s" * 64,
                    source_files=[],
                    artifact_sha256="a" * 64,
                    artifact_files=[],
                    environment_sha256="e" * 64,
                    environment_snapshot={},
                    experiment_sha256="x" * 64,
                    receipt_path=Path("unused.jsonl"),
                    summary_path=Path("unused.summary.json"),
                    promotion=None,
                    allow_unpromoted=False,
                )
        receipt_type.assert_not_called()

    def test_worker_module_is_importable_from_repo_root(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "act.pipeline.verification.hybridz_largecls_gate",
                "--help",
            ],
            cwd=Path(__file__).resolve().parents[3],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20.0,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_single_iid_selector_is_explicit(self):
        args = _build_parser().parse_args(
            ["--family", "cifar100_medium", "--iid", "2", "--dry-run"]
        )
        self.assertEqual(args.families, ["cifar100_medium"])
        self.assertEqual(args.iid, 2)

    def test_parent_only_diagnostic_requires_gate6_single_family_and_iid(self):
        _validate_property_micro_rlt_parent_only_selection(
            enabled=True,
            gate=6,
            selected_families=["cifar100_medium"],
            iid=2,
        )
        invalid = (
            (14, ["cifar100_medium"], 2, "--gate 6"),
            (
                6,
                ["cifar100_medium", "tinyimagenet_medium"],
                2,
                "exactly one --family",
            ),
            (6, ["cifar100_medium"], None, "explicit --iid"),
        )
        for gate, families, iid, reason in invalid:
            with (
                self.subTest(gate=gate, families=families, iid=iid),
                self.assertRaisesRegex(GateConfigError, reason),
            ):
                _validate_property_micro_rlt_parent_only_selection(
                    enabled=True,
                    gate=gate,
                    selected_families=families,
                    iid=iid,
                )
        _validate_property_micro_rlt_parent_only_selection(
            enabled=False,
            gate=40,
            selected_families=FAMILIES,
            iid=None,
        )
        for packet_mode in ("first", "second"):
            _validate_property_micro_rlt_parent_only_selection(
                enabled=False,
                gate=6,
                selected_families=["cifar100_medium"],
                iid=2,
                packet_mode=packet_mode,
            )
            for gate, families, iid, reason in invalid:
                with (
                    self.subTest(
                        packet_mode=packet_mode,
                        gate=gate,
                        families=families,
                        iid=iid,
                    ),
                    self.assertRaisesRegex(GateConfigError, reason),
                ):
                    _validate_property_micro_rlt_parent_only_selection(
                        enabled=False,
                        gate=gate,
                        selected_families=families,
                        iid=iid,
                        packet_mode=packet_mode,
                    )

    def test_operator_phase_clique_trial_is_single_iid_only(self):
        _validate_operator_phase_clique_selection(
            time_limit=20.0,
            gate=6,
            selected_families=["cifar100_medium"],
            iid=2,
        )
        for gate, families, iid, reason in (
            (14, ["cifar100_medium"], 2, "--gate 6"),
            (
                6,
                ["cifar100_medium", "tinyimagenet_medium"],
                2,
                "exactly one --family",
            ),
            (6, ["cifar100_medium"], None, "explicit --iid"),
        ):
            with (
                self.subTest(gate=gate, families=families, iid=iid),
                self.assertRaisesRegex(GateConfigError, reason),
            ):
                _validate_operator_phase_clique_selection(
                    time_limit=20.0,
                    gate=gate,
                    selected_families=families,
                    iid=iid,
                )
        _validate_operator_phase_clique_selection(
            time_limit=0.0,
            gate=40,
            selected_families=FAMILIES,
            iid=None,
        )

    def test_parent_only_diagnostic_cli_boolean_is_unambiguous(self):
        parser = _build_parser()
        self.assertTrue(
            parser.parse_args(
                ["--property-micro-rlt-parent-only-diagnostic"]
            ).property_micro_rlt_parent_only_diagnostic
        )
        self.assertFalse(
            parser.parse_args(
                ["--no-property-micro-rlt-parent-only-diagnostic"]
            ).property_micro_rlt_parent_only_diagnostic
        )

    def test_micro_rlt_packet_mode_cli_is_strict(self):
        parser = _build_parser()
        self.assertIsNone(
            parser.parse_args([]).property_micro_rlt_packet_mode
        )
        for mode in ("both", "first", "second"):
            with self.subTest(mode=mode):
                parsed = parser.parse_args(
                    ["--property-micro-rlt-packet-mode", mode]
                )
                self.assertEqual(
                    parsed.property_micro_rlt_packet_mode,
                    mode,
                )
        with self.assertRaises(SystemExit):
            parser.parse_args(
                ["--property-micro-rlt-packet-mode", "FIRST"]
            )

    def test_materialize_add_boolean_override_is_unambiguous(self):
        parser = _build_parser()
        self.assertFalse(
            parser.parse_args(
                ["--no-operator-materialize-add"]
            ).operator_materialize_add
        )
        self.assertTrue(
            parser.parse_args(
                ["--operator-materialize-add"]
            ).operator_materialize_add
        )

    def test_micro_rlt_default_cli_runtime_and_yaml_are_frozen_off(self):
        parser = _build_parser()
        parsed_default = parser.parse_args([])
        self.assertIsNone(parsed_default.property_micro_rlt_product_cap)
        self.assertIsNone(parsed_default.property_micro_rlt_packet_mode)
        self.assertIsNone(
            parsed_default.property_micro_rlt_parent_prefilter_seconds
        )
        self.assertIsNone(
            parsed_default.property_micro_rlt_parent_only_diagnostic
        )
        runtime = _runtime_from_args(
            {"runtime": _runtime()},
            parsed_default,
        )
        self.assertEqual(runtime["property_micro_rlt_product_cap"], 0)
        self.assertEqual(
            runtime["property_micro_rlt_packet_mode"],
            "both",
        )
        self.assertEqual(
            runtime["property_micro_rlt_parent_prefilter_seconds"],
            0.0,
        )
        self.assertIs(
            runtime["property_micro_rlt_parent_only_diagnostic"],
            False,
        )
        configured = HybridZConfig()
        self.assertEqual(configured.property_micro_rlt_product_cap, 0)
        self.assertEqual(
            configured.property_micro_rlt_packet_mode,
            "both",
        )
        self.assertEqual(
            configured.property_micro_rlt_parent_prefilter_seconds,
            0.0,
        )
        self.assertIs(
            configured.property_micro_rlt_parent_only_diagnostic,
            False,
        )
        raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(
            raw["runtime"]["property_micro_rlt_product_cap"],
            0,
        )
        self.assertEqual(
            raw["runtime"]["property_micro_rlt_packet_mode"],
            "both",
        )
        self.assertEqual(
            raw["runtime"][
                "property_micro_rlt_parent_prefilter_seconds"
            ],
            0.0,
        )
        self.assertIs(
            raw["runtime"][
                "property_micro_rlt_parent_only_diagnostic"
            ],
            False,
        )

    def test_gate_receipts_explicitly_exclude_loaded_ground_truth(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            instance = SimpleNamespace(
                benchmark="cifar100_2024",
                onnx_path=root / "model.onnx",
                vnnlib_path=root / "spec.vnnlib",
                csv_timeout=100.0,
            )
            sentinel = SimpleNamespace(
                family="cifar100_medium",
                iid=2,
                reference_label="S",
                query_dual_feedback_targets=(),
                query_dual_feedback_status="gate1_candidate",
                instance=instance,
            )
            receipt_path = root / "gate.jsonl"
            summary_path = root / "gate.summary.json"
            provenance = {
                "manifest_sha256": "m" * 64,
                "config_sha256": "c" * 64,
                "csv_sha256": {},
                "query_dual_feedback_families": {
                    "cifar100_medium": {
                        "targets": [],
                        "status": "gate1_candidate",
                    }
                },
            }
            with (
                mock.patch(
                    "act.pipeline.verification.hybridz_largecls_gate."
                    "_engine_connected",
                    return_value=(True, "test"),
                ),
                mock.patch(
                    "act.pipeline.verification.hybridz_largecls_gate."
                    "_run_child",
                    return_value=_unknown(),
                ),
                mock.patch(
                    "act.pipeline.verification.hybridz_largecls_gate."
                    "_run_end_integrity",
                    return_value={
                        "passed": True,
                        "checks": {},
                        "expected": {},
                        "observed": {},
                        "errors": [],
                    },
                ),
            ):
                return_code = run_gate(
                    gate=6,
                    sentinels=[sentinel],
                    stages={6: [sentinel]},
                    all_families=["cifar100_medium"],
                    selected_families=["cifar100_medium"],
                    runtime=_runtime(),
                    provenance=provenance,
                    source_sha256="s" * 64,
                    source_files=[],
                    artifact_sha256="a" * 64,
                    artifact_files=[],
                    environment_sha256="e" * 64,
                    environment_snapshot={},
                    experiment_sha256="x" * 64,
                    receipt_path=receipt_path,
                    summary_path=summary_path,
                    promotion=None,
                    allow_unpromoted=False,
                )
            self.assertEqual(return_code, 2)
            records = [
                json.loads(line)
                for line in receipt_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(
                [record["record_type"] for record in records],
                [
                    "run_start",
                    "instance_result",
                    "family_stop",
                    "run_end",
                ],
            )
            ground_truth_records = [
                record
                for record in records
                if record["record_type"]
                in {"run_start", "instance_result", "run_end"}
            ]
            for record in ground_truth_records:
                with self.subTest(record_type=record["record_type"]):
                    self.assertIs(record["ground_truth_loaded"], False)
                    self.assertEqual(
                        record["property_micro_rlt_packet_mode"],
                        "both",
                    )
                    self.assertIs(
                        record["reference_diagnostic_label_present"],
                        True,
                    )
            instance_record = ground_truth_records[1]
            self.assertEqual(
                instance_record["reference_diagnostic_label"],
                "S",
            )
            self.assertIs(
                instance_record[
                    "reference_label_used_for_verdict_or_pass"
                ],
                False,
            )

    def test_parent_only_receipts_are_diagnostic_and_never_promotable(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            instance = SimpleNamespace(
                benchmark="cifar100_2024",
                onnx_path=root / "model.onnx",
                vnnlib_path=root / "spec.vnnlib",
                csv_timeout=100.0,
            )
            sentinel = SimpleNamespace(
                family="cifar100_medium",
                iid=2,
                reference_label="U",
                query_dual_feedback_targets=(),
                query_dual_feedback_status="gate1_candidate",
                instance=instance,
            )
            receipt_path = root / "parent-only.jsonl"
            summary_path = root / "parent-only.summary.json"
            provenance = {
                "manifest_sha256": "m" * 64,
                "config_sha256": "c" * 64,
                "csv_sha256": {},
                "query_dual_feedback_families": {
                    "cifar100_medium": {
                        "targets": [],
                        "status": "gate1_candidate",
                    }
                },
            }
            runtime = _runtime(
                operator_exact_budget=2,
                property_residual_budget=2,
                property_residual_time_limit=1.0,
                property_tail_upper=True,
                property_tail_suffix_blocks=1,
                property_micro_rlt_product_cap=64,
                property_micro_rlt_packet_mode="first",
                property_micro_rlt_parent_prefilter_seconds=1.0,
                property_micro_rlt_parent_only_diagnostic=True,
            )
            with (
                mock.patch(
                    "act.pipeline.verification.hybridz_largecls_gate."
                    "_engine_connected",
                    return_value=(True, "test"),
                ),
                mock.patch(
                    "act.pipeline.verification.hybridz_largecls_gate."
                    "_run_child",
                    return_value=_unknown(),
                ),
                mock.patch(
                    "act.pipeline.verification.hybridz_largecls_gate."
                    "_run_end_integrity",
                    return_value={
                        "passed": True,
                        "checks": {},
                        "expected": {},
                        "observed": {},
                        "errors": [],
                    },
                ),
            ):
                return_code = run_gate(
                    gate=6,
                    sentinels=[sentinel],
                    stages={6: [sentinel]},
                    all_families=["cifar100_medium"],
                    selected_families=["cifar100_medium"],
                    runtime=runtime,
                    provenance=provenance,
                    source_sha256="s" * 64,
                    source_files=[],
                    artifact_sha256="a" * 64,
                    artifact_files=[],
                    environment_sha256="e" * 64,
                    environment_snapshot={},
                    experiment_sha256="x" * 64,
                    receipt_path=receipt_path,
                    summary_path=summary_path,
                    promotion=None,
                    allow_unpromoted=False,
                )
            self.assertEqual(return_code, 2)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(
                summary["run_end"]["status"],
                "DIAGNOSTIC_PARENT_ONLY",
            )
            for record in (
                summary["run_start"],
                summary["instance_results"][0],
                summary["run_end"],
            ):
                with self.subTest(record_type=record["record_type"]):
                    self.assertIs(record["diagnostic_only"], True)
                    self.assertIs(record["promotion_eligible"], False)
                    self.assertIs(
                        record[
                            "property_micro_rlt_parent_only_diagnostic"
                        ],
                        True,
                    )
                    self.assertEqual(
                        record["property_micro_rlt_packet_mode"],
                        "first",
                    )
            self.assertEqual(
                summary["run_end"]["property_micro_rlt_runtime"][
                    "packet_mode"
                ],
                "first",
            )

    def test_packet_isolation_run_is_diagnostic_and_never_promotable(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            sentinel = SimpleNamespace(
                family="cifar100_medium",
                iid=2,
                reference_label="U",
                query_dual_feedback_targets=(),
                query_dual_feedback_status="gate1_candidate",
                instance=SimpleNamespace(
                    benchmark="cifar100_2024",
                    onnx_path=root / "model.onnx",
                    vnnlib_path=root / "spec.vnnlib",
                    csv_timeout=100.0,
                ),
            )
            runtime = _runtime(
                operator_exact_budget=2,
                property_residual_budget=2,
                property_residual_time_limit=1.0,
                property_tail_upper=True,
                property_tail_suffix_blocks=1,
                property_micro_rlt_product_cap=64,
                property_micro_rlt_packet_mode="second",
                property_micro_rlt_parent_prefilter_seconds=1.0,
            )
            summary_path = root / "packet.summary.json"
            with (
                mock.patch(
                    "act.pipeline.verification.hybridz_largecls_gate."
                    "_engine_connected",
                    return_value=(True, "test"),
                ),
                mock.patch(
                    "act.pipeline.verification.hybridz_largecls_gate."
                    "_run_child",
                    return_value=_unknown(),
                ),
                mock.patch(
                    "act.pipeline.verification.hybridz_largecls_gate."
                    "_run_end_integrity",
                    return_value={
                        "passed": True,
                        "checks": {},
                        "expected": {},
                        "observed": {},
                        "errors": [],
                    },
                ),
            ):
                return_code = run_gate(
                    gate=6,
                    sentinels=[sentinel],
                    stages={6: [sentinel]},
                    all_families=["cifar100_medium"],
                    selected_families=["cifar100_medium"],
                    runtime=runtime,
                    provenance={
                        "manifest_sha256": "m" * 64,
                        "config_sha256": "c" * 64,
                        "csv_sha256": {},
                        "query_dual_feedback_families": {
                            "cifar100_medium": {
                                "targets": [],
                                "status": "gate1_candidate",
                            }
                        },
                    },
                    source_sha256="s" * 64,
                    source_files=[],
                    artifact_sha256="a" * 64,
                    artifact_files=[],
                    environment_sha256="e" * 64,
                    environment_snapshot={},
                    experiment_sha256="x" * 64,
                    receipt_path=root / "packet.jsonl",
                    summary_path=summary_path,
                    promotion=None,
                    allow_unpromoted=False,
                )
            self.assertEqual(return_code, 2)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(
                summary["run_end"]["status"],
                "DIAGNOSTIC_PACKET_MODE",
            )
            for record in (
                summary["run_start"],
                summary["instance_results"][0],
                summary["run_end"],
            ):
                with self.subTest(record_type=record["record_type"]):
                    self.assertEqual(
                        record["property_micro_rlt_packet_mode"],
                        "second",
                    )
                    self.assertIs(record["diagnostic_only"], True)
                    self.assertIs(record["promotion_eligible"], False)

    def test_phase_clique_run_is_diagnostic_and_never_promotable(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            sentinel = SimpleNamespace(
                family="cifar100_medium",
                iid=2,
                reference_label="U",
                query_dual_feedback_targets=(),
                query_dual_feedback_status="gate1_candidate",
                instance=SimpleNamespace(
                    benchmark="cifar100_2024",
                    onnx_path=root / "model.onnx",
                    vnnlib_path=root / "spec.vnnlib",
                    csv_timeout=100.0,
                ),
            )
            runtime = _runtime(
                operator_exact_budget=4,
                operator_phase_clique_time_limit=20.0,
                property_residual_budget=4,
                property_residual_time_limit=4.0,
            )
            summary_path = root / "phase-clique.summary.json"
            with (
                mock.patch(
                    "act.pipeline.verification.hybridz_largecls_gate."
                    "_engine_connected",
                    return_value=(True, "test"),
                ),
                mock.patch(
                    "act.pipeline.verification.hybridz_largecls_gate."
                    "_run_child",
                    return_value=_unknown(),
                ),
                mock.patch(
                    "act.pipeline.verification.hybridz_largecls_gate."
                    "_run_end_integrity",
                    return_value={
                        "passed": True,
                        "checks": {},
                        "expected": {},
                        "observed": {},
                        "errors": [],
                    },
                ),
            ):
                return_code = run_gate(
                    gate=6,
                    sentinels=[sentinel],
                    stages={6: [sentinel]},
                    all_families=["cifar100_medium"],
                    selected_families=["cifar100_medium"],
                    runtime=runtime,
                    provenance={
                        "manifest_sha256": "m" * 64,
                        "config_sha256": "c" * 64,
                        "csv_sha256": {},
                        "query_dual_feedback_families": {
                            "cifar100_medium": {
                                "targets": [],
                                "status": "gate1_candidate",
                            }
                        },
                    },
                    source_sha256="s" * 64,
                    source_files=[],
                    artifact_sha256="a" * 64,
                    artifact_files=[],
                    environment_sha256="e" * 64,
                    environment_snapshot={},
                    experiment_sha256="x" * 64,
                    receipt_path=root / "phase-clique.jsonl",
                    summary_path=summary_path,
                    promotion=None,
                    allow_unpromoted=False,
                )
            self.assertEqual(return_code, 2)
            summary = json.loads(
                summary_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                summary["run_end"]["status"],
                "DIAGNOSTIC_OPERATOR_PHASE_CLIQUE",
            )
            for record in (
                summary["run_start"],
                summary["instance_results"][0],
                summary["run_end"],
            ):
                with self.subTest(record_type=record["record_type"]):
                    self.assertIs(
                        record["operator_phase_clique_diagnostic"],
                        True,
                    )
                    self.assertIs(record["diagnostic_only"], True)
                    self.assertIs(record["promotion_eligible"], False)

    def test_parent_only_conclusive_worker_verdict_is_contract_conflict(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            instance = SimpleNamespace(
                benchmark="cifar100_2024",
                onnx_path=root / "model.onnx",
                vnnlib_path=root / "spec.vnnlib",
                csv_timeout=100.0,
            )
            sentinel = SimpleNamespace(
                family="cifar100_medium",
                iid=2,
                reference_label="U",
                query_dual_feedback_targets=(),
                query_dual_feedback_status="gate1_candidate",
                instance=instance,
            )
            certified = {
                "worker_state": "completed",
                "status": "certified",
                "expected_engine": ENGINE,
                "metadata": {
                    "solver": "hybridz",
                    "engine": ENGINE,
                    "hz_verdict": "SAFE",
                    "hz_has_witness": False,
                },
                "has_counterexample": False,
            }
            runtime = _runtime(
                operator_exact_budget=2,
                property_residual_budget=2,
                property_residual_time_limit=1.0,
                property_tail_upper=True,
                property_tail_suffix_blocks=1,
                property_micro_rlt_product_cap=64,
                property_micro_rlt_parent_prefilter_seconds=1.0,
                property_micro_rlt_parent_only_diagnostic=True,
            )
            with (
                mock.patch(
                    "act.pipeline.verification.hybridz_largecls_gate."
                    "_engine_connected",
                    return_value=(True, "test"),
                ),
                mock.patch(
                    "act.pipeline.verification.hybridz_largecls_gate."
                    "_run_child",
                    return_value=certified,
                ),
                mock.patch(
                    "act.pipeline.verification.hybridz_largecls_gate."
                    "_run_end_integrity",
                    return_value={
                        "passed": True,
                        "checks": {},
                        "expected": {},
                        "observed": {},
                        "errors": [],
                    },
                ),
            ):
                return_code = run_gate(
                    gate=6,
                    sentinels=[sentinel],
                    stages={6: [sentinel]},
                    all_families=["cifar100_medium"],
                    selected_families=["cifar100_medium"],
                    runtime=runtime,
                    provenance={
                        "manifest_sha256": "m" * 64,
                        "config_sha256": "c" * 64,
                        "csv_sha256": {},
                        "query_dual_feedback_families": {
                            "cifar100_medium": {
                                "targets": [],
                                "status": "gate1_candidate",
                            }
                        },
                    },
                    source_sha256="s" * 64,
                    source_files=[],
                    artifact_sha256="a" * 64,
                    artifact_files=[],
                    environment_sha256="e" * 64,
                    environment_snapshot={},
                    experiment_sha256="x" * 64,
                    receipt_path=root / "conflict.jsonl",
                    summary_path=root / "conflict.summary.json",
                    promotion=None,
                    allow_unpromoted=False,
                )
            self.assertEqual(return_code, 2)
            summary = json.loads(
                (root / "conflict.summary.json").read_text(
                    encoding="utf-8"
                )
            )
            run_end = summary["run_end"]
            self.assertEqual(run_end["status"], "FAIL_ERROR")
            self.assertEqual(
                run_end["global_failure_reason"],
                (
                    "parent_only_diagnostic_conclusive_verdict_"
                    "contract_conflict"
                ),
            )
            self.assertIs(run_end["promotion_eligible"], False)

    def test_micro_rlt_valid_depth_two_cli_and_fingerprint_binding(self):
        args = _build_parser().parse_args(
            [
                "--property-micro-rlt-product-cap",
                "64",
                "--property-micro-rlt-packet-mode",
                "first",
                "--property-micro-rlt-parent-prefilter-seconds",
                "1.25",
                "--property-micro-rlt-parent-only-diagnostic",
            ]
        )
        phase_runtime = _runtime(
            operator_exact_budget=2,
            property_residual_budget=2,
            property_residual_time_limit=1.0,
            property_tail_upper=True,
            property_tail_suffix_blocks=1,
        )
        enabled = _runtime_from_args(
            {"runtime": phase_runtime},
            args,
        )
        self.assertEqual(enabled["property_micro_rlt_product_cap"], 64)
        self.assertEqual(
            enabled["property_micro_rlt_packet_mode"],
            "first",
        )
        self.assertEqual(
            enabled["property_micro_rlt_parent_prefilter_seconds"],
            1.25,
        )
        self.assertIs(
            enabled["property_micro_rlt_parent_only_diagnostic"],
            True,
        )
        _validate_runtime(enabled)
        _validate_property_micro_rlt_settings(
            enabled,
            context="worker ",
        )
        configured = HybridZConfig(
            engine=ENGINE,
            operator_exact_budget=2,
            property_residual_budget=2,
            property_residual_time_limit=1.0,
            property_tail_upper=True,
            property_tail_suffix_blocks=1,
            property_micro_rlt_product_cap=64,
            property_micro_rlt_packet_mode="first",
            property_micro_rlt_parent_prefilter_seconds=1.25,
            property_micro_rlt_parent_only_diagnostic=True,
        )
        self.assertEqual(configured.property_micro_rlt_product_cap, 64)
        self.assertEqual(
            configured.property_micro_rlt_packet_mode,
            "first",
        )
        self.assertIs(
            configured.property_micro_rlt_parent_only_diagnostic,
            True,
        )

        common = {
            "provenance": {"manifest_sha256": "m" * 64},
            "source_sha256": "s" * 64,
            "artifact_sha256": "a" * 64,
            "environment_sha256": "e" * 64,
            "engine": ENGINE,
        }
        self.assertNotEqual(
            _experiment_fingerprint(runtime=phase_runtime, **common),
            _experiment_fingerprint(runtime=enabled, **common),
        )
        parent_cover_runtime = dict(
            enabled,
            property_micro_rlt_parent_only_diagnostic=False,
        )
        self.assertNotEqual(
            _experiment_fingerprint(
                runtime=parent_cover_runtime,
                **common,
            ),
            _experiment_fingerprint(runtime=enabled, **common),
        )
        second_packet_runtime = dict(
            enabled,
            property_micro_rlt_packet_mode="second",
        )
        self.assertNotEqual(
            _experiment_fingerprint(
                runtime=second_packet_runtime,
                **common,
            ),
            _experiment_fingerprint(runtime=enabled, **common),
        )

    def test_micro_rlt_runtime_contract_fails_closed(self):
        for key in (
            "property_micro_rlt_product_cap",
            "property_micro_rlt_packet_mode",
            "property_micro_rlt_parent_prefilter_seconds",
            "property_micro_rlt_parent_only_diagnostic",
        ):
            with self.subTest(missing=key):
                missing = _runtime()
                del missing[key]
                with self.assertRaisesRegex(
                    GateConfigError,
                    "runtime is missing fields",
                ):
                    _validate_runtime(missing)

        for value in (True, 1.0, "1", None):
            with self.subTest(product_cap_type=value):
                with self.assertRaisesRegex(
                    GateConfigError,
                    "product_cap must be an integer",
                ):
                    _validate_runtime(
                        _runtime(property_micro_rlt_product_cap=value)
                    )
        for value in (-1, 4097):
            with self.subTest(product_cap_range=value):
                with self.assertRaisesRegex(
                    GateConfigError,
                    r"product_cap must lie in \[0, 4096\]",
                ):
                    _validate_runtime(
                        _runtime(property_micro_rlt_product_cap=value)
                    )
        for value in (None, 1, "FIRST", "invalid"):
            with self.subTest(packet_mode=value):
                with self.assertRaisesRegex(
                    GateConfigError,
                    r"packet_mode must be one of both\|first\|second",
                ):
                    _validate_runtime(
                        _runtime(property_micro_rlt_packet_mode=value)
                    )
        for value in ("first", "second"):
            with self.subTest(disabled_packet_mode=value):
                with self.assertRaisesRegex(
                    GateConfigError,
                    "packet_mode first/second requires property micro-RLT",
                ):
                    _validate_runtime(
                        _runtime(property_micro_rlt_packet_mode=value)
                    )
        for value in (True, "1", None):
            with self.subTest(prefilter_type=value):
                with self.assertRaisesRegex(
                    GateConfigError,
                    "parent_prefilter_seconds must be numeric",
                ):
                    _validate_runtime(
                        _runtime(
                            property_micro_rlt_parent_prefilter_seconds=(
                                value
                            )
                        )
                    )
        for value in (
            float("nan"),
            float("inf"),
            -0.1,
            10.1,
        ):
            with self.subTest(prefilter_range=value):
                with self.assertRaisesRegex(
                    GateConfigError,
                    r"parent_prefilter_seconds.*\[0, 10\]",
                ):
                    _validate_runtime(
                        _runtime(
                            property_micro_rlt_parent_prefilter_seconds=(
                                value
                            )
                        )
                    )
        with self.assertRaisesRegex(GateConfigError, "enabled together"):
            _validate_runtime(
                _runtime(property_micro_rlt_product_cap=64)
            )
        with self.assertRaisesRegex(GateConfigError, "enabled together"):
            _validate_runtime(
                _runtime(
                    property_micro_rlt_parent_prefilter_seconds=1.0
                )
            )
        for value in (0, 1, None, "true"):
            with self.subTest(parent_only_type=value):
                with self.assertRaisesRegex(
                    GateConfigError,
                    "parent_only_diagnostic must be a boolean",
                ):
                    _validate_runtime(
                        _runtime(
                            property_micro_rlt_parent_only_diagnostic=value
                        )
                    )
        with self.assertRaisesRegex(
            GateConfigError,
            "parent_only_diagnostic requires property micro-RLT",
        ):
            _validate_runtime(
                _runtime(
                    property_micro_rlt_parent_only_diagnostic=True
                )
            )

        enabled = {
            "property_micro_rlt_product_cap": 64,
            "property_micro_rlt_parent_prefilter_seconds": 1.0,
        }
        common_phase = {
            "operator_exact_budget": 2,
            "property_residual_budget": 2,
            "property_residual_time_limit": 1.0,
            "property_tail_upper": True,
            "property_tail_suffix_blocks": 1,
            **enabled,
        }
        _validate_runtime(
            _runtime(
                **{
                    **common_phase,
                    "property_micro_rlt_packet_mode": "first",
                }
            )
        )
        with self.assertRaisesRegex(
            GateConfigError,
            "engine=operator_hz_objbound",
        ):
            _validate_runtime(
                _runtime(
                    engine="dense_hz_objbound",
                    **common_phase,
                )
            )
        with self.assertRaisesRegex(
            GateConfigError,
            "property_tail_upper=true",
        ):
            _validate_runtime(
                _runtime(**{**common_phase, "property_tail_upper": False})
            )
        with self.assertRaisesRegex(GateConfigError, "depth-2"):
            _validate_runtime(
                _runtime(
                    **{
                        **common_phase,
                        "operator_exact_budget": 1,
                        "property_residual_budget": 1,
                    }
                )
            )

    def test_backend_query_dual_contract_and_yaml_roundtrip(self):
        default = HybridZConfig()
        self.assertEqual(default.query_dual_feedback_targets, ())
        self.assertEqual(default.query_dual_feedback_steps, 0)
        self.assertEqual(default.query_dual_feedback_time_limit, 0.0)
        self.assertEqual(default.query_dual_feedback_block_size, 1024)
        self.assertEqual(default.query_dual_feedback_device, "cuda")

        preconfigured = HybridZConfig(
            query_dual_feedback_targets="10,14,10,22,40"
        )
        self.assertEqual(
            preconfigured.query_dual_feedback_targets,
            (10, 14, 22, 40),
        )
        enabled = HybridZConfig(
            engine=ENGINE,
            property_tail_upper=True,
            operator_exact_budget=0,
            query_dual_feedback_targets=[10, 14, 22, 40],
            query_dual_feedback_steps=8,
            query_dual_feedback_time_limit=12.0,
            query_dual_feedback_block_size=4096,
            query_dual_feedback_device="CPU",
        )
        self.assertEqual(enabled.query_dual_feedback_device, "cpu")

        invalid_kwargs = (
            {"query_dual_feedback_targets": [True]},
            {"query_dual_feedback_targets": [-1]},
            {"query_dual_feedback_targets": "10,,14"},
            {"query_dual_feedback_steps": True},
            {"query_dual_feedback_steps": -1},
            {"query_dual_feedback_steps": 65},
            {"query_dual_feedback_time_limit": float("nan")},
            {"query_dual_feedback_time_limit": 20.0001},
            {"query_dual_feedback_block_size": 0},
            {"query_dual_feedback_block_size": 4097},
            {"query_dual_feedback_device": "auto"},
            {"query_dual_feedback_time_limit": 1.0},
            {"query_dual_feedback_steps": 1},
            {
                "query_dual_feedback_targets": [10],
                "query_dual_feedback_steps": 1,
                "query_dual_feedback_time_limit": 1.0,
            },
            {
                "engine": ENGINE,
                "query_dual_feedback_targets": [10],
                "query_dual_feedback_steps": 1,
                "query_dual_feedback_time_limit": 1.0,
            },
            {
                "engine": ENGINE,
                "property_tail_upper": True,
                "operator_exact_budget": 1,
                "query_dual_feedback_targets": [10],
                "query_dual_feedback_steps": 1,
                "query_dual_feedback_time_limit": 1.0,
            },
        )
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HybridZConfig(**kwargs)

        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "backend.yaml"
            source = BackendConfig(
                hybridz=HybridZConfig(
                    query_dual_feedback_targets=[10, 14, 10, 22, 40]
                )
            )
            source.to_yaml(path)
            serialized = path.read_text(encoding="utf-8")
            self.assertNotIn("!!python/tuple", serialized)
            restored = BackendConfig.from_yaml(path)
            self.assertEqual(
                restored.hybridz.query_dual_feedback_targets,
                (10, 14, 22, 40),
            )

    def test_query_dual_gate_family_effective_settings_are_isolated(self):
        runtime = _runtime(
            property_tail_upper=True,
            query_dual_feedback_steps=8,
            query_dual_feedback_time_limit=12.0,
        )
        families = {
            "cifar100_medium": {
                "query_dual_feedback_targets": [10, 14, 22, 40],
                "query_dual_feedback_status": "gate1_candidate",
            },
            "cifar100_large": {
                "query_dual_feedback_targets": [],
                "query_dual_feedback_status": "not_promoted",
            },
            "tinyimagenet_medium": {
                "query_dual_feedback_targets": [],
                "query_dual_feedback_status": "not_promoted",
            },
        }
        snapshot = _query_dual_family_snapshot(families)
        effective = _query_dual_effective_by_family(runtime, snapshot)
        self.assertEqual(
            effective["cifar100_medium"]["effective_steps"], 8
        )
        self.assertEqual(
            effective["cifar100_medium"]["effective_time_limit"], 12.0
        )
        for family in ("cifar100_large", "tinyimagenet_medium"):
            self.assertEqual(effective[family]["effective_steps"], 0)
            self.assertEqual(effective[family]["effective_time_limit"], 0.0)
            self.assertEqual(effective[family]["requested_steps"], 8)

        cifar_payload = _query_dual_worker_payload(
            runtime,
            family="cifar100_medium",
            targets=[10, 14, 22, 40],
            status="gate1_candidate",
        )
        large_payload = _query_dual_worker_payload(
            runtime,
            family="cifar100_large",
            targets=[],
            status="not_promoted",
        )
        self.assertEqual(cifar_payload["query_dual_feedback_steps"], 8)
        self.assertEqual(large_payload["query_dual_feedback_steps"], 0)
        self.assertEqual(
            large_payload["requested_query_dual_feedback_steps"], 8
        )
        self.assertNotIn(
            10, large_payload["query_dual_feedback_targets"]
        )

    def test_query_dual_gate_override_and_fingerprint_binding(self):
        args = _build_parser().parse_args(
            [
                "--query-dual-feedback-steps",
                "8",
                "--query-dual-feedback-time-limit",
                "12",
                "--query-dual-feedback-block-size",
                "2048",
                "--query-dual-feedback-device",
                "cpu",
            ]
        )
        runtime = _runtime_from_args(
            {"runtime": _runtime(property_tail_upper=True)},
            args,
            query_dual_feedback_targets=[10, 14, 22, 40],
        )
        _validate_runtime(
            runtime,
            query_dual_feedback_targets=[10, 14, 22, 40],
        )
        self.assertEqual(runtime["query_dual_feedback_steps"], 8)
        self.assertEqual(runtime["query_dual_feedback_time_limit"], 12.0)
        self.assertEqual(runtime["query_dual_feedback_block_size"], 2048)
        self.assertEqual(runtime["query_dual_feedback_device"], "cpu")

        common = {
            "provenance": {"manifest_sha256": "m" * 64},
            "source_sha256": "s" * 64,
            "artifact_sha256": "a" * 64,
            "environment_sha256": "e" * 64,
            "engine": ENGINE,
            "runtime": runtime,
        }
        families = {
            "cifar100_medium": {
                "targets": [10, 14, 22, 40],
                "status": "gate1_candidate",
            },
            "cifar100_large": {
                "targets": [],
                "status": "not_promoted",
            },
        }
        original = _experiment_fingerprint(
            query_dual_feedback_families=families,
            **common,
        )
        tampered = {
            **families,
            "cifar100_medium": {
                "targets": [10, 14, 22],
                "status": "gate1_candidate",
            },
        }
        self.assertNotEqual(
            original,
            _experiment_fingerprint(
                query_dual_feedback_families=tampered,
                **common,
            ),
        )
        for key, value in (
            ("query_dual_feedback_steps", 7),
            ("query_dual_feedback_time_limit", 11.0),
            ("query_dual_feedback_block_size", 1024),
            ("query_dual_feedback_device", "cuda"),
        ):
            altered_runtime = dict(runtime)
            altered_runtime[key] = value
            altered_common = dict(common, runtime=altered_runtime)
            self.assertNotEqual(
                original,
                _experiment_fingerprint(
                    query_dual_feedback_families=families,
                    **altered_common,
                ),
            )
        policy = _query_dual_candidate_policy()
        for key, value in (
            ("pipeline_schema", "act.verified_query_dual_feedback.v1"),
            ("candidate_schema", "act.query_dual_candidates.v1"),
            ("candidate_protocol", "frozen_alpha_replay_v1"),
            ("candidate_success_status", "generated"),
            (
                "replay_chunk_size_binding",
                "independent_from_effective_block_size",
            ),
            (
                "candidate_non_authoritative_audit_fields",
                ["solver"],
            ),
            ("optimizer_margins_exported", True),
        ):
            with self.subTest(policy_key=key):
                altered_policy = dict(policy)
                altered_policy[key] = value
                with mock.patch(
                    "act.pipeline.verification.hybridz_largecls_gate."
                    "_query_dual_candidate_policy",
                    return_value=altered_policy,
                ):
                    changed = _experiment_fingerprint(
                        query_dual_feedback_families=families,
                        **common,
                    )
                self.assertNotEqual(original, changed)

    def test_runtime_override_validates_parsed_family_query_targets(self):
        args = _build_parser().parse_args(
            [
                "--query-dual-feedback-steps",
                "4",
                "--query-dual-feedback-time-limit",
                "20",
                "--residual-bound-screen",
                "--property-tail-upper",
            ]
        )
        runtime = _runtime_from_args(
            {"runtime": _runtime()},
            args,
            query_dual_feedback_targets=[10, 14, 22, 40],
        )
        self.assertEqual(runtime["query_dual_feedback_steps"], 4)
        self.assertEqual(runtime["query_dual_feedback_time_limit"], 20.0)
        self.assertTrue(runtime["residual_bound_screen"])
        self.assertTrue(runtime["property_tail_upper"])

    def test_property_tail_add_source_planes_boolean_override_is_unambiguous(
        self,
    ):
        parser = _build_parser()
        self.assertFalse(
            parser.parse_args(
                ["--no-property-tail-add-source-planes"]
            ).property_tail_add_source_planes
        )
        self.assertTrue(
            parser.parse_args(
                ["--property-tail-add-source-planes"]
            ).property_tail_add_source_planes
        )

    def test_property_tail_add_source_planes_dependencies(self):
        with self.assertRaisesRegex(GateConfigError, "must be a boolean"):
            _validate_runtime(
                _runtime(property_tail_add_source_planes=1)
            )
        with self.assertRaisesRegex(
            GateConfigError, "property_tail_upper=true"
        ):
            _validate_runtime(
                _runtime(property_tail_add_source_planes=True)
            )
        with self.assertRaisesRegex(
            GateConfigError, "operator_materialize_add=true"
        ):
            _validate_runtime(
                _runtime(
                    property_tail_add_source_planes=True,
                    property_tail_upper=True,
                    operator_materialize_add=False,
                )
            )
        _validate_runtime(
            _runtime(
                property_tail_add_source_planes=True,
                property_tail_upper=True,
                operator_materialize_add=True,
            )
        )

    def test_backend_config_enforces_add_source_plane_contract(self):
        self.assertFalse(
            HybridZConfig().property_tail_add_source_planes
        )
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            HybridZConfig(property_tail_add_source_planes="false")
        with self.assertRaisesRegex(
            ValueError, "property_tail_upper=true"
        ):
            HybridZConfig(property_tail_add_source_planes=True)
        with self.assertRaisesRegex(
            ValueError, "property_tail_upper=true"
        ):
            HybridZConfig(
                property_tail_add_source_planes=True,
                property_tail_upper="true",
            )
        with self.assertRaisesRegex(
            ValueError, "operator_materialize_add=true"
        ):
            HybridZConfig(
                property_tail_add_source_planes=True,
                property_tail_upper=True,
                operator_materialize_add=False,
            )
        configured = HybridZConfig(
            property_tail_add_source_planes=True,
            property_tail_upper=True,
            operator_materialize_add=True,
        )
        self.assertTrue(configured.property_tail_add_source_planes)

    def test_property_tail_mixture_grid_bits_parser_and_dependencies(self):
        args = _build_parser().parse_args(
            ["--property-tail-mixture-grid-bits", "24"]
        )
        self.assertEqual(args.property_tail_mixture_grid_bits, 24)

        for invalid in (True, "1", -1, 25):
            with self.subTest(invalid=invalid), self.assertRaises(
                GateConfigError
            ):
                _validate_runtime(
                    _runtime(property_tail_mixture_grid_bits=invalid)
                )
        with self.assertRaisesRegex(
            GateConfigError, "property_tail_upper=true"
        ):
            _validate_runtime(
                _runtime(property_tail_mixture_grid_bits=1)
            )
        with self.assertRaisesRegex(
            GateConfigError, "property_tail_alpha_steps>0"
        ):
            _validate_runtime(
                _runtime(
                    property_tail_upper=True,
                    property_tail_mixture_grid_bits=1,
                )
            )
        with self.assertRaisesRegex(GateConfigError, "enabled together"):
            _validate_runtime(
                _runtime(
                    property_tail_upper=True,
                    property_tail_alpha_steps=1,
                    property_tail_mixture_grid_bits=1,
                )
            )
        with self.assertRaisesRegex(
            GateConfigError, "operator_exact_budget=0"
        ):
            _validate_runtime(
                _runtime(
                    operator_exact_budget=1,
                    property_tail_upper=True,
                    property_tail_alpha_steps=1,
                    property_tail_alpha_time_limit=1.0,
                    property_tail_mixture_grid_bits=1,
                )
            )
        for bits in (1, 24):
            with self.subTest(bits=bits):
                _validate_runtime(
                    _runtime(
                        property_tail_upper=True,
                        property_tail_alpha_steps=4,
                        property_tail_alpha_time_limit=1.0,
                        property_tail_mixture_grid_bits=bits,
                    )
                )

    def test_backend_config_enforces_mixture_grid_contract(self):
        self.assertEqual(HybridZConfig().property_tail_mixture_grid_bits, 0)
        for invalid in (True, "1", -1, 25):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                HybridZConfig(property_tail_mixture_grid_bits=invalid)
        with self.assertRaisesRegex(ValueError, "property_tail_upper=true"):
            HybridZConfig(property_tail_mixture_grid_bits=1)
        with self.assertRaisesRegex(
            ValueError, "property_tail_alpha_steps>0"
        ):
            HybridZConfig(
                property_tail_upper=True,
                property_tail_mixture_grid_bits=1,
            )
        with self.assertRaisesRegex(ValueError, "enabled together"):
            HybridZConfig(
                property_tail_upper=True,
                property_tail_alpha_steps=1,
                property_tail_mixture_grid_bits=1,
            )
        with self.assertRaisesRegex(ValueError, "operator_exact_budget=0"):
            HybridZConfig(
                operator_exact_budget=1,
                property_tail_upper=True,
                property_tail_alpha_steps=1,
                property_tail_alpha_time_limit=1.0,
                property_tail_mixture_grid_bits=1,
            )
        configured = HybridZConfig(
            property_tail_upper=True,
            property_tail_alpha_steps=4,
            property_tail_alpha_time_limit=1.0,
            property_tail_mixture_grid_bits=24,
        )
        self.assertEqual(configured.property_tail_mixture_grid_bits, 24)

    def test_property_tail_pairhull_parser_and_runtime_dependencies(self):
        args = _build_parser().parse_args(
            [
                "--property-tail-pairhull-budget",
                "8",
                "--property-tail-pairhull-time-limit",
                "1.5",
            ]
        )
        self.assertEqual(args.property_tail_pairhull_budget, 8)
        self.assertEqual(args.property_tail_pairhull_time_limit, 1.5)

        for invalid in (True, "1", -1, 9):
            with self.subTest(invalid=invalid), self.assertRaises(
                GateConfigError
            ):
                _validate_runtime(
                    _runtime(property_tail_pairhull_budget=invalid)
                )
        for invalid in (True, "1.0", -0.1, 1.500001, float("nan")):
            with self.subTest(invalid=invalid), self.assertRaises(
                GateConfigError
            ):
                _validate_runtime(
                    _runtime(property_tail_pairhull_time_limit=invalid)
                )
        with self.assertRaisesRegex(GateConfigError, "enabled together"):
            _validate_runtime(
                _runtime(property_tail_pairhull_budget=1)
            )
        with self.assertRaisesRegex(GateConfigError, "enabled together"):
            _validate_runtime(
                _runtime(property_tail_pairhull_time_limit=0.5)
            )
        with self.assertRaisesRegex(
            GateConfigError, "property_tail_upper=true"
        ):
            _validate_runtime(
                _runtime(
                    property_tail_pairhull_budget=1,
                    property_tail_pairhull_time_limit=0.5,
                )
            )
        with self.assertRaisesRegex(
            GateConfigError, "operator_exact_budget=0"
        ):
            _validate_runtime(
                _runtime(
                    operator_exact_budget=1,
                    property_tail_upper=True,
                    property_tail_pairhull_budget=1,
                    property_tail_pairhull_time_limit=0.5,
                )
            )
        _validate_runtime(
            _runtime(
                property_tail_upper=True,
                property_tail_pairhull_budget=8,
                property_tail_pairhull_time_limit=1.5,
            )
        )

    def test_backend_config_enforces_pairhull_contract(self):
        default = HybridZConfig()
        self.assertEqual(default.property_tail_pairhull_budget, 0)
        self.assertEqual(default.property_tail_pairhull_time_limit, 0.0)
        for invalid in (True, "1", -1, 9):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                HybridZConfig(property_tail_pairhull_budget=invalid)
        for invalid in (True, "1.0", -0.1, 1.500001, float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                HybridZConfig(property_tail_pairhull_time_limit=invalid)
        with self.assertRaisesRegex(ValueError, "enabled together"):
            HybridZConfig(property_tail_pairhull_budget=1)
        with self.assertRaisesRegex(ValueError, "enabled together"):
            HybridZConfig(property_tail_pairhull_time_limit=0.5)
        with self.assertRaisesRegex(ValueError, "property_tail_upper=true"):
            HybridZConfig(
                property_tail_pairhull_budget=1,
                property_tail_pairhull_time_limit=0.5,
            )
        with self.assertRaisesRegex(ValueError, "operator_exact_budget=0"):
            HybridZConfig(
                operator_exact_budget=1,
                property_tail_upper=True,
                property_tail_pairhull_budget=1,
                property_tail_pairhull_time_limit=0.5,
            )
        configured = HybridZConfig(
            property_tail_upper=True,
            property_tail_pairhull_budget=8,
            property_tail_pairhull_time_limit=1.5,
        )
        self.assertEqual(configured.property_tail_pairhull_budget, 8)
        self.assertEqual(configured.property_tail_pairhull_time_limit, 1.5)

    def test_official_gate_keeps_add_source_planes_disabled(self):
        raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        self.assertIs(
            raw["runtime"]["property_tail_add_source_planes"],
            False,
        )

    def test_official_gate_keeps_mixture_grid_disabled(self):
        raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(
            raw["runtime"]["property_tail_mixture_grid_bits"],
            0,
        )

    def test_official_gate_keeps_pairhull_disabled(self):
        raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(raw["runtime"]["property_tail_pairhull_budget"], 0)
        self.assertEqual(
            raw["runtime"]["property_tail_pairhull_time_limit"],
            0.0,
        )

    def test_official_gate_query_dual_is_property_only_and_disabled(self):
        raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        runtime = raw["runtime"]
        self.assertEqual(runtime["query_dual_feedback_steps"], 0)
        self.assertEqual(runtime["query_dual_feedback_time_limit"], 0.0)
        self.assertEqual(runtime["query_dual_feedback_block_size"], 1024)
        self.assertEqual(runtime["query_dual_feedback_device"], "cuda")
        families = raw["families"]
        self.assertEqual(
            families["cifar100_medium"][
                "query_dual_feedback_targets"
            ],
            [],
        )
        self.assertEqual(
            families["cifar100_medium"]["query_dual_feedback_status"],
            "gate1_candidate",
        )
        for family in ("cifar100_large", "tinyimagenet_medium"):
            self.assertEqual(
                families[family]["query_dual_feedback_targets"], []
            )
            self.assertEqual(
                families[family]["query_dual_feedback_status"],
                "not_promoted",
            )

    def test_add_source_plane_override_reaches_runtime_and_fingerprint(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["--property-tail-add-source-planes"]
        )
        runtime = _runtime_from_args(
            {
                "runtime": _runtime(
                    property_tail_upper=True,
                    property_tail_add_source_planes=False,
                )
            },
            args,
        )
        self.assertTrue(runtime["property_tail_add_source_planes"])
        _validate_runtime(runtime)

        common = {
            "provenance": {"manifest_sha256": "m" * 64},
            "source_sha256": "s" * 64,
            "artifact_sha256": "a" * 64,
            "environment_sha256": "e" * 64,
            "engine": ENGINE,
        }
        disabled = _runtime(property_tail_upper=True)
        enabled = _runtime(
            property_tail_upper=True,
            property_tail_add_source_planes=True,
        )
        self.assertNotEqual(
            _experiment_fingerprint(runtime=disabled, **common),
            _experiment_fingerprint(runtime=enabled, **common),
        )

    def test_mixture_grid_override_reaches_runtime_and_fingerprint(self):
        args = _build_parser().parse_args(
            ["--property-tail-mixture-grid-bits", "8"]
        )
        runtime = _runtime_from_args(
            {
                "runtime": _runtime(
                    property_tail_upper=True,
                    property_tail_alpha_steps=4,
                    property_tail_alpha_time_limit=1.0,
                )
            },
            args,
        )
        self.assertEqual(runtime["property_tail_mixture_grid_bits"], 8)

        common = {
            "provenance": {"manifest_sha256": "m" * 64},
            "source_sha256": "s" * 64,
            "artifact_sha256": "a" * 64,
            "environment_sha256": "e" * 64,
            "engine": ENGINE,
        }
        disabled = dict(runtime, property_tail_mixture_grid_bits=0)
        self.assertNotEqual(
            _experiment_fingerprint(runtime=disabled, **common),
            _experiment_fingerprint(runtime=runtime, **common),
        )

    def test_pairhull_override_reaches_runtime_and_fingerprint(self):
        args = _build_parser().parse_args(
            [
                "--property-tail-pairhull-budget",
                "8",
                "--property-tail-pairhull-time-limit",
                "1.5",
            ]
        )
        runtime = _runtime_from_args(
            {"runtime": _runtime(property_tail_upper=True)},
            args,
        )
        self.assertEqual(runtime["property_tail_pairhull_budget"], 8)
        self.assertEqual(
            runtime["property_tail_pairhull_time_limit"],
            1.5,
        )

        common = {
            "provenance": {"manifest_sha256": "m" * 64},
            "source_sha256": "s" * 64,
            "artifact_sha256": "a" * 64,
            "environment_sha256": "e" * 64,
            "engine": ENGINE,
        }
        disabled = dict(
            runtime,
            property_tail_pairhull_budget=0,
            property_tail_pairhull_time_limit=0.0,
        )
        enabled_fingerprint = _experiment_fingerprint(
            runtime=runtime, **common
        )
        self.assertNotEqual(
            _experiment_fingerprint(runtime=disabled, **common),
            enabled_fingerprint,
        )
        budget_tamper = dict(runtime, property_tail_pairhull_budget=7)
        time_tamper = dict(runtime, property_tail_pairhull_time_limit=1.25)
        self.assertNotEqual(
            _experiment_fingerprint(runtime=budget_tamper, **common),
            enabled_fingerprint,
        )
        self.assertNotEqual(
            _experiment_fingerprint(runtime=time_tamper, **common),
            enabled_fingerprint,
        )

    def test_general_cli_help_exposes_hybridz_candidate_chain(self):
        repository = Path(__file__).resolve().parents[3]
        expected = (
            "--hybridz-operator-materialize-add",
            "--hybridz-property-tail-upper",
            "--hybridz-property-tail-add-source-planes",
            "--hybridz-property-tail-alpha-steps",
            "--hybridz-property-tail-mixture-grid-bits",
            "--hybridz-property-tail-pairhull-budget",
            "--hybridz-property-tail-pairhull-time-limit",
            "--hybridz-query-dual-feedback-targets",
            "--hybridz-query-dual-feedback-steps",
            "--hybridz-query-dual-feedback-time-limit",
            "--hybridz-query-dual-feedback-block-size",
            "--hybridz-query-dual-feedback-device",
        )
        for module in ("act.back_end", "act.pipeline"):
            with self.subTest(module=module):
                completed = subprocess.run(
                    [sys.executable, "-m", module, "--help"],
                    cwd=repository,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=20.0,
                    check=False,
                )
                self.assertEqual(
                    completed.returncode, 0, completed.stderr
                )
                for flag in expected:
                    self.assertIn(flag, completed.stdout)

    def test_float32_is_forbidden(self):
        with self.assertRaisesRegex(GateConfigError, "float32"):
            _validate_runtime(_runtime(dtype="float32"))

    def test_row_workers_must_fit_total_cap(self):
        with self.assertRaisesRegex(GateConfigError, "cannot exceed"):
            _validate_runtime(_runtime(row_workers=4, total_solver_threads=2))

    def test_candidate_budgets_reject_invalid_values(self):
        invalid = (
            {"preactivation_lp_budget": -1},
            {"preactivation_lp_time_limit": float("nan")},
            {"property_residual_budget": -1},
            {"property_residual_time_limit": float("nan")},
            {"property_residual_max_adjoint_cells": 0},
            {"property_residual_pool_per_rival": 0},
            {
                "property_tail_upper": True,
                "property_residual_budget": 1,
            },
            {"property_tail_alpha_steps": -1},
            {"property_tail_alpha_time_limit": float("nan")},
            {"property_tail_alpha_learning_rate": 0.0},
            {"property_tail_alpha_max_cells": 0},
            {"property_tail_alpha_device": "auto"},
            {
                "property_tail_alpha_steps": 1,
                "property_tail_alpha_time_limit": 0.0,
            },
            {
                "property_tail_alpha_steps": 1,
                "property_tail_alpha_time_limit": 1.0,
                "property_tail_upper": False,
            },
            {
                "property_tail_alpha_steps": 1,
                "property_tail_alpha_time_limit": 1.0,
                "property_tail_upper": True,
                "operator_exact_budget": 1,
            },
            {"gpu_dual_steps": -1},
            {"gpu_dual_time_limit": float("inf")},
            {"gpu_dual_row_topk": -1},
            {"gpu_dual_learning_rate": 0.0},
            {"query_dual_feedback_steps": True},
            {"query_dual_feedback_steps": -1},
            {"query_dual_feedback_steps": 65},
            {"query_dual_feedback_time_limit": float("nan")},
            {"query_dual_feedback_time_limit": 20.1},
            {"query_dual_feedback_block_size": 0},
            {"query_dual_feedback_block_size": 4097},
            {"query_dual_feedback_device": "auto"},
            {"query_dual_feedback_time_limit": 1.0},
            {
                "query_dual_feedback_steps": 1,
                "query_dual_feedback_time_limit": 1.0,
            },
            {
                "query_dual_feedback_steps": 1,
                "query_dual_feedback_time_limit": 1.0,
                "property_tail_upper": True,
                "engine": "dense_hz_objbound",
            },
            {
                "query_dual_feedback_steps": 1,
                "query_dual_feedback_time_limit": 1.0,
                "property_tail_upper": True,
                "operator_exact_budget": 1,
            },
            {"lp_prefilter_fraction": float("nan")},
            {"lp_prefilter_max_seconds": float("inf")},
        )
        for update in invalid:
            with self.subTest(update=update), self.assertRaises(
                GateConfigError
            ):
                _validate_runtime(_runtime(**update))

    def test_fixed_hz_environment_is_explicit_and_capped(self):
        environment = _fixed_worker_environment(_runtime())
        hz_keys = {key for key in environment if key.startswith("HZ_")}
        self.assertIn("HZ_HIGHS_OPTIONS", hz_keys)
        self.assertIn("HZ_MILP_EQ_SUBST", hz_keys)
        self.assertEqual(environment["HZ_MILP_BACKEND"], "highs")
        self.assertIn("HZ_RELU_TIGHT_LP_TIMEOUT", hz_keys)
        self.assertIn("HZ_LP_PREFILTER_MAX_SECONDS", hz_keys)
        self.assertLessEqual(
            int(environment["HZ_QUERY_WORKERS"])
            * int(environment["HZ_MILP_THREADS"]),
            20,
        )
        self.assertEqual(environment["OMP_NUM_THREADS"], "1")

    def test_operator_phase_clique_config_is_default_off_and_coupled(self):
        configured = HybridZConfig()
        self.assertEqual(
            configured.operator_phase_clique_time_limit, 0.0
        )
        raw = yaml.safe_load(
            DEFAULT_CONFIG.read_text(encoding="utf-8")
        )
        self.assertEqual(
            raw["runtime"]["operator_phase_clique_time_limit"],
            0.0,
        )
        parsed = _build_parser().parse_args(["--dry-run"])
        self.assertIsNone(
            parsed.operator_phase_clique_time_limit
        )
        runtime = _runtime_from_args(
            {"runtime": _runtime()},
            parsed,
        )
        self.assertEqual(
            runtime["operator_phase_clique_time_limit"], 0.0
        )

        enabled = _runtime(
            operator_exact_budget=4,
            operator_phase_clique_time_limit=20.0,
            property_residual_budget=4,
            property_residual_time_limit=4.0,
        )
        _validate_runtime(enabled)
        configured = HybridZConfig(
            engine=ENGINE,
            operator_exact_budget=4,
            operator_phase_clique_time_limit=20.0,
            property_residual_budget=4,
            property_residual_time_limit=4.0,
        )
        self.assertEqual(
            configured.operator_phase_clique_time_limit, 20.0
        )
        with self.assertRaisesRegex(
            ValueError, "preactivation_lp_time_limit=0"
        ):
            HybridZConfig(
                engine=ENGINE,
                operator_exact_budget=4,
                operator_phase_clique_time_limit=20.0,
                preactivation_lp_time_limit=0.25,
                property_residual_budget=4,
                property_residual_time_limit=4.0,
            )

        invalid_updates = (
            {"operator_phase_clique_time_limit": True},
            {"operator_phase_clique_time_limit": -1.0},
            {"operator_phase_clique_time_limit": 40.1},
            {"operator_phase_clique_time_limit": 1.0},
            {
                "operator_phase_clique_time_limit": 1.0,
                "operator_exact_budget": 4,
                "property_residual_budget": 4,
                "property_residual_time_limit": 1.0,
                "query_dual_feedback_steps": 1,
                "query_dual_feedback_time_limit": 1.0,
                "property_tail_upper": True,
            },
            {
                "operator_phase_clique_time_limit": 1.0,
                "operator_exact_budget": 4,
                "property_residual_budget": 4,
                "property_residual_time_limit": 1.0,
                "gpu_dual_steps": 1,
            },
            {
                "operator_phase_clique_time_limit": 1.0,
                "operator_exact_budget": 4,
                "property_residual_budget": 4,
                "property_residual_time_limit": 1.0,
                "preactivation_lp_time_limit": 0.25,
            },
        )
        for update in invalid_updates:
            with self.subTest(update=update), self.assertRaises(
                GateConfigError
            ):
                _validate_runtime(_runtime(**update))


class PromotionAudit(unittest.TestCase):
    def _validate(self, path: Path, gate: int):
        return validate_promotion(
            path,
            gate=gate,
            provenance={"manifest_sha256": "m" * 64},
            experiment_sha256="x" * 64,
            source_sha256="s" * 64,
            artifact_sha256="a" * 64,
            environment_sha256="e" * 64,
            expected_families=FAMILIES,
        )

    def test_gate40_requires_embedded_gate6_chain(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            gate6_path = directory / "gate6.json"
            gate6_path.write_text(
                json.dumps(_pass_record(6, chain=[])),
                encoding="utf-8",
            )
            gate6 = self._validate(gate6_path, 14)
            self.assertEqual(
                [item["gate"] for item in gate6["chain"]],
                [6],
            )

            gate14_path = directory / "gate14.json"
            gate14_path.write_text(
                json.dumps(_pass_record(14, chain=gate6["chain"])),
                encoding="utf-8",
            )
            gate14 = self._validate(gate14_path, 40)
            self.assertEqual(
                [item["gate"] for item in gate14["chain"]],
                [6, 14],
            )

            gate6_path.write_text("mutated", encoding="utf-8")
            with self.assertRaisesRegex(
                GateConfigError, "promotion_chain_receipts_unchanged"
            ):
                self._validate(gate14_path, 40)

            gate14_path.write_text(
                json.dumps(_pass_record(14, chain=[])),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                GateConfigError, "promotion_chain"
            ):
                self._validate(gate14_path, 40)

    def test_parent_only_or_diagnostic_receipt_cannot_promote(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "gate6.json"
            for key, value, failed_check in (
                ("diagnostic_only", True, "not_diagnostic_only"),
                ("promotion_eligible", False, "promotion_eligible"),
                (
                    "property_micro_rlt_parent_only_diagnostic",
                    True,
                    "not_parent_only_diagnostic",
                ),
                (
                    "property_micro_rlt_packet_mode",
                    "first",
                    "production_micro_rlt_packet_mode",
                ),
            ):
                with self.subTest(key=key):
                    record = _pass_record(6, chain=[])
                    record[key] = value
                    path.write_text(json.dumps(record), encoding="utf-8")
                    with self.assertRaisesRegex(
                        GateConfigError,
                        failed_check,
                    ):
                        self._validate(path, 14)


if __name__ == "__main__":
    unittest.main()
