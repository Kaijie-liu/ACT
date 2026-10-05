from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from generate_formal_unsolved_structure_manifest_v1 import (
    AUTHORITY_FILES,
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_HYZOR_ROOT,
    EXPERIMENT_ROOT,
    build_manifest,
    classify_signature,
    exclusive_atomic_write,
    matching_cohorts,
    render_manifest,
    sha256_json,
    validate_manifest,
    verify_authority,
)


MANIFEST_PATH = (
    EXPERIMENT_ROOT / "manifests" / "formal_unsolved_structure_manifest_v1.json"
)


def _signature(op_counts, *, add=0, concat=0, matmul=0):
    return {
        "op_counts": dict(op_counts),
        "dynamic_merge_counts": {
            "Add": add,
            "Concat": concat,
            "MatMul": matmul,
        },
    }


@pytest.mark.parametrize(
    ("signature", "expected"),
    [
        (_signature({"Conv": 8, "Relu": 7, "Gemm": 1}), "A"),
        (_signature({"ConvTranspose": 4, "Conv": 4, "Relu": 7}), "A"),
        (_signature({"Add": 26, "MatMul": 26, "Relu": 12}), "B"),
        (_signature({"Add": 9, "MatMul": 8, "Relu": 7}), "C"),
        (_signature({"Gemm": 8, "MatMul": 3, "Relu": 7, "Concat": 1}, concat=1), "D"),
        (
            _signature(
                {"Conv": 1, "MatMul": 16, "Relu": 2, "Softmax": 2, "Add": 18},
                add=4,
                matmul=4,
            ),
            "E",
        ),
        (_signature({"Conv": 8, "Relu": 7, "Sigmoid": 1}), "F"),
    ],
)
def test_structure_classifier_uses_disjoint_operator_rules(signature, expected):
    assert matching_cohorts(signature) == [expected]
    assert classify_signature(signature) == expected


def test_attention_precedes_smooth_and_smooth_precedes_conv():
    attention_with_smooth = _signature(
        {"Conv": 30, "Relu": 13, "Sigmoid": 1, "Tanh": 1, "Softmax": 2},
        add=13,
        concat=8,
        matmul=4,
    )
    smooth_conv = _signature({"Conv": 8, "Relu": 7, "Sigmoid": 1})
    assert matching_cohorts(attention_with_smooth) == ["E"]
    assert matching_cohorts(smooth_conv) == ["F"]


def test_unsupported_signature_fails_closed():
    signature = _signature({"AveragePool": 1})
    assert matching_cohorts(signature) == []
    with pytest.raises(ValueError, match="0 A-F matches"):
        classify_signature(signature)


def test_checked_in_manifest_rebuilds_byte_identically():
    existing = MANIFEST_PATH.read_bytes()
    rebuilt = build_manifest(DEFAULT_HYZOR_ROOT, DEFAULT_BENCHMARK_ROOT)
    assert render_manifest(rebuilt) == existing
    assert hashlib.sha256(existing).hexdigest() == hashlib.sha256(
        render_manifest(rebuilt)
    ).hexdigest()


def test_manifest_closes_all_543_rows_and_family_cohorts():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="ascii"))
    validate_manifest(manifest)
    assert manifest["coverage_proof"] == {
        "composite_source_rows": 2413,
        "excluded_solved_rows": 1870,
        "included_unsolved_rows": 543,
        "unknown": 269,
        "timeout": 274,
        "unassigned_rows": 0,
        "multiply_assigned_rows": 0,
        "unique_row_identity_count": 543,
        "unique_row_identity_sha256_count": 543,
    }
    assert manifest["cohort_family_matrix"] == {
        "A": {
            "cgan": {"UNKNOWN": 4, "TIMEOUT": 0, "unsolved": 4},
            "metaroom": {"UNKNOWN": 0, "TIMEOUT": 5, "unsolved": 5},
            "relusplitter": {"UNKNOWN": 75, "TIMEOUT": 44, "unsolved": 119},
        },
        "B": {"tllverify": {"UNKNOWN": 15, "TIMEOUT": 0, "unsolved": 15}},
        "C": {
            "acasxu": {"UNKNOWN": 62, "TIMEOUT": 4, "unsolved": 66},
            "cora": {"UNKNOWN": 9, "TIMEOUT": 131, "unsolved": 140},
            "relusplitter": {"UNKNOWN": 23, "TIMEOUT": 33, "unsolved": 56},
            "safenlp": {"UNKNOWN": 1, "TIMEOUT": 0, "unsolved": 1},
        },
        "D": {
            "cersyve": {"UNKNOWN": 1, "TIMEOUT": 0, "unsolved": 1},
            "linearizenn": {"UNKNOWN": 18, "TIMEOUT": 2, "unsolved": 20},
        },
        "E": {
            "cgan": {"UNKNOWN": 2, "TIMEOUT": 0, "unsolved": 2},
            "vit": {"UNKNOWN": 57, "TIMEOUT": 53, "unsolved": 110},
        },
        "F": {
            "cgan": {"UNKNOWN": 2, "TIMEOUT": 0, "unsolved": 2},
            "dist_shift": {"UNKNOWN": 0, "TIMEOUT": 2, "unsolved": 2},
        },
    }


def test_A_direct_reach_is_structurally_separate_from_convtranspose_four():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="ascii"))
    rows = [row for row in manifest["instances"] if row["cohort"] == "A"]
    direct = [
        row
        for row in rows
        if row["a_reach_scope"] == "current_direct_implicit_conv2d"
    ]
    extension = [
        row for row in rows if row["a_reach_scope"] == "convtranspose_extension"
    ]
    assert (len(direct), len(extension), len(rows)) == (124, 4, 128)
    assert all(
        row["model"]["operator_signature"]["op_counts"].get("ConvTranspose", 0)
        == 0
        for row in direct
    )
    assert all(
        row["model"]["operator_signature"]["op_counts"].get("ConvTranspose", 0)
        > 0
        for row in extension
    )


def test_vit_uses_strict_wall_verdict_not_non_strict_status():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="ascii"))
    row = next(row for row in manifest["instances"] if row["row_identity"] == "vit:100")
    # The authority's non-strict status is CERTIFIED, but strict_status is
    # TIMEOUT because wall time exceeded 100 seconds.  Only the latter belongs
    # in the locked 1,870 baseline.
    assert row["verdict"] == "TIMEOUT"
    assert row["source"]["authority_key"] == "vit_strict_rows"


def _rehash_manifest(manifest):
    payload = dict(manifest)
    payload.pop("manifest_payload_sha256", None)
    manifest["manifest_payload_sha256"] = sha256_json(payload)


@pytest.mark.parametrize(
    "tamper",
    [
        "row_identity",
        "cohort",
        "cohort_name",
        "a_scope",
        "source_family",
        "row_order",
        "family_summary",
        "coverage",
        "authority",
        "benchmark",
    ],
)
def test_self_rehashed_semantic_tampering_fails_closed(tamper):
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="ascii"))
    candidate = copy.deepcopy(manifest)
    if tamper == "row_identity":
        candidate["instances"][0]["row_identity"] = "cora:invented"
    elif tamper == "cohort":
        candidate["instances"][0]["cohort"] = "A"
    elif tamper == "cohort_name":
        candidate["instances"][0]["cohort_name"] = "invented"
    elif tamper == "a_scope":
        first_A = next(row for row in candidate["instances"] if row["cohort"] == "A")
        first_A["a_reach_scope"] = "convtranspose_extension"
    elif tamper == "source_family":
        candidate["instances"][0]["source"]["source_family"] = "cora_2024"
    elif tamper == "row_order":
        candidate["instances"][0], candidate["instances"][1] = (
            candidate["instances"][1],
            candidate["instances"][0],
        )
    elif tamper == "family_summary":
        candidate["family_summary"]["safenlp"]["solved"] -= 1
    elif tamper == "coverage":
        candidate["coverage_proof"]["included_unsolved_rows"] = 542
    elif tamper == "authority":
        candidate["composite_authority"]["vit_strict_rows"]["sha256"] = "0" * 64
    else:
        candidate["benchmark_provenance"]["git_commit"] = "0" * 40
    _rehash_manifest(candidate)
    with pytest.raises(ValueError):
        validate_manifest(candidate)


def test_authority_hash_mismatch_fails_before_missing_later_sources(tmp_path):
    first = next(iter(AUTHORITY_FILES.values()))
    target = tmp_path / str(first["relative_path"])
    target.parent.mkdir(parents=True)
    target.write_bytes(b"tampered authority")
    with pytest.raises(ValueError, match="authority hash mismatch"):
        verify_authority(tmp_path)


def test_exclusive_output_refuses_overwrite_and_escape(tmp_path):
    allowed = tmp_path / "isolated"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    output = allowed / "manifest.json"
    exclusive_atomic_write(output, b"first", allowed_root=allowed)
    with pytest.raises(FileExistsError):
        exclusive_atomic_write(output, b"second", allowed_root=allowed)
    assert output.read_bytes() == b"first"
    assert not list(allowed.glob(".*.tmp.*"))
    with pytest.raises(ValueError, match="output must remain below"):
        exclusive_atomic_write(outside / "manifest.json", b"no", allowed_root=allowed)
