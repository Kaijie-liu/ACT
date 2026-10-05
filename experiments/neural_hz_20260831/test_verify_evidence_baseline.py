from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from test_revalidate_largecnn_witnesses import _fixture
from revalidate_largecnn_witnesses import build_evidence_ledger
from verify_evidence_baseline import verify_evidence_baseline


def _ledger(tmp_path):
    universe, sidecar, _, _ = _fixture(tmp_path)
    value = build_evidence_ledger(universe, sidecar, expected_witnesses=1)
    ledger = tmp_path / "experiment" / "evidence" / "ledger.json"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    # Validator paths are relative to the real experiment root. Repoint the
    # ledger location there logically by using the real source closure below.
    return universe, ledger, value


def _sha256_json(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _copy_validator_sources(ledger, value) -> None:
    experiment = ledger.parents[1]
    real_root = Path(__file__).resolve().parent
    for source in value["validator_sources"]:
        source_path = real_root / source["relative_path"]
        destination = experiment / source["relative_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source_path.read_bytes())


def _write_rehashed_ledger(ledger, payload) -> None:
    payload["rows_payload_sha256"] = _sha256_json(payload["rows"])
    payload.pop("manifest_payload_sha256", None)
    payload["manifest_payload_sha256"] = _sha256_json(payload)
    ledger.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def test_real_source_closure_and_rows_verify(tmp_path):
    universe, ledger, value = _ledger(tmp_path)
    # The verifier resolves dependencies relative to ledger.parents[1]. For a
    # unit fixture, replace paths with absolute paths relative to that root.
    _copy_validator_sources(ledger, value)
    result = verify_evidence_baseline(universe, ledger)
    assert result["validated_adv_count"] == 1
    assert result["unknown_count"] == 0
    assert result["neural_hz_gain_credit"] == 0


def test_manifest_tamper_is_detected_before_row_claims(tmp_path):
    universe, ledger, _ = _ledger(tmp_path)
    payload = json.loads(ledger.read_text())
    payload["validated_adv_count"] = 2
    ledger.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="manifest payload hash mismatch"):
        verify_evidence_baseline(universe, ledger)


def test_namespace_tamper_is_detected_after_self_rehash(tmp_path):
    universe, ledger, _ = _ledger(tmp_path)
    payload = json.loads(ledger.read_text())
    payload["namespace"] = "external.wrong/evidence_baseline_v2"
    _write_rehashed_ledger(ledger, payload)
    with pytest.raises(ValueError, match="namespace mismatch"):
        verify_evidence_baseline(universe, ledger)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("problem_content_key", "0" * 64, "row problem content key mismatch"),
        (
            "evidence.problem_content_key",
            "0" * 64,
            "nested problem content key mismatch",
        ),
        ("evidence.evidence_origin", "PGD", "evidence origin mismatch"),
        ("evidence.content_match_status", "FUZZY", "content match status mismatch"),
        (
            "evidence.neural_hz_gain_credit",
            True,
            "nested Neural-HZ gain credit changed",
        ),
        (
            "historical_evidence_status",
            "NO_REVALIDATED_WITNESS",
            "validated ADV historical evidence status mismatch",
        ),
    ],
)
def test_rehashed_row_semantic_tamper_is_detected(
    tmp_path, field, replacement, message
):
    universe, ledger, value = _ledger(tmp_path)
    _copy_validator_sources(ledger, value)
    payload = json.loads(ledger.read_text())
    target = payload["rows"][0]
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = replacement
    _write_rehashed_ledger(ledger, payload)
    with pytest.raises(ValueError, match=message):
        verify_evidence_baseline(universe, ledger)


def test_unknown_historical_status_tamper_is_detected(tmp_path):
    universe, ledger, value = _ledger(tmp_path)
    _copy_validator_sources(ledger, value)
    payload = json.loads(ledger.read_text())
    row = payload["rows"][0]
    row.update(
        {
            "baseline_verdict": "UNKNOWN",
            "baseline_credit": "NONE",
            "historical_evidence_status": "INDEPENDENT_REPLAY_PASS",
            "evidence": None,
        }
    )
    payload["validated_adv_count"] = 0
    payload["unknown_count"] = 1
    _write_rehashed_ledger(ledger, payload)
    with pytest.raises(ValueError, match="UNKNOWN row historical evidence status"):
        verify_evidence_baseline(universe, ledger)
