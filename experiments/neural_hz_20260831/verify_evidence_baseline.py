#!/usr/bin/env python3
"""Read-only closure verification for one external evidence-baseline ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    )


def _problem_content_key(
    family: str, model_sha256: str, spec_sha256: str
) -> str:
    return _sha256_json(
        {
            "format": "neural_hz_problem_content_key_v1",
            "family": family,
            "model_sha256": model_sha256,
            "spec_sha256": spec_sha256,
        }
    )


def _load_self_hashed(path: Path, expected_format: str) -> tuple[dict, bytes]:
    payload = path.resolve(strict=True).read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict) or value.get("format_version") != expected_format:
        raise ValueError(f"unsupported format: {path}")
    claimed = value.get("manifest_payload_sha256")
    unhashed = dict(value)
    unhashed.pop("manifest_payload_sha256", None)
    if claimed != _sha256_json(unhashed):
        raise ValueError(f"manifest payload hash mismatch: {path}")
    return value, payload


def _tree_payload(root: Path) -> tuple[int, str]:
    resolved = root.resolve(strict=True)
    records = []
    for path in sorted(candidate for candidate in resolved.rglob("*") if candidate.is_file()):
        records.append(
            {
                "relative_path": path.relative_to(resolved).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        )
    return len(records), _sha256_json(records)


def verify_evidence_baseline(universe_path: Path, ledger_path: Path) -> dict[str, object]:
    universe, universe_bytes = _load_self_hashed(
        universe_path, "neural_hz_family_universe_v1"
    )
    ledger, ledger_bytes = _load_self_hashed(
        ledger_path, "neural_hz_external_evidence_baseline_v2"
    )
    if ledger.get("ledger_kind") != "EXTERNAL_CONTENT_KEY_EVIDENCE_BASELINE":
        raise ValueError("unexpected evidence ledger kind")
    if ledger.get("neural_hz_baseline_claim") is not False:
        raise ValueError("evidence ledger claims a Neural-HZ baseline")
    required_nonaliases = {
        "historical_large_classification_59_of_400",
        "formal_13_family_1870_of_2413",
    }
    if set(ledger.get("not_alias_of", [])) != required_nonaliases:
        raise ValueError("evidence ledger namespace exclusions changed")
    if ledger["family"] != universe["family"]:
        raise ValueError("family mismatch")
    if ledger["source_universe_file_sha256"] != _sha256_bytes(universe_bytes):
        raise ValueError("evidence ledger references a different universe file")
    if ledger["ordered_universe_sha256"] != universe["ordered_universe_sha256"]:
        raise ValueError("ordered universe mismatch")
    expected_namespace = (
        "external.vnncomp2025_content_v1/"
        f"{universe['family']}@{universe['ordered_universe_sha256']}"
        "/evidence_baseline_v2"
    )
    if ledger.get("namespace") != expected_namespace:
        raise ValueError("evidence ledger namespace mismatch")
    if ledger.get("zero_tolerance") != 0.0:
        raise ValueError("evidence ledger is not literal zero tolerance")
    session = ledger.get("ort_session_config")
    if not isinstance(session, dict) or session.get("providers") != [
        "CPUExecutionProvider"
    ]:
        raise ValueError("evidence ledger is not CPU-only ORT")
    if ledger.get("ort_session_config_sha256") != _sha256_json(session):
        raise ValueError("ORT session config hash mismatch")

    validator_sources = ledger.get("validator_sources")
    if not isinstance(validator_sources, list) or len(validator_sources) != 3:
        raise ValueError("validator dependency closure is incomplete")
    if ledger.get("validator_sources_sha256") != _sha256_json(validator_sources):
        raise ValueError("validator source-list hash mismatch")
    experiment_root = ledger_path.resolve(strict=True).parents[1]
    for source in validator_sources:
        path = (experiment_root / source["relative_path"]).resolve(strict=True)
        try:
            path.relative_to(experiment_root)
        except ValueError as exc:
            raise ValueError("validator source escapes experiment root") from exc
        if int(path.stat().st_size) != int(source["size_bytes"]):
            raise ValueError(f"validator source size mismatch: {path}")
        if _sha256_file(path) != source["sha256"]:
            raise ValueError(f"validator source hash mismatch: {path}")

    tree_count, tree_sha256 = _tree_payload(Path(ledger["source_sidecar_root"]))
    if tree_count != ledger["source_sidecar_file_count"]:
        raise ValueError("source sidecar tree count mismatch")
    if tree_sha256 != ledger["source_sidecar_files_sha256"]:
        raise ValueError("source sidecar tree hash mismatch")
    results_record = ledger.get("source_results_csv")
    if results_record is not None:
        results_path = Path(results_record["path"]).resolve(strict=True)
        if int(results_path.stat().st_size) != int(results_record["size_bytes"]):
            raise ValueError("source results CSV size mismatch")
        if _sha256_file(results_path) != results_record["sha256"]:
            raise ValueError("source results CSV hash mismatch")

    universe_rows = sorted(
        universe["instances"], key=lambda row: int(row["row_index"])
    )
    rows = sorted(ledger["rows"], key=lambda row: int(row["row_index"]))
    if ledger.get("rows_payload_sha256") != _sha256_json(ledger["rows"]):
        raise ValueError("evidence rows payload hash mismatch")
    if len(rows) != len(universe_rows) or len(rows) != ledger["instance_count"]:
        raise ValueError("evidence row count mismatch")
    keys = [row["instance_key"] for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate evidence instance key")

    counts = {"VALIDATED_ADV": 0, "UNKNOWN": 0}
    minimum_input_slack = float("inf")
    minimum_unsafe_slack = float("inf")
    for expected_index, (source, row) in enumerate(
        zip(universe_rows, rows, strict=True)
    ):
        if int(row["row_index"]) != expected_index:
            raise ValueError("evidence row order is not contiguous")
        for field in (
            "family",
            "instance_key",
            "model_sha256",
            "spec_sha256",
            "timeout_seconds",
        ):
            expected = universe["family"] if field == "family" else source[field]
            if row[field] != expected:
                raise ValueError(f"evidence row identity mismatch: {field}")
        expected_problem_key = _problem_content_key(
            universe["family"], source["model_sha256"], source["spec_sha256"]
        )
        if row.get("problem_content_key") != expected_problem_key:
            raise ValueError("evidence row problem content key mismatch")
        verdict = row["baseline_verdict"]
        if verdict not in counts:
            raise ValueError(f"forbidden evidence verdict: {verdict}")
        counts[verdict] += 1
        if row.get("neural_hz_gain_credit") is not False:
            raise ValueError("historical evidence received Neural-HZ gain credit")
        evidence = row.get("evidence")
        if verdict == "UNKNOWN":
            if evidence is not None or row.get("baseline_credit") != "NONE":
                raise ValueError("UNKNOWN row carries solved evidence")
            if row.get("historical_evidence_status") != "NO_REVALIDATED_WITNESS":
                raise ValueError("UNKNOWN row historical evidence status mismatch")
            continue
        if row.get("baseline_credit") != "EXTERNAL_RETENTION_ANCHOR":
            raise ValueError("validated ADV is not a retention anchor")
        if row.get("historical_evidence_status") != "INDEPENDENT_REPLAY_PASS":
            raise ValueError("validated ADV historical evidence status mismatch")
        if not isinstance(evidence, dict):
            raise ValueError("validated ADV lacks evidence")
        if evidence.get("problem_content_key") != expected_problem_key:
            raise ValueError("validated ADV nested problem content key mismatch")
        if evidence.get("evidence_origin") != "LEGACY_SIDECAR_ATTACK_WITNESS":
            raise ValueError("validated ADV evidence origin mismatch")
        if evidence.get("content_match_status") != "EXACT_UNIQUE":
            raise ValueError("validated ADV content match status mismatch")
        if evidence.get("neural_hz_gain_credit") is not False:
            raise ValueError("validated ADV nested Neural-HZ gain credit changed")
        if evidence.get("replay_status") != "ORT_VNNLIB_PASS_ZERO_TOL":
            raise ValueError("validated ADV lacks independent replay pass")
        if evidence.get("all_values_finite") is not True:
            raise ValueError("validated ADV contains non-finite values")
        if evidence.get("failure_reason") is not None:
            raise ValueError("validated ADV records a failure")
        if evidence.get("stored_output_array_equal") is not True:
            raise ValueError("stored/current ORT output differs")
        input_slack = float(evidence["input_min_slack"])
        unsafe_slack = float(evidence["unsafe_min_slack"])
        if input_slack < 0.0 or unsafe_slack < 0.0:
            raise ValueError("validated ADV has a negative zero-tolerance slack")
        minimum_input_slack = min(minimum_input_slack, input_slack)
        minimum_unsafe_slack = min(minimum_unsafe_slack, unsafe_slack)

    if counts["VALIDATED_ADV"] != ledger["validated_adv_count"]:
        raise ValueError("validated ADV count mismatch")
    if counts["UNKNOWN"] != ledger["unknown_count"]:
        raise ValueError("UNKNOWN count mismatch")
    if ledger.get("cert_count") != 0 or ledger.get("invalid_adv_count") != 0:
        raise ValueError("evidence ledger contains a forbidden CERT/invalid ADV")
    return {
        "family": ledger["family"],
        "instance_count": len(rows),
        "cert_count": 0,
        "validated_adv_count": counts["VALIDATED_ADV"],
        "unknown_count": counts["UNKNOWN"],
        "invalid_adv_count": 0,
        "minimum_input_slack": minimum_input_slack,
        "minimum_unsafe_slack": minimum_unsafe_slack,
        "ledger_file_sha256": _sha256_bytes(ledger_bytes),
        "ledger_payload_sha256": ledger["manifest_payload_sha256"],
        "rows_payload_sha256": ledger["rows_payload_sha256"],
        "neural_hz_gain_credit": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe-manifest", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()
    result = verify_evidence_baseline(args.universe_manifest, args.ledger)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
