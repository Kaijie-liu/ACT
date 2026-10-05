#!/usr/bin/env python3
"""Build an external evidence-only baseline from independently replayed witnesses."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from freeze_family_manifest import EXPERIMENT_ROOT, exclusive_atomic_write
from independent_vnnlib_replay import evaluate_vnnlib


FORMAT_VERSION = "neural_hz_external_evidence_baseline_v2"
NAMESPACE = "external.vnncomp2025_content_v1"


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


def problem_content_key(family: str, model_sha256: str, spec_sha256: str) -> str:
    return _sha256_json(
        {
            "format": "neural_hz_problem_content_key_v1",
            "family": family,
            "model_sha256": model_sha256,
            "spec_sha256": spec_sha256,
        }
    )


def _load_universe(path: Path) -> tuple[dict, bytes]:
    payload = path.resolve(strict=True).read_bytes()
    manifest = json.loads(payload)
    if not isinstance(manifest, dict):
        raise ValueError("universe manifest must be a JSON object")
    if manifest.get("format_version") != "neural_hz_family_universe_v1":
        raise ValueError("unsupported universe manifest format")
    claimed = manifest.get("manifest_payload_sha256")
    unhashed = dict(manifest)
    unhashed.pop("manifest_payload_sha256", None)
    if claimed != _sha256_json(unhashed):
        raise ValueError("universe manifest payload hash mismatch")
    if manifest.get("baseline_vector_status") != "UNFROZEN":
        raise ValueError("evidence builder requires an UNFROZEN universe")
    return manifest, payload


def _contained_file(root: Path, relative: str) -> Path:
    resolved_root = root.resolve(strict=True)
    candidate = (resolved_root / relative).resolve(strict=True)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"path escapes root: {relative}") from exc
    if not candidate.is_file():
        raise ValueError(f"missing regular file: {relative}")
    return candidate


def _evidence_tree(root: Path) -> list[dict[str, object]]:
    resolved = root.resolve(strict=True)
    files = []
    for path in sorted(candidate for candidate in resolved.rglob("*") if candidate.is_file()):
        files.append(
            {
                "relative_path": path.relative_to(resolved).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        )
    if not files:
        raise ValueError("sidecar artifact root contains no files")
    return files


def _artifact_jsons(root: Path) -> list[Path]:
    resolved = root.resolve(strict=True)
    return sorted(
        path
        for path in resolved.rglob("*.json")
        if path.name != "MANIFEST.json"
    )


def _input_dtype(ort_type: str):
    mapping = {
        "tensor(float)": np.float32,
        "tensor(double)": np.float64,
    }
    try:
        return mapping[ort_type]
    except KeyError as exc:
        raise ValueError(f"unsupported ONNX input dtype: {ort_type}") from exc


def _concrete_shape(ort_shape, size: int) -> tuple[int, ...]:
    shape = tuple(
        int(value) if isinstance(value, int) and value > 0 else 1
        for value in ort_shape
    )
    if int(np.prod(shape, dtype=np.int64)) != int(size):
        raise ValueError(f"witness size {size} does not match ONNX shape {ort_shape}")
    return shape


def _session(model_path: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def _publish_path(path: Path) -> Path:
    output = path.resolve(strict=False)
    try:
        output.relative_to(EXPERIMENT_ROOT)
    except ValueError as exc:
        raise ValueError(f"output must remain below {EXPERIMENT_ROOT}: {output}") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def build_evidence_ledger(
    universe_path: Path,
    sidecar_root: Path,
    *,
    expected_witnesses: int,
    source_results_csv: Path | None = None,
) -> dict[str, object]:
    """Replay every winning sidecar and return a 200-row evidence ledger."""

    universe, universe_bytes = _load_universe(universe_path)
    family = str(universe["family"])
    bench_root = Path(universe["source_benchmark_root"]).resolve(strict=True)
    family_root = (bench_root / family).resolve(strict=True)
    rows = sorted(universe["instances"], key=lambda row: int(row["row_index"]))
    if len(rows) != int(universe["instance_count"]):
        raise ValueError("universe instance count mismatch")

    by_content: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_content[(row["model_sha256"], row["spec_sha256"])].append(row)

    sidecar = sidecar_root.resolve(strict=True)
    artifact_paths = _artifact_jsons(sidecar)
    if len(artifact_paths) != expected_witnesses:
        raise ValueError(
            f"expected {expected_witnesses} witness artifacts, found {len(artifact_paths)}"
        )

    mapped: list[tuple[dict, dict, Path]] = []
    seen_instance_keys: set[str] = set()
    for artifact_path in artifact_paths:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        if artifact.get("sidecar_verdict") != "sat_zero_tol":
            raise ValueError(f"non-zero-tolerance artifact: {artifact_path}")
        historical_zero = artifact.get("spec_result_zero_tol")
        if not isinstance(historical_zero, dict) or not historical_zero.get("ast_holds"):
            raise ValueError(f"artifact lacks historical zero-tolerance pass: {artifact_path}")
        witness_id = artifact.get("witness_id")
        if not isinstance(witness_id, dict) or witness_id.get("benchmark") != family:
            raise ValueError(f"artifact family mismatch: {artifact_path}")
        historical_model = Path(artifact["model_path"]).resolve(strict=True)
        historical_spec = Path(artifact["spec_path"]).resolve(strict=True)
        model_sha256 = _sha256_file(historical_model)
        spec_sha256 = _sha256_file(historical_spec)
        if model_sha256 != artifact.get("model_sha256"):
            raise ValueError(f"historical model hash mismatch: {artifact_path}")
        if spec_sha256 != artifact.get("spec_sha256"):
            raise ValueError(f"historical spec hash mismatch: {artifact_path}")
        matches = by_content.get((model_sha256, spec_sha256), [])
        if len(matches) != 1:
            status = "UNMATCHED_CONTENT" if not matches else "AMBIGUOUS_CONTENT"
            raise ValueError(f"{status}: {artifact_path}")
        row = matches[0]
        instance_key = str(row["instance_key"])
        if instance_key in seen_instance_keys:
            raise ValueError(f"duplicate current instance witness: {instance_key}")
        seen_instance_keys.add(instance_key)
        mapped.append((row, artifact, artifact_path))

    grouped: dict[str, list[tuple[dict, dict, Path]]] = defaultdict(list)
    for item in mapped:
        grouped[str(item[0]["model_sha256"])].append(item)

    import onnxruntime as ort

    evidence_by_key: dict[str, dict[str, object]] = {}
    for model_sha256 in sorted(grouped):
        group = sorted(grouped[model_sha256], key=lambda item: int(item[0]["row_index"]))
        model_path = _contained_file(family_root, group[0][0]["model_relative_path"])
        if _sha256_file(model_path) != model_sha256:
            raise ValueError(f"current model hash mismatch: {model_path}")
        session = _session(model_path)
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("witness validator requires one ONNX input and output")
        input_meta = inputs[0]
        dtype = _input_dtype(input_meta.type)
        for row, artifact, artifact_path in group:
            relative_artifact = artifact_path.relative_to(sidecar).as_posix()
            x_path = _contained_file(artifact_path.parent, artifact["x_star_npy"])
            x_original = np.load(x_path, allow_pickle=False)
            x_array_sha256 = _sha256_bytes(
                np.ascontiguousarray(x_original).tobytes(order="C")
            )
            if x_array_sha256 != artifact.get("x_star_sha256"):
                raise ValueError(
                    f"historical witness payload hash mismatch: {artifact_path}"
                )
            if not np.all(np.isfinite(x_original)):
                raise ValueError(f"non-finite witness: {artifact_path}")
            input_shape = _concrete_shape(input_meta.shape, x_original.size)
            x_feed = np.asarray(x_original, dtype=dtype).reshape(input_shape)
            if not np.all(np.isfinite(x_feed)):
                raise ValueError(f"non-finite cast witness: {artifact_path}")
            raw_outputs = session.run(None, {input_meta.name: x_feed})
            if len(raw_outputs) != 1:
                raise ValueError("ONNX Runtime returned an unexpected output count")
            y_current = np.asarray(raw_outputs[0], dtype=np.float64).reshape(-1)
            if not np.all(np.isfinite(y_current)):
                raise ValueError(f"non-finite ORT output: {artifact_path}")

            spec_path = _contained_file(family_root, row["spec_relative_path"])
            if _sha256_file(spec_path) != row["spec_sha256"]:
                raise ValueError(f"current spec hash mismatch: {spec_path}")
            replay = evaluate_vnnlib(
                spec_path.read_text(encoding="utf-8"),
                np.asarray(x_feed, dtype=np.float64).reshape(-1),
                y_current,
            )
            if not replay.input_holds:
                raise ValueError(f"independent input-domain replay failed: {artifact_path}")
            if not replay.unsafe_holds:
                raise ValueError(f"independent UNSAFE replay failed: {artifact_path}")

            y_stored_path = _contained_file(artifact_path.parent, artifact["y_ort_npy"])
            y_stored = np.asarray(
                np.load(y_stored_path, allow_pickle=False), dtype=np.float64
            ).reshape(-1)
            if not np.all(np.isfinite(y_stored)) or y_stored.shape != y_current.shape:
                raise ValueError(f"invalid stored historical output: {artifact_path}")
            key = str(row["instance_key"])
            evidence_by_key[key] = {
                "baseline_verdict": "VALIDATED_ADV",
                "baseline_credit": "EXTERNAL_RETENTION_ANCHOR",
                "neural_hz_gain_credit": False,
                "evidence_origin": "LEGACY_SIDECAR_ATTACK_WITNESS",
                "content_match_status": "EXACT_UNIQUE",
                "problem_content_key": problem_content_key(
                    family, row["model_sha256"], row["spec_sha256"]
                ),
                "source_artifact_relative_path": relative_artifact,
                "source_artifact_sha256": _sha256_file(artifact_path),
                "source_witness_relative_path": x_path.relative_to(sidecar).as_posix(),
                "source_witness_file_sha256": _sha256_file(x_path),
                "source_witness_array_sha256": x_array_sha256,
                "source_stored_output_relative_path": y_stored_path.relative_to(
                    sidecar
                ).as_posix(),
                "source_stored_output_sha256": _sha256_file(y_stored_path),
                "historical_instance_id": witness_id.get("instance_id"),
                "historical_repair_applied": bool(artifact.get("repair_applied")),
                "historical_repair_method": artifact.get("repair_method"),
                "replay_status": "ORT_VNNLIB_PASS_ZERO_TOL",
                "ort_version": ort.__version__,
                "execution_provider": "CPUExecutionProvider",
                "ort_input_name": input_meta.name,
                "ort_input_dtype": input_meta.type,
                "input_shape": list(input_shape),
                "all_values_finite": True,
                "input_min_slack": replay.input_min_slack,
                "unsafe_min_slack": replay.unsafe_min_slack,
                "input_assertion_count": replay.input_assertion_count,
                "unsafe_assertion_count": replay.unsafe_assertion_count,
                "current_output_sha256": _sha256_bytes(y_current.tobytes(order="C")),
                "stored_output_array_equal": bool(np.array_equal(y_current, y_stored)),
                "stored_output_max_abs_diff": float(
                    np.max(np.abs(y_current - y_stored), initial=0.0)
                ),
                "failure_reason": None,
            }
        del session

    if len(evidence_by_key) != expected_witnesses:
        raise ValueError("validated evidence count does not match the pre-registered count")
    ledger_rows = []
    for row in rows:
        key = str(row["instance_key"])
        base = {
            "family": family,
            "row_index": int(row["row_index"]),
            "instance_key": key,
            "problem_content_key": problem_content_key(
                family, row["model_sha256"], row["spec_sha256"]
            ),
            "model_sha256": row["model_sha256"],
            "spec_sha256": row["spec_sha256"],
            "timeout_seconds": row["timeout_seconds"],
        }
        evidence = evidence_by_key.get(key)
        if evidence is None:
            base.update(
                {
                    "baseline_verdict": "UNKNOWN",
                    "baseline_credit": "NONE",
                    "neural_hz_gain_credit": False,
                    "historical_evidence_status": "NO_REVALIDATED_WITNESS",
                    "evidence": None,
                }
            )
        else:
            base.update(
                {
                    "baseline_verdict": "VALIDATED_ADV",
                    "baseline_credit": "EXTERNAL_RETENTION_ANCHOR",
                    "neural_hz_gain_credit": False,
                    "historical_evidence_status": "INDEPENDENT_REPLAY_PASS",
                    "evidence": evidence,
                }
            )
        ledger_rows.append(base)

    tree = _evidence_tree(sidecar)
    result_csv_record = None
    if source_results_csv is not None:
        csv_path = source_results_csv.resolve(strict=True)
        result_csv_record = {
            "path": str(csv_path),
            "size_bytes": int(csv_path.stat().st_size),
            "sha256": _sha256_file(csv_path),
        }
    validator_path = Path(__file__).resolve()
    validator_sources = []
    for dependency in (
        validator_path,
        validator_path.with_name("independent_vnnlib_replay.py"),
        validator_path.with_name("freeze_family_manifest.py"),
    ):
        validator_sources.append(
            {
                "relative_path": dependency.relative_to(EXPERIMENT_ROOT).as_posix(),
                "size_bytes": int(dependency.stat().st_size),
                "sha256": _sha256_file(dependency),
            }
        )
    ort_session_config = {
        "providers": ["CPUExecutionProvider"],
        "intra_op_num_threads": 1,
        "inter_op_num_threads": 1,
        "graph_optimization_level": "ORT_ENABLE_ALL",
    }
    ledger: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "namespace": f"{NAMESPACE}/{family}@{universe['ordered_universe_sha256']}/evidence_baseline_v2",
        "ledger_kind": "EXTERNAL_CONTENT_KEY_EVIDENCE_BASELINE",
        "not_alias_of": [
            "historical_large_classification_59_of_400",
            "formal_13_family_1870_of_2413",
        ],
        "neural_hz_baseline_claim": False,
        "family": family,
        "source_universe_file_sha256": _sha256_bytes(universe_bytes),
        "ordered_universe_sha256": universe["ordered_universe_sha256"],
        "source_sidecar_root": str(sidecar),
        "source_sidecar_file_count": len(tree),
        "source_sidecar_files_sha256": _sha256_json(tree),
        "source_results_csv": result_csv_record,
        "validator_sources": validator_sources,
        "validator_sources_sha256": _sha256_json(validator_sources),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "onnxruntime_version": ort.__version__,
        "ort_session_config": ort_session_config,
        "ort_session_config_sha256": _sha256_json(ort_session_config),
        "zero_tolerance": 0.0,
        "instance_count": len(ledger_rows),
        "cert_count": 0,
        "validated_adv_count": len(evidence_by_key),
        "unknown_count": len(ledger_rows) - len(evidence_by_key),
        "invalid_adv_count": 0,
        "rows": ledger_rows,
        "rows_payload_sha256": _sha256_json(ledger_rows),
    }
    ledger["manifest_payload_sha256"] = _sha256_json(ledger)
    return ledger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe-manifest", type=Path, required=True)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--source-results-csv", type=Path)
    parser.add_argument("--expected-witnesses", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.expected_witnesses <= 0:
        raise ValueError("expected_witnesses must be positive")
    ledger = build_evidence_ledger(
        args.universe_manifest,
        args.sidecar_root,
        expected_witnesses=args.expected_witnesses,
        source_results_csv=args.source_results_csv,
    )
    output = _publish_path(args.output)
    exclusive_atomic_write(
        output,
        (json.dumps(ledger, indent=2, sort_keys=True) + "\n").encode("ascii"),
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "family": ledger["family"],
                "validated_adv_count": ledger["validated_adv_count"],
                "unknown_count": ledger["unknown_count"],
                "invalid_adv_count": ledger["invalid_adv_count"],
                "manifest_payload_sha256": ledger["manifest_payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
