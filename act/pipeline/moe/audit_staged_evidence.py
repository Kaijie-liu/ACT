"""Independent structural audit for one staged-verifier evidence package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from act.back_end.moe import load_output_moe_checkpoint


ALLOWED_ROOT = Path("/data1/Kane/MOE")


def _inside(path: Path, root: Path = ALLOWED_ROOT) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes allowed root {root}: {resolved}")
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_identity(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "sha256": digest.hexdigest(),
    }


def _record_issue(issues: list[str], condition: bool, message: str) -> None:
    if not condition:
        issues.append(message)


def _load_bound_artifact(
    package: Path,
    record: Mapping[str, Any],
    expected_name: str,
    issues: list[str],
) -> Path | None:
    try:
        path = _inside(Path(str(record["path"])))
    except Exception as exc:
        issues.append(f"invalid {expected_name} path: {exc}")
        return None
    _record_issue(
        issues,
        path == package / expected_name,
        f"{expected_name} is not bound to its package location",
    )
    if not path.is_file():
        issues.append(f"missing {expected_name}")
        return None
    _record_issue(
        issues,
        _file_sha256(path) == record.get("sha256"),
        f"{expected_name} hash mismatch",
    )
    return path


def _audit_safe_structure(evidence: Mapping[str, Any], issues: list[str]) -> None:
    coverage = evidence.get("route_coverage", {})
    _record_issue(
        issues,
        coverage.get("coverage_complete") is True,
        "SAFE lacks complete route coverage",
    )
    _record_issue(
        issues,
        coverage.get("candidate_set_minimal") is True,
        "SAFE lacks minimal exact candidates",
    )
    _record_issue(
        issues,
        coverage.get("route_sets_exact") is True,
        "SAFE lacks exact route sets",
    )

    candidates = coverage.get("candidate_experts") or []
    route_sets = coverage.get("feasible_route_sets") or []
    canonical_sets = [
        tuple(sorted(int(value) for value in pair)) for pair in route_sets
    ]
    _record_issue(
        issues,
        len(canonical_sets) == len(set(canonical_sets)),
        "duplicate feasible route set",
    )
    _record_issue(
        issues,
        all(len(pair) == 2 for pair in canonical_sets),
        "non-top2 route set in v1 evidence",
    )
    _record_issue(
        issues,
        all(value in candidates for pair in canonical_sets for value in pair),
        "feasible route set contains a non-candidate expert",
    )

    verdict = evidence.get("verdict", {})
    tier1 = evidence.get("tier1", {})
    tier2 = evidence.get("tier2", {})
    if verdict.get("decision_tier") == "TIER1_GATE_ELIMINATION":
        _record_issue(
            issues,
            tier1.get("status") == "SAFE",
            "Tier-1 SAFE verdict disagrees with Tier 1",
        )
        branches = tier1.get("branches") or []
        _record_issue(
            issues,
            {row.get("candidate") for row in branches} == set(candidates),
            "Tier-1 SAFE branches do not cover every candidate",
        )
        _record_issue(
            issues,
            all(row.get("unknown_reason") == "SAFE_PROVED" for row in branches),
            "Tier-1 SAFE contains a non-proved branch",
        )
        _record_issue(
            issues,
            tier2.get("invoked") is False,
            "Tier-1 SAFE unexpectedly invokes F0",
        )
    elif verdict.get("decision_tier") == "TIER2_F0":
        _record_issue(issues, tier2.get("invoked") is True, "F0 SAFE did not invoke F0")
        _record_issue(issues, tier2.get("status") == "SAFE", "F0 SAFE verdict disagrees with F0")
        pairs = tier2.get("pairs") or []
        observed_pairs = {tuple(row.get("pair", [])) for row in pairs}
        _record_issue(
            issues,
            observed_pairs == set(canonical_sets),
            "F0 SAFE does not cover exactly the feasible route pairs",
        )
        tolerance = float(evidence["numerical_safety"]["safe_positive_margin"])
        classes = int(evidence["identity"]["property"]["classes"])
        for pair in pairs:
            _record_issue(
                issues,
                pair.get("status") == "SAFE",
                "F0 SAFE contains a non-safe pair",
            )
            rows = pair.get("property_rows") or []
            _record_issue(
                issues,
                len(rows) == classes - 1,
                "F0 pair has incomplete property rows",
            )
            _record_issue(
                issues,
                {row.get("property_index") for row in rows} == set(range(classes - 1)),
                "F0 property indices are incomplete",
            )
            for row in rows:
                accepted = row.get("accepted_minimum")
                _record_issue(
                    issues,
                    row.get("status") == "SAFE"
                    and accepted is not None
                    and float(accepted) > tolerance
                    and row.get("solver_status") == 0
                    and row.get("solver_bound_kind")
                    in {"lp_status0_optimum", "mip_dual_bound"},
                    "F0 SAFE property lacks a strictly accepted certified bound",
                )
                _record_issue(
                    issues,
                    row.get("full_model_witness_valid") is False,
                    "F0 SAFE property also records a violating witness",
                )
    else:
        issues.append("SAFE has an unknown decision tier")


def audit_evidence_package(
    package_dir: Path,
    *,
    replay_unsafe: bool = False,
) -> dict[str, Any]:
    """Recompute package identities and fail closed on malformed verdicts."""
    package = _inside(package_dir)
    issues: list[str] = []
    manifest_path = package / "manifest.json"
    evidence_path = package / "evidence.json"
    if not manifest_path.is_file() or not evidence_path.is_file():
        return {"status": "FAIL", "issues": ["missing manifest or evidence"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    _record_issue(
        issues,
        manifest.get("schema_version") == 1,
        "unsupported manifest schema",
    )
    _record_issue(
        issues,
        evidence.get("schema_version") == 1,
        "unsupported evidence schema",
    )
    try:
        recorded_evidence_path = _inside(Path(str(manifest.get("evidence_path"))))
    except Exception as exc:
        recorded_evidence_path = None
        issues.append(f"invalid evidence path: {exc}")
    _record_issue(
        issues,
        recorded_evidence_path == evidence_path,
        "evidence is not bound to its package location",
    )
    _record_issue(
        issues,
        manifest.get("evidence_sha256") == _file_sha256(evidence_path),
        "evidence hash mismatch",
    )
    _record_issue(
        issues,
        manifest.get("request_id") == evidence.get("request_id"),
        "request id differs across files",
    )
    _record_issue(
        issues,
        evidence.get("request_id")
        == _canonical_sha256(evidence.get("identity")),
        "request identity hash mismatch",
    )
    verdict = evidence.get("verdict", {})
    _record_issue(
        issues,
        manifest.get("status") == verdict.get("status"),
        "manifest status mismatch",
    )
    _record_issue(
        issues,
        manifest.get("reason") == verdict.get("reason"),
        "manifest reason mismatch",
    )

    request_path = _load_bound_artifact(
        package, manifest.get("request", {}), "request.pt", issues
    )
    request = None
    if request_path is not None:
        request = torch.load(request_path, map_location="cpu", weights_only=True)
        _record_issue(
            issues,
            request.get("request_id") == evidence.get("request_id"),
            "request artifact id mismatch",
        )
        for name in ("center", "lower", "upper"):
            value = request.get(name)
            _record_issue(issues, isinstance(value, torch.Tensor), f"request lacks tensor {name}")
            if isinstance(value, torch.Tensor):
                _record_issue(
                    issues,
                    _tensor_identity(value) == evidence["identity"].get(name),
                    f"{name} tensor identity mismatch",
                )
        if all(
            isinstance(request.get(name), torch.Tensor)
            for name in ("center", "lower", "upper")
        ):
            center, lower, upper = request["center"], request["lower"], request["upper"]
            epsilon = float(evidence["request"]["epsilon"])
            _record_issue(
                issues,
                torch.equal(lower, (center - epsilon).clamp(0, 1)),
                "represented lower box mismatch",
            )
            _record_issue(
                issues,
                torch.equal(upper, (center + epsilon).clamp(0, 1)),
                "represented upper box mismatch",
            )

    status = verdict.get("status")
    _record_issue(
        issues,
        status in {"SAFE", "UNSAFE", "UNKNOWN", "TIMEOUT"},
        "unknown verdict status",
    )
    if status == "SAFE":
        _record_issue(
            issues,
            verdict.get("certificate_complete") is True,
            "SAFE is not marked complete",
        )
        _record_issue(
            issues,
            verdict.get("full_model_witness_valid") is False,
            "SAFE conflicts with a witness",
        )
        _record_issue(
            issues,
            manifest.get("witness") is None,
            "SAFE unexpectedly contains a witness artifact",
        )
        _audit_safe_structure(evidence, issues)
    elif status == "UNSAFE":
        _record_issue(
            issues,
            verdict.get("full_model_witness_valid") is True,
            "UNSAFE lacks validated witness flag",
        )
        _record_issue(
            issues,
            manifest.get("witness") is not None,
            "UNSAFE lacks witness artifact",
        )

    witness_path = None
    if manifest.get("witness") is not None:
        witness_path = _load_bound_artifact(package, manifest["witness"], "witness.pt", issues)
        if witness_path is not None:
            saved = torch.load(witness_path, map_location="cpu", weights_only=True)
            _record_issue(
                issues,
                saved.get("request_id") == evidence.get("request_id"),
                "witness request id mismatch",
            )

    if replay_unsafe and status == "UNSAFE":
        checkpoint = evidence.get("identity", {}).get("checkpoint", {})
        checkpoint_path = checkpoint.get("path")
        if request is None or witness_path is None or not checkpoint_path:
            issues.append("UNSAFE replay lacks checkpoint, request, or witness")
        else:
            model_path = _inside(Path(checkpoint_path))
            _record_issue(issues, model_path.is_file(), "checkpoint missing for replay")
            if model_path.is_file():
                _record_issue(
                    issues,
                    _file_sha256(model_path) == checkpoint.get("sha256"),
                    "checkpoint hash mismatch",
                )
                model, _ = load_output_moe_checkpoint(model_path, map_location="cpu")
                model.cpu().double().eval()
                witness = torch.load(witness_path, map_location="cpu", weights_only=True)["input"]
                value = (
                    witness.unsqueeze(0)
                    if witness.shape != request["center"].shape
                    else witness
                )
                _record_issue(
                    issues,
                    bool((value >= request["lower"] - 1e-7).all()),
                    "witness below represented box",
                )
                _record_issue(
                    issues,
                    bool((value <= request["upper"] + 1e-7).all()),
                    "witness above represented box",
                )
                with torch.no_grad():
                    output, _ = model.forward_with_routing(value.double())
                prediction = int(output.argmax(dim=1).item())
                _record_issue(
                    issues,
                    prediction != int(evidence["request"]["clean_prediction"]),
                    "witness does not violate the requested prediction property",
                )

    return {
        "schema_version": 1,
        "package": str(package),
        "request_id": evidence.get("request_id"),
        "verdict": status,
        "replay_unsafe": bool(replay_unsafe),
        "status": "PASS" if not issues else "FAIL",
        "issue_count": len(issues),
        "issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--replay-unsafe", action="store_true")
    args = parser.parse_args()
    result = audit_evidence_package(args.package, replay_unsafe=args.replay_unsafe)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
