"""Independent structural audit for one staged-verifier evidence package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from act.back_end.moe import load_output_moe_checkpoint
from act.util.device_manager import initialize_device


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
                if row.get("solver_bound_kind") == "scoped_tier1_interval":
                    try:
                        from act.pipeline.moe.scoped_f0_proofs import audit_reused_property
                        audit_reused_property(row, pair["pair"], evidence)
                    except Exception as exc:
                        issues.append(f"invalid scoped property reuse: {exc}")
                    continue
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
    elif verdict.get("decision_tier") == "MONOLITHIC_F0":
        _record_issue(issues, tier2.get("invoked") is True and tier2.get("status") == "SAFE",
                      "monolithic SAFE lacks successful invocation")
        supplied = [tuple(pair) for pair in tier2.get("feasible_route_sets", [])]
        _record_issue(issues, supplied == canonical_sets and bool(supplied),
                      "monolithic branches differ from exact route coverage")
        rows = tier2.get("property_rows", [])
        classes = int(evidence["identity"]["property"]["classes"])
        _record_issue(issues, len(rows) == classes - 1 and
                      {r.get("property_index") for r in rows} == set(range(classes - 1)),
                      "monolithic property coverage incomplete")
        tolerance = float(evidence["numerical_safety"]["safe_positive_margin"])
        for row in rows:
            partition = row.get("coverage_partition")
            if partition is not None:
                try:
                    _audit_monolithic_partition(row, canonical_sets, evidence)
                except Exception as exc:
                    issues.append(f"invalid monolithic proof partition: {exc}")
                continue
            minimum = row.get("minimum")
            _record_issue(issues, row.get("status") == "SAFE" and minimum is not None
                          and math.isfinite(float(minimum)) and float(minimum) > tolerance
                          and row.get("solver_status") == 0
                          and row.get("solver_bound_kind") in {"lp_status0_optimum", "mip_dual_bound"}
                          and row.get("pair_count") == len(supplied)
                          and row.get("full_model_witness_valid") is False,
                          "monolithic SAFE lacks accepted complete bound")
    else:
        issues.append("SAFE has an unknown decision tier")

    method = evidence.get("algorithm", {}).get("comparison_method", "staged")
    if method == "route_invariance":
        _record_issue(issues, len(canonical_sets) == 1,
                      "route-invariance SAFE has multiple legal sets")
    if method == "tier1_only":
        _record_issue(issues, tier2.get("invoked") is False,
                      "Tier-1-only SAFE invokes fallback")


def _audit_monolithic_partition(row, canonical_sets, evidence):
    from act.pipeline.moe.scoped_f0_proofs import audit_reused_property
    part = row["coverage_partition"]
    reused = part["reused"]
    proved_pairs = [tuple(v["pair"]) for v in reused]
    solved_pairs = [tuple(v) for v in part["solved_pairs"]]
    combined = proved_pairs + solved_pairs
    if len(set(combined)) != len(combined) or set(combined) != set(canonical_sets):
        raise ValueError("overlap, duplicate or missing route obligation")
    tolerance = float(evidence["numerical_safety"]["safe_positive_margin"])
    minimum = row.get("minimum")
    if (row.get("status") != "SAFE" or minimum is None or not math.isfinite(float(minimum))
            or minimum <= tolerance or row.get("full_model_witness_valid") is not False):
        raise ValueError("invalid global property acceptance")
    for value in reused:
        proof = value["proof"]
        if proof["property_index"] != row["property_index"]:
            raise ValueError("wrong reused property")
        audit_reused_property(proof, value["pair"], evidence)
        if minimum > proof["accepted_minimum"]:
            raise ValueError("combined bound exceeds reused branch bound")
    if row.get("pair_count") != len(solved_pairs):
        raise ValueError("solver obligation count mismatch")
    if solved_pairs:
        if row.get("solver_status") != 0 or row.get("solver_bound_kind") not in {"lp_status0_optimum", "mip_dual_bound"}:
            raise ValueError("residual branches lack accepted solver bound")
    elif not reused or row.get("solver_bound_kind") != "scoped_pair_partition" or row.get("solver_status") is not None:
        raise ValueError("all-reused branch lacks a complete proof partition")


def _audit_schedule(evidence, issues):
    schedule = evidence.get("route_complexity_schedule")
    if schedule is None:
        return
    try:
        from act.pipeline.moe.route_complexity_schedule import validate_schedule
        validate_schedule({"route_complexity_schedule": schedule["config"],
                           "scoped_proof_reuse": evidence["proof_reuse"]["enabled"],
                           "comparison_method": evidence["algorithm"]["comparison_method"]})
        coverage = evidence["route_coverage"]
        if coverage["coverage_complete"]:
            count = len(coverage["feasible_route_sets"])
            expected = ("MONOLITHIC_MATCHED" if evidence["algorithm"]["comparison_method"] == "monolithic_f0"
                        else "SINGLE_PAIR_DIRECT" if count == 1 else "MULTI_PAIR_STAGED")
            if schedule["selected_path"] != expected or schedule["exact_pair_count"] != count:
                raise ValueError("unregistered route-complexity decision")
        if evidence["verdict"]["status"] == "SAFE":
            if not schedule["common_fact_prelude_complete"]:
                raise ValueError("SAFE before common fact prelude completion")
            if schedule["selected_path"] != "MULTI_PAIR_STAGED" and evidence["verdict"]["decision_tier"] != "MONOLITHIC_F0":
                raise ValueError("single/matched path did not use weighted obligations")
        budget = schedule["budget"]
        if budget["total_seconds"] != schedule["config"]["total_seconds"]:
            raise ValueError("total budget mismatch")
        if not 0 <= budget["remaining_seconds"] <= budget["total_seconds"]:
            raise ValueError("invalid remaining budget")
        previous = -1.0
        for event in budget["events"]:
            elapsed, remaining, grant = (event[k] for k in ("elapsed_seconds", "remaining_seconds", "granted_seconds"))
            if not all(math.isfinite(float(v)) for v in (elapsed, remaining, grant)):
                raise ValueError("nonfinite budget event")
            if elapsed < previous or not 0 < grant <= remaining <= budget["total_seconds"]:
                raise ValueError("invalid budget event")
            if elapsed + remaining > budget["total_seconds"] + 1e-5:
                raise ValueError("budget was reset")
            previous = elapsed
        if evidence["verdict"]["status"] == "SAFE" and budget["remaining_seconds"] <= 0:
            raise ValueError("late SAFE after deadline")
    except Exception as exc:
        issues.append(f"invalid route-complexity schedule: {exc}")


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
    _audit_schedule(evidence, issues)
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
                initialize_device("cpu", "float64")
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
