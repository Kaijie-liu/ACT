"""Durable pre-solve observations, not standalone network certificates.

Publish once after common preparation and before arm-specific solves. Hashes
bind stored values; route completeness and network-to-HZ propagation remain
trusted. An absent snapshot is unavailable, never proof of equal facts.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from act.pipeline.moe.experiment1 import WRITE_ROOT, _inside
from act.pipeline.moe.scoped_f0_proofs import facts_from_branches, make_scope


POLICY = "COMMON_GUARDED_INTERVAL_NO_SUPPORT_SOLVES"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def build_snapshot(record, reuse, identity, config, elapsed):
    if not record["schedule"]["common_fact_prelude_complete"]:
        raise ValueError("cannot snapshot incomplete prelude")
    body = copy.deepcopy({
        "schema": "common_fact_snapshot_v1", "identity": identity,
        "request_id": digest(identity), "config": config, "scope": reuse["scope"],
        "completion_elapsed_seconds": elapsed,
        "candidate_experts": record["candidate_experts"],
        "feasible_route_sets": record["feasible_route_sets"],
        "route_sets_exact": True,
        "branches": [{k: b[k] for k in ("candidate", "proof_output_bounds", "source_policy")}
                     for b in record["branches"]],
        "available_fact_count": len(reuse["facts"]),
    })
    result = {"payload": body, "payload_sha256": digest(body)}
    check_snapshot(result)
    return result


def check_snapshot(value, *, expected_identity=None, expected_config=None, evidence=None):
    """Check binding and interval-fact arithmetic, without invoking a solver."""
    body = value["payload"]
    if value["payload_sha256"] != digest(body) or body["schema"] != "common_fact_snapshot_v1":
        raise ValueError("snapshot hash/schema mismatch")
    identity, config = body["identity"], body["config"]
    if body["request_id"] != digest(identity) or identity["config_sha256"] != digest(config):
        raise ValueError("snapshot request/config mismatch")
    if expected_identity is not None and identity != expected_identity:
        raise ValueError("snapshot belongs to another request")
    if expected_config is not None and config != expected_config:
        raise ValueError("snapshot belongs to another config")
    elapsed = body["completion_elapsed_seconds"]
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("invalid snapshot completion time")
    scope = make_scope(body["request_id"], identity, body["scope"]["frame_id"], config["numerical_safety"])
    if scope != body["scope"]:
        raise ValueError("snapshot fact scope mismatch")
    candidates, pairs, branches = body["candidate_experts"], body["feasible_route_sets"], body["branches"]
    if (not candidates or any(type(i) is not int or i < 0 for i in candidates)
            or candidates != sorted(set(candidates)) or not pairs or body["route_sets_exact"] is not True):
        raise ValueError("invalid candidate/route inventory")
    if any(len(p) != 2 or p != sorted(set(p)) or not set(p) <= set(candidates) for p in pairs):
        raise ValueError("invalid legal pair")
    if pairs != sorted(map(list, set(map(tuple, pairs)))):
        raise ValueError("duplicate or unordered legal pairs")
    if [b["candidate"] for b in branches] != candidates or any(b["source_policy"] != POLICY for b in branches):
        raise ValueError("missing/duplicate/source-mismatched fact branch")
    facts = facts_from_branches(branches, scope)
    if len(facts) != body["available_fact_count"]:
        raise ValueError("snapshot fact count mismatch")
    if evidence is not None:
        if identity != evidence["identity"] or scope["frame_id"] != evidence["proof_reuse"]["frame_id"]:
            raise ValueError("snapshot/final package identity mismatch")
        from act.pipeline.moe.route_complexity_paired import common_facts
        if fact_view(value) != common_facts(evidence):
            raise ValueError("snapshot/final package facts mismatch")
    return {"status": "PASS", "available_fact_count": len(facts),
            "scope": "Stored prelude identity and interval arithmetic; not independent route/network proof."}


def fact_view(value):
    body = value["payload"]
    return {"candidates": body["candidate_experts"], "pairs": body["feasible_route_sets"],
            "facts": [{"expert": b["candidate"], "bounds": b["proof_output_bounds"],
                       "source_policy": b["source_policy"]} for b in body["branches"]],
            "available_count": body["available_fact_count"]}


def publish_snapshot(path, value):
    """Atomic no-clobber publication. A killed writer cannot expose half JSON."""
    path = _inside(Path(path), WRITE_ROOT)
    check_snapshot(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        # link(), unlike replace(), fails if another observation already exists.
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)
