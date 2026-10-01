"""Independent reception of support bounds for a *given* serialized HybridZ.

No candidate optimizer or HZ exporter is called here. This is not a network
lowering, route-coverage or native floating-point proof. Imports through ACT's
package namespace are not a solver-free portable entry point.
"""
from fractions import Fraction

from act.back_end.solver.check_hz_lp_export import check_export
from scoped_source.rowwise_bound import check_bound, clock, identity, rational


SCHEMA = "HZ_SUPPORT_BATCH_V1"
SCOPE = "CHECKED_GIVEN_HZ_CONTINUOUS_RELAXATION"


def validated_records(batch, *, expected_batch_sha256, deadline):
    """Check every objective against the anchored source and original factors."""
    tick = clock(deadline)
    if identity(batch) != expected_batch_sha256 or batch.get("schema") != SCHEMA:
        raise ValueError("support batch identity/schema")
    context = batch.get("context")
    if (type(context) is not dict or not context
            or any(type(context.get(k)) is not str or not context[k]
                   for k in ("request", "domain", "guard"))):
        raise ValueError("explicit caller-owned request/domain/guard identity required")
    source = batch["source"]
    if identity(source) != batch["source_sha256"]:
        raise ValueError("support source identity")
    base = batch["base"]
    n = len(base["lower"])
    if (not 1 <= n <= 128 or len(source["c"]) > 128
            or len(base["b"]) + len(base["h"]) > 256):
        raise ValueError("finite control capacity exceeded")
    queries = batch["queries"]
    if type(queries) is not list or not 1 <= len(queries) <= 8:
        raise ValueError("finite nonempty query roster required")
    seen = set()
    records = []
    for query in queries:
        tick()
        key = query["id"]
        if type(key) is not str or not key or key in seen:
            raise ValueError("duplicate or empty support query id")
        seen.add(key)
        if query["side"] not in ("min", "max"):
            raise ValueError("support side must be min/max")
        sign = 1 if query["side"] == "min" else -1
        q = [str(sign * rational(v)) for v in query["q"]]
        offset = str(sign * rational(query["offset"]))
        lp = dict(base, c=query["c"], offset=query["constant"])
        record = {"source": source, "source_sha256": batch["source_sha256"],
                  "q": q, "offset": offset, "lp": lp,
                  "relaxation": batch["relaxation"],
                  "n_relaxed_binaries": batch["n_relaxed_binaries"]}
        check_export(record, expected_source_sha256=batch["source_sha256"])
        tick()
        records.append((query, lp))
    if identity(batch) != expected_batch_sha256:
        raise ValueError("support batch changed during validation")
    tick()
    return records


def check_batch(batch, candidates, *, expected_batch_sha256, deadline):
    """All roster entries must pass; incomplete batches never receive this status."""
    tick = clock(deadline)
    candidate_hash = identity(candidates)
    records = validated_records(batch, expected_batch_sha256=expected_batch_sha256,
                                deadline=deadline)
    if candidates.get("batch_sha256") != expected_batch_sha256:
        raise ValueError("candidate batch binding")
    entries = candidates["entries"]
    if type(entries) is not list or len(entries) != len(records):
        raise ValueError("incomplete support certificate roster")
    by_id = {}
    for item in entries:
        key = item["id"]
        if key in by_id:
            raise ValueError("duplicate support certificate")
        by_id[key] = item
    if set(by_id) != {query["id"] for query, _ in records}:
        raise ValueError("wrong support certificate roster")
    results = []
    for query, lp in records:
        checked = check_bound(lp, by_id[query["id"]]["certificate"], deadline=deadline)
        lower = Fraction(checked["checked_lower_bound"])
        bound = lower if query["side"] == "min" else -lower
        results.append({"id": query["id"], "side": query["side"],
                        "bound": str(bound), "bound_kind": "lower" if query["side"] == "min" else "upper",
                        "lp_sha256": checked["lp_sha256"],
                        "rows_checked": checked["rows_checked"],
                        "entries_checked": checked["entries_checked"]})
    if identity(batch) != expected_batch_sha256 or identity(candidates) != candidate_hash:
        raise ValueError("support input changed during reception")
    tick()
    return {"status": SCOPE, "batch_sha256": expected_batch_sha256,
            "candidate_sha256": candidate_hash, "results": results,
            "n_relaxed_binaries": batch["n_relaxed_binaries"],
            "hard_budget_supervision": False,
            "trusted": ["supplied HZ and factor semantics", "caller request/domain/guard binding"],
            "network_or_complete_moe_proof": False}
