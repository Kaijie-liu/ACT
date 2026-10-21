"""Opt-in, standard-library specifications for the nonlinear pair baseline.

Outputs are [F_pair, all router scores], with one UNSAFE AND per property.
Pairs/properties form an OR. No backend import or automatic execution occurs.
The receiver checks coverage/identity, not the numerical verifier's mathematics.
"""
from fractions import Fraction
import hashlib
from itertools import combinations
import json
import math
import re
import time

SCHEMA = "GENBAB_PAIR_SPEC_V1"
ENGINE = "BAB_NONLINEAR_CPU_FLOAT64_V1"
MODES = ("whole_box", "guarded")
STATUSES = ("UNSAT", "SAT", "UNKNOWN", "TIMEOUT", "ERROR")
STAGES = ("incomplete_only", "nonlinear_bab", "not_started")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def tick(deadline):
    require(type(deadline) in (int, float) and math.isfinite(deadline), "finite deadline")
    if time.monotonic() >= deadline:
        raise TimeoutError("same request deadline")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def number(value):
    require(type(value) in (int, float, str, Fraction), "explicit finite coefficient")
    if type(value) is float:
        require(math.isfinite(value), "finite coefficient")
    if type(value) is str:
        require(len(value) <= 2500, "bounded coefficient text")
    result = Fraction(value)
    require(max(abs(result.numerator).bit_length(), result.denominator.bit_length()) <= 4096,
            "original coefficient bit limit")
    return result


def make_request(*, model_sha256, input_sha256, graph_sha256, experts, classes,
                 properties, mode, deadline):
    """Bind a COMPLETE all-pair roster; no pruning authority is inferred."""
    tick(deadline)
    for identity in (model_sha256, input_sha256, graph_sha256):
        require(type(identity) is str and re.fullmatch(r"[0-9a-f]{64}", identity),
                "model/input/graph identity")
    require(type(experts) is int and experts >= 2, "expert dimension")
    require(type(classes) is int and classes >= 1, "output dimension")
    require(mode in MODES, "explicit whole-box or guarded mode")
    require(type(properties) in (list, tuple) and bool(properties), "nonempty property roster")
    parsed = []
    for prop in properties:
        tick(deadline)
        require(type(prop) is dict and set(prop) == {"id", "q", "c"}, "property schema")
        require(type(prop["id"]) is str and 0 < len(prop["id"]) <= 128, "property id")
        require(type(prop["q"]) in (list, tuple) and len(prop["q"]) == classes,
                "all output coordinates retained")
        parsed.append(dict(id=prop["id"], q=[str(number(x)) for x in prop["q"]],
                           c=str(number(prop["c"]))))
    require(len({p["id"] for p in parsed}) == len(parsed), "unique property ids")
    return dict(schema=SCHEMA, model_sha256=model_sha256, input_sha256=input_sha256,
                graph_sha256=graph_sha256, experts=experts, classes=classes,
                properties=parsed, mode=mode, target="qF+c>0", tie="ANY_LEGAL_TOP2",
                pairs=[list(p) for p in combinations(range(experts), 2)], engine=ENGINE)


def validate_request(request, deadline):
    tick(deadline)
    require(type(request) is dict, "request dictionary")
    rebuilt = make_request(model_sha256=request["model_sha256"],
                           input_sha256=request["input_sha256"],
                           graph_sha256=request["graph_sha256"], experts=request["experts"],
                           classes=request["classes"], properties=request["properties"],
                           mode=request["mode"], deadline=deadline)
    require(request == rebuilt, "request schema/target/complete-pair binding")


def clauses(request, *, deadline):
    """Return raw exact C Z <= rhs clauses, not a safe-property DSL expression."""
    validate_request(request, deadline)
    width = request["classes"] + request["experts"]
    binding = digest(request)
    result = []
    for pair in request["pairs"]:
        tick(deadline)
        guards = []
        if request["mode"] == "guarded":
            for selected in pair:
                for outside in range(request["experts"]):
                    if outside not in pair:
                        row = ["0"] * width
                        row[request["classes"] + outside] = "1"
                        row[request["classes"] + selected] = "-1"
                        guards.append(row)
        for prop in request["properties"]:
            tick(deadline)
            value = dict(request_sha256=binding, pair=list(pair), property_id=prop["id"],
                         output_layout="F_then_all_router_scores", logic="UNSAFE_AND",
                         C=guards + [prop["q"] + ["0"] * request["experts"]],
                         rhs=["0"] * len(guards) + [str(-number(prop["c"]))])
            # Independent rows per exported clause; no mutable shared guard lists.
            result.append(json.loads(json.dumps(value)))
    return result


def raw_binary64(request, clause, *, deadline):
    """Reject inexact conversion rather than silently changing the bad set.

    The external caller must separately bind torch.float64 and exact input
    endpoints. Classification coefficients are exactly representable.
    """
    validate_clause(request, clause, deadline=deadline)
    def cast(value):
        tick(deadline)
        exact = number(value)
        rounded = float(exact)
        require(math.isfinite(rounded) and Fraction(rounded) == exact,
                "coefficient not exactly representable as binary64")
        return rounded
    return [[cast(v) for v in row] for row in clause["C"]], [cast(v) for v in clause["rhs"]]


def validate_clause(request, clause, *, deadline):
    """Recompute signs, constants, coordinates, pairs and Boolean grouping."""
    require(type(clause) is dict, "clause dictionary")
    expected = clauses(request, deadline=deadline)
    matches = [row for row in expected if row["pair"] == clause.get("pair")
               and row["property_id"] == clause.get("property_id")]
    require(len(matches) == 1 and matches[0] == clause, "clause source/semantics binding")
    return digest(clause)


def receive(request, receipts, *, deadline):
    """All-pair/all-property structural reception. SAT needs separate replay.

    The future runtime adapter supplies solver receipts. These are not rational
    certificates and cannot elevate the result to formal SAFE. Missing rows
    stay UNKNOWN even if every received row is positive.
    """
    expected = clauses(request, deadline=deadline)
    by_hash = {digest(row): row for row in expected}
    require(type(receipts) is list, "receipt list")
    seen = set()
    counts = {status: 0 for status in STATUSES}
    stages = {stage: 0 for stage in STAGES}
    for receipt in receipts:
        tick(deadline)
        require(type(receipt) is dict and set(receipt) == {
            "request_sha256", "clause_sha256", "engine", "stage", "status"},
            "receipt schema")
        require(receipt["request_sha256"] == digest(request), "receipt request identity")
        key = receipt["clause_sha256"]
        require(key in by_hash and key not in seen, "unknown/duplicate obligation")
        require(receipt["engine"] == ENGINE, "explicit nonlinear CPU binary64 engine")
        status, stage = receipt["status"], receipt["stage"]
        require(status in STATUSES and stage in STAGES, "known status/stage")
        require(stage != "not_started" or status in {"UNKNOWN", "TIMEOUT", "ERROR"},
                "unstarted query cannot decide")
        seen.add(key)
        counts[status] += 1
        stages[stage] += 1
    missing = sorted(set(by_hash) - seen)
    if counts["ERROR"]:
        status = "ERROR"
    elif counts["SAT"]:
        status = "CANDIDATE_REPLAY_REQUIRED"
    elif not missing and counts["UNSAT"] == len(expected):
        status = "POSITIVE"
    elif counts["TIMEOUT"]:
        status = "TIMEOUT"
    else:
        status = "UNKNOWN"
    tick(deadline)
    return dict(status=status, evidence_grade="GENBAB_NUMERICAL_FILTER", formal_SAFE=False,
                request_sha256=digest(request), expected=len(expected), received=len(seen),
                missing=missing, counts=counts, stages=stages,
                scope="ALL_PAIRS_WHOLE_BOX_SUFFICIENT" if request["mode"] == "whole_box"
                else "ALL_TIE_LEGAL_GUARDED_OBLIGATIONS")
