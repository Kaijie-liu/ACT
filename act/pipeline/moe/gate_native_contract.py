"""Opt-in M1 native accounting; no new mathematical acceptance rule.

The Work unit covers signed-fact capture/combination/check, not upstream HZ
instructions. Wall-clock supervision must cover those instructions as well.
"""
import time

from scoped_source.hz_request_store import Work, POLICY, require
from act.pipeline.moe.request_budget import BudgetExhausted


class GateCapacity(ValueError):
    """Explicit capacity exhaustion, distinct from a malformed source."""


class GateSourceError(ValueError):
    """Invalid source binding is an execution error, not a solver outcome."""


def source_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except GateCapacity:
        raise
    except ValueError as exc:
        raise GateSourceError(str(exc)) from exc


def validate_work_counts(counts):
    require(set(counts) == {"row_visits", "coefficient_visits", "arithmetic"}
            and all(type(v) is int and v >= 0 for v in counts.values())
            and sum(counts.values()) <= POLICY["visits"], "invalid cumulative Work")


def require_continuation(work, recorded):
    validate_work_counts(recorded)
    validate_work_counts(work.counts)
    require(all(work.counts[k] >= recorded[k] for k in recorded),
            "Work ledger reset before receipt")
    work.tick()


class GateWork(Work):
    def tick(self):
        if time.monotonic() >= self.deadline:
            raise BudgetExhausted("gate fact/check deadline")

    def visit(self, kind, amount=1):
        self.tick()
        require(kind in self.counts and type(amount) is int and amount >= 0,
                "invalid Work visit")
        if sum(self.counts.values()) + amount > POLICY["visits"]:
            raise GateCapacity("gate fact/check Work capacity")
        self.counts[kind] += amount

    def exact(self, q):
        self.visit("arithmetic")
        if max(q.numerator.bit_length(), q.denominator.bit_length()) > POLICY["work_bits"]:
            raise GateCapacity("gate fact/check intermediate rational capacity")
        return q


def terminal(exc):
    if isinstance(exc, BudgetExhausted):
        return {"status": "TIMEOUT", "reason": "REQUEST_BUDGET_EXHAUSTED"}
    if isinstance(exc, GateCapacity):
        return {"status": "UNKNOWN", "reason": "REQUEST_WORK_CAPACITY"}
    raise exc


def validate_counts(counts):
    require(set(counts) == {"attempted", "returned", "status0_returned"}
            and all(type(v) is int and v >= 0 for v in counts.values()),
            "invalid F0 call counters")
    require(counts["status0_returned"] <= counts["returned"] <= counts["attempted"],
            "inconsistent F0 call counters")


class F0Calls:
    """Count actual joint-F0 function entries/returns, NOT support queries."""
    def __init__(self):
        self.counts = dict(attempted=0, returned=0, status0_returned=0)

    def attempted(self):
        self.counts["attempted"] += 1

    def returned(self, status):
        self.counts["returned"] += 1
        self.counts["status0_returned"] += int(status == 0)

    def record(self):
        validate_counts(self.counts)
        return dict(self.counts)


def validate_config(config):
    enabled = config.get("gate_native_contract", False)
    require(type(enabled) is bool, "gate_native_contract must be boolean")
    if enabled:
        schedule = config.get("route_complexity_schedule", {})
        require(schedule.get("total_seconds") == 300
                and schedule.get("multi_pair_tier1_fraction") == .25
                and config.get("scoped_proof_reuse") is True
                and config.get("comparison_method", "staged") in {"staged", "monolithic_f0"},
                "native gate contract requires registered budget/reuse/method")
    return enabled
