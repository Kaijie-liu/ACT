"""Opt-in 2026-10-21 development accounting; never changes legacy Work.

This object does not install process limits or authorize a run. A separately
frozen supervisor must bind the same policy and enforce the 300s hard budget.
"""
from scoped_source.hz_request_store import Work, require

POLICY = "DEV_MARGIN_20261021_V1"


class DevelopmentWork(Work):
    """Same deadline/exact-bit checks, with accumulated visits but no 8M gate."""

    def __init__(self, deadline, *, policy):
        require(policy == POLICY, "explicit frozen development policy")
        super().__init__(deadline)
        self._continued = False

    def visit(self, kind, amount=1):
        self.tick()
        require(kind in self.counts and type(amount) is int and amount >= 0,
                "nonnegative development Work accounting")
        self.counts[kind] += amount

    def continue_from(self, counts):
        """One-time reception continuation; caller must bind the worker receipt."""
        self.tick()
        require(not self._continued and all(v == 0 for v in self.counts.values()),
                "cannot reset existing Work")
        require(type(counts) is dict and set(counts) == set(self.counts)
                and all(type(v) is int and v >= 0 for v in counts.values()),
                "complete continued Work counters")
        self.counts = dict(counts)
        self._continued = True
        self.tick()
