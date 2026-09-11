"""Cooperative request deadline; callers still need an outer process watchdog.

Native propagation/solver calls cannot be preempted here. Never pass zero to a
solver as a stand-in for exhaustion (some interfaces interpret it as unlimited).
"""
import math
import time


class BudgetExhausted(TimeoutError):
    pass


class RequestBudget:
    def __init__(self, seconds, *, started=None, clock=None):
        self.clock = clock or time.monotonic
        now = self.clock()
        self.started = now if started is None else float(started)
        if not math.isfinite(self.started) or self.started > now:
            raise ValueError("budget start must be a finite past monotonic timestamp")
        self.seconds = float(seconds)
        if not math.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError("request budget must be finite and positive")
        self.deadline = self.started + self.seconds
        self.events = []

    def remaining(self):
        return max(0.0, self.deadline - self.clock())

    def limit(self, stage, cap=None, obligations=1, until=None):
        now = self.clock()
        remaining = max(0.0, min(self.deadline, until or self.deadline) - now)
        if remaining <= 0:
            raise BudgetExhausted(stage)
        if obligations < 1:
            raise ValueError("positive obligation count required")
        value = remaining / obligations
        if cap is not None:
            if not math.isfinite(float(cap)) or cap <= 0:
                raise ValueError("positive finite cap required")
            value = min(value, float(cap))
        self.events.append({"stage": stage, "elapsed_seconds": now - self.started,
                            "remaining_seconds": remaining, "granted_seconds": value,
                            "obligations": obligations})
        return value

    def check(self, stage):
        if self.remaining() <= 0:
            raise BudgetExhausted(stage)

    def record(self):
        return {"total_seconds": self.seconds, "remaining_seconds": self.remaining(),
                "events": list(self.events), "kind": "COOPERATIVE_WITH_EXTERNAL_WATCHDOG",
                "native_calls_preemptible": False}
