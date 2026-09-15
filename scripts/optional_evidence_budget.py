"""One-request clock for an opt-in evidence path, separate from production."""
import math
import time


class EvidenceBudgetExpired(RuntimeError):
    pass


class EvidenceBudget:
    def __init__(self, started, total=300., clock=time.monotonic):
        if not math.isfinite(started) or not math.isfinite(total) or total <= 0 or started > clock():
            raise ValueError('invalid request clock')
        self.started, self.total, self.clock = started, total, clock
        self.deadline = started + total

    def remaining(self, reserve=0.):
        if reserve < 0 or not math.isfinite(reserve):
            raise ValueError('invalid reserve')
        left = self.deadline - self.clock() - reserve
        if left <= .001:
            raise EvidenceBudgetExpired('request evidence budget exhausted')
        return left

    def grant(self, cap, reserve=0.):
        if not math.isfinite(cap) or cap <= 0:
            raise ValueError('invalid proposal cap')
        return min(cap, self.remaining(reserve))


def terminal_status(candidate, elapsed, total, complete_check):
    if elapsed > total:
        return 'TIMEOUT'
    if candidate == 'CHECKED_CONDITIONAL' and not complete_check:
        return 'UNKNOWN_INCOMPLETE_EVIDENCE'
    return candidate
