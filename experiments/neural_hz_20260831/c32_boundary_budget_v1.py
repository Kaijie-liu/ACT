"""Explicit native payload vs incremental generator work, same boundary.

Native transfers are paid and reported separately from the historical generator
budget. The forwarding adapter is needed ONLY for the ~300 writer charges;
discovery/lineage use a local pool capped by remaining coupled capacity, then
debit exactly once. No per-row wrapper overhead is hidden or double counted.
"""

from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool


class WriterPool:
    def __init__(self,coupled,*,payload_cap):
        if type(payload_cap) is not int or not 0<=payload_cap<=256_000_000:
            raise ValueError('invalid unchanged native payload bound')
        self.coupled=coupled
        self.native=WorkPool(payload_cap)
        self.used=0;self.parts={};self.dispatch_work=0

    def charge(self,name,amount):
        # Extra adapter name dispatch and dual-ledger bookkeeping. C30's own
        # tariffs remain untouched. No blanket wrapper on its large discovery.
        self.coupled.charge('native_writer_budget_dispatch',4)
        self.dispatch_work+=4
        if name.startswith('native_'):
            self.native.charge(name,amount)
        else:
            self.coupled.charge(name,amount)
        self.used+=amount;self.parts[name]=self.parts.get(name,0)+amount


def remaining_pool(coupled):
    return WorkPool(coupled.capacity-coupled.used)


def finish_local(coupled,local,name):
    # The local cap was the remaining global/branch capacity at entry. With
    # this sequential single transaction, successful local work is prepaid at
    # every operation and is now accounted globally exactly once.
    coupled.charge(name,local.used)
