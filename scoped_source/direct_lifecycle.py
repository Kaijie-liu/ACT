"""Checked root leases and fixed stdlib proposer. No native dispatch.

Revocable in-process API, not a sandbox against private-memory manipulation.
No writable backing map or Fraction alias is returned to the proposer.
"""
from collections.abc import Mapping, Sequence
from fractions import Fraction as F
import time

from scoped_source.hz_request_store import require, identity
from scoped_source.direct_candidates import Driver
from scoped_source.direct_method import validate


class ReadOnly:
    __slots__ = ()

    def __setattr__(self, name, value):
        raise AttributeError('read-only checked root lease')

    def __delattr__(self, name):
        raise AttributeError('read-only checked root lease')


class RootMap(ReadOnly, Mapping):
    __slots__ = ('_lease', '_row')

    def __init__(self, lease, row):
        object.__setattr__(self, '_lease', lease)
        object.__setattr__(self, '_row', row)

    def __len__(self):
        return len(self._lease._get(self._row)[1])

    def __iter__(self):
        # Do not capture the underlying map in a generator retained by callers.
        keys = tuple(self._lease._get(self._row)[1])
        self._lease._work.visit('coefficient_visits', len(keys))
        for key in keys:
            self._lease._check()
            yield key
        self._lease._check()

    def __getitem__(self, key):
        roots = self._lease._get(self._row)
        self._lease._work.visit('coefficient_visits')
        q = roots[1][key]
        return F(q.numerator, q.denominator)


class RootLease(ReadOnly, Sequence):
    __slots__ = ('_driver', '_roots', '_token', '_tasks', '_ordinal', '_binding', '_work', '_live')

    def __init__(self, *args, **kw):
        raise TypeError('roots issued only by lifecycle prepare')

    def _check(self):
        require(self._live, 'revoked root lease')
        d = self._driver
        require(type(d) is Driver and not d.closed and d.pending is not None
                and d.pending[1] is self._token and d.pending[0] is self._tasks
                and d.ordinal == self._ordinal and d.work is self._work
                and (id(d.forest), d.forest.request, d.forest.identity, d.contract) == self._binding,
                'root lease request/tile/owner binding')
        d.owner.check()
        self._work.tick()

    def _get(self, row):
        self._check()
        require(type(row) is int and 0 <= row < len(self._roots), 'root lease row')
        self._work.visit('row_visits')
        return self._roots[row]

    def __len__(self):
        self._check()
        return len(self._roots)

    def __getitem__(self, row):
        d, _ = self._get(row)
        return F(d.numerator, d.denominator), RootMap(self, row)

    def __iter__(self):
        for row in range(len(self)):
            yield self[row]
        self._check()


def revoke(lease):
    if lease is not None:
        object.__setattr__(lease, '_live', False)
        object.__setattr__(lease, '_roots', None)
        object.__setattr__(lease, '_driver', None)


def prepare(driver):
    """Obtain authoritative roots directly, not from a caller-supplied tuple."""
    require(type(driver) is Driver, 'exact Driver issuer')
    tasks, roots, token = driver.prepare()
    lease = object.__new__(RootLease)
    values = dict(_driver=driver, _roots=roots, _token=token, _tasks=tasks,
                  _ordinal=driver.ordinal, _work=driver.work, _live=True,
                  _binding=(id(driver.forest), driver.forest.request, driver.forest.identity, driver.contract))
    for key, value in values.items():
        object.__setattr__(lease, key, value)
    lease._check()
    return tasks, lease, token


FAULTS = ('', 'setup_error', 'propose_error', 'close_error', 'setup_block',
          'propose_block', 'close_block', 'missing', 'partial', 'stale_roots')


def inject(fault, stage, work):
    if fault == stage + '_error':
        raise RuntimeError('fixed ' + fault)
    if fault == stage + '_block':
        # Intentionally non-cooperative. Only the owned process supervisor can stop it.
        while True:
            time.sleep(.05)


class FixedProposer:
    """Closed deterministic candidate generator; roots read, never optimized."""
    def __init__(self, config, fault=''):
        self.method = validate(config)
        require(fault in FAULTS, 'fixed lifecycle fault')
        self.fault = fault
        self.forest = self.work = self.saved = None
        self.state = 'NEW'
        self.stats = dict(setup_calls=0, propose_calls=0, close_calls=0, root_terms=0)

    def setup(self, forest, work):
        require(self.state == 'NEW', 'setup exactly once')
        self.state = 'STARTING'
        self.forest, self.work = forest, work
        self.stats['setup_calls'] += 1
        from scoped_source.check_scaled_references import checked
        checked(forest, work)
        from scoped_source.bounded_runtime import shared_source
        require(shared_source(forest) == (self.method['schema'] == 'SHARED_LIFECYCLE_METHOD_V1'),
                'method and actual source representation binding')
        inject(self.fault, 'setup', work)
        self.state = 'READY'

    def propose(self, tasks, roots):
        require(self.state == 'READY' and type(roots) is RootLease, 'ready proposer/issued roots')
        roots._check()
        require(roots._driver.forest is self.forest and roots._work is self.work
                and tasks is roots._tasks, 'proposer source/Work/task binding')
        self.state = 'PROPOSING'
        self.stats['propose_calls'] += 1
        if self.fault == 'stale_roots' and self.saved is not None:
            len(self.saved)
        self.saved = roots
        inject(self.fault, 'propose', self.work)
        records = []
        general = self.method['producer']['profile'] == 1
        for task, (center, weights) in zip(tasks, roots, strict=True):
            # Exercise the actual prepared objectives, without changing the fixed candidates.
            for _, q in weights.items():
                self.work.exact(q)
                self.stats['root_terms'] += 1
            self.work.exact(center)
            if self.fault == 'missing':
                records.append(dict(slot=task.slot, proposed=None, zero=None))
                continue
            eq, ine = (len(self.forest.rows(task.pair, kind)) for kind in ('eq', 'ine'))
            self.work.visit('coefficient_visits', 2 * (eq + ine))
            proposed = dict(query=task.query_id, lower='1/3',
                            equality=[str(F((j % 3) - 1, 3 if general else 8)) for j in range(eq)],
                            inequality=[str(F(-1, 3) if general else F(-(j % 2), 16)) for j in range(ine)])
            zero = dict(query=task.query_id, lower='0', equality=['0'] * eq, inequality=['0'] * ine)
            records.append(dict(slot=task.slot, proposed=proposed, zero=zero))
        if self.fault == 'partial':
            records.pop()
        self.state = 'READY'
        return records

    def close(self):
        require(self.state != 'CLOSED', 'close exactly once')
        self.stats['close_calls'] += 1
        self.state = 'CLOSED'
        revoke(self.saved)
        try:
            inject(self.fault, 'close', self.work)
        finally:
            self.saved = self.forest = self.work = None


class Lifecycle:
    """Restricted controller, not an arbitrary callback adapter."""
    def __init__(self, config, fault=''):
        self.proposer = FixedProposer(config, fault)
        self.lease = None
        self.closed = False

    def setup(self, forest, work):
        require(not self.closed, 'open lifecycle')
        self.proposer.setup(forest, work)

    def propose(self, driver):
        require(not self.closed and self.lease is None, 'one proposal at a time')
        from scoped_source.bounded_runtime import Workspace
        from scoped_source.pair_workspace import PairWorkspace
        expected = PairWorkspace if self.proposer.method['production']['kind'] == 'pair_scoped' else Workspace
        require(type(driver) is Driver and type(driver.workspace) is expected, 'actual producer consumer binding')
        tasks, self.lease, token = prepare(driver)
        try:
            records = self.proposer.propose(tasks, self.lease)
            return token, records
        finally:
            revoke(self.lease)
            self.lease = None

    def close(self):
        require(not self.closed, 'lifecycle close exactly once')
        self.closed = True
        revoke(self.lease)
        self.lease = None
        self.proposer.close()
