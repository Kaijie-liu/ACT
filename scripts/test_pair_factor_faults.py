"""Exactly the two unfinished R1 factor faults; no candidates or performance runs."""
import copy
import time
import unittest

from scoped_source.hz_request_store import Work, POLICY
from scoped_source.forest_endpoint_controls import capture
from scoped_source.shared_build import build
from scoped_source.check_shared_source import CheckedSharedForest
from scoped_source.bounded_runtime import Owner
from scoped_source.pair_workspace import Scope, U32
from scripts.test_support_traversal import special

ROOT = None
DEADLINE = None
RECORDS = []
SOURCE = '3a9df8651ea6781098377244b410a1cfa821bc4f431fbf1d067fa5c4a76cb360'
FOREST = 'cfd36bba9a2702c13a4d80e2f093cc4a764f77e85d66a19e12ac51c37d3ef157'


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        started = time.monotonic()
        doc = special('eight_private')
        for expert in doc['experts']:
            expert[-1]['rows'].extend(copy.deepcopy(expert[-1]['rows'][1:2])*4)
        cls.path = ROOT/'eight_private_six_outputs'
        source = capture(cls.path, doc)
        if source != SOURCE:
            raise ValueError('unchanged R1 multi-factor source identity')
        work = Work(DEADLINE)
        cls.packet, construction = build(cls.path, source, work=work, enabled=True)
        if cls.packet.identity != FOREST:
            raise ValueError('unchanged R1 forest identity')
        RECORDS.append(dict(kind='source', name='eight_private_six_outputs',
                            source=source, forest=cls.packet.identity,
                            work=dict(work.counts), construction=construction,
                            seconds=time.monotonic()-started))

    def factor_fault(self, mode):
        started = time.monotonic()
        work = Work(DEADLINE)
        forest = CheckedSharedForest(self.packet, self.path, SOURCE,
            expected=FOREST, work=work, enabled=True)
        self.assertEqual((POLICY['visits'], POLICY['total_bytes'],
                          POLICY['stored_bits'], POLICY['work_bits']),
                         (8000000, 134217728, 4096, 8192))
        owner = Owner(forest, work)
        scope = None
        try:
            scope = Scope(owner, (0, 1))
            _, selected, forward, reverse = scope.banks['factors']
            self.assertGreater(selected, 1)  # Never weaken the failed prerequisite.
            a, b = (scope.global_id('factors', k) for k in (0, 1))
            self.assertLess(a, b)
            before = [forward.get(4*a, 4), forward.get(4*b, 4), reverse.get(0, 8)]
            self.assertEqual(before, [U32.pack(1), U32.pack(2), U32.pack(a)+U32.pack(b)])
            if mode == 'missing':
                forward.put(4*a, U32.pack(0))
                expected = 'scope membership binding'
                expected_after = [U32.pack(0), U32.pack(2), before[2]]
            else:
                self.assertEqual(mode, 'permuted')
                forward.put(4*a, U32.pack(2)); forward.put(4*b, U32.pack(1))
                reverse.put(0, U32.pack(b)); reverse.put(4, U32.pack(a))
                expected = 'canonical traversal rank/order'
                expected_after = [U32.pack(2), U32.pack(1), U32.pack(b)+U32.pack(a)]
            after = [forward.get(4*a, 4), forward.get(4*b, 4), reverse.get(0, 8)]
            self.assertEqual(after, expected_after)
            self.assertNotEqual(before, after)
            counts_before = dict(work.counts)
            with self.assertRaises(ValueError) as caught:
                scope.prepare_traversal(owner, (0, 1))
            self.assertEqual(str(caught.exception), expected)
            self.assertFalse(scope.traversal_ready)
            self.assertTrue(all(v._data is None and v._sealed is not None
                                for v in scope.banks['nodes'][2:]))
            self.assertTrue(all(v._data is not None and v._sealed is None
                                for v in (forward, reverse)))
            self.assertTrue(all(work.counts[k] >= v for k, v in counts_before.items()))
            record = dict(kind='factor_fault', mode=mode, pair=[0, 1],
                source=SOURCE, forest=FOREST, selected_factors=selected,
                factor_ids=[a, b], before_hex=[x.hex() for x in before],
                after_hex=[x.hex() for x in after], error=expected,
                traversal_ready=scope.traversal_ready, nodes_sealed=True,
                factor_maps_unsealed=True, check_started=True)
        finally:
            if scope is not None:
                scope.close()
            owner.close()
        self.assertTrue(scope.closed and owner.closed)
        self.assertEqual(owner.stats['live_bytes'], 0)
        self.assertEqual(owner.stats['allocated_bytes'], owner.stats['released_bytes'])
        self.assertTrue(all(v._data is None and v._sealed is None for v in scope._blocks))
        work.tick()
        record.update(work=dict(work.counts), owner=dict(owner.stats),
                      seconds=time.monotonic()-started, scope_closed=True, owner_closed=True,
                      all_buffers_released=True)
        RECORDS.append(record)

    def test_01_factor_missing(self):
        self.factor_fault('missing')

    def test_02_factor_permuted(self):
        self.factor_fault('permuted')
