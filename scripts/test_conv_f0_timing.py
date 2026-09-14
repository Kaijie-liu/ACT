"""Observer controls: no trained-model endpoint queries."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts.f0_timing_trace import Recorder, TraceFailure, install, restore, scalar
from scripts.audit_conv_f0_timing import check_trace
from scripts.conv_three_arm_contract import ROOT


class TimingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results')
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'trace.jsonl'
        self.started = time.monotonic()
        self.rec = Recorder(self.path, self.started, {'test':True})
        self.addCleanup(self.rec.close)

    def finish(self):
        self.rec.emit('WORKER_COMPLETE')
        return check_trace(self.path, time.monotonic()-self.started, {'test':True})

    def test_forward_identity(self):
        sentinel = object()
        def fn(a, *, b):
            self.assertIs(a, sentinel); self.assertEqual(b, 7)
            return sentinel
        self.assertIs(self.rec.wrap(fn, 'test')(sentinel, b=7), sentinel)
        self.assertEqual(self.finish()['aggregates']['test']['closed_calls'], 1)

    def test_exception_identity(self):
        exc = ValueError('sentinel')
        def fail():
            raise exc
        try:
            self.rec.wrap(fail, 'test')()
        except ValueError as got:
            self.assertIs(got, exc)
        self.assertEqual(self.finish()['spans'][0]['end_kind'], 'RAISE')

    def test_nested_and_tamper(self):
        inner = self.rec.wrap(lambda: 1, 'inner')
        self.rec.wrap(lambda: inner(), 'outer')()
        result = self.finish()
        self.assertEqual(result['spans'][1]['parent'], result['spans'][0]['id'])
        data = self.path.read_bytes().replace(b'"value":1', b'"value":2')
        with patch.object(Path, 'read_bytes', return_value=data):
            with self.assertRaises(ValueError):check_trace(self.path, 10)

    def test_nonfinite_explicit(self):
        self.assertEqual(scalar(float('nan')), {'nonfinite':'nan'})
        self.rec.wrap(lambda: float('inf'), 'test')()
        self.assertEqual(self.finish()['spans'][0]['result']['value'], {'nonfinite':'inf'})

    def test_failure_not_swallowed(self):
        with patch('scripts.f0_timing_trace.os.fsync', side_effect=OSError('disk')):
            with self.assertRaises(TraceFailure): self.rec.wrap(lambda: 1, 'test')()
        self.assertFalse(issubclass(TraceFailure, Exception))

    def test_budget_unchanged(self):
        from act.pipeline.moe.request_budget import RequestBudget
        budget = RequestBudget(300, started=1, clock=lambda: 10)
        fn = self.rec.wrap(RequestBudget.limit, 'budget')
        self.assertEqual(fn(budget, 'F0', obligations=9), 291/9)
        self.assertEqual(budget.events[0]['granted_seconds'], 291/9)
        self.finish()

    def test_aliases_restore_and_native_solver(self):
        import numpy as np
        from scipy.optimize import Bounds
        from act.back_end.solver import solver_hz
        from act.back_end.moe import weighted_top2, monolithic_f0
        native = solver_hz.milp
        original_support = weighted_top2.hz_support_bounds
        kw = {'c':np.array([1.]), 'integrality':np.array([1]), 'bounds':Bounds([0.], [2.]),
              'options':{'time_limit':1., 'mip_rel_gap':0.}}
        reference = native(**kw)
        patches = install(self.rec)
        try:
            self.assertIs(solver_hz.milp, monolithic_f0.milp)
            self.assertIs(weighted_top2.hz_support_bounds, solver_hz.hz_support_bounds)
            self.assertIsNot(solver_hz.milp, native)
            result = monolithic_f0.milp(**kw)
            self.assertEqual(result.status, reference.status)
            self.assertEqual(result.fun, reference.fun)
            self.assertTrue(np.array_equal(result.x, reference.x))
        finally:
            restore(patches)
        self.assertIs(solver_hz.milp, native)
        self.assertIs(weighted_top2.hz_support_bounds, original_support)
        self.assertTrue(any('milp' in s['name'] for s in self.finish()['spans']))

    def test_open_span_and_partial_tail(self):
        self.rec.emit('BEGIN', name='native', parent=None, arguments={})
        result = check_trace(self.path, 10, killed=True)
        self.assertTrue(result['spans'][0]['right_censored'])
        self.assertNotIn('seconds', result['spans'][0])
        with self.assertRaises(ValueError): check_trace(self.path, 10, killed=False)
        with patch.object(Path, 'read_bytes', return_value=self.path.read_bytes()+b'{"seq":'):
            self.assertEqual(check_trace(self.path, 10, killed=True)['partial_tail_bytes'], 7)

    def test_real_toy_f0_same_encoding_and_verdict(self):
        import numpy as np
        from act.back_end.moe.test_weighted_top2 import _equal_expert_encoding
        from act.back_end.moe import monolithic_f0
        original = _equal_expert_encoding(offset=2.)
        baseline = monolithic_f0.solve_monolithic_weighted_top2_f0(
            [original], input_shape=(1, 1), time_limit=2.)
        patches = install(self.rec)
        try:
            observed = _equal_expert_encoding(offset=2.)
            # No tensor/constraint/source arithmetic changed by timing wrappers.
            for name in ('c', 'b', 'ub'):
                self.assertTrue(np.array_equal(getattr(original.output_hz, name),
                                               getattr(observed.output_hz, name)))
            for name in ('Gc', 'Gb', 'Ac', 'Ab', 'Auc', 'Aub'):
                self.assertEqual((getattr(original.output_hz, name) !=
                                  getattr(observed.output_hz, name)).nnz, 0)
            decision = monolithic_f0.solve_monolithic_weighted_top2_f0(
                [observed], input_shape=(1, 1), time_limit=2.)
            self.assertEqual(decision.status, baseline.status)
            self.assertAlmostEqual(decision.minimum, baseline.minimum, places=10)
        finally:
            restore(patches)
        names = {s['name'].split('.')[-1] for s in self.finish()['spans']}
        self.assertTrue({'_build_disjunction', '_lower_hz_milp', 'hz_support_bounds', 'milp'} <= names)

    def test_wrong_identity_and_clock(self):
        self.rec.emit('WORKER_COMPLETE')
        with self.assertRaises(ValueError):check_trace(self.path, 10, {'other':True})
        with self.assertRaises(ValueError):check_trace(self.path, -1)

    def test_real_kill_leaves_begin(self):
        killed = Path(self.tmp.name)/'killed.jsonl'
        code = ('import time; from pathlib import Path; '
                'from scripts.f0_timing_trace import Recorder; '
                f'r=Recorder(Path({str(killed)!r}),time.monotonic(),{{"test":True}}); '
                'r.wrap(lambda:time.sleep(60),"native")()')
        proc = subprocess.Popen([sys.executable, '-c', code], cwd=ROOT)
        try:
            until = time.monotonic()+10
            while time.monotonic()<until:
                if killed.exists() and b'BEGIN' in killed.read_bytes(): break
                time.sleep(.01)
            else: self.fail('child did not enter span')
            os.kill(proc.pid, signal.SIGKILL); proc.wait(timeout=5)
            result = check_trace(killed, 10, killed=True)
            self.assertEqual(len(result['open_span_ids']), 1)
        finally:
            if proc.poll() is None: proc.kill(); proc.wait()


if __name__ == '__main__':
    unittest.main()
