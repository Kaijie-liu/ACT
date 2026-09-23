"""No large model: tiny native differential + fake deadline/fault controls."""
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from act.back_end.core import Bounds
from act.back_end.solver import solver_hz as sh
from act.back_end.solver.protected_hz_solver import ProtectedHZSolver, protected_experts
from act.back_end.solver.isolated_feasibility import NativeSession, save_npz
from act.back_end.solver.native_feasibility_worker import digest, publish
from act.front_end.specs import OutKind, OutputSpec
from act.util.stats import VerifyStatus


class ProtectedControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='protected-hz-control-', dir='/data1/Kane/MOE')
        self.root = Path(self.tmp.name)
        self.hz = sh.sparse_hz_from_bounds(Bounds(torch.tensor([[-1.]], dtype=torch.float64),
                                                torch.tensor([[1.]], dtype=torch.float64)), frame_id=910)

    def tearDown(self):
        self.tmp.cleanup()

    def spec(self, thresholds=(2., 2.)):
        return OutputSpec(kind=OutKind.LINEAR_LE, c=torch.tensor([[1.], [-1.]], dtype=torch.float64),
                          d=torch.tensor(thresholds, dtype=torch.float64))

    def evaluate(self, solver, spec=None, **kw):
        return solver.evaluate_spec(self.hz, spec or self.spec(), batch_size=1, n_out=1,
            input_hz=self.hz, input_shape=(1, 1), **kw)[0]

    def test_tiny_safe_native_matches_legacy(self):
        old = self.evaluate(sh.HZSolver(time_limit=10))
        new = self.evaluate(ProtectedHZSolver(directory=self.root/'eval', time_limit=10))
        self.assertEqual(old.status, VerifyStatus.CERTIFIED)
        self.assertEqual(new.status, old.status)
        self.assertEqual(new.metadata['required_properties'], 2)
        self.assertEqual(new.metadata['base_status'], 'feasible')
        self.assertEqual([p['status'] for p in new.metadata['properties']], ['infeasible']*2)
        self.assertEqual(new.metadata['native_queries_started'], 3)

    def test_tiny_unsafe_native_matches_legacy_and_recovers_input(self):
        old = self.evaluate(sh.HZSolver(time_limit=10), self.spec((-.5, 2.)))
        new = self.evaluate(ProtectedHZSolver(directory=self.root/'eval', time_limit=10), self.spec((-.5, 2.)))
        self.assertEqual(old.status, VerifyStatus.FALSIFIED)
        self.assertEqual(new.status, old.status)
        self.assertIsNotNone(new.counterexample)
        self.assertTrue(-1. <= new.counterexample.item() <= 1.)
        self.assertGreater(new.counterexample.item(), -.5)

    def test_real_base_hang_then_real_property_queries(self):
        launches = [0]
        def commands(original):
            launches[0] += 1
            return [sys.executable, '-c', 'import time; time.sleep(60)'] if launches[0] == 1 else original
        factory = lambda model, directory: NativeSession(model, directory, command_factory=commands)
        result = self.evaluate(ProtectedHZSolver(directory=self.root/'eval', time_limit=3, session_factory=factory))
        self.assertEqual(result.status, VerifyStatus.UNKNOWN)
        self.assertEqual(result.metadata['reason'], 'properties_excluded_but_base_unproved')
        self.assertEqual(result.metadata['queries'][0]['terminal'], 'TIMEOUT')
        self.assertEqual([p['status'] for p in result.metadata['properties']], ['infeasible']*2)
        self.assertEqual(result.metadata['native_queries_started'], 2)

    def fake(self, base_status='unknown', property_status='infeasible', expire_base=False):
        clock = [100.]
        class Session:
            def __init__(this, model, directory):
                this.model_hash, this.queries = 'SYNTHETIC', []
            def query(this, until, *, scope, **kwargs):
                status = base_status if scope['phase'] == 'base' else property_status
                if expire_base and scope['phase'] == 'base':
                    clock[0] = until+.01
                else:
                    clock[0] += .01
                this.queries.append({'scope': scope, 'native_started': True,
                    'deadline_monotonic': until, 'returned_status': status})
                return sh._MILPResult(status, np.zeros(1) if status == 'feasible' else None)
            def close(this):
                pass
        with patch('act.back_end.solver.protected_hz_solver.time.monotonic', side_effect=lambda: clock[0]):
            solver = ProtectedHZSolver(directory=self.root/'eval', time_limit=30, session_factory=Session)
            result = self.evaluate(solver)
        return result

    def test_expired_base_does_not_starve_properties_or_license_safe(self):
        result = self.fake(expire_base=True)
        self.assertEqual(result.status, VerifyStatus.UNKNOWN)
        self.assertEqual(result.metadata['reason'], 'properties_excluded_but_base_unproved')
        self.assertEqual([r['status'] for r in result.metadata['properties']], ['infeasible']*2)
        self.assertEqual(result.metadata['queries'][0]['deadline_monotonic'], 103.)
        self.assertEqual(result.metadata['allocation_seconds'], 30.)

    def test_base_feasible_all_properties_required(self):
        result = self.fake(base_status='feasible', property_status='unknown')
        self.assertEqual(result.status, VerifyStatus.UNKNOWN)
        self.assertEqual(len(result.metadata['properties']), 2)

    def test_infeasible_base_retains_nonvacuity_gate(self):
        result = self.fake(base_status='infeasible')
        self.assertEqual(result.status, VerifyStatus.UNKNOWN)
        self.assertEqual(result.metadata['reason'], 'empty_hz')
        self.assertEqual(result.metadata['native_queries_started'], 1)

    def test_constant_safe_no_native(self):
        self.hz = sh.sparse_hz_from_bounds(Bounds(torch.tensor([[0.]], dtype=torch.float64),
                                                torch.tensor([[0.]], dtype=torch.float64)), frame_id=911)
        result = self.evaluate(ProtectedHZSolver(directory=self.root/'eval', time_limit=3))
        self.assertEqual(result.status, VerifyStatus.CERTIFIED)
        self.assertEqual(result.metadata['native_queries_started'], 0)

    def test_unsupported_scope_no_fallback_to_unbounded_path(self):
        solver = ProtectedHZSolver(directory=self.root/'eval', time_limit=2)
        result = solver.evaluate_spec(self.hz, self.spec(), batch_size=2, n_out=1)[0]
        self.assertEqual(result.status, VerifyStatus.UNKNOWN)
        self.assertEqual(result.metadata['reason'], 'unsupported_protected_scope')

    def test_expert_budget_cannot_be_expanded(self):
        for value in (0., -1., 30.01, float('inf'), float('nan')):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'allocation'):
                self.evaluate(ProtectedHZSolver(directory=self.root/'eval'), timelimit=value)
        self.assertFalse((self.root/'eval').exists())

    def test_frozen_tolerance_cannot_be_changed(self):
        with self.assertRaisesRegex(ValueError, 'tolerance'):
            ProtectedHZSolver(directory=self.root/'eval', tolerance=1e-5)

    def test_context_restores_and_rejects_nested_installation(self):
        original = sh.HZSolver
        with protected_experts(self.root):
            self.assertNotEqual(sh.HZSolver, original)
            self.assertIsInstance(sh.HZSolver(), ProtectedHZSolver)
            with self.assertRaises(RuntimeError):
                with protected_experts(self.root):
                    pass
        self.assertIs(sh.HZSolver, original)

    def session(self, child_code):
        folder = self.root/'native'
        folder.mkdir()
        session = NativeSession(sh._lower_hz_milp(self.hz), folder,
            command_factory=lambda original: [sys.executable, '-c', child_code])
        self.addCleanup(session.close)
        return session

    def test_native_hang_stopped_without_consuming_expert_budget(self):
        session = self.session('import time; time.sleep(60)')
        start = time.monotonic()
        result = session.query(start+.2, scope={'phase': 'base'})
        self.assertEqual(result.status, 'unknown')
        self.assertLess(time.monotonic()-start, 2.)
        self.assertEqual(session.queries[0]['terminal'], 'TIMEOUT')
        self.assertIsNone(session.proc)
        self.assertTrue((session.root/'query_000/return.json').exists())

    def test_child_exception_and_partial_candidate_retained(self):
        session = self.session('raise RuntimeError("synthetic child error")')
        self.assertEqual(session.query(time.monotonic()+2, scope={'phase': 'base'}).status, 'unknown')
        self.assertEqual(session.queries[0]['terminal'], 'ERROR')

    def test_truncated_candidate_is_error_not_positive(self):
        session = self.session('import sys,json; from pathlib import Path; '
            'm=json.loads(sys.stdin.readline()); p=Path(m["folder"]); '
            '(p/"native_result.json").write_text("{"); import time; time.sleep(60)')
        self.assertEqual(session.query(time.monotonic()+2, scope={'phase': 'base'}).status, 'unknown')
        self.assertEqual(session.queries[0]['terminal'], 'ERROR')
        self.assertTrue((session.root/'query_000/native_result.json').exists())

    def test_wrong_candidate_binding_rejected(self):
        session = self.session('import sys,json,time; from pathlib import Path; '
            'm=json.loads(sys.stdin.readline()); p=Path(m["folder"]); '
            'r=json.loads((p/"request.json").read_text()); r.update(token="WRONG",status=2,finished_monotonic=time.monotonic()); '
            '(p/"native_result.json").write_text(json.dumps(r)); time.sleep(60)')
        result = session.query(time.monotonic()+2, scope={'phase': 'base'})
        self.assertEqual(result.status, 'unknown')
        self.assertIn('identity', session.queries[0]['message'])

    def test_expired_query_does_not_spawn(self):
        session = self.session('raise RuntimeError("MUST NOT START")')
        result = session.query(time.monotonic()-1, scope={'phase': 'base'})
        self.assertEqual(result.status, 'unknown')
        self.assertEqual(session.launches, 0)
        self.assertEqual(session.queries[0]['terminal'], 'LOCAL_DEADLINE')

    def proposed(self, x, *, late=False, slow_publication=False):
        folder = self.root/'proposal'
        folder.mkdir()
        session = NativeSession(sh._lower_hz_milp(self.hz), folder)
        self.addCleanup(session.close)
        def start():
            q = folder/'query_000'
            req = json.loads((q/'request.json').read_text())
            save_npz(q/'candidate.npz', x=np.array([x]))
            publish(q/'native_result.json', {**req, 'status': 1, 'success': False,
                'message': 'synthetic time limit', 'mip_node_count': 0,
                'candidate_sha256': digest(q/'candidate.npz'),
                'finished_monotonic': req['deadline_monotonic']+1 if late else time.monotonic()})
            session.proc = SimpleNamespace(stdin=SimpleNamespace(write=lambda s: None, flush=lambda: None, close=lambda: None),
                poll=lambda: None, kill=lambda: None, wait=lambda: 0)
        def publisher(path, value):
            publish(path, value)
            if slow_publication and Path(path).name == 'return.json':
                time.sleep(.3)
        with patch.object(session, 'start', side_effect=start), patch(
                'act.back_end.solver.isolated_feasibility.publish', side_effect=publisher):
            result = session.query(time.monotonic()+(.2 if slow_publication else 2), scope={'phase': 'base'})
        return result, session

    def test_status_one_valid_candidate_keeps_original_feasibility_gate(self):
        result, session = self.proposed(0.)
        self.assertEqual(result.status, 'feasible')
        self.assertTrue(session.queries[0]['incumbent_valid'])

    def test_status_one_invalid_candidate_is_unknown(self):
        result, session = self.proposed(2.)
        self.assertEqual(result.status, 'unknown')
        self.assertFalse(session.queries[0]['incumbent_valid'])

    def test_late_native_candidate_cannot_be_accepted(self):
        result, session = self.proposed(0., late=True)
        self.assertEqual(result.status, 'unknown')
        self.assertEqual(session.queries[0]['terminal'], 'ERROR')

    def test_publication_cost_can_demote_early_candidate(self):
        result, session = self.proposed(0., slow_publication=True)
        self.assertEqual(result.status, 'unknown')
        self.assertEqual(session.queries[0]['terminal'], 'RETURN_PUBLICATION_DEADLINE')
        self.assertTrue((session.root/'query_000/late_return_rejected.json').exists())

    def test_existing_evaluation_directory_never_overwritten(self):
        folder = self.root/'eval'
        folder.mkdir()
        with self.assertRaises(FileExistsError):
            self.evaluate(ProtectedHZSolver(directory=folder))


if __name__ == '__main__':
    unittest.main()
