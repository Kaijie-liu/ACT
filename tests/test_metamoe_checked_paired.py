import copy
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import torch
from act.back_end.moe.checked_execution import CheckedExecutionOptions, checked_execution
from act.back_end.moe import class_separated_top1 as entry
from act.back_end.moe import hz_routing
from act.back_end.solver import solver_hz as sh
import metamoe_checked_paired as runner
from audit_metamoe_checked_paired import check_author
from metamoe_paired_execution_r2 import spec_text
from recent_moe_deployment import sha256


class CompositionControls(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory(prefix='checked-paired-', dir='/data1/Kane/MOE')
        self.addCleanup(t.cleanup)
        self.root = Path(t.name)
        self.args = dict(request_sha256='1'*64, input_sha256='2'*64)

    def test_explicit_flags_and_identity(self):
        for bad in (1, None, 'true'):
            with self.assertRaises(ValueError): CheckedExecutionOptions(expert_base=bad)
        with self.assertRaises(TypeError):
            with checked_execution(self.root, **self.args, options={}): pass
        with self.assertRaises(ValueError):
            with checked_execution(self.root, request_sha256='bad', input_sha256='2'*64,
                                   options=CheckedExecutionOptions()): pass

    def test_disabled_changes_nothing(self):
        old = sh.HZSolver, hz_routing.hz_check_feasibility, entry.selected_score_support
        with checked_execution(self.root, **self.args, options=CheckedExecutionOptions()) as cls:
            self.assertIsNone(cls)
            self.assertEqual(old, (sh.HZSolver, hz_routing.hz_check_feasibility, entry.selected_score_support))
        self.assertFalse(list(self.root.iterdir()))

    def test_partial_full_restore_and_nested_refusal(self):
        old = sh.HZSolver, hz_routing.hz_check_feasibility, entry.selected_score_support
        native = sh._solve_hz_feasibility
        for i, opts in enumerate((CheckedExecutionOptions(True), CheckedExecutionOptions(False, True),
                                CheckedExecutionOptions(False, False, True), CheckedExecutionOptions(True, True, True))):
            with self.assertRaisesRegex(ValueError, 'synthetic'):
                with checked_execution(self.root/str(i), **self.args, options=opts):
                    self.assertEqual([a is not b for a,b in zip(old, (sh.HZSolver, hz_routing.hz_check_feasibility,
                                     entry.selected_score_support))], [opts.expert_base, opts.route_feasibility, opts.score_nonzero])
                    self.assertIs(sh._solve_hz_feasibility, native)
                    with self.assertRaises(RuntimeError):
                        with checked_execution(self.root/'nested', **self.args, options=opts): pass
                    raise ValueError('synthetic')
            self.assertEqual(old, (sh.HZSolver, hz_routing.hz_check_feasibility, entry.selected_score_support))

    def test_later_context_failure_unwinds_earlier(self):
        from act.back_end.solver import checked_route_feasibility as route
        original = sh.HZSolver
        with patch.object(route, 'checked_route_feasibility', side_effect=RuntimeError('later')):
            with self.assertRaisesRegex(RuntimeError, 'later'):
                with checked_execution(self.root, **self.args, options=CheckedExecutionOptions(True, True)): pass
        self.assertIs(sh.HZSolver, original)

    def test_real_tie_legal_all_output_obligations(self):
        from tests.test_class_separated_top1 import affine
        from act.util.device_manager import initialize_device
        initialize_device('cpu', 'float64')
        model = entry.ClassSeparatedTop1(affine([0., 0.], [1., 1.]),
                  [affine([0., 0.], [2., 1.]), affine([0.], [3.])], (2, 1))
        x = torch.zeros((1,1), dtype=torch.float64)
        with checked_execution(self.root, **self.args, options=CheckedExecutionOptions(True, True, True)):
            result = entry.verify_class_separated_box(model, center=x, lower=x-.1, upper=x+.1,
                rows=torch.ones((1,3), dtype=torch.float64), thresholds=torch.tensor([.1]), total_seconds=20)
        self.assertEqual(result['status'], 'POSITIVE')
        self.assertEqual(result['candidates'], [0,1])
        self.assertEqual([r['expert'] for r in result['nonzero_obligations']], [0,1])
        self.assertEqual(len(list((self.root/'protected').glob('evaluation_*/properties.npz'))),2)

    def test_author_exact_box_property_and_identity_refusals(self):
        file = self.root/'tensor.npz'
        lo, hi = np.array([[.1]]), np.array([[.2]])
        np.savez(file, center=(lo+hi)/2, lower=lo, upper=hi)
        v = {'route':1, 'sign':1, 'output_rows':19, 'router_rows':2,
             'checkpoint_sha256':'3'*64, 'probe':np.ones((1,22)).tolist()}
        (self.root/'obligation_identity.json').write_text(json.dumps(v))
        spec = self.root/'request.vnnlib'
        spec.write_text(spec_text(lo,hi,22))
        cfg={'checkpoint':'model', 'files':{'model':'3'*64}}
        req={'tensor_file':str(file)}
        result={'status':'UNKNOWN','clean_route':1}
        check_author(self.root,cfg,req,result)
        spec.write_text(spec.read_text().replace('(>= Y_0 Y_21)', '(>= Y_0 Y_20)'))
        with self.assertRaisesRegex(ValueError,'box/property'): check_author(self.root,cfg,req,result)
        spec.write_text(spec_text(lo,hi,22))
        result['clean_route']=0
        with self.assertRaisesRegex(ValueError,'joint identity'): check_author(self.root,cfg,req,result)


class OuterControls(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory(prefix='checked-outer-',dir='/data1/Kane/MOE')
        self.addCleanup(t.cleanup)
        self.root=Path(t.name)
        self.path=self.root/'config.json'
        self.cfg={'protocol':'synthetic', 'output_root':str(self.root/'results'), 'execution_commit':'HEAD',
            'roster':[['a','act'],['a','author']], 'seconds':.05, 'group_rss_limit_bytes':2**30,
            'python':{'act':sys.executable,'author':sys.executable}, 'requests':[{'id':'a'}]}
        self.path.write_text(json.dumps(self.cfg))

    def run_synthetic(self, script):
        def command(cfg,path,rid,arm):
            return [sys.executable,'-c',script,str(Path(cfg['output_root'])/f'{rid}_{arm}')]
        with patch.object(runner,'validate'),patch.object(runner,'require_clean'),patch.object(runner,'command',side_effect=command):
            runner.run(self.path)
        return json.loads((self.root/'results/summary.json').read_text())['rows']

    def test_outer_deadline_overrides_late_positive_partial_and_full_cost(self):
        script="import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); (p/'result.json').write_text('{'); time.sleep(1)"
        rows=self.run_synthetic(script)
        self.assertEqual([r['status'] for r in rows],['TIMEOUT','TIMEOUT'])
        cost=json.loads((self.root/'results/batch_cost.json').read_text())
        self.assertEqual(cost['charged_request_seconds'],sum(r['seconds'] for r in rows))
        self.assertGreaterEqual(cost['batch_wall_through_summary_seconds'],cost['charged_request_seconds'])
        with self.assertRaises(FileExistsError): self.run_synthetic(script)

    def test_error_stops_remaining_roster(self):
        self.cfg['seconds']=2
        self.path.write_text(json.dumps(self.cfg))
        rows=self.run_synthetic('raise RuntimeError("synthetic")')
        self.assertEqual([r['status'] for r in rows],['ERROR','NOT_STARTED_AFTER_ERROR'])
        self.assertFalse((self.root/'results/a_author').exists())

    def test_completed_without_valid_identity_cannot_pass(self):
        self.cfg['seconds']=2
        self.path.write_text(json.dumps(self.cfg))
        rows=self.run_synthetic("import pathlib,sys; (pathlib.Path(sys.argv[1])/'result.json').write_text('{\"status\":\"POSITIVE\"}')")
        self.assertEqual(rows[0]['status'],'ERROR')
        self.assertTrue(rows[0]['result_parse_error'])

    def test_frozen_smoke_rejects_changes(self):
        cfg=runner.build_smoke()
        runner.validate(cfg)
        for field,value in [('seconds',301),('act_options',{'expert_base':False}),('margin',0),('roster',[])]:
            bad=copy.deepcopy(cfg);bad[field]=value
            with self.assertRaises(ValueError): runner.validate(bad)
        bad=copy.deepcopy(cfg);bad['files']['act/back_end/moe/checked_execution.py']='0'*64
        with self.assertRaises(ValueError): runner.validate(bad)

    def test_order_rotates_by_input_without_result_access(self):
        self.assertEqual(runner.roster([{'id':'x'},{'id':'y'}]),
                         [['x','act'],['x','author'],['y','author'],['y','act']])


if __name__=='__main__': unittest.main()
