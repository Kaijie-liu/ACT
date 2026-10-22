"""Stdlib controls for mandatory query-boundary publication and post-prefix labels."""
import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import multilayer_stage1 as stage
from act.pipeline.moe.multilayer_study import mechanism_decomposition,mechanism_endpoint,endpoint


class Controls(unittest.TestCase):
    def test_subquarter_starts_same_phase_new_history_and_return(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE/results',prefix='telemetry_fixed_') as name:
            folder=Path(name);clock=SimpleNamespace(value=100.)
            with patch.object(stage,'time',SimpleNamespace(monotonic=lambda:clock.value)):
                publish=stage.Progress(folder,'genbab_pruned',9)
                value={};publish(value,force=True)
                clock.value+=.01
                value.update(active=dict(phase='prefix_CROWN',prefix=[],entered=clock.value),
                             attempted_prefix_queries=1)
                publish(value);saved=json.loads((folder/'progress.json').read_text())
                self.assertEqual(saved['coverage']['active']['phase'],'prefix_CROWN')
                self.assertEqual(saved['last_query_start']['entered'],100.01)
                old=(folder/'progress.json').read_bytes();clock.value+=.01;publish(value)
                self.assertEqual((folder/'progress.json').read_bytes(),old)
                for prefix in ([[0,1]],[[0,2]]):
                    clock.value+=.01
                    value['active']=dict(phase='history_solver',prefix=prefix,entered=clock.value)
                    publish(value)
                    self.assertEqual(json.loads((folder/'progress.json').read_text())['coverage']['active']['prefix'],prefix)
                # An exception/native timeout needs no later callback: start is durable.
                self.assertEqual(json.loads((folder/'progress.json').read_text())['coverage']['active']['phase'],'history_solver')
                clock.value+=.01;value['active']=None
                value.update(attempted_histories=1,returned_histories=1,
                             histories=[dict(outcome=dict(seconds=151.))])
                publish(value);saved=json.loads((folder/'progress.json').read_text())
                self.assertIsNone(saved['coverage']['active'])
                self.assertEqual(saved['last_query_start']['prefix'],[[0,2]])
                self.assertEqual(saved['max_completed_history_solver_seconds'],151.)
                self.assertLess(clock.value-100,.25)

    def test_classification_boundaries_and_missing_not_aggregate(self):
        cov=dict(traversal_complete=False,returned=5,active=None)
        def label(residual,returned=5,maximum=None):
            t=dict(phase_seconds={'history_solver':300-residual})
            if maximum is not None:t['max_completed_history_solver_seconds']=maximum
            return mechanism_decomposition(300,dict(cov,returned=returned),t)['category']
        self.assertEqual(label(150),'A_SINGLE_QUERY_DOMINATED')
        self.assertEqual(label(149.999),'C_UNSEPARATED')
        self.assertEqual(label(49.999),'B_MANY_HISTORIES_DOMINATED')
        self.assertEqual(label(50),'C_UNSEPARATED')
        self.assertEqual(label(1,4),'C_UNSEPARATED')
        self.assertEqual(label(1,5,150),'A_SINGLE_QUERY_DOMINATED')
        self.assertEqual(label(1,5,149.999),'B_MANY_HISTORIES_DOMINATED')
        self.assertEqual(mechanism_decomposition(300,cov,None)['category'],'C_UNSEPARATED')
        self.assertEqual(mechanism_decomposition(300,cov,{'phase_seconds':{'x':float('nan')}})['category'],'C_UNSEPARATED')
        self.assertEqual(mechanism_decomposition(300,dict(cov,traversal_complete=True),None)['category'],'TRAVERSAL_COMPLETE')

    def test_same_unsafe_denominator_and_supplement_not_original_rewrite(self):
        rows=[dict(model='m',index=i,coverage={'traversal_complete':False},
                   mechanism={'category':'A_SINGLE_QUERY_DOMINATED'}) for i in range(60)]
        unsafe={('m',0)};primary=endpoint(rows,unsafe)
        self.assertEqual(primary['adjusted']['decision'],'ENTER_STAGE2')
        result=mechanism_endpoint(rows,unsafe,primary)
        self.assertEqual(result['adjusted_many']['n'],59)
        self.assertEqual(result['decision'],'STOP_REPORT_SINGLE_ROUTE_NOT_COMBINATIONS_OPERATIONAL')
        for r in rows:r['mechanism']['category']='B_MANY_HISTORIES_DOMINATED'
        self.assertEqual(mechanism_endpoint(rows,unsafe,primary)['decision'],'ELIGIBLE_ALPHA_CROWN_ENTRANCE')
        self.assertEqual(endpoint(rows,unsafe),primary)

    def test_boundary_reservation_stops_before_request_or_worker(self):
        old=Path('/data1/Kane/MOE/executions/multilayer_stage1_20261022_r5/scripts/multilayer_stage1.py')
        method=lambda p:next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='invoke')
        self.assertEqual(ast.dump(method(old)),ast.dump(method(Path(stage.__file__))))
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE/results',prefix='boundary_fixed_') as name:
            target=Path(name)/'verify_001';target.mkdir()
            with patch.object(stage,'execute',side_effect=AssertionError('must not launch')) as launch:
                with self.assertRaises(FileExistsError):
                    stage.invoke(target,{},300,Path(name)/'unused_manifest',stage.time.monotonic()+1000)
                launch.assert_not_called()
            self.assertEqual(list(target.iterdir()),[])


if __name__=='__main__':unittest.main()
