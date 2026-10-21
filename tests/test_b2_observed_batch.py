"""Ordinary B2 wiring tests, no model/checkpoint/solver or external imports."""
import importlib.util
import ast
import inspect
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


batch=module('b2_observed_batch');worker=module('b2_observed_worker')


class B2WiringControls(unittest.TestCase):
    def test_suffix_keeps_error_and_does_not_rerun_old_rows(self):
        queries=[dict(id=f'q{i}',kind='genbab_aggregate',full_slot=True) for i in range(4)]
        rows=[dict(query_id=q['id'],kind=q['kind'],full_slot=True,status='UNEXECUTED') for q in queries]
        rows[0].update(status='POSITIVE',session={},accepted=True)
        rows[1].update(status='ERROR',session={},accepted=False)
        parent=dict(status='PREFIX_TOOLING_OR_IDENTITY_STOP',rows=rows,full_attempts=2,local_attempts=0)
        self.assertEqual(batch.retained_prefix({'queries':queries},parent),rows[:2])
        for mutate in ('identity','gap','extra_session','count'):
            bad=json.loads(json.dumps(parent))
            if mutate=='identity':bad['rows'][1]['query_id']='other'
            elif mutate=='gap':bad['rows'][0]['status']='UNEXECUTED'
            elif mutate=='extra_session':bad['rows'][3]['session']={}
            else:bad['full_attempts']=1
            with self.assertRaises(ValueError):batch.retained_prefix({'queries':queries},bad)

    def test_explicit_unchanged_control_reuse_rejects_changed_identity(self):
        check=dict(commit='previous',result=dict(kind='synthetic_vector_control',accepted=True))
        binding=dict(control_commit='previous',control_identity=batch.digest(check),
            repaired_batch_sha256=batch.sha(batch.__file__),
            invoke_source_identity=batch.digest(inspect.getsource(batch.invoke)),
            unchanged_files={p:batch.sha(ROOT/p) for p in batch.CONTROL_FILES})
        batch.validate_control(check,'new',binding)
        for key in ('control_commit','control_identity','repaired_batch_sha256','invoke_source_identity'):
            changed={**binding,key:'wrong'}
            with self.assertRaises(ValueError):batch.validate_control(check,'new',changed)
        for changed_files in ({},{**binding['unchanged_files'],batch.CONTROL_FILES[0]:'wrong'}):
            with self.assertRaises(ValueError):
                batch.validate_control(check,'new',{**binding,'unchanged_files':changed_files})

    def test_cost_includes_external_failed_and_controller_logs(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as temporary:
            root=Path(temporary);folder=root/'run';folder.mkdir()
            (folder/'receipt').write_bytes(b'12345');log=root/'failed.log';log.write_bytes(b'123')
            self.assertEqual(sum(batch.sizes(p) for p in (folder,log)),8)

    def test_backend_binding_json_roundtrip_keeps_all_identity_checks(self):
        config={'cpu': True, 'seconds': 300}
        source={'files': {'worker.py': 'bound-hash'},
                'compatibility_edits': {'file.py': ('before', 'after')}}
        spec=json.loads(json.dumps(dict(genbab_configuration=config,source_binding=source)))
        batch.validate_backend_binding(spec,config,source)
        for changed in (
                {**source,'files':{'worker.py':'wrong-hash'}},
                {**source,'compatibility_edits':{'file.py':('before','different')}}):
            with self.assertRaises(ValueError):batch.validate_backend_binding(spec,config,changed)
        with self.assertRaises(ValueError):batch.validate_backend_binding(spec,{'cpu':False,'seconds':300},source)

    def test_worker_reuses_timer_finalizer_after_all_outcomes(self):
        from b1_observed_batch import finish_worker_io
        self.assertIs(worker.finish_worker_io,finish_worker_io)
        tree=ast.parse(inspect.getsource(worker.worker))
        self.assertTrue(any(isinstance(node,ast.Try) and any(
            isinstance(call,ast.Call) and isinstance(call.func,ast.Name)
            and call.func.id=='finish_worker_io'
            for statement in node.finalbody for call in ast.walk(statement))
            for node in ast.walk(tree)))

    def test_helper_uses_executing_checkout(self):
        self.assertEqual(batch.backend.ROOT,ROOT)
        self.assertEqual(worker.backend.ROOT,ROOT)

    def test_fixed_local_groups_then_41_and_19_full_requests(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as temporary:
            root=Path(temporary);items=[dict(job_id=f'{i:02d}') for i in range(41)]
            (root/'start.json').write_text('{}')
            for i in range(6):
                folder=root/items[i]['job_id']/'diagnostics';folder.mkdir(parents=True)
                obligations=[dict(state='RETURNED',result='UNKNOWN',pair=pair,property_index=k)
                    for pair in ([3,4],[0,2]) for k in (1,0)]
                obligations.extend([dict(state='REUSED',pair=[0,1],property_index=0),
                    dict(state='RETURNED',result='SAFE',pair=[0,1],property_index=1)])
                (folder/'000_inventory_last.json').write_text(json.dumps({'obligations':obligations}))
            rows=batch.query_catalog({'requests':items},root)
            self.assertEqual(len(rows),70)
            self.assertEqual(sum(r['full_slot'] for r in rows),60)
            self.assertEqual(rows[0]['pair'],[0,2]);self.assertEqual(rows[0]['property_indices'],[0,1])
            self.assertEqual(rows[9]['item'],items[4]);self.assertEqual(rows[9]['pair'],[3,4])
            self.assertEqual([r['item'] for r in rows[10:51]],items)
            self.assertTrue(all(r['kind']=='genbab_aggregate' for r in rows[10:51]))
            self.assertEqual([r['item'] for r in rows[51:]],items[:19])
            self.assertTrue(all(r['seconds']==300 for r in rows))

    def test_all_classes_and_selected_property_identity(self):
        rows=worker.properties(10,4)
        self.assertEqual(len(rows),9)
        self.assertTrue(all(len(p['q'])==10 and p['q'][4]==1 for p in rows))
        self.assertEqual(worker.properties(10,4,[0,8]),[rows[0],rows[8]])

    def test_unknown_control_is_not_new_positive_gate(self):
        check=dict(commit='frozen',result=dict(kind='synthetic_vector_control',accepted=True,
            outcome={'verifier':{'status':'unknown'}}))
        batch.validate_control(check,'frozen')
        with self.assertRaises(ValueError):batch.validate_control(check,'changed')
        check['result']['kind']='genbab_aggregate'
        with self.assertRaises(ValueError):batch.validate_control(check,'frozen')

    def test_unknown_and_timeout_continue_with_owned_cleanup(self):
        row=dict(status='UNKNOWN',session={'cleanup_status':'LEADER_REAPED_NO_LIVE_SESSION','cleanup_error':None})
        self.assertFalse(batch.should_stop(row))
        row['status']='TIMEOUT';self.assertFalse(batch.should_stop(row))
        row['session']['cleanup_status']='CLEANUP_INCOMPLETE';self.assertTrue(batch.should_stop(row))
        row['session']['cleanup_status']='LEADER_REAPED_NO_LIVE_SESSION'
        row['status']='ERROR';self.assertTrue(batch.should_stop(row))

    def test_success_field_cannot_promote_falsified_static_query(self):
        self.assertEqual(worker.obligation_status('verified'),'UNSAT')
        self.assertEqual(worker.obligation_status('falsified'),'UNKNOWN')
        self.assertEqual(worker.obligation_status('unknown'),'UNKNOWN')
        with self.assertRaises(ValueError):worker.obligation_status('unexpected_engine_error')


if __name__=='__main__':unittest.main()
