"""New continuation-only controls using immutable saved prefix, no model loads."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from scoped_proof.io import load,save,sha
from multilayer_stage1_audit import audit,result_chain,check_continuation

PARENT=Path('/data1/Kane/MOE/results/multilayer_stage1_20261022_r1')
MANIFEST=Path('/data1/Kane/MOE/executions/multilayer_stage1_20261022_r1/docs/multilayer_stage1_freeze_20261022_r2.json')


class ResumeControls(unittest.TestCase):
    def test_second_continuation_and_cumulative_binding(self):
        import math
        parent=Path('/data1/Kane/MOE/results/multilayer_stage1_20261022_r3')
        roots=result_chain(parent,MANIFEST)
        self.assertEqual(roots,[PARENT,parent])
        prefix=audit(parent,MANIFEST)
        self.assertEqual(prefix['full_calls'],9)
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE/results',prefix='resume_chain_') as name:
            temp=Path(name);root=temp/'r4';root.mkdir()
            control=temp/'control.json';save(control,dict(status='PASS',seconds=1.))
            carried=math.ceil(load(parent/'stopped.json')['cumulative_seconds']+1.)
            console=parent.with_suffix('.controller.log')
            size=lambda r:sum(p.stat().st_size for p in r.rglob('*') if p.is_file())+r.with_suffix('.controller.log').stat().st_size
            binding=dict(schema='multilayer-stage1-continuation-r2',parent_root=str(parent),
                manifest_sha256=sha(MANIFEST),parent_files={str(p.relative_to(parent)):sha(p) for p in parent.rglob('*') if p.is_file()},
                parent_console=dict(path=str(console),sha256=sha(console)),catalog_sha256=sha(PARENT/'catalog.json'),
                next_verify=9,carried_seconds=carried,prior_bytes=sum(size(r) for r in roots),
                control=dict(path=str(control),sha256=sha(control),seconds=1.))
            save(root/'continuation.json',binding)
            pointer=dict(path=str(root/'continuation.json'),sha256=sha(root/'continuation.json'))
            save(root/'start.json',dict(manifest_sha256=sha(MANIFEST),execution_binding=pointer,
                carried_seconds=carried,entry=1000.,deadline=1000.+52*3600-carried))
            q=load(PARENT/'catalog.json')['queries'][9];target=root/'verify_009';target.mkdir()
            save(target/'request.json',dict(**q,manifest_sha256=sha(MANIFEST),execution_binding=pointer,worker_deadline=100))
            terminal=dict(status='TIMEOUT',accepted=False,through_reception_seconds=.9,hard_budget=300,
                files={'request.json':sha(target/'request.json')},partial=None)
            save(target/'terminal.json',terminal)
            ledger=dict(id='verify_009',kind='verification',**terminal,seconds=1.,model=q['model']['id'],
                index=q['sample']['index'],rank=q['sample']['rank'],arm=q['arm'],terminal_sha256=sha(target/'terminal.json'))
            save(root/'ledger_0035.json',ledger)
            good=audit(root,MANIFEST)
            self.assertEqual((good['full_calls'],good['unexecuted'][0]),(10,'verify_010'))
            self.assertEqual(len(good['unsafe_union']),20)
            self.assertEqual([r['status'] for r in good['rows'] if r['status']=='ERROR'],['ERROR','ERROR'])
            for field,value in [('id','verify_008'),('id','verify_010'),('seconds',301.)]:
                bad=dict(ledger);bad[field]=value;(root/'ledger_0035.json').write_text(json.dumps(bad))
                with self.assertRaises(AssertionError):audit(root,MANIFEST)
            (root/'ledger_0035.json').write_text(json.dumps(ledger))
            for field,value in [('carried_seconds',614),('prior_bytes',binding['prior_bytes']+size(PARENT)),
                                ('next_verify',8),('catalog_sha256','0'*64)]:
                bad=copy.deepcopy(binding);bad[field]=value
                with self.assertRaises(AssertionError):check_continuation(bad,MANIFEST)
            bad=copy.deepcopy(binding);bad['parent_files']['continuation.json']='0'*64
            with self.assertRaises(AssertionError):check_continuation(bad,MANIFEST)
            bad=copy.deepcopy(binding);bad['parent_console']['sha256']='0'*64
            with self.assertRaises(AssertionError):check_continuation(bad,MANIFEST)
            # The ancestor is already in the traversal: refuse a cycle before recursion.
            with self.assertRaises(AssertionError):result_chain(parent,MANIFEST,{PARENT})

    def test_saved_prefix_and_synthetic_continuation_mutations(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE/results',prefix='resume_control_') as name:
            root=Path(name)
            binding=dict(parent_root=str(PARENT),manifest_sha256=sha(MANIFEST),
                parent_files={str(p.relative_to(PARENT)):sha(p) for p in PARENT.rglob('*') if p.is_file()},
                catalog_sha256=sha(PARENT/'catalog.json'),next_verify=5,
                carried_seconds=642.,prior_bytes=sum(p.stat().st_size for p in PARENT.rglob('*') if p.is_file()))
            save(root/'continuation.json',binding)
            pointer=dict(path=str(root/'continuation.json'),sha256=sha(root/'continuation.json'))
            save(root/'start.json',dict(manifest_sha256=sha(MANIFEST),execution_binding=pointer,
                carried_seconds=642.,entry=1000.,deadline=1000.+52*3600-642.))
            q=load(PARENT/'catalog.json')['queries'][5]
            target=root/'verify_005';target.mkdir()
            save(target/'request.json',dict(**q,manifest_sha256=sha(MANIFEST),execution_binding=pointer,worker_deadline=100))
            terminal=dict(status='TIMEOUT',accepted=False,through_reception_seconds=.9,hard_budget=300,
                files={'request.json':sha(target/'request.json')},partial=None)
            save(target/'terminal.json',terminal)
            ledger=dict(id='verify_005',kind='verification',**terminal,seconds=1.,model=q['model']['id'],
                index=q['sample']['index'],rank=q['sample']['rank'],arm=q['arm'],terminal_sha256=sha(target/'terminal.json'))
            save(root/'ledger_0031.json',ledger)
            good=audit(root,MANIFEST)
            self.assertEqual(good['full_calls'],6);self.assertEqual(len(good['unsafe_union']),20)
            self.assertEqual(good['unexecuted'][0],'verify_006')
            self.assertEqual(good['rows'][4]['status'],'ERROR')
            for field,value in [('id','verify_004'),('id','verify_006'),('seconds',301.)]:
                bad=dict(ledger);bad[field]=value;(root/'ledger_0031.json').write_text(json.dumps(bad))
                with self.assertRaises(AssertionError):audit(root,MANIFEST)
            (root/'ledger_0031.json').write_text(json.dumps(ledger))
            bad=copy.deepcopy(binding);bad['parent_files']['stopped.json']='0'*64
            (root/'continuation.json').write_text(json.dumps(bad))
            with self.assertRaises(AssertionError):audit(root,MANIFEST)
            (root/'continuation.json').write_text(json.dumps(binding))
            bad=dict(binding);bad['carried_seconds']=0
            (root/'continuation.json').write_text(json.dumps(bad))
            with self.assertRaises(AssertionError):audit(root,MANIFEST)


if __name__=='__main__':unittest.main()
