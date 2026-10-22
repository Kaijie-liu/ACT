"""New continuation-only controls using immutable saved prefix, no model loads."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from scoped_proof.io import load,save,sha
from multilayer_stage1_audit import audit

PARENT=Path('/data1/Kane/MOE/results/multilayer_stage1_20261022_r1')
MANIFEST=Path('/data1/Kane/MOE/executions/multilayer_stage1_20261022_r1/docs/multilayer_stage1_freeze_20261022_r2.json')


class ResumeControls(unittest.TestCase):
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
