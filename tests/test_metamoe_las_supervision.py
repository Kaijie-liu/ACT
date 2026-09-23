import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import metamoe_las_paired as control
from metamoe_csr_execution import supervise, terminal
from audit_metamoe_las_paired import check_events


class SupervisorTests(unittest.TestCase):
    def test_protocol_hooks_restore_after_exception(self):
        original=control.paired.validate,control.paired.command
        with self.assertRaises(RuntimeError):
            with control.protocol_context():
                self.assertIs(control.paired.validate,control.validate)
                raise RuntimeError('control')
        self.assertEqual(original,(control.paired.validate,control.paired.command))

    def test_deadline_partial_and_exception_costs(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE',prefix='las-controls-') as t:
            root=Path(t)
            code="import pathlib,time; pathlib.Path('partial').write_text('observed'); time.sleep(2)"
            r=supervise([sys.executable,'-c',code],str(root),root/'deadline',.15,8*2**30)
            self.assertEqual(r['status'],'TIMEOUT')
            self.assertGreaterEqual(r['execution_including_preflight_seconds'],.15)
            (root/'deadline/result.json').write_text(json.dumps({'status':'BACKEND_POSITIVE'}))
            self.assertEqual(terminal(root/'deadline',r)['status'],'TIMEOUT')
            r=supervise([sys.executable,'-c',"raise RuntimeError('synthetic')"],str(root),root/'exception',2.,8*2**30)
            self.assertEqual(r['status'],'ERROR')
            self.assertGreater(r['total_with_postflight_seconds'],0)

    def test_event_identity_and_unjustified_restore_refused(self):
        install={'seq':0,'request_config_sha256':'h','monotonic':1.,'event':'INSTALLED',
                 'repair':'disconnected_zero_branching_metadata_v1','bounds_unchanged':True}
        self.assertEqual(check_events([install],'h')['restorations'],0)
        with self.assertRaisesRegex(ValueError,'identity'):check_events([install],'bad')
        bad={'seq':1,'request_config_sha256':'h','monotonic':2.,'event':'RESTORE_DISCONNECTED_ZERO'}
        with self.assertRaisesRegex(ValueError,'schema'):check_events([install,bad],'h')


if __name__=='__main__':unittest.main()
