"""Process-lifetime/fault controls; no native optimization or real LP queries."""
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest

from evidence_cohort.ownership import info,stop_owned
from soplex_execution.runtime import ROOT,PYTHON,env
from soplex_fidelity.io import save,sha
from soplex_detached.run import launch_tmux,reconcile,postmortem,boot,same_process,same_science

RAW=Path(tempfile.mkdtemp(prefix='soplex_life_',dir='/data1/Kane/MOE'))
JOBS=[{'job_id':str(i)} for i in range(4)]


def until(test,seconds=8):
    end=time.monotonic()+seconds
    while not test():
        if time.monotonic()>end:raise TimeoutError('control condition')
        time.sleep(.05)


class Controls(unittest.TestCase):
    def test_only_execution_path_can_change(self):
        original={'output':'old','policy':{'seconds':300},'jobs':[1,2,3,4]}
        same_science({**original,'output':'new'},original)
        for change in ({'policy':{'seconds':301}},{'jobs':[1,2,3]},{'new_option':True}):
            with self.assertRaises(ValueError):same_science({**original,'output':'new',**change},original)

    def test_detached_survives_launch_tree_death_and_records_completion(self):
        root=RAW/'detach';root.mkdir();socket=RAW/'d.sock'
        child="import time,pathlib; time.sleep(2); pathlib.Path("+repr(str(root/'batch.json'))+").write_text('{}')"
        target="from pathlib import Path; from soplex_detached.run import guard; guard(Path("+repr(str(root))+"),"+repr(JOBS)+","+repr([str(PYTHON),'-c',child])+")"
        launcher="import time; from pathlib import Path; from soplex_detached.run import launch_tmux; launch_tmux(Path("+repr(str(socket))+"),Path("+repr(str(root))+"),"+repr([str(PYTHON),'-c',target,str(root)])+"); time.sleep(30)"
        proc=subprocess.Popen([str(PYTHON),'-c',launcher],cwd=ROOT,env=env(),start_new_session=True)
        record=info(proc.pid)
        try:
            until(lambda:(root/'heartbeat.json').exists());g=json.loads((root/'guardian_identity.json').read_text())['process']
            stop_owned(record);proc.wait(timeout=5)
            self.assertTrue(same_process(g))
            until(lambda:(root/'guardian_terminal.json').exists())
            self.assertEqual(json.loads((root/'guardian_terminal.json').read_text())['status'],'CONTROLLER_COMPLETE')
            self.assertFalse((root/'interruption.json').exists())
        finally:
            if proc.poll() is None:stop_owned(record);proc.wait()
            if socket.exists():subprocess.run(['/usr/bin/tmux','-S',str(socket),'kill-server'],capture_output=True)

    def test_controller_sigkill_keeps_four_slots_and_kills_orphan(self):
        root=RAW/'killed';root.mkdir()
        child="import pathlib,subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True); pathlib.Path("+repr(str(root/'orphan.pid'))+").write_text(str(p.pid)); time.sleep(30)"
        code="from pathlib import Path; from soplex_detached.run import guard; guard(Path("+repr(str(root))+"),"+repr(JOBS)+","+repr([str(PYTHON),'-c',child])+")"
        proc=subprocess.Popen([str(PYTHON),'-c',code],cwd=ROOT,env=env(),start_new_session=True);record=info(proc.pid)
        try:
            until(lambda:(root/'orphan.pid').exists())
            controller=json.loads((root/'controller_identity.json').read_text())['process'];os.kill(controller['pid'],signal.SIGKILL)
            proc.wait(timeout=8);r=json.loads((root/'interruption.json').read_text())
            self.assertEqual(len(r['rows']),4);self.assertTrue(all(v['upper_bound'] is None for v in r['rows']))
            orphan=info(int((root/'orphan.pid').read_text()));self.assertTrue(orphan is None or orphan['state']=='Z')
        finally:
            if proc.poll() is None:stop_owned(record);proc.wait()

    def test_reconcile_live_identity_and_stale_pid(self):
        root=RAW/'identity';root.mkdir();r=info(os.getpid())
        save(root/'guardian_identity.json',dict(process=r,boot_id=boot()))
        self.assertEqual(reconcile(root,JOBS),'LIVE_NO_MUTATION')
        r['start']+=1
        (root/'guardian_identity.json').write_text(json.dumps(dict(process=r,boot_id=boot())))
        self.assertEqual(reconcile(root,JOBS),'RECORDED_INTERRUPTION')
        digest=sha(root/'interruption.json');self.assertEqual(reconcile(root,JOBS),'RECORDED_INTERRUPTION')
        self.assertEqual(sha(root/'interruption.json'),digest)

    def test_partial_costs_and_malformed_tail_not_zero_or_success(self):
        root=RAW/'partial';root.mkdir();jr=root/'0';jr.mkdir()
        save(jr/'spec.json',{});save(jr/'worker_result.json',{'status':'CHECKED','upper_bound':'-1'})
        (root/'resource_wait.jsonl').write_text(json.dumps({'job_id':'0','seconds':42,'passed':False})+'\n{"partial"')
        r=postmortem(root,JOBS,'control death')
        self.assertEqual(r['rows'][0]['status'],'INTERRUPTED_DURING_JOB')
        self.assertIsNone(r['rows'][0]['complete_request_seconds']);self.assertIsNone(r['rows'][0]['upper_bound'])
        self.assertEqual(r['rows'][0]['last_resource_observation']['seconds'],42)
        self.assertTrue(r['malformed_last_wait_line'])

    def test_completed_record_requires_audit_not_blind_acceptance(self):
        root=RAW/'completed';root.mkdir();save(root/'0.summary.json',{'status':'CHECKED','upper_bound':'-1'})
        r=postmortem(root,JOBS,'late interruption')
        self.assertEqual(r['rows'][0]['status'],'COMPLETED_RECORD_REQUIRES_AUDIT');self.assertIsNone(r['rows'][0]['upper_bound'])

    def test_duplicate_socket_rejected(self):
        root=RAW/'dup';root.mkdir();socket=root/'exists';socket.touch()
        with self.assertRaises(FileExistsError):launch_tmux(socket,root,['false'])


if __name__=='__main__':
    start=time.monotonic();stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
    (RAW/'unittest.txt').write_text(stream.getvalue())
    receipt=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        errors=len(result.errors),failures=len(result.failures),raw_root=str(RAW),seconds=time.monotonic()-start,
        real_queries=0,native_queries=0,sources={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')})
    save(RAW/'controls.json',receipt);print(stream.getvalue());print(json.dumps(receipt));raise SystemExit(0 if result.wasSuccessful() else 1)
