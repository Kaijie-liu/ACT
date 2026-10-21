"""Queue and receipt controls; no trained model, dataset or solver calls."""
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('b1_observed_batch',
    Path(__file__).resolve().parents[1]/'scripts/b1_observed_batch.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class BatchControls(unittest.TestCase):
    def alarm_child(self, body):
        code=(f'import sys; sys.path.insert(0, {str(Path(m.__file__).parent)!r}); '
              'import b1_observed_batch as m; import signal,time; '+body)
        return subprocess.run([sys.executable,'-I','-B','-c',code],capture_output=True,timeout=5)

    def test_completed_work_cancels_alarm_before_slow_teardown(self):
        result=self.alarm_child('signal.signal(signal.SIGALRM,m.worker_cutoff); '
            'signal.setitimer(signal.ITIMER_REAL,.05); m.finish_worker_io(); '
            'signal.signal(signal.SIGALRM,signal.SIG_DFL); time.sleep(.12)')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertNotIn(b'B1_AUTHORIZED_WORKER_CUTOFF',result.stderr)

    def test_work_deadline_still_uses_authorized_cutoff(self):
        result=self.alarm_child('signal.signal(signal.SIGALRM,m.worker_cutoff); '
            'signal.setitimer(signal.ITIMER_REAL,.05); time.sleep(.12)')
        self.assertEqual(result.returncode,124,result.stderr)
        self.assertIn(b'B1_AUTHORIZED_WORKER_CUTOFF',result.stderr)

    def fixture(self, root):
        prefix=root/'prefix';prefix.mkdir()
        items=[dict(ordinal=i,job_id=f'{i:02d}',model='seed0',dataset_index=4200+i) for i in range(3)]
        def write(p,v):
            p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
        write(prefix/'start.json',dict(freeze_sha256='freeze',commit='old'))
        write(prefix/'online_complete.json',dict(status='PREFIX_STOP',completed_calls=2,
              reserved_full_ordinals=1,stage_elapsed=600.5))
        for item in items[:2]:
            folder=prefix/item['job_id'];status='UNKNOWN' if item['ordinal']==0 else 'ERROR'
            write(folder/'launch.json',dict(item=item,freeze_sha256='freeze',commit='old',started_at=10.))
            session=dict(returncode=0 if status=='UNKNOWN' else -signal.SIGALRM,
                         exit_observed_at=304.2,execution_error=None,
                         cleanup_status='LEADER_REAPED_NO_LIVE_SESSION')
            write(folder/'terminal.json',dict(ordinal=item['ordinal'],status=status,session=session))
            write(folder/'reception.json',dict(status=status,accepted=status=='UNKNOWN',
                  through_terminal_ledger_seconds=294.3))
            write(folder/'diagnostics/000_inventory_end.json',dict(full_capture_events=[{}] if item['ordinal']==0 else []))
        return prefix,dict(requests=items)

    def test_finished_prefix_is_not_retried_and_capture_is_not_reset(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            root=Path(tmp);prefix,spec=self.fixture(root)
            retained=m.retained_prefix(prefix,spec,'freeze')
            self.assertEqual([r['status'] for r in retained['rows']],['UNKNOWN','ERROR'])
            self.assertEqual(spec['requests'][len(retained['rows']):][0]['ordinal'],2)
            self.assertEqual(retained['reserved'],1)
            self.assertEqual(retained['stage_elapsed'],600.5)
            current=root/'current';current.mkdir()
            (current/'start.json').write_text(json.dumps(dict(freeze_sha256='freeze',
                retained_requests=retained['refs'],retained_online_roots=[str(prefix),str(prefix)])))
            self.assertEqual(m.resolve_request_folder(current,spec['requests'][0]),prefix/'00')
            self.assertEqual(m.resolve_request_folder(current,spec['requests'][2]),current/'02')
            self.assertEqual(m.online_sizes(current),m.sizes(current)+m.sizes(prefix))
            (prefix/'00/reception.json').write_text('{}')
            with self.assertRaises(ValueError):m.resolve_request_folder(current,spec['requests'][0])

    def test_prefix_rejects_changed_model_or_incomplete_cleanup(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            root=Path(tmp);prefix,spec=self.fixture(root)
            spec['requests'][0]['model']='other'
            with self.assertRaises(ValueError):m.retained_prefix(prefix,spec,'freeze')
            spec['requests'][0]['model']='seed0'
            p=prefix/'01/terminal.json';v=json.loads(p.read_text())
            v['session']['cleanup_status']='UNKNOWN';p.write_text(json.dumps(v))
            with self.assertRaises(ValueError):m.retained_prefix(prefix,spec,'freeze')

    def test_late_alarm_is_diagnostic_not_execution_acceptance(self):
        r=dict(status='ERROR',returncode=-signal.SIGALRM,exit_observed_at=304.2,
               execution_error=None,cleanup_error=None,cleanup_status='LEADER_REAPED_NO_LIVE_SESSION')
        self.assertTrue(m.expired_alarm_observation(r,10.))
        self.assertFalse(m.session_ok(r))
        self.assertFalse(m.expired_alarm_observation(dict(r,exit_observed_at=303.),10.))
        self.assertFalse(m.expired_alarm_observation(dict(r,returncode=-signal.SIGSEGV),10.))

    def test_order_and_sealed_exclusion(self):
        rows = [dict(method='adaptive',status='UNKNOWN',dataset_index=i,model=s)
                for i,s in [(4150,'seed2'),(4096,'seed1'),(4141,'seed2'),(4150,'seed1')]]
        rows += [dict(method='matched',status='UNKNOWN',dataset_index=4100,model='seed0'),
                 dict(method='adaptive',status='SAFE',dataset_index=4100,model='seed0')]
        self.assertEqual([(r['dataset_index'],r['model']) for r in m.select(rows)],
                         [(4141,'seed2'),(4150,'seed1'),(4150,'seed2')])

    def test_outer_terminal_cannot_be_replaced_by_outcome_file(self):
        r = dict(status='COMPLETED', returncode=0, execution_error=None,
                 cleanup_error=None,cleanup_status='LEADER_REAPED_NO_LIVE_SESSION')
        self.assertTrue(m.session_ok(r))
        for k,v in [('status','TIMEOUT'),('returncode',1),('returncode',124),('execution_error',{}),
                    ('cleanup_error',{}),('cleanup_status','CLEANUP_INCOMPLETE')]:
            self.assertFalse(m.session_ok(dict(r,**{k:v})))


if __name__ == '__main__': unittest.main()
