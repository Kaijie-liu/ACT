"""Fixed standard-library stage4 ledger controls; zero native/model calls."""
import contextlib
import io
import os
from pathlib import Path
import shutil
import time
import unittest

import gate_decoupling_stage4 as stage


def objects(count=100):
    return [dict(id=f'fixed_{i:03d}', group='singlelayer' if i % 2 else 'multilayer') for i in range(count)]


def row(obj, variant, status):
    return dict(object_id=obj['id'], variant=variant, status=status, seconds=.01,
        accepted=status in ('POSITIVE', 'DIAGNOSTIC_CLOSED', 'UNKNOWN'), last_phase=None,
        evidence=dict(closed=status in ('POSITIVE', 'DIAGNOSTIC_CLOSED')))


class BatchControls(unittest.TestCase):
    def test_full_denominator_missing_errors_and_unexecuted(self):
        obs = objects(4)
        rows = [row(obs[0], 'V2', 'POSITIVE'), row(obs[0], 'V3', 'DIAGNOSTIC_CLOSED'),
                row(obs[1], 'V2', 'ERROR'), row(obs[1], 'V3', 'DIAGNOSTIC_CLOSED'),
                row(obs[2], 'V2', 'SOURCE_MISSING')]
        result = stage.summarize(obs, rows, terminal=True)['groups']['all']
        self.assertEqual(result['objects'], 4)
        self.assertEqual(result['counts'], dict(v3_timeout=0, both_closed=1, v3_closed_v2_observed_unclosed=0))
        self.assertEqual(result['status_counts']['V3']['UNEXECUTED'], 2)
        self.assertEqual(result['decision'], 'GRAY_REPORT')
        with self.assertRaises(ValueError): stage.summarize(obs, rows+[rows[0]], terminal=True)

    def test_registered_decisions_and_no_prefix_decision(self):
        obs = objects()
        for kind, expected in [('a', 'STOP_ALGORITHM_LINE_USER_RULE_A'),
                               ('b', 'ENTER_STAGE5_USER_RULE_B'),
                               ('c', 'MATH_CONTRACT_FOR_USER_REVIEW_RULE_C')]:
            rows = []
            for i, obj in enumerate(obs):
                v2 = 'POSITIVE' if kind == 'b' and i < 50 else 'UNKNOWN'
                v3 = ('TIMEOUT' if kind == 'a' and i < 70 else
                      'DIAGNOSTIC_CLOSED' if kind == 'b' and i < 50 or kind == 'c' and i < 60 else 'UNKNOWN')
                rows.extend((row(obj, 'V2', v2), row(obj, 'V3', v3)))
            self.assertEqual(stage.summarize(obs, rows, terminal=True)['primary_decision'], expected)
            self.assertEqual(stage.summarize(obs, rows, terminal=False)['primary_decision'], 'PENDING_REGISTERED_QUEUE')
        self.assertAlmostEqual(stage.wilson(0, 10)[0], 0.)
        self.assertAlmostEqual(stage.wilson(10, 10)[1], 1.)
        self.assertIsNone(stage.wilson(0, 0))
        with self.assertRaises(ValueError): stage.wilson(3, 2)

    def test_fixed_order_cap_and_error_prefix(self):
        obs = objects(3); plan = stage.calls(obs)
        self.assertEqual([r['variant'] for r in plan], ['V2', 'V3', 'V3', 'V2', 'V2', 'V3'])
        root = Path(os.environ['GATE_CONTROL_ROOT'])/self._testMethodName; root.mkdir()
        spec = dict(calls=plan, budgets=dict(local_calls=4, artifacts=1024*1024))
        for name, error_at, count, expected in [('cap', None, 4, 'LOCAL_CALL_CAP'),
                ('error', 1, 2, 'TOOL_OR_RECEPTION_ERROR')]:
            folder = root/name; folder.mkdir()
            def invoke(child, manifest, index, *, stage_deadline):
                call = plan[index]
                return row(next(o for o in obs if o['id'] == call['object_id']), call['variant'],
                           'ERROR' if index == error_at else 'UNKNOWN')
            with contextlib.redirect_stdout(io.StringIO()):
                rows, reason = stage.execute_queue(folder, spec, None, time.monotonic()+600, invoke=invoke)
            self.assertEqual((len(rows), reason), (count, expected))
            self.assertEqual(len(list(folder.glob('started_*.json'))), count)
        folder = root/'no-time'; folder.mkdir()
        rows, reason = stage.execute_queue(folder, spec, None, time.monotonic()+1,
            invoke=lambda *args, **kwargs: self.fail('must not start shortened request'))
        self.assertEqual((rows, reason), ([], 'STAGE_TIME_CAP'))

    def test_failed_start_is_retained_and_not_retried(self):
        root = Path(os.environ['GATE_CONTROL_ROOT'])/self._testMethodName; root.mkdir()
        spec = dict(calls=stage.calls(objects(1)), budgets=dict(local_calls=250, artifacts=1024*1024))
        seen = []
        def fail(*args, **kwargs):
            seen.append(1); raise OSError('fixed launch failure')
        with self.assertRaises(OSError):
            stage.execute_queue(root, spec, None, time.monotonic()+600, invoke=fail)
        self.assertEqual(len(seen), 1)
        self.assertEqual(len(list(root.glob('started_*.json'))), 1)
        self.assertFalse(list(root.glob('ledger_*.json')))

    def test_continuation_counts_old_attempts_and_artifacts(self):
        obs=objects(3);plan=stage.calls(obs)
        root=Path(os.environ['GATE_CONTROL_ROOT'])/self._testMethodName;root.mkdir()
        spec=dict(calls=plan,budgets=dict(local_calls=4,artifacts=1024*1024),
                  prior_accounting=dict(attempts=3,bytes=0,seconds=10))
        def invoke(child,manifest,index,*,stage_deadline):
            return row(obs[0],plan[index]['variant'],'UNKNOWN')
        first=root/'attempts';first.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            rows,reason=stage.execute_queue(first,spec,None,time.monotonic()+600,invoke=invoke)
        self.assertEqual((len(rows),reason),(1,'LOCAL_CALL_CAP'))
        second=root/'bytes';second.mkdir();spec['prior_accounting']['bytes']=900*1024
        rows,reason=stage.execute_queue(second,spec,None,time.monotonic()+600,
            invoke=lambda *a,**kw:self.fail('must account for archived bytes before launching'))
        self.assertEqual((rows,reason),([],'ARTIFACT_CAP'))
        for key,value in (('seconds',-1),('attempts',-1),('bytes',-1),('seconds',float('nan'))):
            bad=dict(spec,prior_accounting=dict(spec['prior_accounting'],**{key:value}))
            with self.assertRaises(ValueError):
                stage.execute_queue(second,bad,None,time.monotonic()+600,
                    invoke=lambda *a,**kw:self.fail('invalid prior cost must not launch'))

    def test_audit_binds_raw_terminal_and_failed_attempt(self):
        root = Path(os.environ['GATE_CONTROL_ROOT'])/self._testMethodName; root.mkdir()
        w = stage.worker; obj = dict(id='missing', group='singlelayer', history=None)
        catalog = root/'catalog.json'; w.save(catalog, dict(objects=[obj]))
        spec = dict(source_files={}, backend_identity='synthetic',
            catalog=dict(path=str(catalog), sha256=w.sha(catalog)), calls=stage.calls([obj]),
            budgets=dict(artifacts=1024*1024, stage=21*3600))
        manifest = root/'manifest.json'; w.save(manifest, spec)
        sid = w.digest(dict(object=obj, sources={}, backend='synthetic'))
        folder = root/'run'; folder.mkdir()
        w.save(folder/'start.json', dict(manifest_sha256=w.sha(manifest)))
        call = spec['calls'][0]; child = folder/'call_000'; child.mkdir()
        now = time.monotonic()
        w.save(folder/'started_000.json', dict(index=0, call=call))
        w.save(child/'request.json', dict(worker_deadline=now+100))
        session = dict(status='COMPLETED', returncode=0, execution_error=None,
                       cleanup_error=None, cleanup_status='LEADER_REAPED_NO_LIVE_SESSION')
        w.save(child/'session.json', session)
        value = dict(source_id=sid, group='V2', status='SOURCE_MISSING', closed=False,
            original_history_closed=False, formal_SAFE=False, complete_MoE=False, endpoints=[])
        w.save(child/'outcome.json', dict(value=value, request_sha256=w.sha(child/'request.json'), clock=now))
        w.save(child/'published.json', dict(sha256=w.sha(child/'outcome.json'), clock=now))
        terminal = dict(object_id='missing', variant='V2', status='SOURCE_MISSING', accepted=True,
            evidence=value, files={p.name:w.sha(p) for p in child.iterdir() if p.is_file()})
        w.save(child/'terminal.json', terminal)
        w.save(folder/'ledger_000.json', dict(terminal, seconds=.1, terminal_sha256=w.sha(child/'terminal.json')))
        w.save(folder/'started_001.json', dict(index=1, call=spec['calls'][1]))
        result = stage.audit(folder, manifest)
        self.assertEqual((result['attempted_local_calls'], result['actual_local_calls']), (2, 1))
        self.assertEqual(result['started_without_terminal'], 1)
        self.assertEqual(result['unexecuted'], [])
        self.assertEqual(result['groups']['all']['status_counts']['V3']['STARTED_NO_TERMINAL'], 1)
        self.assertEqual(result['primary_decision'], 'PENDING_REGISTERED_QUEUE')
        bad = root/'mutated'; shutil.copytree(folder, bad)
        (bad/'call_000'/'outcome.json').write_text('{}')
        with self.assertRaises(ValueError): stage.audit(bad, manifest)


if __name__ == '__main__':
    unittest.main()
