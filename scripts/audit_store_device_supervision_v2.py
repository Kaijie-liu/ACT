"""Read-only V2 accounting for existing R1 logs; never regenerate candidates.

Nested event times are inclusive. Only depth-zero durations are additive;
child rows are diagnostic and open durations remain right-censored.
"""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import time
import traceback
import unittest
from unittest.mock import patch

from scoped_proof.io import load, save, sha
from scoped_source.hz_request_store import require
from scripts import run_hz_store_device_supervision as prior
from scripts import hz_store_device_supervised as policy


def parse(events,stage):
    stack=[]; rows=[]; previous=stage['start_seconds']
    for e in events:
        policy.finite(e['elapsed'])
        require(previous<=e['elapsed']<=stage['end_seconds'],'event clock order')
        previous=e['elapsed']
        if e['event']=='ENTER':
            stack.append(dict(event=e,depth=len(stack),child_seconds=0.))
        elif e['event'] in ('EXIT','EXIT_ERROR'):
            policy.finite(e['seconds'])
            require(bool(stack),'exit without entry'); frame=stack.pop()
            require(e['operation']==frame['event']['operation']
                    and e['seconds']<=e['elapsed']-frame['event']['elapsed']+.01
                    and frame['child_seconds']<=e['seconds']+.01,'nested duration/identity')
            rows.append(dict(operation=e['operation'],seconds=e['seconds'],status=e['event'],
                             depth=frame['depth'],additive=frame['depth']==0,
                             nested_child_seconds=frame['child_seconds']))
            if stack: stack[-1]['child_seconds']+=e['seconds']
    if stack:
        require(stage['status']!='COMPLETED','unfinished completed operation')
        for frame in stack:
            rows.append(dict(operation=frame['event']['operation'],seconds=None,status='RIGHT_CENSORED',
                depth=frame['depth'],additive=frame['depth']==0,
                observed_lower_seconds=max(0.,stage['end_seconds']-frame['event']['elapsed']-stage['cleanup_seconds'])))
    if stage['status']=='COMPLETED':
        require(events and events[-1].get('event')=='WORKER_COMPLETE'
                and events[-1].get('phase')==stage['phase'],'complete phase event receipt')
        required=['final_serialization']+(['imports','source','plan','candidates'] if stage['phase']=='produce' else [])
        for name in required:
            require(len([r for r in rows if r['operation']==name and r['status']=='EXIT' and r['depth']==0])==1,
                    'complete top-level operation inventory')
    require(sum(r['seconds'] for r in rows if r['additive'] and r['seconds'] is not None)
            <=stage['seconds']+.01,'top-level costs inside phase')
    return rows


def costs(root,terminal):
    result={}
    for stage in terminal['stages']:
        path=root/(stage['phase']+'_events.jsonl')
        if not path.is_file():
            require(stage['status']!='COMPLETED','completed phase lacks events')
            result[stage['phase']]=[]; continue
        result[stage['phase']]=parse([json.loads(s) for s in path.read_text().splitlines()],stage)
    return result


def prefix(name,terminal,events,invocation):
    complete='COMPLETED'
    phases=['admit','produce','release','check','receive']; statuses=[complete]*5
    if name=='admission_busy_stub': phases=phases[:1]; statuses=statuses[:1]
    elif name not in ('half','negative','partial'):
        phases=phases[:3]; statuses=statuses[:3]
        if name=='candidate_exception': statuses[1]='ERROR'
        if name=='readback_delay': statuses[1]='TIMEOUT'
        if name=='release_delay': statuses[2]='TIMEOUT'
    require([(s['phase'],s['status']) for s in terminal['stages']]==list(zip(phases,statuses)),
            'registered stage/status prefix')
    reached=[(phase,e) for phase,e in events if e.get('event')=='FAULT_REACHED']
    if name in ('half','negative','partial'): require(not reached,'normal invocation has fault'); return
    target='admit' if name=='admission_busy_stub' else 'release' if name in ('release_owned_stays','release_delay') else 'produce'
    require(len(reached)==1 and reached[0][0]==target and reached[0][1].get('control')==name
            and reached[0][1].get('invocation')==invocation and reached[0][1].get('physical_cuda') is False,
            'one correctly placed fault witness')


class Controls(unittest.TestCase):
    def sample(self):
        stage=dict(phase='check',start_seconds=0.,end_seconds=3.,seconds=3.,cleanup_seconds=.1,status='COMPLETED')
        e=lambda event,t,op='',seconds=None:dict(event=event,elapsed=t,operation=op,**({} if seconds is None else dict(seconds=seconds)))
        events=[e('ENTER',.1,'outer'),e('ENTER',.2,'inner'),e('EXIT',.4,'inner',.2),e('EXIT',1.1,'outer',1.),
                e('ENTER',1.2,'final_serialization'),e('EXIT',1.3,'final_serialization',.1),
                dict(event='WORKER_COMPLETE',elapsed=1.4,phase='check')]
        return stage,events

    def test_nested_cost_not_double_counted(self):
        stage,events=self.sample(); rows=parse(events,stage)
        self.assertAlmostEqual(sum(r['seconds'] for r in rows if r['additive']),1.1)
        self.assertEqual(rows[0]['depth'],1); self.assertFalse(rows[0]['additive'])

    def test_malformed_costs_rejected(self):
        import copy
        stage,events=self.sample()
        for kind in ('order','duration','identity','missing','nan','stage'):
            bad=copy.deepcopy(events); s=dict(stage)
            if kind=='order': bad[2]['elapsed']=.15
            elif kind=='duration': bad[2]['seconds']=2.
            elif kind=='identity': bad[2]['operation']='foreign'
            elif kind=='missing': bad.pop(1)
            elif kind=='nan': bad[2]['seconds']=float('nan')
            else: s['seconds']=.2
            with self.assertRaises(ValueError): parse(bad,s)

    def test_nested_censoring_not_zero(self):
        stage,events=self.sample(); stage['status']='TIMEOUT'
        rows=parse(events[:2],stage)
        self.assertEqual([r['seconds'] for r in rows],[None,None])
        self.assertEqual([r['additive'] for r in rows],[True,False])
        stage['status']='COMPLETED'
        with self.assertRaises(ValueError): parse(events[:2],stage)

    def test_fault_prefix_and_duplicate_marker(self):
        t=dict(stages=[dict(phase=p,status='COMPLETED') for p in ('admit','produce','release')])
        event=dict(event='FAULT_REACHED',control='pre_cuda_busy_stub',invocation='i',physical_cuda=False)
        prefix('pre_cuda_busy_stub',t,[('produce',event)],'i')
        for es in ([],[('admit',event)],[('produce',event),('produce',event)]):
            with self.assertRaises(ValueError): prefix('pre_cuda_busy_stub',t,es,'i')
        t['stages'].append(dict(phase='check',status='COMPLETED'))
        with self.assertRaises(ValueError): prefix('pre_cuda_busy_stub',t,[('produce',event)],'i')

    def test_batch_api_exception_stops_once(self):
        self.batch(True)

    def test_batch_unknown_release_stops_once(self):
        self.batch(False)

    def batch(self,exception):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            root=Path(tmp); seen=[]
            def invoke(*a,**k):
                seen.append(a)
                if exception: raise RuntimeError('injected API error')
                return dict(status='DEVICE_RELEASE_UNCONFIRMED')
            with patch.object(prior,'load',return_value=dict(stages=[])):
                calls=prior.execute_calls(root,invoke=invoke)
            self.assertEqual(len(seen),1); self.assertEqual(len(calls),1)
            self.assertEqual(load(root/'batch_stop.json')['pending'],[n for n,_ in prior.roster()][1:])
            self.assertFalse(load(root/'pending_after_half.json')['continuation_allowed'])


def run(root):
    started=time.monotonic(); root=Path(root)
    source=Path(__file__); archived=root/'post_audit_v2'/source.name
    archived.parent.mkdir(exist_ok=False)
    shutil.copyfile(source,archived)
    require(sha(source)==sha(archived),'audit extension source snapshot')
    receipt=save(root/'audit_v2_source.json',dict(file=str(source.relative_to(policy.ROOT)),
        sha256=sha(source),archived_file=str(archived.relative_to(root))))
    # Preserve and reproduce the original failure; do not overwrite its code or logs.
    try: prior.audit(root)
    except ValueError as exc:
        require(str(exc)=='unexpected nested operation','unexpected initial audit error')
        save(root/'audit_v1_failure.json',dict(status='FAIL',reason=str(exc),traceback=traceback.format_exc(),
            calls_sha256=sha(root/'calls.json'),kind='offline reproduction of tool-observed initial failure'))
    else: raise ValueError('expected original accounting failure absent')
    class Result(unittest.TextTestResult):
        outcomes=[]
        def startTest(self,t): self.state='PASS'; super().startTest(t)
        def addError(self,t,e): self.state='ERROR'; super().addError(t,e)
        def addFailure(self,t,e): self.state='FAIL'; super().addFailure(t,e)
        def addSkip(self,t,e): self.state='SKIP'; super().addSkip(t,e)
        def stopTest(self,t): self.outcomes.append(dict(test=t.id(),status=self.state)); super().stopTest(t)
    with (root/'audit_v2_tests.log').open('x') as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2,resultclass=Result).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
    save(root/'audit_v2_tests.json',dict(tests=result.testsRun,outcomes=result.outcomes))
    require(result.wasSuccessful() and result.testsRun==6 and all(r['status']=='PASS' for r in result.outcomes),'six accounting controls')
    for name,_ in prior.roster():
        path=root/name; events=[(f.name.removesuffix('_events.jsonl'),json.loads(s))
            for f in path.glob('*_events.jsonl') for s in f.read_text().splitlines()]
        prefix(name,load(path/'terminal.json'),events,load(path/'invocation.json')['invocation'])
    with patch.object(prior,'costs',costs): report=prior.audit(root)
    report.update(audit_version=2,audit_extension=receipt,audit_extension_source_sha256=sha(source),
                  extra_tests=6,extra_tests_sha256=sha(root/'audit_v2_tests.json'),
                  original_audit_failure_sha256=sha(root/'audit_v1_failure.json'),numerical_reruns=0,
                  cost_rule='depth-zero times additive; children diagnostic; missing durations censored',
                  offline_total_before_final_publication_seconds=time.monotonic()-started,
                  offline_cost_scope='includes initial failure replay, pure controls and complete offline audit; not request costs')
    save(root/'audit_v2.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','sources')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('root'); a=p.parse_args(); run(a.root)
