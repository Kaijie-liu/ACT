"""Analytic native calls and fault controls only; never imports a real study LP."""
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from lp_sandwich.check import identity,check
from soplex_compat.controls import cases,lp
from soplex_fidelity.io import save,sha
from soplex_execution.candidate import parse,token,LimitError,bundle
from soplex_execution.runtime import ROOT,PYTHON,output_sizes
from soplex_execution.supervisor import supervise,FREEZE,publication_valid
from soplex_execution.supervisor import run_jobs,resource_gate
from soplex_execution.audit import audit_job

RAW=Path(tempfile.mkdtemp(prefix='soplex_execution_controls_',dir=ROOT/'data/moe/results'))
FROZEN=json.loads(FREEZE.read_text())
RUNTIME={k:FROZEN['runtime'][k] for k in ('soplex','reader','settings')}
RUNTIME['checker']=FROZEN['independent_checker']
POLICY=FROZEN['policy']
NATIVE=[]


def fixture(name,p):
    source={'analytic_fixture':name};v=dict(source=source,source_sha256=identity(source),q=[1,-1],offset=0,lp=p)
    path=RAW/(name+'.export.json');save(path,v)
    s=dict(schema='LP_OBLIGATION_IDENTITY_V1',request_id='1'*64,source_sha256=identity(source),
           export_sha256=sha(path),pair=[0,1],property_index=0,property={'q':[1,-1],'constant':0},
           lp_sha256=identity(p),acceptance_threshold=0)
    return dict(job_id=name,export=dict(path=str(path),sha256=sha(path)),statement=s,statement_sha256=identity(s))


class Controls(unittest.TestCase):
    def point(self,text):
        path=RAW/f'point{time.time_ns()}.txt';path.write_text(text);return path

    def test_native_all_analytic_controls_with_full_cost_audit(self):
        for name,p,expected in cases():
            with self.subTest(case=name):
                job=fixture(name,p);root=RAW/name
                r=supervise(job,root,RUNTIME,POLICY)
                self.assertEqual(r['status'],'CHECKED',r)
                self.assertEqual(r['upper_bound'],expected)
                reviewed=audit_job(job,root,RAW/(name+'_review'),POLICY,RUNTIME)
                self.assertEqual(reviewed['upper_bound'],expected)
                self.assertEqual(reviewed['solver_attempts'],1)
                self.assertGreater(r['whole_request_seconds'],r['accounted_phase_seconds'])
                self.assertFalse(r['network_UNSAFE']);self.assertFalse(r['network_SAFE'])
                NATIVE.append(dict(case=name,upper_bound=expected,seconds=r['whole_request_seconds']))

    def test_complete_sparse_zero_and_fraction(self):
        p=self.point('\nPrimal solution (name, value):\nx1\t-1/3\nAll other variables are zero. Solution has 1 nonzero entries.\n')
        x,r=parse(p,3);self.assertEqual(list(map(str,x)),['0','-1/3','0']);self.assertEqual(r['max_bits'],2)
        p=self.point('\nPrimal solution (name, value):\nAll other variables are zero. Solution has 0 nonzero entries.\n')
        self.assertEqual(list(map(str,parse(p,2)[0])),['0','0'])

    def test_duplicate_unknown_count_and_nonexact_rejection(self):
        head='\nPrimal solution (name, value):\n';foot='All other variables are zero. Solution has 1 nonzero entries.\n'
        for middle in ('x0 0.5\n','x0 1e-8\n','x9 1\n','x00 1\n','x0 0\n','x0 1/0\n','x0 1\nx0 1\n'):
            with self.subTest(middle=middle),self.assertRaises(ValueError):parse(self.point(head+middle+foot),1)
        for text in (head+'x0 1\n',head+'x0 1\n'+foot.rstrip(),head+'x0 1\n'+foot+'junk\n',
                     'Primal ray (name, value):\nx0 1\nAll other entries are zero.\n'):
            with self.assertRaises(ValueError):parse(self.point(text),1)

    def test_missing_and_no_point_not_feasible(self):
        for path in (RAW/'absent.txt',self.point('No primal (rational) solution available.\n')):
            self.assertIsNone(parse(path,1)[0])

    def test_bit_and_byte_caps_before_fraction(self):
        for value in (str(2**4096),'1/'+str(2**4096),'9'*10000):
            with self.assertRaises(LimitError):token(value)
        with self.assertRaises(LimitError):parse(self.point('x'*50),1,byte_cap=10)

    def test_wrong_binding_and_inexact_candidate_never_upper(self):
        p=lp(e=[[3]],h=[1]);j=fixture('binding',p)
        b=bundle(p,j['statement'],[token('1/2')])
        r=check(b,j['statement_sha256']);self.assertIsNone(r['upper_bound']);self.assertEqual(r['primal_status'],'NOT_EXACTLY_FEASIBLE')
        b['primal']['statement_sha256']='0'*64
        with self.assertRaises(ValueError):check(b,j['statement_sha256'])

    def test_parser_and_packaging_deadline(self):
        def tick():raise TimeoutError('test deadline')
        with self.assertRaises(TimeoutError):parse(RAW/'absent',1,tick)
        with self.assertRaises(TimeoutError):bundle(lp(),{},None,tick)

    def test_outer_deadline_preserves_censored_phase_and_kills_owned_child(self):
        job=fixture('deadline',lp());root=RAW/'deadline'
        policy={**POLICY,'proposal_seconds':.4,'work_seconds':.8,'total_seconds':2}
        script=("import pathlib,json,time,subprocess,sys; r=pathlib.Path(sys.argv[1]); "
                "s=json.loads((r/'spec.json').read_text()); "
                "(r/'solve.entered.json').write_text(json.dumps({'start_offset':time.monotonic()-s['start'],'deadline_offset':s['proposal']})); "
                "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True); "
                "(r/'child.pid').write_text(str(p.pid)); (r/'point.txt').write_text('partial'); time.sleep(30)")
        r=supervise(job,root,RUNTIME,policy,[str(PYTHON),'-c',script,str(root)])
        self.assertEqual(r['status'],'TIMEOUT');self.assertIsNone(r['upper_bound'])
        self.assertEqual(r['phase_costs']['solve']['state'],'CENSORED');self.assertIsNone(r['phase_costs']['solve']['seconds'])
        self.assertGreater(r['phase_costs']['solve']['observed_seconds'],0)
        self.assertTrue((root/'point.txt').exists())
        pid=int((root/'child.pid').read_text());path=Path(f'/proc/{pid}/stat')
        if path.exists():self.assertEqual(path.read_text().split(')')[1].split()[0],'Z')
        audit_job(job,root,RAW/'deadline_review',policy,RUNTIME)

    def test_shared_work_deadline_not_new_clock(self):
        job=fixture('check_deadline',lp());root=RAW/'check_deadline'
        policy={**POLICY,'proposal_seconds':.3,'work_seconds':.7,'total_seconds':2}
        script=("import pathlib,json,time,sys; r=pathlib.Path(sys.argv[1]); s=json.loads((r/'spec.json').read_text()); "
                "time.sleep(.12); (r/'package.entered.json').write_text(json.dumps({'start_offset':time.monotonic()-s['start'],'deadline_offset':s['work']})); time.sleep(30)")
        r=supervise(job,root,RUNTIME,policy,[str(PYTHON),'-c',script,str(root)])
        self.assertEqual(r['status'],'TIMEOUT');self.assertGreater(r['observed_worker_seconds'],.6);self.assertLess(r['observed_worker_seconds'],1.2)
        audit_job(job,root,RAW/'check_deadline_review',policy,RUNTIME)

    def test_error_and_partial_never_checked(self):
        job=fixture('error',lp());root=RAW/'error'
        r=supervise(job,root,RUNTIME,POLICY,[str(PYTHON),'-c','raise RuntimeError("control failure")'])
        self.assertEqual(r['status'],'ERROR');self.assertIsNone(r['upper_bound'])
        self.assertIsNone(r['phase_costs']['solve']['seconds'])
        audit_job(job,root,RAW/'error_review',POLICY,RUNTIME)

    def test_source_drift_fails_before_solver(self):
        job=fixture('drift',lp());job['export']['sha256']='0'*64
        r=supervise(job,RAW/'drift',RUNTIME,POLICY)
        self.assertEqual(r['status'],'ERROR');self.assertFalse((RAW/'drift/solver.command.json').exists())

    def test_late_publication_rejects_even_valid_hash(self):
        self.assertFalse(publication_valid({'sha256':'a'},dict(terminal_sha256='a',published_offset=300.1,serialization_started_offset=299),300))
        self.assertFalse(publication_valid({'sha256':'a'},dict(terminal_sha256='b',published_offset=10,serialization_started_offset=9),300))

    def test_output_limits_and_null_cost_tamper(self):
        r=RAW/'size_cap';r.mkdir()
        with (r/'point.txt').open('wb') as f:f.truncate(64*1024**2+1)
        with self.assertRaises(LimitError):output_sizes(r)

    def test_fail_stop_preserves_four_slots(self):
        root=RAW/'batch_control';root.mkdir()
        jobs=[{'job_id':str(i)} for i in range(4)]
        with patch('soplex_execution.supervisor.resource_gate',return_value={'seconds':0}),patch(
             'soplex_execution.supervisor.supervise',return_value={'job_id':'0','status':'ERROR'}) as worker:
            result=run_jobs(jobs,root,{},POLICY,lambda:None)
        self.assertEqual(worker.call_count,1);self.assertEqual(result['denominator'],4)
        self.assertEqual([r['status'] for r in result['rows']],['ERROR']+['NOT_RUN_AFTER_ERROR']*3)

    def test_limits_continue_roster_without_retry(self):
        root=RAW/'batch_limits';root.mkdir();jobs=[{'job_id':str(i)} for i in range(4)]
        results=[{'job_id':str(i),'status':s} for i,s in enumerate(('TIMEOUT','LIMIT','CHECKED','CHECKED'))]
        with patch('soplex_execution.supervisor.resource_gate',return_value={'seconds':0}),patch(
             'soplex_execution.supervisor.supervise',side_effect=results) as worker:
            result=run_jobs(jobs,root,{},POLICY,lambda:None)
        self.assertEqual(worker.call_count,4);self.assertEqual(len(result['rows']),4)

    def test_resource_gate_no_free_solver_time(self):
        root=RAW/'resource_gate';root.mkdir()
        bad=dict(available_ram_gib=100,free_disk_gib=100,load_per_core=.6)
        with patch('soplex_execution.supervisor.resources',return_value=bad):
            with self.assertRaises(TimeoutError):resource_gate(root,'analytic',{**POLICY['resource'],'wait_limit_seconds':0})
        good={**bad,'load_per_core':.4}
        with patch('soplex_execution.supervisor.resources',return_value=good):
            self.assertEqual(resource_gate(root,'analytic',POLICY['resource'])['polls'],1)

    def test_worker_address_space_threads_and_nice(self):
        j=fixture('process_limits',lp());root=RAW/'process_limits'
        script="import os,resource,json; print(json.dumps([resource.getrlimit(resource.RLIMIT_AS),resource.getrlimit(resource.RLIMIT_FSIZE),os.getpriority(os.PRIO_PROCESS,0),os.environ['OMP_NUM_THREADS'],os.environ['CUDA_VISIBLE_DEVICES']]))"
        supervise(j,root,RUNTIME,POLICY,[str(PYTHON),'-c',script])
        values=json.loads((root/'worker.stdout').read_text())
        self.assertEqual(values,[[8*1024**3]*2,[128*1024**2]*2,10,'1',''])

    def test_cost_tampering_is_rejected(self):
        j=fixture('cost_tamper',lp());root=RAW/'cost_tamper'
        supervise(j,root,RUNTIME,POLICY,[str(PYTHON),'-c','pass'])
        t=json.loads((root/'terminal.json').read_text());t['phase_costs']['solve']['seconds']=0
        (root/'terminal.json').write_text(json.dumps(t))
        p=json.loads((root/'publication.json').read_text());p['terminal_sha256']=sha(root/'terminal.json')
        (root/'publication.json').write_text(json.dumps(p))
        with self.assertRaisesRegex(ValueError,'null'):audit_job(j,root,RAW/'cost_tamper_review',POLICY,RUNTIME)


def main():
    started=time.monotonic();stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
    (RAW/'unittest.txt').write_text(stream.getvalue())
    receipt=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),raw_root=str(RAW),
        analytic_native_queries=len(list(RAW.glob('*/solver.command.json'))),real_queries=0,
        seconds=time.monotonic()-started,native_rows=NATIVE,
        sources={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')},
        artifacts={str(p.relative_to(RAW)):sha(p) for p in RAW.rglob('*') if p.is_file() and p.stat().st_size<1024**2})
    save(RAW/'controls.json',receipt);print(stream.getvalue());print(json.dumps({k:v for k,v in receipt.items() if k not in ('artifacts','sources','native_rows')}))
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':sys.exit(main())
