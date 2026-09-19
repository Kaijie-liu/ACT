"""Analytic supervised LP controls; never select or solve real network LPs."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from single_check_portable.execution import ROOT,ACT,read,save_new
from portable_proof.runtime import digest,original_bytes
from lp_sandwich.tests import fixture,csr
from lp_sandwich.check import identity
from lp_diagnostic.flow import supervise,audit,costs,review_candidate,phase_state
from lp_diagnostic.study import loop
from lp_diagnostic import study

OBSERVATIONS=[]

@contextmanager
def case(third=False):
    with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as tmp:
        base=Path(tmp);b=fixture();lp=b['lp'];s=b['statement']
        if third:
            lp['E']=csr([[1]],1);lp['h']=['1/3'];lp['offset']=-1
        source={'analytic':True}
        ex={'lp':lp,'source':source,'source_sha256':identity(source),
            'q':s['property']['q'],'offset':s['property']['constant']}
        p=base/'export.json';save_new(p,ex)
        s.update(source_sha256=identity(source),export_sha256=digest(p.read_bytes()),lp_sha256=identity(lp))
        spec={'job_id':'analytic','dataset_index':-1,'export':{'path':str(p),'sha256':s['export_sha256']},
              'statement':s,'statement_sha256':identity(s)}
        yield base,spec

def overwrite(path,value):path.write_bytes(original_bytes(value))

class Controls(unittest.TestCase):
    def test_full_flow_move_and_accounting(self):
        with case() as (base,spec):
            start=time.monotonic();root=base/'run';v=supervise(spec,root,started=start)
            self.assertEqual(v['status'],'CHECKED_LP_DIAGNOSTIC',(v,(root/'driver.log').read_text()))
            self.assertTrue(audit(root)['complete_independent_check']);c=review_candidate(root);t=costs(root)
            self.assertEqual([r['name'] for r in c['stages']],['load','propose','package','check'])
            self.assertEqual(t['native_calls'],1)
            self.assertEqual(read(root/'proposal/terminal.json')['deadline_monotonic'],start+218)
            self.assertAlmostEqual(t['whole_diagnostic_seconds'],t['observed_phase_sum_seconds']+t['residual_clock_seconds'])
            self.assertGreater(t['phases']['propose']['seconds'],t['native_seconds'])
            result=read(root/'check.log');self.assertEqual(result['classification'],'LP_NONPOSITIVE_FEASIBLE_POINT')
            self.assertFalse(result['network_UNSAFE']);self.assertFalse(result['network_SAFE'])
            moved=base/'moved';shutil.copytree(root/'portable',moved);pack=read(root/'packing.json')
            out=subprocess.run([ACT,'-I','-S',str(moved/'verify.py'),str(moved/'bundle.json'),
                '--bundle-sha256',pack['bundle_sha256'],'--statement-sha256',pack['statement_sha256'],
                '--timeout-seconds','30'],cwd=moved,text=True,capture_output=True,timeout=30,check=True)
            re=json.loads(out.stdout)
            self.assertEqual({k:v for k,v in re.items() if k!='seconds'},
                             {k:v for k,v in result.items() if k!='seconds'})
            OBSERVATIONS.append({'status':v['status'],'costs':t,'moved_isolated':True})
            for name in ('proposal/native.json','portable/bundle.json','check.log','load_stage.json'):
                p=root/name;raw=p.read_bytes();p.write_bytes(raw+b' ')
                with self.assertRaises(ValueError):audit(root)
                p.write_bytes(raw)
            candidate=read(root/'candidate.json');bad=deepcopy(candidate)
            bad['artifact_sha256'].pop('proposal/native.json');overwrite(root/'candidate.json',bad)
            with self.assertRaises(ValueError):review_candidate(root)
            overwrite(root/'candidate.json',candidate)
            stage=read(root/'load_stage.json');bad=deepcopy(candidate)
            stage['elapsed_seconds']=0;overwrite(root/'load_stage.json',stage)
            bad['stages'][0]=stage;bad['artifact_sha256']['load_stage.json']=digest((root/'load_stage.json').read_bytes())
            overwrite(root/'candidate.json',bad)
            with self.assertRaises(ValueError):review_candidate(root)

    def test_nonexact_native_point_not_an_upper_bound(self):
        with case(third=True) as (base,spec):
            root=base/'run';v=supervise(spec,root,started=time.monotonic())
            self.assertEqual(v['status'],'CHECKED_LP_DIAGNOSTIC',(v,(root/'driver.log').read_text()))
            out=read(root/'check.log');self.assertEqual(out['primal_status'],'NOT_EXACTLY_FEASIBLE')
            self.assertIsNone(out['upper_bound'])
            self.assertEqual(out['classification'],'UNRESOLVED_CANDIDATE_VS_LP_RELAXATION')
            self.assertEqual(costs(root)['native_calls'],1)

    def test_binding_failure_before_native_and_no_retry(self):
        with case() as (base,spec):
            spec['export']['sha256']='0'*64;root=base/'run'
            v=supervise(spec,root,started=time.monotonic());self.assertEqual(v['status'],'ERROR')
            self.assertEqual(audit(root)['status'],'ERROR');self.assertFalse((root/'proposal').exists())
            self.assertIsNone(costs(root)['native_calls'])
            with self.assertRaises(FileExistsError):supervise(spec,root,started=time.monotonic())

    def test_outer_deadline_kills_owned_process(self):
        with case() as (base,spec):
            root=base/'run';v=supervise(spec,root,started=time.monotonic()-297.995)
            self.assertEqual(v['status'],'TIMEOUT');self.assertTrue(v['outer_process']['killed'])
            self.assertFalse(audit(root)['complete_independent_check']);self.assertIsNone(costs(root)['native_calls'])

    def test_expired_clock_missing_cost_never_zero(self):
        with case() as (base,spec):
            root=base/'run';v=supervise(spec,root,started=time.monotonic()-301)
            self.assertEqual(v['status'],'TIMEOUT');self.assertIsNone(v['outer_process'])
            t=costs(root);self.assertIsNone(t['native_seconds'])
            self.assertTrue(all(p['seconds'] is None for p in t['phases'].values()))
            self.assertGreater(t['whole_diagnostic_seconds'],300)

    def test_proposal_reserve_not_reset(self):
        with case() as (base,spec):
            root=base/'run';v=supervise(spec,root,started=time.monotonic()-219)
            self.assertEqual(v['status'],'TIMEOUT');self.assertEqual(audit(root)['status'],'TIMEOUT')
            self.assertFalse((root/'prepared.json').exists());self.assertFalse((root/'proposal').exists())

    def test_future_clock_rejected(self):
        with case() as (base,spec):
            with self.assertRaises(ValueError):supervise(spec,base/'run',started=time.monotonic()+10)
            self.assertFalse((base/'run').exists())

    def test_late_publication_revokes_acceptance(self):
        with case() as (base,spec):
            root=base/'run';v=supervise(spec,root,started=time.monotonic())
            self.assertEqual(v['status'],'CHECKED_LP_DIAGNOSTIC')
            pub=read(root/'publication.json');pub['observed_seconds']=301;overwrite(root/'publication.json',pub)
            self.assertEqual(audit(root)['status'],'TIMEOUT');self.assertFalse(audit(root)['complete_independent_check'])

    def test_partial_timeout_records_censored_not_zero(self):
        with case() as (base,spec):
            root=base/'run';supervise(spec,root,started=time.monotonic()-301)
            save_new(root/'propose_entered.json',{'phase':'propose','seconds':200})
            (root/'propose_stage.json').write_bytes(b'{"incomplete":')
            t=costs(root);self.assertIsNone(t['phases']['propose']['seconds'])
            self.assertTrue(t['phases']['propose']['censored'])
            self.assertGreater(t['phases']['propose']['observed_window_seconds'],100)
            self.assertEqual(t['unreadable_interrupted_records'],['propose_stage.json'])

    def test_failed_process_and_deadline_priority(self):
        def p(code,killed=False):return {'return_code':code,'killed':killed}
        self.assertEqual(phase_state(p(0),298,298),'TIMEOUT')
        self.assertEqual(phase_state(p(0,True),2,298),'TIMEOUT')
        self.assertEqual(phase_state(p(3),2,298),'TIMEOUT')
        self.assertEqual(phase_state(p(1),2,298),'ERROR')

    def test_roster_error_stops_without_dropping_denominator(self):
        jobs=[{'job_id':str(i)} for i in range(4)];seen=[]
        def fail(j):raise ValueError('injected')
        out=loop(jobs,fail,seen.append)
        self.assertEqual(out,seen);self.assertEqual(len(out),4)
        self.assertEqual([r['status'] for r in out],['ERROR']+['NOT_RUN_AFTER_ERROR']*3)

    def test_roster_unresolved_and_timeout_continue(self):
        jobs=[{'job_id':str(i)} for i in range(4)];states=iter(['TIMEOUT','CHECKED_LP_DIAGNOSTIC']*2)
        out=loop(jobs,lambda j:{'status':next(states)},lambda r:None)
        self.assertEqual(len(out),4);self.assertNotIn('NOT_RUN_AFTER_ERROR',[r['status'] for r in out])

    def test_batch_terminal_cost_reconstruction_and_mutation(self):
        with case() as (base,spec):
            output=base/'batch';output.mkdir();freeze=base/'freeze.json';review=base/'review.json'
            save_new(freeze,{'analytic_freeze':True});save_new(review,{'analytic_review':True})
            jobs=[{**spec,'job_id':str(i)} for i in range(4)]
            save_new(output/'launch.json',{'freeze_sha256':digest(freeze.read_bytes()),
                'review_sha256':digest(review.read_bytes()),'preflight_seconds':.01})
            for i,j in enumerate(jobs):
                if i==3:row={'status':'NOT_RUN_AFTER_ERROR','complete_independent_check':False}
                else:
                    root=output/j['job_id']
                    if i==2:
                        # Keep the frozen request identity, corrupt the input bytes.
                        original=Path(spec['export']['path']).read_bytes()
                        Path(spec['export']['path']).write_bytes(original+b' ')
                    try:row=supervise(j,root,started=time.monotonic()-(301 if i==1 else 0))
                    finally:
                        if i==2:Path(spec['export']['path']).write_bytes(original)
                    row['post_terminal_audit_seconds']=.002
                    save_new(output/(j['job_id']+'_resource.json'),{'observed_seconds':.01,'wait':{},'events':[],'error':None})
                save_new(output/(j['job_id']+'_row.json'),{'job_id':j['job_id'],**row})
            with patch.object(study,'verify',return_value={'jobs':jobs}),patch.object(study,'OUTPUT',output),\
                 patch.object(study,'FREEZE',freeze),patch.object(study,'REVIEW',review):
                r=study.summarize()
                self.assertEqual(r['denominator'],4);self.assertEqual(r['status'],'AUDITED_WITH_ERRORS')
                self.assertEqual(r['status_counts'],{'CHECKED_LP_DIAGNOSTIC':1,'TIMEOUT':1,'ERROR':1,'NOT_RUN_AFTER_ERROR':1})
                self.assertEqual(sum(r['classification_counts'].values()),1)
                self.assertEqual(r['cost_totals']['diagnostics_with_cost_record'],3)
                self.assertAlmostEqual(r['cost_totals']['resource_seconds'],.03)
                self.assertAlmostEqual(r['cost_totals']['post_terminal_audit_seconds'],.006)
                save_new(output/'summary.json',{**r,'final_audit_seconds':.01})
                self.assertEqual(study.audit_saved(),r)
                bad_summary=read(output/'summary.json');bad_summary['cost_totals']['diagnostic_publication_seconds']=0
                overwrite(output/'summary.json',bad_summary)
                with self.assertRaises(ValueError):study.audit_saved()
                p=output/'1_row.json';old=read(p);bad=deepcopy(old)
                bad['status']='CHECKED_LP_DIAGNOSTIC';bad['complete_independent_check']=True;overwrite(p,bad)
                with self.assertRaises(ValueError):study.summarize()
                overwrite(p,old)
                p=output/'3_row.json';p.rename(output/'missing_row.json')
                with self.assertRaises(FileNotFoundError):study.summarize()

    def test_success_without_checker_output_rejected(self):
        with case() as (base,spec):
            root=base/'run';supervise(spec,root,started=time.monotonic())
            p=root/'check.log';p.rename(root/'missing_check.log')
            with self.assertRaises(FileNotFoundError):audit(root)

    def test_checker_flag_tamper_resigned_inventory_rejected(self):
        with case() as (base,spec):
            root=base/'run';supervise(spec,root,started=time.monotonic())
            c=read(root/'candidate.json');o=read(root/'check.log');o['solver_or_model_imported']=True
            overwrite(root/'check.log',o);c['artifact_sha256']['check.log']=digest((root/'check.log').read_bytes())
            overwrite(root/'candidate.json',c)
            with self.assertRaises(ValueError):review_candidate(root)

if __name__=='__main__':unittest.main()
