"""Analytic batch controls; no real-LP solve or reconstruction."""
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch
from single_check_portable.execution import ROOT,read,save_new
from portable_proof.runtime import original_bytes
from lp_sandwich.check import identity
from exact_basis.tests import case
from modular_supervised.flow import supervise
from modular_diagnostic import contract as C
from modular_diagnostic.run import execute,summarize,audit_saved,launch_gate
from modular_diagnostic.archive import comparison,partial,detail

ARTIFACT_ROOT=None
JOBS=[]
VALID_BATCHES=[]


def overwrite(p,v):p.write_bytes(original_bytes(v))


def ready(record):
    r={'ram_gib':32,'disk_gib':20,'load_per_core':.1}
    record({'state':'RESOURCE_WAIT','seconds':0,'resource':r})
    return {'seconds':0,'at_launch':r}


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global ARTIFACT_ROOT,JOBS
        ARTIFACT_ROOT=Path(tempfile.mkdtemp(prefix='modular_diagnostic_controls_',dir=ROOT/'data/moe/results'))
        cls.base=ARTIFACT_ROOT;cls.jobs=[]
        for i in range(4):
            lp,s,_,_=case();lp['c']=[-1,0];lp['offset']=i;s['lp_sha256']=identity(lp)
            path=cls.base/f'input{i}.json';save_new(path,{'lp':lp,'statement':s})
            cls.jobs.append({'job_id':f'analytic{i}','input':C.ref(path),'statement':s,'statement_sha256':identity(s)})
        JOBS=cls.jobs
        cls.good=cls.base/'complete'
        result=execute(cls.jobs,cls.good,{'preflight_seconds':0},wait=ready)
        if result['aggregates']['complete_checks']!=4:raise AssertionError(result)
        VALID_BATCHES.append(str(cls.good))

    def clone(self,name):
        p=self.base/name;shutil.copytree(self.good,p);return p

    def test_complete_roster_full_cost_and_exact_checks(self):
        result=audit_saved(self.jobs,self.good)
        self.assertEqual(result['status'],'PASS')
        self.assertEqual(result['aggregates']['checked_upper_bounds'],4)
        self.assertEqual(result['aggregates']['recorded_native_calls'],4)
        self.assertEqual(result['aggregates']['checked_nonpositive_upper_bounds'],1)
        from fractions import Fraction
        for i,row in enumerate(result['rows']):
            self.assertEqual(Fraction(row['diagnostic']['upper_bound']),Fraction(i)-Fraction(1,3))
            self.assertFalse(row['diagnostic']['network_SAFE'] or row['diagnostic']['network_UNSAFE'])
            self.assertGreaterEqual(row['attempt_seconds'],row['costs']['whole_supplied_LP_seconds'])
            self.assertTrue(row['costs']['arithmetic_progress']['complete'])

    def test_no_resume_duplicate_or_escape(self):
        with self.assertRaises(FileExistsError):execute(self.jobs,self.good,{},wait=ready)
        for jobs in (self.jobs[:-1],[self.jobs[0]]*4):
            with self.assertRaises(ValueError):execute(jobs,self.base/'invalid',{},wait=ready)
        bad=deepcopy(self.jobs);bad[0]['job_id']='../escape'
        with self.assertRaises(ValueError):execute(bad,self.base/'invalid_path',{},wait=ready)

    def test_resource_error_stops_all_unstarted_costs_null(self):
        def fail(record):ready(record);raise TimeoutError('synthetic resource cap')
        root=self.base/'resource_error';r=execute(self.jobs,root,{'preflight_seconds':0},wait=fail)
        self.assertEqual(r['aggregates']['status_counts'],{'ERROR':1,'NOT_RUN_AFTER_ERROR':3})
        self.assertEqual(r['aggregates']['cost_records'],0)
        self.assertTrue(all(row['costs'] is None for row in r['rows']))
        self.assertIsNone(r['rows'][1]['attempt_seconds'])
        VALID_BATCHES.append(str(root))

    def test_timeout_continues_then_exception_stops(self):
        calls=[]
        def invoke(job,root,*,started):
            calls.append(job['job_id'])
            if len(calls)==1:
                # Synthetic no-work TIMEOUT, retaining a consistent wrapper clock.
                start=started-301
                overwrite(root.parent/(job['job_id']+'_invocation.json'),{'original_started':start})
                entered=root.parent/(job['job_id']+'_entered.json');e=read(entered);e['started']-=301;overwrite(entered,e)
                return supervise(job,root,started=start)
            raise OSError('synthetic invocation failure')
        # Advancing start without spending301s cannot be a true attempt clock.
        # Use an accelerated fake whole clock solely for deterministic wrapper logic.
        root=self.base/'timeout_then_error'
        from modular_diagnostic import run
        original=run.summarize
        def synthetic_clock(jobs,folder):
            p=folder/(jobs[0]['job_id']+'_row.json');row=read(p);row['attempt_seconds']+=301;overwrite(p,row)
            b=read(folder/'batch_terminal.json');b['seconds']+=301;overwrite(folder/'batch_terminal.json',b)
            return original(jobs,folder)
        with patch.object(run,'summarize',synthetic_clock):
            result=execute(self.jobs,root,{'preflight_seconds':0},wait=ready,invoke=invoke)
        self.assertEqual([r['status'] for r in result['rows']],['TIMEOUT','ERROR','NOT_RUN_AFTER_ERROR','NOT_RUN_AFTER_ERROR'])
        self.assertEqual(len(calls),2)
        VALID_BATCHES.append(str(root))

    def test_missing_extra_reordered_and_status_rows_rejected(self):
        for mode in ('missing','extra','reorder','status','late_started'):
            root=self.clone('bad_'+mode)
            p=root/'analytic0_row.json';r=read(p)
            if mode=='missing':p.rename(root/'removed.json')
            elif mode=='extra':save_new(root/'extra_row.json',r)
            elif mode=='reorder':r['job_id']='analytic1';overwrite(p,r)
            elif mode=='status':r['status']='TIMEOUT';r['complete_independent_check']=False;overwrite(p,r)
            else:overwrite(root/'analytic0_invocation.json',{'original_started':0})
            with self.assertRaises(ValueError):summarize(self.jobs,root)

    def test_cost_overclaims_and_summary_mutation_rejected(self):
        for mode in ('negative','zero','batch'):
            root=self.clone('cost_'+mode);p=root/'analytic0_row.json';row=read(p)
            if mode=='batch':
                b=read(root/'batch_terminal.json');b['seconds']=0;overwrite(root/'batch_terminal.json',b)
            else:
                row['attempt_seconds']=-1 if mode=='negative' else 0
                overwrite(p,row)
            with self.assertRaises(ValueError):summarize(self.jobs,root)
        root=self.clone('summary_drift');s=read(root/'summary.json');s['denominator']=3;overwrite(root/'summary.json',s)
        with self.assertRaises(ValueError):audit_saved(self.jobs,root)

    def test_arithmetic_comparison_never_assumes_same_basis(self):
        root=self.good/'analytic0';m=read(root/'mapping.json');c=read(root/'construction.json')
        self.assertTrue(comparison(m,c,m,c)['assembled_system_equal'])
        changed=deepcopy(m);changed['hint']['basic_columns'].reverse()
        self.assertFalse(comparison(changed,c,m,c)['basis_structure_equal'])
        self.assertIn('no pure',comparison(changed,c,m,c)['interpretation'])
        changed_c={**c,'assembled_system_sha256':'different'}
        self.assertFalse(comparison(m,changed_c,m,c)['assembled_system_equal'])
        self.assertIsNone(comparison(None,None,m,c)['assembled_system_equal'])
        d=detail(root,root)
        self.assertNotIn('upper_bound',d)
        self.assertIn('native_objective_untrusted',d)

    def test_archive_partial_is_not_checked_or_zero(self):
        root=self.base/'partial';root.mkdir();(root/'construction.json').write_bytes(b'{"partial":')
        r=partial(root,'construction.json');self.assertEqual(r['state'],'PARTIAL_UNPARSEABLE');self.assertIsNone(r['value'])
        self.assertIsNone(detail(root)['construction'])
        self.assertEqual(partial(root,'mapping.json')['state'],'MISSING')

    def test_launch_gate_rejects_dirty_unpushed_and_wrong_branch(self):
        v={};review={'status':'PASS','issues':[],'freeze':{'fake':1},'sources':C.sources()}
        for failure in ('dirty','branch','remote','review'):
            def git(*args):
                if args==('branch','--show-current'):return 'main' if failure=='branch' else 'feat/moe-route-verification'
                if args==('status','--porcelain'):return ' M something' if failure=='dirty' else ''
                if args==('rev-parse','HEAD'):return 'local'
                if args[0]=='ls-remote':return ('other' if failure=='remote' else 'local')+' refs/heads/x'
                raise AssertionError(args)
            r=deepcopy(review)
            if failure=='review':r['issues']=['not ready']
            with patch.object(C,'verify',return_value=v),patch.object(C,'ref',return_value={'fake':1}),\
                 patch('modular_diagnostic.run.read',return_value=r),patch('scripts.optional_evidence_dev_contract.git',git):
                with self.assertRaises(ValueError):launch_gate()

    def test_frozen_roster_and_production_no_fault_flag(self):
        jobs=[{'job_id':j,'dataset_index':i} for j,i in zip(C.POLICY['job_ids'],C.POLICY['indices'])]
        C.check_roster(jobs)
        for bad in (jobs[::-1],jobs[:-1]):
            with self.assertRaises(ValueError):C.check_roster(bad)
        self.assertEqual(C.POLICY['arithmetic']['max_bits'],4096)
        self.assertFalse(C.POLICY['resume'] or C.POLICY['retry'])
        self.assertIsNone(C.POLICY['fallback'])

    def test_freeze_rejects_policy_source_runtime_mutations(self):
        from modular_diagnostic.contract import validate
        base={'schema':'MODULAR_DIAGNOSTIC_FREEZE_V1','status':'FROZEN_NOT_EXECUTED',
              'policy':deepcopy(C.POLICY),'sources':C.sources(),'output':str(C.OUTPUT),'runtime':C.runtime()}
        for name in ('policy','sources','output','runtime','status'):
            v=deepcopy(base)
            if name=='policy':v[name]['arithmetic']['max_bits']=8192
            elif name=='sources':v[name]['modular_supervised/flow.py']='changed'
            elif name=='runtime':v[name]['native_version']='other'
            else:v[name]='changed'
            with self.assertRaises(ValueError):validate(v)


if __name__=='__main__':unittest.main()
