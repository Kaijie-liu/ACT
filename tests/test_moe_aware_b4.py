"""B4 paired subjects, partial evidence and nontransferable execution costs."""
import copy
import gzip
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import moe_aware_b4 as B
from scoped_proof.io import save,sha,load


def admission():
    good=dict(gained=list(range(5)),paired_median_bound_improvement=.1,passes=True)
    return dict(status='AUDITED_DIAGNOSTIC_TERMINALS',primary_decision='COMPLETE_FIXED_MECHANISM_GATE',
        completed_calls=103,planned_calls=103,unexecuted=[],started_without_terminal=0,
        mb=copy.deepcopy(good),mc2=copy.deepcopy(good),mc4=copy.deepcopy(good),admitted_mechanisms=['M-B','M-C-2'])


class Paired(unittest.TestCase):
    def test_public_partial_domains_use_original_receipt_only(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent/'results') as tmp:
            folder=Path(tmp);manifest=folder/'manifest.json';save(manifest,dict(census_multipliers=[.5,1.,2.]))
            query=dict(kind='census',model=dict(id='m'),samples=[dict(index=5000)])
            expected=dict(query=query,seconds=300)
            save(folder/'request.json',dict(**query,manifest_sha256=sha(manifest),worker_deadline=294.))
            def domain(multiplier):
                return dict(index=5000,multiplier=multiplier,request_sha256=sha(folder/'request.json'),
                    accepted=True,clock=30.,deadline=31.,seconds=2.,result=dict(observed_lower=2,candidate_upper=99))
            save(folder/'domain_001.json',domain(.5))
            terminal=dict(accepted=False,status='TIMEOUT',through_reception_seconds=295.,hard_budget=300,
                files={p.name:sha(p) for p in folder.iterdir()})
            save(folder/'terminal.json',terminal)
            ledger=dict(**terminal,seconds=296.,terminal_sha256=sha(folder/'terminal.json'))
            save(folder/'domain_002.json',domain(1.))  # absent from the original terminal snapshot
            result=B.audit_aux(folder,expected,ledger,manifest,time.monotonic()+5)
            self.assertEqual(len(result['domains']),1)
            self.assertEqual(result['unreceived_domains'],[[5000,1.],[5000,2.]])
            with self.assertRaises(ValueError):B.audit_aux(folder,dict(expected,seconds=100),ledger,manifest,time.monotonic()+5)

    def test_public_route_strata_and_conflicting_replay(self):
        rows=[dict(model='m',index=i,arm=arm,status='POSITIVE' if arm=='on' else 'TIMEOUT')
              for i in (1,2,3) for arm in ('off','on')]
        aux=[dict(kind='census',domains=[dict(model='m',index=i,multiplier=1,
            result=dict(observed_lower=i,candidate_upper=99)) for i in (1,2)]),
            dict(kind='pgd',domains=[dict(model='m',index=2,multiplier=1,result=dict(status='UNSAFE_REPLAYED'))])]
        value=B.explain_rows(rows,aux)
        self.assertEqual(value['public_positive_conflicts'],[['m',2]])
        self.assertEqual(value['strata']['ONE_OBSERVED_NOT_PROVEN_STABLE']['gained'],1)
        self.assertEqual(value['strata']['OBSERVED_MULTIPLE']['gained'],1)
        self.assertEqual(value['strata']['NO_USABLE_WITNESS_COUNT']['gained'],1)
        self.assertIsNone(value['strata']['OBSERVED_MULTIPLE']['observed'][0]['exact_legal_count'])

    def test_independent_full_catalog_audit_keeps_stopped_prefix(self):
        from gate_decoupling_worker import digest
        with tempfile.TemporaryDirectory(dir=ROOT.parent/'results') as tmp:
            base=Path(tmp);root=base/'result';root.mkdir();selection=root/'selection';selection.mkdir()
            prior=base/'synthetic5a';prior.mkdir();prior_manifest=base/'prior_manifest.json';save(prior_manifest,dict(synthetic=True))
            summary=admission();save(prior/'audit.json',summary)
            save(prior/'start.json',dict(manifest_sha256=sha(prior_manifest)))
            save(prior/'final_accounting.json',dict(audit_sha256=sha(prior/'audit.json')))
            control=base/'control.json';save(control,dict(status='PASS',synthetic=True))
            excluded=list(range(5000));models=[dict(id=m,architecture=dict(experts=2,layers=1,classes=3))
                for m in B.S.MULTILAYER+B.S.SINGLELAYER]
            spec=dict(schema=B.SCHEMA,policy=B.POLICY,models=models,excluded_ranges=[[0,4999]],
                history_union_sha256=B.hashlib.sha256(json.dumps(excluded,separators=(',',':')).encode()).hexdigest(),
                stage5a_summary=summary,admission=B.admitted(summary),
                stage5a=dict(root=str(prior),manifest=str(prior_manifest),manifest_sha256=sha(prior_manifest),
                    start_sha256=sha(prior/'start.json'),audit_sha256=sha(prior/'audit.json'),final_sha256=sha(prior/'final_accounting.json')),
                part_caps={p:dict(slots=n,seconds=t) for p,(n,t) in B.S.CAPS.items()},
                budgets=dict(request=300,stage=110*3600,slots=1320,selection=300,pgd_model=100,census_model=300,artifacts=B.Q.CAP),
                source_files={p:sha(ROOT/p) for p in B.sources()},genbab_config=B.Q.backend.CONFIG,
                backend_identity=digest(B.Q.backend.source_identity()),source_manifests=[],historical_cohorts=[],
                control=dict(path=str(control),sha256=sha(control)),census_multipliers=[.5,1.,2.])
            manifest=base/'manifest.json';save(manifest,spec);save(root/'start.json',dict(manifest_sha256=sha(manifest)))
            request=dict(kind='selection',manifest_sha256=sha(manifest),worker_deadline=100.)
            save(selection/'request.json',request);samples=[]
            for rank,index in enumerate(range(5000,5020)):
                one=dict(index=index,rank=rank,label=0,file=f'{index}.pt',tensors=dict(center='c',rows='q',thresholds='t'))
                (selection/one['file']).write_text('synthetic materialization placeholder')
                one['sha256']=sha(selection/one['file']);extra=copy.deepcopy(one);extra['file']=f'{index}_4.pt'
                (selection/extra['file']).write_text('synthetic larger box placeholder');extra['sha256']=sha(selection/extra['file'])
                one['additional_domains']={'4/255':extra};samples.append(one)
            value=dict(status='COMPLETE',samples=samples,scanned=[dict(index=s['index'],label=0,predictions=[0]*18) for s in samples])
            with gzip.open(selection/'outcome.json.gz','wt') as f:json.dump(dict(value=value,clock=50.,request_sha256=sha(selection/'request.json')),f)
            save(selection/'published.json',dict(clock=51.,sha256=sha(selection/'outcome.json.gz')))
            save(selection/'session.json',dict(status='COMPLETED',returncode=0,execution_error=None,cleanup_error=None,
                cleanup_status='LEADER_REAPED_NO_LIVE_SESSION'))
            terminal=dict(status='COMPLETE',accepted=True,through_reception_seconds=52.,hard_budget=300,
                files={p.name:sha(p) for p in selection.iterdir() if p.is_file()})
            save(selection/'terminal.json',terminal)
            save(root/'selection_ledger.json',dict(**terminal,seconds=53.,terminal_sha256=sha(selection/'terminal.json')))
            table=B.catalog(spec,samples,selection)
            save(root/'catalog.json',dict(manifest_sha256=sha(manifest),entries=table,
                selection_outcome_sha256=sha(selection/'outcome.json.gz')))
            item=table[0];name='main_verify_0000';child=root/name;child.mkdir()
            save(root/(name+'_started.json'),dict(index=0,part='main',entry=item,clock=60.))
            save(child/'request.json',dict(**item['query'],manifest_sha256=sha(manifest),worker_deadline=360.))
            t=dict(status='TIMEOUT',accepted=False,through_reception_seconds=294.,hard_budget=300,
                files={'request.json':sha(child/'request.json')})
            save(child/'terminal.json',t)
            save(root/(name+'_ledger.json'),dict(**t,id=name,seconds=295.,terminal_sha256=sha(child/'terminal.json')))
            result=B.audit(root,manifest)
            self.assertEqual(result['status'],'PASS_RETAINED_PREFIX');self.assertEqual(result['full_calls'],1)
            self.assertEqual(result['reports']['main']['summary']['contrast']['decision'],'PENDING_INCOMPLETE_QUEUE')
            self.assertEqual(result['reports']['boundary']['checked']['planned'],120)
            self.assertEqual(len(result['reports']['main']['checked']['unexecuted']),599)
            # A late positive-looking file cannot change original TIMEOUT.
            with gzip.open(child/'outcome.json.gz','wt') as f:json.dump(dict(value=dict(status='POSITIVE')),f)
            again=B.audit(root,manifest);self.assertEqual(again['full_calls'],1)
            self.assertEqual(again['reports']['main']['checked']['rows'][0]['status'],'TIMEOUT')
            path=root/'catalog.json';bad=load(path);bad['entries'][0]['query']['guard_dual']=True
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):B.audit(root,manifest)
            path.write_text(json.dumps(dict(manifest_sha256=sha(manifest),entries=table,
                selection_outcome_sha256=sha(selection/'outcome.json.gz'))))
            save(root/'stopped.json',dict(seconds=400.))
            controls=base/'controls';controls.mkdir()
            failed_control=base/'failed_control';failed_control.mkdir()
            save(failed_control/'complete.json',dict(status='FAIL',seconds=1.))
            (failed_control/'log.txt').write_text('retained synthetic tooling failure')
            receipt=controls/'complete.json';save(receipt,dict(status='PASS',trained_calls=0,seconds=3.,
                source_files={p:sha(ROOT/p) for p in B.sources()},retained_control=dict(
                    path=str(failed_control/'complete.json'),sha256=sha(failed_control/'complete.json'))))
            binding=base/'continuation.json'
            B.freeze_resume(binding,manifest,root,ROOT,receipt)
            frozen=load(binding)
            self.assertEqual(frozen['next_indices']['main'],1)
            self.assertGreaterEqual(frozen['carried_part_seconds']['main'],403.)
            self.assertGreaterEqual(frozen['prior_bytes'],B.Q.size(root)+receipt.stat().st_size+B.Q.size(failed_control))
            self.assertIn(str(failed_control/'log.txt'),[r['path'] for r in frozen['external_records']])
            later=base/'suffix';later.mkdir();save(later/'continuation.json',frozen)
            pointer=dict(path=str(binding),sha256=sha(binding))
            save(later/'start.json',dict(manifest_sha256=sha(manifest),execution_root=str(ROOT),execution_binding=pointer))
            calls=[]
            def invoke(folder,query,seconds,manifest,deadline):
                calls.append(query);folder.mkdir()
                save(folder/'request.json',dict(**query,manifest_sha256=sha(manifest),
                    execution_binding=pointer,timing_reserve=list(B.RESUME_TIMING)))
                terminal=dict(status='TIMEOUT',accepted=False,through_reception_seconds=3.,hard_budget=300,
                    files={'request.json':sha(folder/'request.json')})
                save(folder/'terminal.json',terminal)
                return dict(**terminal,seconds=4.,terminal_sha256=sha(folder/'terminal.json')),None
            B.execute_part(later,manifest,table[:3],part='main',deadline=time.monotonic()+1000,
                start_index=1,invoke=invoke)
            self.assertEqual(calls,[r['query'] for r in table[1:3]])
            reviewed=B.audit(later,manifest)
            self.assertEqual(reviewed['full_calls'],3)
            self.assertEqual(reviewed['bytes'],B.Q.size(root)+B.Q.size(later)
                +frozen['external_retained_bytes']+binding.stat().st_size)
            self.assertTrue(all(sha(root/p)==h for p,h in frozen['parent_files'].items()))
            save(later/'main_verify_0003_ledger.json',dict(id='orphan'))
            with self.assertRaisesRegex(ValueError,'orphan'):B.audit(later,manifest)
            (later/'main_verify_0003_ledger.json').unlink()
            (root/'new_unbound_file').write_text('unregistered late parent change')
            with self.assertRaisesRegex(ValueError,'unbound additions'):B.audit(later,manifest)

    def test_admission_and_paired_subjects(self):
        summary=admission();self.assertEqual(B.admitted(summary)['margin_pieces'],2)
        for key,value in [('completed_calls',102),('started_without_terminal',1),('admitted_mechanisms',['M-C-4'])]:
            bad=copy.deepcopy(summary);bad[key]=value
            with self.assertRaises(ValueError):B.admitted(bad)
        samples=[]
        for rank,i in enumerate(range(5000,5020)):
            sample=dict(rank=rank,index=i,label=0,file=f'{i}.pt',sha256='x',
                tensors=dict(center='c',rows='q',thresholds='t',lower='lo',upper='hi'))
            extra=copy.deepcopy(sample);extra['file']=f'{i}_4.pt';extra['tensors']['lower']='lo4'
            sample['additional_domains']={'4/255':extra};samples.append(sample)
        spec=dict(models=[dict(id=m) for m in B.S.MULTILAYER+B.S.SINGLELAYER],excluded_ranges=[[98,98]],
            stage5a_summary=summary,budgets=dict(pgd_model=100,census_model=300))
        rows=B.catalog(spec,samples,'/synthetic');self.assertEqual(len(rows),1320)
        for part in B.S.CAPS:
            by={}
            for row in rows:
                if row['part']==part:by.setdefault((row['model'],row['index']),[]).append(row)
            for pair in by.values():self.assertTrue(all(r['query']['sample']==pair[0]['query']['sample'] for r in pair))
        minus=[r for r in rows if r['arm']=='minus_M-A'][0]['query']
        self.assertFalse(minus['gate_cascade']);self.assertTrue(minus['guard_dual']);self.assertEqual(minus['margin_branch_pieces'],2)
        bad=copy.deepcopy(samples);bad[0]['additional_domains']['4/255']['tensors']['rows']='different-property'
        with self.assertRaises(ValueError):B.catalog(spec,bad,'/synthetic')
        self.assertEqual(len(B.auxiliary(spec,samples,'/synthetic','ablation')),0)
        self.assertEqual(len(B.auxiliary(spec,samples,'/synthetic','radius')),12)
        self.assertTrue(all(r['query']['epsilon']==4/255 for r in B.auxiliary(spec,samples,'/synthetic','radius')))

    def test_part_budget_full_calls_failures_and_no_transfer(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent/'results') as tmp:
            root=Path(tmp);now=[0.];calls=[]
            entries=[dict(part='boundary',model='m',index=i,rank=i,arm='off',query=dict(id=i)) for i in range(3)]
            def invoke(folder,query,seconds,manifest,deadline):
                self.assertEqual(seconds,300);self.assertEqual(deadline,600);calls.append(query['id']);now[0]+=300
                return dict(status='TIMEOUT',accepted=False,seconds=300),None
            r=B.execute_part(root,root/'manifest',entries,part='boundary',deadline=600,invoke=invoke,clock=lambda:now[0])
            self.assertEqual(calls,[0,1]);self.assertEqual(r['seconds'],600)
            self.assertEqual(r['unexecuted'],entries[2:]);self.assertEqual(r['stop_reason'],'PART_TIME_CAP')
            failure=root/'failure';failure.mkdir();now[0]=0
            def error(*args):now[0]+=2;return dict(status='ERROR',accepted=False,seconds=2),None
            r=B.execute_part(failure,root/'manifest',entries,part='boundary',deadline=600,invoke=error,clock=lambda:now[0])
            self.assertEqual(len(r['rows']),1);self.assertEqual(r['stop_reason'],'TOOL_OR_RECEPTION_ERROR')
            self.assertEqual(load(failure/'boundary_verify_0000_ledger.json')['status'],'ERROR')
            with self.assertRaises(ValueError):B.execute_part(root,root/'manifest',entries,part='main',deadline=600,invoke=error,clock=lambda:0)

    def test_terminal_reception_does_not_upgrade_timeout_or_accept_other_engine(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent/'results') as tmp:
            root=Path(tmp);manifest=root/'manifest.json';save(manifest,dict(frozen=True))
            entry=dict(part='main',model='synthetic',index=5000,rank=0,arm='off',query=dict(kind='verification',
                arm='genbab_pruned',model=dict(architecture=dict(layers=1,experts=2,classes=3)),sample=dict(id=5000),
                gate_cascade=False,guard_dual=False))
            request=dict(**entry['query'],manifest_sha256=sha(manifest),worker_deadline=100.)
            save(root/'request.json',request)
            value=dict(status='POSITIVE',expected_histories=1,attempted_histories=1,returned_histories=1,
                verified_histories=1,pruned_histories=0,exclusions=[],prefix_bounds=[],histories=[dict(history=[[0,1]],
                outcome=dict(status='verified',actual_raw_spec=[[[[1.,0.]],[0.]],[[[0.,1.]],[0.]]]))])
            wrapped=dict(status='POSITIVE',result=value,coverage=B.coverage(value,'genbab_pruned',1),
                guarantee='GENBAB_NUMERICAL_FILTER',replayed=False)
            with gzip.open(root/'outcome.json.gz','wt') as f:json.dump(dict(value=wrapped,clock=50.,request_sha256=sha(root/'request.json')),f)
            save(root/'published.json',dict(clock=51.,sha256=sha(root/'outcome.json.gz')))
            save(root/'session.json',dict(status='COMPLETED',returncode=0,execution_error=None,cleanup_error=None,
                cleanup_status='LEADER_REAPED_NO_LIVE_SESSION'))
            save(root/'environment.json',dict(status='PASS',sources_canonical_sha256='a'*64))
            def ledger(status,accepted):
                r=dict(status=status,accepted=accepted,hard_budget=300,through_reception_seconds=52.,
                    files={p.name:sha(p) for p in root.iterdir() if p.is_file() and p.name!='terminal.json'})
                path=root/'terminal.json'
                if path.exists():path.unlink()  # mutable synthetic receipt only
                save(path,r);return dict(**r,seconds=53.,terminal_sha256=sha(path))
            end=time.monotonic()+8
            self.assertEqual(B.audit_call(root,entry,ledger('POSITIVE',True),manifest,'a'*64,end)['status'],'POSITIVE')
            with self.assertRaises(ValueError):B.audit_call(root,entry,ledger('POSITIVE',True),manifest,'b'*64,end)
            self.assertEqual(B.audit_call(root,entry,ledger('TIMEOUT',False),manifest,'a'*64,end)['status'],'TIMEOUT')
            with self.assertRaises(ValueError):B.audit_call(root,entry,ledger('POSITIVE',False),manifest,'a'*64,end)
            bad=copy.deepcopy(entry);bad['query']['guard_dual']=True
            with self.assertRaises(ValueError):B.audit_call(root,bad,ledger('TIMEOUT',False),manifest,'a'*64,end)

    def test_physical_two_radius_domains_keep_input_and_property(self):
        import torch
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir=ROOT.parent/'results') as tmp:
            root=Path(tmp);x=torch.tensor([[.001,.5,.999]],dtype=torch.float64)
            props=torch.tensor([[1.,-1.]],dtype=torch.float64)
            a=B.Q.materialize_sample(root,0,0,0,x,props,2/255,torch)
            b=B.Q.materialize_sample(root,0,0,0,x,props,4/255,torch,suffix='_radius4')
            va=B.Q.tensor_row(root,a,torch);vb=B.Q.tensor_row(root,b,torch)
            for k in ('center','rows','thresholds'):torch.testing.assert_close(va[k],vb[k],rtol=0,atol=0)
            self.assertTrue(bool((vb['lower']<=va['lower']).all() and (vb['upper']>=va['upper']).all()))
            self.assertFalse(torch.equal(va['lower'],vb['lower']));self.assertNotEqual(a['sha256'],b['sha256'])

    def test_repaired_tails_keep_cap_and_reject_late_publication(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent/'results') as tmp:
            root=Path(tmp);manifest=root/'manifest.json';save(manifest,dict(synthetic=True))
            now=[100.];seen=[]
            def execute(*args,**kwargs):
                seen.append(kwargs);now[0]=394.
                return dict(status='TIMEOUT',returncode=124,cleanup_status='LEADER_REAPED_NO_LIVE_SESSION')
            with patch.object(B.Q,'time',SimpleNamespace(monotonic=lambda:now[0])), \
                 patch.object(B.Q,'execute',side_effect=execute),patch.object(B.Q,'PRIOR_BYTES',0), \
                 patch.object(B.Q,'ARTIFACT_ROOT',None),patch.object(B.Q,'EXECUTION_BINDING',None):
                folder=root/'good'
                row,value=B.Q.invoke(folder,dict(kind='verification',arm='genbab_pruned'),300,
                    manifest,1000,timing_reserve=B.RESUME_TIMING)
                self.assertEqual((row['status'],row['accepted'],row['seconds']),('TIMEOUT',False,294.))
                self.assertIsNone(value)
                self.assertEqual(load(folder/'request.json')['worker_deadline'],388.)
                self.assertEqual((seen[0]['run_deadline'],seen[0]['cleanup_deadline']),(390.,395.))
                now[0]=100.
                def late(*args,**kwargs):
                    result=execute(*args,**kwargs);now[0]=401.;return result
                with patch.object(B.Q,'execute',side_effect=late):
                    with self.assertRaisesRegex(ValueError,'complete request budget'):
                        B.Q.invoke(root/'late',dict(kind='verification',arm='genbab_pruned'),300,
                            manifest,1000,timing_reserve=B.RESUME_TIMING)
                self.assertFalse(load(root/'late/terminal.json')['accepted'])

    def test_failed_publication_is_charged_error_not_retried_or_accepted(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent/'results') as tmp:
            root=Path(tmp);manifest=root/'manifest.json';save(manifest,dict(synthetic=True))
            name='main_verify_0000';child=root/name;child.mkdir()
            item=dict(part='main',model='m',index=5000,rank=0,arm='on',query=dict(gate_cascade=True))
            save(root/(name+'_started.json'),dict(entry=item,clock=10.))
            save(root/(name+'_failed_start.json'),dict(clock=311.,kind='ValueError',message='budget'))
            save(root/'start.json',dict(entry=0.));save(root/'stopped.json',dict(seconds=311.5))
            save(child/'request.json',dict(**item['query'],manifest_sha256=sha(manifest)))
            save(child/'terminal.json',dict(status='TIMEOUT',accepted=False,files={'request.json':sha(child/'request.json')}))
            r=B.failed_attempt(root,name,item,manifest)
            self.assertEqual((r['status'],r['accepted'],r['seconds']),('ERROR',False,301.5))
            self.assertEqual(r['observed_terminal_status'],'TIMEOUT')
            self.assertEqual(B.S.contrast([r],['m'],[5000])['states']['on'],{'ERROR':1})
            for update in (dict(status='TIMEOUT'),dict(accepted=True),dict(accounting=''),dict(status='POSITIVE')):
                with self.assertRaises(ValueError):B.S.contrast([dict(r,**update)],['m'],[5000])
            (child/'terminal.json').unlink()  # Deliberately mutated synthetic receipt only.
            save(child/'terminal.json',dict(status='POSITIVE',accepted=True,files={'request.json':sha(child/'request.json')}))
            with self.assertRaisesRegex(ValueError,'cannot supply a positive'):B.failed_attempt(root,name,item,manifest)

    def test_owned_deadline_reaps_only_synthetic_child(self):
        from scoped_proof.owned_session import execute
        began=time.monotonic()
        result=execute([sys.executable,'-I','-S','-c','import time; time.sleep(30)'],
            run_deadline=began+1.5,cleanup_deadline=began+6.5,
            env={'PATH':'/usr/bin:/bin'},rss_limit=None,development_policy=B.POLICY)
        self.assertNotEqual(result['status'],'COMPLETED')
        self.assertEqual(result['cleanup_status'],'LEADER_REAPED_NO_LIVE_SESSION')
        self.assertIsNone(result['cleanup_error'])
        self.assertLess(time.monotonic()-began,6.5)


if __name__=='__main__':unittest.main()
