"""B4 paired subjects, partial evidence and nontransferable execution costs."""
import copy
import gzip
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
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


if __name__=='__main__':unittest.main()
