"""Selection controls and complete synthetic lifecycle; no real request run."""
import copy
from fractions import Fraction as F
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

from router_source.tests import encode
from router_source.capture import sha
from router_source.build import save
from source_enclosure.format import compact, identity, unpack, pack
from source_enclosure.produce import box
from full_source.tests import fixture
from checked_gate.candidate_run import ACT, execute
from property_ranges.select import select
from property_ranges.check_selection import check
from property_ranges.run import commands, supervise
from property_ranges.review import audit, compare_selections

CONTEXT={'request_id':'synthetic-selection-only','scope':'pair[0,1]/expert0','layer':'6'}


def classifier(weights):
    return {'kind':'Linear','training':False,'weight':encode(sum(weights,[]),[len(weights),len(weights[0])]),
            'bias':encode([0.]*len(weights),[len(weights)])}


def toy():
    s=box([-1],[1]);rows=[{0:1}]*4;bias=[-2,0,0,0]
    w=classifier([[0.,0.,0.,0.],[100.,1.,3.,3.],[0.,2.,0.,0.]])
    return s,rows,bias,w


class SelectionControls(unittest.TestCase):
    def test_formula_property_union_fixed_quota_and_order(self):
        args=toy()
        for arm,expected in [('prefix',[0,1]),('property',[2,3])]:
            record=select(*args,CONTEXT,3,0,arm)
            self.assertEqual(record['selected'],expected)
            self.assertEqual([r['score'] for r in record['rows']],['0','1','3/2','3/2'])
            self.assertEqual(check(*args,CONTEXT,3,0,arm,record),expected)
        # All classification competitors matter, not just a chosen blocking row.
        record=select(*args,CONTEXT,3,1,'property')
        self.assertEqual(check(*args,CONTEXT,3,1,'property',record),record['selected'])

    def test_ties_zero_width_and_width_change_do_not_expand_queries(self):
        for width in (1,2,5):
            rows=[{0:1}]*width;w=classifier([[0.]*width,[0.]*width])
            for limits in ((0,0),(-1,1),(0,1),(-1,0)):
                s=box([limits[0]],[limits[1]])
                record=select(s,rows,[0]*width,w,CONTEXT,2,0,'property')
                self.assertEqual(record['selected'],list(range(min(2,width))))
                check(s,rows,[0]*width,w,CONTEXT,2,0,'property',record)

    def test_shared_factor_cancellation_binary_and_checker_independence(self):
        s=box([-1],[1]);h,ci,bi=unpack(s)
        h['c']=[F(0),F(0)];h['Gc']=[{0:F(1)},{0:F(1)}];h['Gb']=[{0:F(1,2)},{0:F(1,2)}]
        s=pack(h,ci,['private-sign']);rows=[{0:1,1:-1},{0:1}];w=classifier([[0.,0.],[2.,2.]])
        args=(s,rows,[0,0],w,CONTEXT,2,0,'property');record=select(*args)
        self.assertEqual(record['rows'][0]['range'],['0','0'])
        self.assertEqual(record['rows'][1]['range'],['-3/2','3/2'])
        with patch('property_ranges.select.select',side_effect=AssertionError('producer forbidden')):
            self.assertEqual(check(*args,record),[0,1])

    def test_identity_score_and_roster_mutations_reject(self):
        args=(*toy(),CONTEXT,3,0,'property');record=select(*args)
        for mode in ('score','range','classifier','request','scope','source','label','missing','duplicate','wrong_row','quota','order'):
            bad=copy.deepcopy(record)
            if mode=='score':bad['rows'][2]['score']='2'
            elif mode=='range':bad['rows'][2]['range']=['0','1']
            elif mode=='classifier':bad['classifier_sha256']='other'
            elif mode in ('request','scope'):bad['context']['request_id' if mode=='request' else 'scope']='other'
            elif mode=='source':bad['source_sha256']='old'
            elif mode=='label':bad['label']=1
            elif mode=='missing':bad['rows'].pop()
            elif mode=='duplicate':bad['selected']=[2,2]
            elif mode=='wrong_row':bad['selected']=[0,1]
            elif mode=='quota':bad['quota']=3
            else:bad['selected']=[3,2]
            with self.subTest(mode=mode),self.assertRaises(ValueError):check(*args,bad)

    def test_outer_deadline_exception_and_partial_publication(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            for mode in ('deadline','error','malformed'):
                root=Path(tmp)/mode;root.mkdir()
                def command(phase):
                    if mode=='error':raise RuntimeError('injected command failure')
                    script='import time; print("partial",flush=True); time.sleep(2)' if mode=='deadline' else 'print("partial")'
                    return [ACT,'-I','-S','-c',script]
                budget=.2 if mode=='deadline' else 2
                t=supervise(root,command,dict(os.environ),budget)
                self.assertIn(t['status'],('ERROR','TIMEOUT'))
                result=audit(root,budget=budget,recheck=False)
                self.assertFalse(result['complete']);self.assertEqual(result['status'],'PASS')
                self.assertGreater(result['total_seconds'],0)

    def test_partial_call_outside_snapshot_rejects(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            root=Path(tmp);queries=root/'range_queries';queries.mkdir()
            record=select(*toy(),CONTEXT,3,0,'property')
            save(root/'selection_expert0.json',{'record':record,'record_sha256':identity(record),'seconds_since_build_start':0})
            save(queries/'e0_r0_lower_entered.json',{'context':CONTEXT,'row':0,'source_sha256':record['source_sha256'],
                 'side':'lower','native_limit_seconds':3,'lp_sha256':'irrelevant'})
            def fail(phase):raise RuntimeError('stopped partial execution')
            supervise(root,fail,dict(os.environ))
            with self.assertRaisesRegex(ValueError,'outside prepublished'):audit(root,recheck=False)

    def test_cross_arm_prequery_identity_and_missing_are_distinct(self):
        a=select(*toy(),CONTEXT,3,0,'prefix');b=select(*toy(),CONTEXT,3,0,'property')
        arms=[{'selection_snapshots':{0:{'record':v}}} for v in (a,b)]
        result=compare_selections(arms)
        self.assertEqual(len(result['compared']),1);self.assertFalse(result['complete_two_expert_comparison'])
        self.assertEqual(compare_selections([arms[0],{}])['unavailable_experts'],[0])
        b['source_sha256']='different'
        with self.assertRaises(ValueError):compare_selections(arms)

    def test_partial_output_call_cap_is_still_audited(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            root=Path(tmp);(root/'relocated/source').mkdir(parents=True)
            save(root/'relocated/source/obligations.json',{'rows':[{'competitor':1}]})
            save(root/'entered_01.json',{'competitor':1,'native_seconds':17})
            def fail(phase):raise RuntimeError('stopped partial execution')
            supervise(root,fail,dict(os.environ))
            with self.assertRaisesRegex(ValueError,'output call roster'):audit(root,required=1,recheck=False)

    def test_forged_positive_terminal_rejects(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            root=Path(tmp)
            def fail(phase):raise RuntimeError('injected')
            supervise(root,fail,dict(os.environ))
            t=json.loads((root/'terminal.json').read_bytes());p=json.loads((root/'publication.json').read_bytes())
            t.update(status='CHECKED_POSITIVE_DECLARED_REAL_MOE',error=None,complete_declared_real_output_proof=True)
            (root/'terminal.json').write_bytes(compact(t));p['terminal_sha256']=sha(root/'terminal.json')
            (root/'publication.json').write_bytes(compact(p))
            with self.assertRaisesRegex(ValueError,'terminal positive claim'):audit(root,recheck=False)

    def test_complete_two_arms_moved_and_semantic_mutations(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            root=Path(tmp);up=root/'fixture';up.mkdir();_,doc=fixture(up)
            old=up/'old';docpath=up/'full_experts.json';save(docpath,doc)
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            results=[]
            for arm in ('prefix','property'):
                dest=root/arm;dest.mkdir()
                save(dest/'job.json',{'arm':arm,'prefix_source':str(old),'prefix_source_hash':sha(old/'manifest.json'),
                     'expert_document':str(docpath),'input_files':{str(old/'manifest.json'):sha(old/'manifest.json'),str(docpath):sha(docpath)}})
                terminal=supervise(dest,commands(dest),env)
                self.assertEqual(terminal['status'],'CHECKED_POSITIVE_DECLARED_REAL_MOE',
                     [(p.name,p.read_text()[-2200:]) for p in dest.glob('*.log')])
                reviewed=audit(dest,required=1)
                self.assertEqual(reviewed['status'],'PASS');self.assertTrue(reviewed['complete'])
                self.assertEqual((reviewed['range_calls_entered'],reviewed['accepted_range_facts'],reviewed['output_calls']),(8,4,1))
                self.assertGreater(reviewed['selection_seconds'],0)
                self.assertLess(reviewed['total_seconds'],300)
                results.append(json.loads((dest/'relocated/source/joint.state.json').read_bytes()))
            self.assertEqual(*results)  # width2 fixture: quotas cover both rows under either rule.
            moved=root/'moved';shutil.copytree(root/'property/relocated',moved)
            shutil.rmtree(up);shutil.rmtree(root/'prefix');shutil.rmtree(root/'property')
            def run(path,name):
                return execute([ACT,'-I','-S',str(path/'verify_bounds.py'),'--manifest-hash',sha(path/'manifest.json')],
                               root/(name+'.log'),time.monotonic()+15,env)
            self.assertEqual(run(moved,'valid')['state'],'COMPLETED')
            for mode in ('score','wrong_source','missing_selection','missing_property','unselected_fact'):
                target=root/mode;shutil.copytree(moved,target)
                sm=json.loads((target/'source/manifest.json').read_bytes())
                if mode in ('score','wrong_source','missing_selection'):
                    name='source/expert0_selection.json';obj=json.loads((target/name).read_bytes())
                    if mode=='score':obj['rows'][0]['score']='12345'
                    elif mode=='wrong_source':obj['source_sha256']='old'
                    else:obj['selected']=[]
                elif mode=='missing_property':
                    name='source/obligations.json';obj=json.loads((target/name).read_bytes());obj['rows']=[]
                else:
                    name='source/trace.json';obj=json.loads((target/name).read_bytes())
                    row=next(s for s in obj['steps'] if s['layer']==6)
                    row['proof']['facts'][0]['query']['source_sha256']='foreign'
                (target/name).write_bytes(compact(obj));sm['files'][name[7:]]=sha(target/name)
                (target/'source/manifest.json').write_bytes(compact(sm))
                m=json.loads((target/'manifest.json').read_bytes());m['files'][name]=sha(target/name)
                m['parent_manifest_sha256']=sha(target/'source/manifest.json');m['files']['source/manifest.json']=sha(target/'source/manifest.json')
                (target/'manifest.json').write_bytes(compact(m))
                with self.subTest(mode=mode):self.assertEqual(run(target,mode)['state'],'ERROR')


if __name__=='__main__':unittest.main()
