"""Complete synthetic chains, native candidate lifecycle, and honest terminal costs."""
import copy
from fractions import Fraction as F
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from source_enclosure.format import identity,compact,unpack
from source_ranges.tests import constrained,interval_fact,CONTEXT
from source_ranges.produce import affine as scalar_affine
from source_ranges.check import check_affine as scalar_check,projection
from source_enclosure.produce import relu
from full_source.tests import fixture
from full_source.build import build as baseline_build
from router_source.capture import sha
from router_source.build import save
from checked_gate.candidate_run import ACT,execute
from range_pipeline.steps import affine
from range_pipeline.check_step import check
from range_pipeline.candidates import program,propose
from range_pipeline.run import supervise,commands
from range_pipeline.review import audit


class IntegrationControls(unittest.TestCase):
    def test_bulk_exact_differential_and_wrong_frame(self):
        s=constrained();fact=interval_fact(s,{0:1},0,F(1,4),F(3,4))
        for facts in ([fact,None],[None,None]):
            args=(s,[{0:1},{0:-2}],[0,F(1,3)],facts,CONTEXT,'a')
            t,p=affine(*args);v,q=scalar_affine(*args)
            self.assertEqual((t,p),(v,q));self.assertEqual(check(s,t,args[1],args[2],p,CONTEXT,'a'),scalar_check(s,t,args[1],args[2],p,CONTEXT,'a'))
        s,_=relu(s,'b');t,p=affine(s,[{0:2}],[1],[None],CONTEXT,'c');check(s,t,[{0:2}],[1],p,CONTEXT,'c')
        bad=copy.deepcopy(t);bad['hz']['frame_id']+=1
        with self.assertRaises(ValueError):check(s,bad,[{0:2}],[1],p,CONTEXT,'c')
        with patch('range_pipeline.steps.affine',side_effect=AssertionError('producer called')):
            check(s,t,[{0:2}],[1],p,CONTEXT,'c')

    def test_numeric_program_matches_independent_projection(self):
        for s in (constrained(),relu(constrained(),'r')[0]):
            self.assertEqual(program(s,{0:F(3,2)},F(1,3)),projection(s,{0:F(3,2)},F(1,3)))

    def test_range_partial_exception_and_bad_evidence(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            root=Path(tmp);s=constrained()
            def side(lp,limit):
                y=[-1,0] if F(lp['c'][0])>0 else [0,-1]
                return {'candidate':{'lp_sha256':identity(lp),'inequality_dual':y,'equality_dual':[]},
                    'native_seconds':.001,'conversion_seconds':.001,'status':0,'success':True,'message':'synthetic','native_limit_seconds':limit}
            for mode in ('good','partial','error','bad_sign','expired'):
                dest=root/mode;dest.mkdir();counter=[0]
                def solve(lp,limit):
                    counter[0]+=1;r=side(lp,limit)
                    if counter[0]==2 and mode=='partial':r.update(candidate=None,status=1,success=False)
                    if counter[0]==2 and mode=='error':raise RuntimeError('synthetic exception')
                    if mode=='bad_sign':r['candidate']['inequality_dual']=[1,0]
                    return r
                with patch('range_pipeline.candidates.solve',side_effect=solve):
                    if mode in ('error','bad_sign'):
                        with self.assertRaises((ValueError,RuntimeError)):propose(s,{0:1},0,CONTEXT,0,dest,'q',time.monotonic()+10)
                        self.assertTrue((dest/'q_lower.json').exists())
                    else:
                        f,r=propose(s,{0:1},0,CONTEXT,0,dest,'q',time.monotonic()+(-1 if mode=='expired' else 10))
                        self.assertEqual(f is not None,mode=='good')
                        self.assertEqual(r['calls'],0 if mode=='expired' else 2)
                        if mode=='partial':self.assertTrue((dest/'q_lower.json').exists());self.assertIsNone(f)

    def test_outer_deadline_exception_and_partial_evidence(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            root=Path(tmp)
            for mode in ('build_timeout','error','partial'):
                dest=root/mode;dest.mkdir()
                def cmd(phase):
                    if mode=='error':raise RuntimeError('injected command exception')
                    if mode=='build_timeout':return [ACT,'-I','-S','-c','import time; print("partial",flush=True); time.sleep(2)']
                    return [ACT,'-I','-S','-c','print("incomplete")']
                budget=.2 if mode=='build_timeout' else 2
                t=supervise(dest,cmd,dict(os.environ),budget)
                self.assertIn(t['status'],('TIMEOUT','ERROR'));self.assertFalse(t['complete_declared_real_output_proof'])
                reviewed=audit(dest,budget=budget,recheck=False)
                self.assertEqual(reviewed['status'],'PASS');self.assertFalse(reviewed['complete'])

    def test_incomplete_terminal_cannot_be_relabelled_positive(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            root=Path(tmp)
            def command(phase):raise RuntimeError('injected before launch')
            supervise(root,command,dict(os.environ))
            t=json.loads((root/'terminal.json').read_bytes());p=json.loads((root/'publication.json').read_bytes())
            t.update(status='CHECKED_POSITIVE_DECLARED_REAL_MOE',error=None,complete_declared_real_output_proof=True)
            (root/'terminal.json').write_bytes(compact(t));p['terminal_sha256']=sha(root/'terminal.json')
            (root/'publication.json').write_bytes(compact(p))
            with self.assertRaisesRegex(ValueError,'terminal positive claim'):
                audit(root,recheck=False)

    def test_complete_both_arms_moved_checks_and_costs(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            root=Path(tmp);froot=root/'fixture';froot.mkdir();prefix,doc=fixture(froot)
            docpath=froot/'full_experts.json';save(docpath,doc);old=froot/'old'
            baseline=root/'baseline';baseline.mkdir();baseline_build(baseline,prefix,sha(prefix/'manifest.json'),doc)
            good=None
            for arm in ('range_off','range_on'):
                dest=root/arm;dest.mkdir();save(dest/'job.json',{'arm':arm,'prefix_source':str(old),
                    'prefix_source_hash':sha(old/'manifest.json'),'expert_document':str(docpath),
                    'input_files':{str(old/'manifest.json'):sha(old/'manifest.json'),str(docpath):sha(docpath)}})
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
                t=supervise(dest,commands(dest),env)
                self.assertEqual(t['status'],'CHECKED_POSITIVE_DECLARED_REAL_MOE',[(p.name,p.read_text()[-2500:]) for p in dest.glob('*.log')])
                self.assertEqual(t['check']['required_obligations'],1)
                self.assertFalse(t['deployed_floating_point_proof']);self.assertFalse(t['production_verdict_changed'])
                reviewed=audit(dest,required=1)
                self.assertTrue(reviewed['complete']);self.assertEqual(reviewed['status'],'PASS')
                self.assertEqual(reviewed['range_calls_entered'],0 if arm=='range_off' else 8)
                self.assertEqual(reviewed['accepted_range_facts'],0 if arm=='range_off' else 4)
                self.assertEqual(reviewed['output_calls'],1)
                if arm=='range_off':
                    self.assertEqual(json.loads((dest/'relocated/source/joint.state.json').read_bytes()),json.loads((baseline/'relocated/joint.state.json').read_bytes()))
                else:good=dest/'relocated'
            # No original source data, producer modules, model, solver or historical dirs needed.
            moved=root/'moved';shutil.copytree(good,moved)
            shutil.rmtree(froot);shutil.rmtree(baseline);shutil.rmtree(root/'range_off');shutil.rmtree(root/'range_on')
            def rebind(target,name,obj):
                (target/name).write_bytes(compact(obj));sm=json.loads((target/'source/manifest.json').read_bytes())
                if name.startswith('source/'):
                    sm['files'][name[7:]]=sha(target/name);(target/'source/manifest.json').write_bytes(compact(sm))
                m=json.loads((target/'manifest.json').read_bytes());m['files'][name]=sha(target/name)
                m['parent_manifest_sha256']=sha(target/'source/manifest.json');m['files']['source/manifest.json']=sha(target/'source/manifest.json')
                (target/'manifest.json').write_bytes(compact(m))
            run=lambda path,name:execute([ACT,'-I','-S',str(path/'verify_bounds.py'),'--manifest-hash',sha(path/'manifest.json')],root/(name+'.log'),time.monotonic()+15,env)
            self.assertEqual(run(moved,'valid')['state'],'COMPLETED')
            for mode in ('missing_layer','wrong_range','disabled','missing_output'):
                target=root/mode;shutil.copytree(moved,target)
                name='source/obligations.json' if mode=='missing_output' else 'source/manifest.json' if mode=='disabled' else 'source/trace.json'
                obj=json.loads((target/name).read_bytes())
                if mode=='missing_layer':obj['steps'].pop()
                elif mode=='wrong_range':
                    row=next(r for r in obj['steps'] if r['layer']==6);row['proof']['facts'][0]['query']['source_sha256']='old'
                elif mode=='disabled':obj['policy']['enabled']=False
                else:obj['rows']=[]
                if mode=='disabled':
                    (target/name).write_bytes(compact(obj));m=json.loads((target/'manifest.json').read_bytes())
                    m['parent_manifest_sha256']=sha(target/name);m['files'][name]=sha(target/name);(target/'manifest.json').write_bytes(compact(m))
                else:rebind(target,name,obj)
                with self.subTest(mode=mode):self.assertEqual(run(target,mode)['state'],'ERROR')


if __name__=='__main__':unittest.main()
