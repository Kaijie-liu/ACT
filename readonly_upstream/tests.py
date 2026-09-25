"""Full upstream tiny controls; no profiling fixtures or real-model runs here."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import time
import unittest

from source_cost_supervised import tests as prior
from scoped_proof.io import ROOT,PYTHON,load,save,sha
from source_enclosure.format import identity
from parsed_source_reuse.check import clone
from readonly_upstream import audit
from readonly_upstream.execution import supervise


class Controls(prior.Controls):
    def setUp(self):
        super().setUp(); self.method={'schema':'READONLY_UPSTREAM_R1','readonly':True}

    def call(self,**kwargs):
        hook=kwargs.pop('command_factory',None)
        r=supervise(self.root/'call',self.spec,self.method,command_hook=hook,**kwargs)
        save(self.root/'returned.json',r); return r

    def factory(self,code,phase='profile'):
        # Preserve previous fault injections, retarget ONLY the new worker.
        code=code.replace('from source_cost_supervised.worker import run','from readonly_upstream.worker import run')
        code=code.replace('from source_cost_controls import profile','from readonly_upstream import worker as profile')
        code=code.replace('run(sys.argv[1])','run(sys.argv[1],sys.argv[2])')
        def command(which,root,deadline,args):
            return [PYTHON,'-S','-c',code,str(root),identity(self.method)] if which==phase else args
        return command

    def review(self,result): return audit.review(self.root/'call',result)

    def test_complete_relocated_fresh_solver_free_recheck(self):
        r=self.call(); self.assertEqual(r['status'],'COMPLETED')
        v=audit.review(self.root/'call',r,recheck=True); self.assertTrue(v['source_rechecked'])
        shutil.copytree(self.root/'call',self.root/'relocated')
        code='''import sys
from pathlib import Path
from scoped_proof.io import load
def forbid(event,args):
    if event=='import' and (args[0].split('.')[0] in ('torch','numpy','scipy','act','highspy') or args[0] in ('residual_proof.build','checked_route_frontier.build','shared_route_residual.propose','source_construction_lab.fixtures','readonly_source.check','readonly_source.cache','readonly_source.view','readonly_upstream.worker')):raise ImportError(args[0])
sys.addaudithook(forbid)
from readonly_upstream.audit import review
out=review(Path(sys.argv[1]),load(Path(sys.argv[2])),recheck=True)
assert out['audit']=='PASS' and out['source_rechecked'] and not out['complete_output_positive_proof']
'''
        out=subprocess.run([PYTHON,'-S','-c',code,str(self.root/'relocated'),str(self.root/'returned.json')],
            cwd=ROOT,env=prior.supervisor.ENV,capture_output=True,text=True,timeout=30)
        self.assertEqual(out.returncode,0,out.stderr)

    def test_both_modes_fresh_generation_same_bytes_and_original_checker(self):
        from source_cost_controls.profile import run as old_profile
        from source_construction_lab.fixtures import document
        old=old_profile(document(**self.spec['fixture']),self.root/'reference',deadline=time.monotonic()+30)
        results=[]
        for enabled in (False,True):
            method=dict(self.method,readonly=enabled); folder=self.root/str(enabled)
            r=supervise(folder,self.spec,method); self.assertEqual(r['status'],'COMPLETED')
            audit.review(folder,r,recheck=True,expected=identity(method))
            report=load(folder/'profile/report.json'); stats=load(folder/'profile/parser_stats.json')['stats']
            for name in ('source','construction'):
                self.assertEqual(report['files'][name],old['files'][name])
            self.assertEqual(report['source_check'],old['source_check'])
            self.assertEqual(stats['parser']['hits']>0,enabled)
            self.assertEqual(stats['parser']['scope'][0],load(folder/'invocation.json')['invocation'])
            self.assertGreaterEqual(r['seconds_including_terminal'],sum(p['seconds'] for p in report['phases']))
            results.append(report['source_check'])
        self.assertEqual(results[0],results[1])

    def test_ties_multiple_pairs_and_dimension_differential(self):
        from readonly_upstream.worker import profile
        from source_cost_supervised.worker import Journal
        from source_construction_lab.fixtures import document
        from source_cost_controls.profile import run as old_profile
        for name,doc in [('ties',document(experts=3,classes=3,width=2,depth=1,seed=91,tied=True)),
                         ('zero',document(experts=4,classes=4,width=3,depth=1,seed=91,constant=True))]:
            folder=self.root/name; folder.mkdir()
            inv={'invocation':name,'spec_sha256':identity(doc),'work_deadline_monotonic':time.monotonic()+30}
            report=profile(doc,folder/'profile',inv,self.method,Journal(folder,inv))
            old=old_profile(doc,folder/'old',deadline=time.monotonic()+30)
            self.assertEqual(report['source_check'],old['source_check']); self.assertEqual(report['files'],old['files'])
            self.assertGreater(report['source_check']['frontier']['retained_pairs'],1)

    def test_weighted_construction_cutoff_keeps_partial_obligations(self):
        code='''import sys,time
from residual_proof import build
from readonly_upstream.worker import run
build.output_lp=lambda *a,**k:time.sleep(30)
run(sys.argv[1],sys.argv[2])'''
        r=self.call(budget=.8,command_factory=self.factory(code)); self.assertEqual(r['status'],'TIMEOUT')
        v=self.review(r)
        self.assertTrue(any(x['name']=='weighted_lp' and x['seconds'] is None for x in v['journal']['open']))
        self.assertFalse((self.root/'call/received.json').exists())

    def test_readonly_failure_retains_stats_and_cannot_publish_success(self):
        code='''import sys
from readonly_source.cache import SourceParser
from readonly_upstream.worker import run
def fail(*a,**k):raise ValueError('injected borrow failure')
SourceParser.unpack.__globals__['thaw']=fail
run(sys.argv[1],sys.argv[2])'''
        r=self.call(command_factory=self.factory(code)); self.assertEqual(r['status'],'ERROR'); self.review(r)
        p=load(self.root/'call/profile/parser_stats.json')['stats']
        self.assertEqual(p['status'],'ERROR'); self.assertTrue(p['after_close']['closed'])
        self.assertFalse((self.root/'call/received.json').exists())

    def test_missing_output_at_production_is_rejected(self):
        code='''import sys
from residual_proof import build
from readonly_upstream.worker import run
old=build.output_lp
def corrupt(*a,**k):
    base,rows=old(*a,**k);rows['rows'].pop();return base,rows
build.output_lp=corrupt
run(sys.argv[1],sys.argv[2])'''
        r=self.call(command_factory=self.factory(code)); self.assertEqual(r['status'],'ERROR')
        self.review(r); self.assertFalse((self.root/'call/received.json').exists())

    def test_mode_parser_and_component_binding_mutations(self):
        self.call(); folder=self.root/'call'; inv=load(folder/'invocation.json')
        report=load(folder/'profile/report.json'); candidate=load(folder/'candidate.json'); stats=load(folder/'profile/parser_stats.json')
        for kind in ('scope','policy','closed','mode','missing_component','cost'):
            r=copy.deepcopy(report); p=copy.deepcopy(stats); c=copy.deepcopy(candidate)
            if kind=='scope': p['stats']['parser']['scope'][0]='other'
            elif kind=='policy': p['stats']['parser']['policy']='other'
            elif kind=='closed': p['stats']['after_close']['closed']=False
            elif kind=='mode': r['upstream_method']['readonly']=False
            elif kind=='missing_component': r['operations'][0]['role']='fake'
            else: p['stats']['parser']['seconds']['total']=-1
            (folder/'profile/parser_stats.json').write_text(json.dumps(p))
            r['upstream_method']['parser_stats_record']={'sha256':sha(folder/'profile/parser_stats.json'),'bytes':(folder/'profile/parser_stats.json').stat().st_size}
            (folder/'profile/report.json').write_text(json.dumps(r))
            c['records']['report.json']={'sha256':sha(folder/'profile/report.json'),'bytes':(folder/'profile/report.json').stat().st_size}
            (folder/'candidate.json').write_text(json.dumps(c))
            with self.assertRaises(ValueError): audit.receive(folder,inv,self.spec,deadline=time.monotonic()+30,expected=identity(self.method))


for name,fn in vars(prior.Controls).items():
    if name.startswith('test_') and name not in vars(Controls):
        setattr(Controls,name,clone(fn,{'audit':audit} if 'audit' in fn.__globals__ else {}))


if __name__=='__main__': unittest.main()
