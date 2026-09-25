"""Separate tiny saved-source supervision controls, not a timing comparison."""
import copy
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from scoped_proof.io import PYTHON,save,load,sha
from source_enclosure.format import identity
from source_cost_supervised.audit import cost_check
from source_cost_supervised import supervisor as outer
from parsed_source_reuse.execution import supervise,receive
from parsed_source_reuse.tests import built


class SupervisionControls(unittest.TestCase):
    def setUp(self):
        parent=Path(os.environ.get('PARSED_SOURCE_CONTROL_ROOT','/data1/Kane/MOE'))
        parent.mkdir(parents=True,exist_ok=True)
        self.root=Path(tempfile.mkdtemp(prefix=self._testMethodName+'_',dir=parent))
        self.fixture=dict(experts=2,classes=2,width=1,depth=0,seed=91)
        doc,b,inv=built(self.fixture,'saved_control')
        source=save(self.root/'source.json',doc);construction=save(self.root/'construction.json',b)
        from residual_proof.check import check
        r=check(doc,b,invocation=inv,expected_source_sha256=identity(doc),deadline=time.monotonic()+30)
        self.spec={'id':'tiny','fixture':self.fixture,'source_sha256':identity(doc)}
        self.method={'schema':'PARSED_SOURCE_SAVED_CHECK_R1','enabled':True,
                     'source':dict(source,path=str(self.root/'source.json')),
                     'construction':dict(construction,path=str(self.root/'construction.json')),
                     'expected_result_sha256':identity(r)}

    def run_call(self,name='call',**kwargs):
        result=supervise(self.root/name,self.spec,self.method,**kwargs)
        save(self.root/(name+'_return.json'),result)
        folder=self.root/name
        inv=load(folder/'invocation.json');cost=load(folder/'cost.json');terminal=load(folder/'terminal.json')
        cost_check(inv,cost,terminal,result)
        if result['status']=='COMPLETED':
            self.assertEqual(receive(folder,identity(self.method),deadline=time.monotonic()+30),load(folder/'received.json'))
        return result

    def test_complete_modes_same_result_full_cost_and_fresh_scope(self):
        for enabled in (False,True):
            self.method['enabled']=enabled;name='on' if enabled else 'off'
            r=self.run_call(name);self.assertEqual(r['status'],'COMPLETED')
            p=load(self.root/name/'parse_stats.json')['stats']['parser']
            self.assertEqual(p['scope'][0],load(self.root/name/'invocation.json')['invocation'])
            self.assertEqual(p['hits']>0,enabled)

    def test_hard_cutoff_and_partial_stats_not_accepted(self):
        def hook(phase,root,deadline,args):
            if phase!='profile':return args
            code='''import sys,time
from pathlib import Path
from parsed_source_reuse import execution,cache
root=Path(sys.argv[1]);old=cache.SourceParser._timed
def pause(self,name,fn):
    if name=='freeze':
        (root/'parse_stats.json.partial').write_text('partial freeze')
        time.sleep(30)
    return old(self,name,fn)
cache.SourceParser._timed=pause
execution.work(root,sys.argv[2])'''
            return [PYTHON,'-S','-c',code,str(root),identity(self.method)]
        r=self.run_call(budget=.8,command_hook=hook);self.assertEqual(r['status'],'TIMEOUT')
        self.assertTrue((self.root/'call/parse_stats.json.partial').exists())
        self.assertFalse((self.root/'call/received.json').exists())

    def test_exception_retains_failed_stats(self):
        def hook(phase,root,deadline,args):
            if phase!='profile':return args
            code='''import sys
from pathlib import Path
from parsed_source_reuse import execution,cache
def fail(*a,**k):raise ValueError('injected parser failure')
cache.reference_unpack=fail
execution.work(Path(sys.argv[1]),sys.argv[2])'''
            return [PYTHON,'-S','-c',code,str(root),identity(self.method)]
        r=self.run_call(command_hook=hook);self.assertEqual(r['status'],'ERROR')
        self.assertEqual(load(self.root/'call/parse_stats.json')['stats']['status'],'ERROR')

    def test_receipt_cutoff_and_exception(self):
        for name,code,status in [('cutoff','import time;time.sleep(30)','TIMEOUT'),('error',"raise ValueError('receiver')",'ERROR')]:
            def hook(phase,root,deadline,args):return [PYTHON,'-S','-c',code] if phase=='receive' else args
            r=self.run_call(name,budget=.8,command_hook=hook);self.assertEqual(r['status'],status)
            self.assertTrue((self.root/name/'candidate.json').exists())
            self.assertFalse((self.root/name/'received.json').exists())

    def test_wrong_saved_source_or_expected_result(self):
        for name in ('source','result'):
            original=copy.deepcopy(self.method)
            if name=='source':self.method['construction']['sha256']='0'*64
            else:self.method['expected_result_sha256']='0'*64
            r=self.run_call(name);self.assertEqual(r['status'],'ERROR');self.method=original

    def test_candidate_and_cost_mutations(self):
        self.run_call();folder=self.root/'call';c=load(folder/'candidate.json')
        for key,value in [('method_sha256','0'*64),('invocation','other'),('complete_output_positive_proof',True),('read_seconds',-1)]:
            bad=copy.deepcopy(c);bad[key]=value;(folder/'candidate.json').write_text(json.dumps(bad))
            with self.assertRaises(ValueError):receive(folder,identity(self.method),deadline=time.monotonic()+30)
        (folder/'candidate.json').write_text(json.dumps(c))
        inv=load(folder/'invocation.json');cost=load(folder/'cost.json');t=load(folder/'terminal.json');r=load(self.root/'call_return.json')
        for key in ('overhead_seconds','budget_seconds','seconds_before_ledger'):
            bad=copy.deepcopy(cost);bad[key]=-1
            with self.assertRaises(ValueError):cost_check(inv,bad,t,r)

    def test_late_terminal_and_no_overwrite(self):
        original=outer.save
        def slow(path,value):
            if Path(path).name=='terminal.json':time.sleep(1.1)
            return original(path,value)
        with patch.object(outer,'save',slow):r=self.run_call(budget=1.)
        self.assertEqual(r['status'],'TIMEOUT')
        with self.assertRaises(FileExistsError):self.run_call()

    def test_resource_limit_and_no_candidate(self):
        r=self.run_call('rss',rss_limit=1);self.assertEqual(r['status'],'RESOURCE_LIMIT')
        def hook(phase,root,deadline,args):return [PYTHON,'-S','-c','pass'] if phase=='profile' else args
        r=self.run_call('missing',command_hook=hook);self.assertEqual(r['status'],'ERROR')


if __name__=='__main__':unittest.main()
