"""Fixed saved-object relocation and corruption controls; no proposal calls."""
import copy
import json
import os
from pathlib import Path
import shutil
import time
import unittest
from unittest.mock import patch

from scoped_proof.io import load,save,PYTHON
from scripts import hz_source_portable as p

ROOT=None
OBS={}
MATH_ERRORS={'pair':'proof request coverage','property':'endpoint source/domain/property binding',
             'factor':'source factors/frame not preserved','input':'input enclosure inward'}


def rewrite(path,value):
    path.write_bytes(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode())


def restamp(root,anchors):
    m=load(root/'manifest.json')
    for name in m['files']:
        raw=(root/name).read_bytes(); m['files'][name]={'bytes':len(raw),'sha256':p.v.digest(raw)}
    m['source_sha256']=p.v.identity(load(root/'source.json'))
    m['package_sha256']=p.v.identity(load(root/'package.json'))
    m['code_sha256']=p.v.identity({n:m['files'][n] for n in sorted(p.v.CODE_NAMES)})
    rewrite(root/'manifest.json',m)
    return {'manifest_sha256':p.sha(root/'manifest.json'),
            **{k:m[k] for k in ('source_sha256','package_sha256','code_sha256')}}


class PortableSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if ROOT is None: raise ValueError('fixed output root required')
        names=list(p.protocol()['cases'])
        for i,name in enumerate(names):
            start=time.monotonic()
            try: result=p.run_case(ROOT/name,name)
            except Exception as exc:
                save(ROOT/'batch_stop.json',{'after':name,'pending':names[i+1:],'error':repr(exc)})
                raise
            end=time.monotonic()
            OBS[name]={'start':start,'end':end,'result':result}
            save(ROOT/('observed_'+name+'.json'),OBS[name])
            if result['status']!='CHECKED_OFFLINE_RELOCATION':
                save(ROOT/'batch_stop.json',{'after':name,'pending':names[i+1:]})
                raise RuntimeError('relocation did not complete; no automatic retry')
        (ROOT/'mutations').mkdir()

    def base(self,name='weighted_sign'):
        root=ROOT/name/'relocated'; r=load(ROOT/name/'checker.log')
        return root,r['anchors'],r['result']

    def copied(self,label):
        root,a,expected=self.base(); dest=ROOT/'mutations'/label
        shutil.copytree(root,dest)
        return dest,copy.deepcopy(a),expected

    def test_five_relocated_results(self):
        for name in p.protocol()['cases']:
            root,a,expected=self.base(name); t=load(ROOT/name/'terminal.json')
            stage=load(ROOT/name/'stage.json')
            self.assertEqual(p.receive(root,a,stage,ROOT/name/'checker.log',time.monotonic()+30,p.reference(name)[2])['result'],expected)
            self.assertLess(OBS[name]['end'],t['deadline'])
            self.assertTrue(t['generation_cost_not_in_this_measurement'])
        partial=self.base('partial')[2]
        self.assertEqual((partial['positive'],partial['required'],partial['missing_endpoints']),(15,18,3))
        self.assertEqual(partial['status'],'UNKNOWN_MISSING_EVIDENCE')
        self.assertEqual(self.base('unsafe_tied')[2]['status'],'UNKNOWN_NONPOSITIVE')

    def test_unchanged_checker_bytes(self):
        root,a,_=self.base()
        _,bodies,_=p.preflight(root,a,time.monotonic()+30)
        for n in p.v.CODE: self.assertEqual(bodies['code/'+n],(p.ROOT/n).read_bytes())
        for n in p.v.INITS: self.assertEqual(bodies['code/'+n],b'')
        self.assertFalse((root/'code/act/back_end/moe/batched_support.py').exists())

    def test_external_code_and_data_anchors(self):
        root,a,_=self.base()
        for key in a:
            bad=dict(a,**{key:'0'*64})
            with self.assertRaises(ValueError): p.preflight(root,bad,time.monotonic()+30)
        for target in ('verify.py','code/scoped_source/hz_source_check.py'):
            dest,a,_=self.copied('code_'+target.replace('/','_'))
            (dest/target).write_bytes(b'print("FORGED")\n')
            signed=restamp(dest,a)
            with self.assertRaisesRegex(ValueError,'caller checker identity'):
                p.preflight(dest,signed,time.monotonic()+30)
        dest,a,_=self.copied('source_change'); source=load(dest/'source.json')
        source['request']['label']=1; rewrite(dest/'source.json',source)
        signed=restamp(dest,a); signed['source_sha256']=a['source_sha256']
        with self.assertRaises(ValueError): p.preflight(dest,signed,time.monotonic()+30)

    def test_complete_file_inventory(self):
        for what in ('missing','extra','symlink','traversal','duplicate'):
            dest,a,_=self.copied('inventory_'+what)
            if what=='missing': (dest/'package.json').unlink()
            elif what=='extra': (dest/'unlisted.py').write_text('raise RuntimeError("shadow")')
            elif what=='symlink':
                (dest/'package.json').unlink(); (dest/'package.json').symlink_to(self.base()[0]/'package.json')
            elif what=='traversal':
                m=load(dest/'manifest.json'); m['files']['../outside']=m['files'].pop('package.json')
                rewrite(dest/'manifest.json',m); a['manifest_sha256']=p.sha(dest/'manifest.json')
            else:
                raw=(dest/'manifest.json').read_bytes(); (dest/'manifest.json').write_bytes(b'{"schema":"wrong",'+raw[1:])
                a['manifest_sha256']=p.sha(dest/'manifest.json')
            with self.assertRaises((ValueError,FileNotFoundError)): p.preflight(dest,a,time.monotonic()+30)

    def test_mathematical_mutations(self):
        for what in ('pair','property','factor','input'):
            dest,a,_=self.copied('math_'+what); q=load(dest/'package.json')
            if what=='pair': q['proof']['pairs'].pop()
            elif what=='property': q['endpoint_request']['properties'][0]['offset']='-1'
            elif what=='factor': q['pairs'][0]['a'][-1]['target']['continuous_ids'][0]='foreign/input'
            else: q['input']['hz']['c'][0]='999'
            rewrite(dest/'package.json',q); a=restamp(dest,a)
            log=ROOT/'mutations'/('math_'+what+'.log')
            deadline=time.monotonic()+30; stage=p.launch(dest,a,log,deadline)
            save(ROOT/'mutations'/('math_'+what+'_stage.json'),stage)
            p.validate_stage(dest,a,stage,log,deadline,success=False)
            self.assertEqual(log.read_text().splitlines()[-1],'ValueError: '+MATH_ERRORS[what])

    def test_import_and_read_isolation(self):
        root,a,expected=self.base(); shadow=ROOT/'mutations'/'shadow'; shadow.mkdir()
        for n in ('local_check','proof_format','scoped_source','act'):
            (shadow/(n+'.py')).write_text('raise RuntimeError("SHADOW_EXECUTED")\n')
        end=time.monotonic()+30; cmd=p.command(root,a,end-.5); cmd[1]='--chdir='+str(shadow)
        log=ROOT/'mutations'/'shadow.log'
        stage=p.run_child(cmd,root,a,log,end,dict(os.environ,PYTHONPATH=str(shadow),PYTHONDONTWRITEBYTECODE='1'))
        save(ROOT/'mutations'/'shadow_stage.json',stage)
        self.assertEqual(p.receive(root,a,stage,log,end,expected,end)['result'],expected)
        # Fixed trusted probe uses the same allowlist/read policy, no producer.
        probe='''import importlib.util,sys,time,json,types
from pathlib import Path
path=Path(sys.argv[1]); spec=importlib.util.spec_from_file_location("portable_probe",path)
v=importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
sys.modules["local_check"]=types.ModuleType("local_check")
try: v.clean_namespace(); raise AssertionError("polluted namespace accepted")
except ValueError: pass
del sys.modules["local_check"]
root=path.parent; anchors=json.loads(sys.argv[2]); _,bodies,_=v.envelope(root,anchors,time.monotonic()+30)
sys.meta_path.insert(0,v.FrozenModules(root,bodies))
guard,_=v.read_guard(root,tuple(Path(p).resolve() for p in sys.path if p)); sys.addaudithook(guard)
blocked=[]
for name in ("numpy","scipy","torch","local_check","proof_format","act.back_end.moe.batched_support"):
 try: __import__(name); raise AssertionError("forbidden import accepted")
 except ImportError: blocked.append(name)
try: open(sys.argv[3],"rb"); raise AssertionError("outside read accepted")
except PermissionError: blocked.append("outside_read")
try: open(root/"new_file","w"); raise AssertionError("write accepted")
except PermissionError: blocked.append("write")
print(json.dumps({"blocked":blocked}))
'''
        end=time.monotonic()+30; log=ROOT/'mutations'/'isolation.log'
        stage=p.run_child([PYTHON,'-B','-I','-S','-c',probe,str(root/'verify.py'),json.dumps(a),str(p.ROOT/p.CONFIG)],
                          root,a,log,end,dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        save(ROOT/'mutations'/'isolation_stage.json',stage)
        p.validate_stage(root,a,stage,log,end)
        self.assertEqual(load(log)['blocked'],['numpy','scipy','torch','local_check','proof_format',
                         'act.back_end.moe.batched_support','outside_read','write'])

    def test_expired_deadline(self):
        root,a,_=self.base()
        with self.assertRaises(TimeoutError): p.preflight(root,a,time.monotonic()-1)
        end=time.monotonic()+30; log=ROOT/'mutations'/'expired.log'
        stage=p.run_child(p.command(root,a,time.monotonic()-1),root,a,log,end,
                          dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        save(ROOT/'mutations'/'expired_stage.json',stage)
        p.validate_stage(root,a,stage,log,end,success=False)
        self.assertEqual(log.read_text().splitlines()[-1],'TimeoutError: portable shared deadline')

    def test_result_receipt_binding(self):
        root,a,expected=self.base(); stage=load(ROOT/'weighted_sign'/'stage.json'); log=ROOT/'weighted_sign'/'checker.log'
        real=p.load
        for what in ('source','result','modules','reads','isolation'):
            def damaged(path,*args,**kwargs):
                result=real(path,*args,**kwargs)
                if Path(path)==log:
                    if what=='source': result['anchors']['source_sha256']='0'*64
                    elif what=='result': result['result']['positive']=0
                    elif what=='modules': result['loaded_modules'].pop('act')
                    elif what=='reads': result['bundle_reads'].append('/outside')
                    else: result['isolated']=False
                return result
            with patch.object(p,'load',side_effect=damaged),self.assertRaises(ValueError):
                p.receive(root,a,stage,log,time.monotonic()+30,expected)
        with self.assertRaises(TimeoutError): p.receive(root,a,stage,log,time.monotonic()-1,expected)
        edits=[('cleanup_status','CLEANUP_INCOMPLETE'),('pid',None),('exit_observed_at',None),
               ('remaining_group',{'live':[123]}),('descendant_on_leader_exit',True),
               ('cleanup_seconds',-1.),('execution_seconds',-1.),('seconds',100.),
               ('run_deadline',stage['run_deadline']+1),('cleanup_deadline',stage['cleanup_deadline']+1)]
        for key,value in edits:
            with self.assertRaises(ValueError):
                p.receive(root,a,dict(stage,**{key:value}),log,time.monotonic()+30,expected)
        for key,value in [('stdout_sha256','0'*64),('root','/wrong'),('end',stage['receipt']['deadline']+1)]:
            bad=copy.deepcopy(stage); bad['receipt'][key]=value
            with self.assertRaises(ValueError): p.receive(root,a,bad,log,time.monotonic()+30,expected)
        # Final publication must be charged before an API return can pass.
        with patch.object(p,'save') as saved,patch.object(p.time,'monotonic',return_value=101.):
            with self.assertRaises(TimeoutError): p.return_observed(Path('/unused'),{'status':'CHECKED'},100.)
            saved.assert_called_once()
