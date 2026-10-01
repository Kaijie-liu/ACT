"""Isolated stdlib bootstrap for byte-bound saved source-to-HybridZ evidence.

Trust the externally anchored code, not arbitrary Python in a proof bundle.
This read/import policy is not a sandbox against malicious trusted code.
"""
import argparse
import hashlib
import importlib.abc
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import sys
import time

CODE = ('scoped_source/hz_source_check.py', 'scoped_source/graph.py',
        'scoped_source/rowwise_bound.py', 'source_enclosure/format.py',
        'source_enclosure/check.py', 'upstream_source/checker.py', 'router_source/checker.py',
        'act/back_end/moe/check_hz_endpoints.py', 'act/back_end/moe/check_batched_support.py',
        'act/back_end/solver/check_hz_lp_export.py', 'act/back_end/solver/lp_certificate.py')
PACKAGES = ('act', 'act/back_end', 'act/back_end/moe', 'act/back_end/solver',
            'scoped_source', 'source_enclosure', 'upstream_source', 'router_source')
INITS = tuple(p+'/__init__.py' for p in PACKAGES)
CODE_NAMES = {'code/'+n for n in CODE+INITS} | {'verify.py'}
MEMBERS = CODE_NAMES | {'source.json', 'package.json'}
MEMBER_LIMIT, TOTAL_LIMIT = 4*2**20, 32*2**20
ALIASES = {'local_check', 'proof_format'}
ROOTS = {p.split('/')[0] for p in PACKAGES}
FORBIDDEN = {'torch','numpy','scipy','highspy','gurobipy','ctypes'}


def tick(deadline):
    if not math.isfinite(deadline) or time.monotonic() >= deadline:
        raise TimeoutError('portable shared deadline')


def digest(raw): return hashlib.sha256(raw).hexdigest()
def identity(obj): return digest(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode())


def hash_value(v):
    if type(v) is not str or len(v)!=64 or any(c not in '0123456789abcdef' for c in v):
        raise ValueError('required external SHA identity')
    return v


def decode(raw):
    def pairs(items):
        d={}
        for k,v in items:
            if k in d: raise ValueError('duplicate JSON member')
            d[k]=v
        return d
    return json.loads(raw,object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def member(root,name):
    rel=Path(name)
    if rel.is_absolute() or str(rel)!=name or '..' in rel.parts: raise ValueError('member path')
    path=Path(root)
    if path.is_symlink(): raise ValueError('symlink bundle')
    for part in rel.parts:
        path=path/part
        if path.is_symlink(): raise ValueError('symlink member')
    st=path.stat()
    if not stat.S_ISREG(st.st_mode) or st.st_size>MEMBER_LIMIT or st.st_nlink!=1:
        raise ValueError('nonregular/large/aliased member')
    with path.open('rb') as stream: raw=stream.read(MEMBER_LIMIT+1)
    if len(raw)>MEMBER_LIMIT: raise ValueError('growing member')
    return raw


def envelope(root, anchors, deadline):
    tick(deadline)
    if set(anchors)!={'manifest_sha256','source_sha256','package_sha256','code_sha256'}:
        raise ValueError('external anchor inventory')
    for v in anchors.values(): hash_value(v)
    raw=member(root,'manifest.json')
    if digest(raw)!=anchors['manifest_sha256']: raise ValueError('external manifest identity')
    m=decode(raw)
    if (set(m)!={'schema','source_sha256','package_sha256','code_sha256','files'}
            or m['schema']!='HZ_SOURCE_PORTABLE_V1'
            or any(m[k]!=anchors[k] for k in ('source_sha256','package_sha256','code_sha256'))
            or set(m['files'])!=MEMBERS): raise ValueError('manifest context/inventory')
    actual=set(); directories=set()
    permitted_dirs={str(p) for n in MEMBERS for p in Path(n).parents if str(p)!='.'}
    for p in Path(root).rglob('*'):
        tick(deadline)
        if p.is_symlink(): raise ValueError('symlink inventory')
        name=str(p.relative_to(root))
        if p.is_file(): actual.add(name)
        elif p.is_dir(): directories.add(name)
        else: raise ValueError('nonregular inventory')
    if actual!=MEMBERS|{'manifest.json'} or directories!=permitted_dirs:
        raise ValueError('complete member inventory')
    bodies={}; total=len(raw)
    for name,ref in sorted(m['files'].items()):
        tick(deadline)
        zero=name in {'code/'+n for n in INITS}
        if (set(ref)!={'sha256','bytes'} or type(ref['bytes']) is not int
                or not (0 if zero else 1)<=ref['bytes']<=MEMBER_LIMIT
                or zero and ref['bytes']!=0): raise ValueError('member size descriptor')
        data=member(root,name)
        if len(data)!=ref['bytes'] or digest(data)!=hash_value(ref['sha256']):
            raise ValueError('member identity')
        total+=len(data); bodies[name]=data
        if total>TOTAL_LIMIT: raise ValueError('total bundle capacity')
    if identity({n:m['files'][n] for n in sorted(CODE_NAMES)})!=anchors['code_sha256']:
        raise ValueError('external code identity')
    for name,key in [('source.json','source_sha256'),('package.json','package_sha256')]:
        if identity(decode(bodies[name]))!=anchors[key]: raise ValueError('external data identity')
    tick(deadline)
    return m,bodies,total


class FrozenModules(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Execute precisely the bytes already checked, never reopen import paths."""
    def __init__(self,root,bodies):
        self.root=Path(root); self.modules={}
        for p in PACKAGES: self.modules[p.replace('/','.')]=('code/'+p+'/__init__.py',True)
        for p in CODE: self.modules[p[:-3].replace('/','.')]=('code/'+p,False)
        self.bodies=bodies

    def find_spec(self,fullname,path=None,target=None):
        if fullname in self.modules:
            name,package=self.modules[fullname]
            return importlib.util.spec_from_loader(fullname,self,origin=str(self.root/name),is_package=package)
        if fullname.split('.')[0] in ROOTS|ALIASES: raise ImportError('unlisted local module')
        return None

    def create_module(self,spec): return None

    def exec_module(self,module):
        name,package=self.modules[module.__name__]
        module.__file__=str(self.root/name)
        if package: module.__path__=[]
        exec(compile(self.bodies[name],module.__file__,'exec'),module.__dict__)


def clean_namespace():
    if any(n.split('.')[0] in ROOTS|ALIASES|FORBIDDEN for n in sys.modules):
        raise ValueError('preloaded local/numerical namespace')


def read_guard(root,stdlib):
    root=Path(root).resolve(); reads=set()
    def guard(event,args):
        if event=='open' and not isinstance(args[0],int):
            path=Path(os.fsdecode(args[0])).resolve(); mode,flags=args[1:3]
            if (isinstance(mode,str) and any(c in mode for c in 'wax+')) or flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT):
                raise PermissionError('read-only checker')
            standard=not {'site-packages','dist-packages'}.intersection(path.parts) and any(path.is_relative_to(p) for p in stdlib)
            if path.is_relative_to(root): reads.add(str(path.relative_to(root)))
            elif not standard: raise PermissionError('outside bundle/stdlib')
        if event.startswith(('subprocess.','socket.','ctypes.')) or event in (
                'os.system','os.exec','os.fork','os.posix_spawn','os.remove','os.rename','os.mkdir','os.rmdir','os.link','os.symlink','os.chdir'):
            raise PermissionError('external execution/mutation forbidden')
        if event=='import' and args[0].split('.')[0] in FORBIDDEN|ALIASES:
            raise ImportError('numerical/compatibility import forbidden')
    return guard,reads


def verify(root,anchors,deadline):
    clean_namespace()
    m,bodies,total=envelope(root,anchors,deadline)
    loader=FrozenModules(root,bodies); sys.meta_path.insert(0,loader)
    from scoped_source.hz_source_check import check
    result=check(decode(bodies['source.json']),decode(bodies['package.json']),
                 expected_source_sha256=anchors['source_sha256'],deadline=deadline)
    loaded={}
    for name,module in list(sys.modules.items()):
        if name.split('.')[0] in FORBIDDEN|ALIASES: raise ValueError('forbidden imported module')
        if name.split('.')[0] in ROOTS:
            if name not in loader.modules or module.__loader__ is not loader: raise ValueError('local import loader')
            path,package=loader.modules[name]
            if module.__file__!=str(Path(root)/path) or package and module.__path__!=[]:
                raise ValueError('local module origin/path')
            loaded[name]={'path':path,'sha256':digest(bodies[path])}
    if set(loaded)!=set(loader.modules): raise ValueError('checker import closure changed')
    final,_,count=envelope(root,anchors,deadline)
    if final!=m or total!=count: raise ValueError('bundle changed during checking')
    tick(deadline)
    return {'schema':'HZ_SOURCE_PORTABLE_CHECK_V1','anchors':anchors,'result':result,
            'bundle_bytes':total,'loaded_modules':loaded,'numerical_modules_loaded':[],
            'isolated':bool(sys.flags.isolated),'site_disabled':bool(sys.flags.no_site),
            'offline_recheck_only':True,'deployed_float_SAFE':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('manifest','source','package','code'): parser.add_argument('--'+k+'-sha',required=True)
    parser.add_argument('--deadline',type=float,required=True)
    a=parser.parse_args(); start=time.monotonic(); root=Path(__file__).resolve().parent
    if not sys.flags.isolated or not sys.flags.no_site: raise ValueError('python -I -S required')
    if not math.isfinite(a.deadline) or a.deadline>start+300: raise ValueError('bounded checker deadline')
    sys.dont_write_bytecode=True
    guard,reads=read_guard(root,tuple(Path(p).resolve() for p in sys.path if p))
    sys.addaudithook(guard)
    anchors={k+'_sha256':getattr(a,k+'_sha') for k in ('manifest','source','package','code')}
    result=verify(root,anchors,a.deadline)
    result['bundle_reads']=sorted(reads); result['check_seconds']=time.monotonic()-start
    raw=json.dumps(result,sort_keys=True,allow_nan=False); tick(a.deadline); print(raw,flush=True)


if __name__=='__main__': main()
