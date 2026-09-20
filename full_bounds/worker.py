"""Fresh numeric candidates only. Old matrices/duals never changed or imported."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import time
import resource
from fractions import Fraction as F

from router_source.capture import ROOT,sha
from router_source.build import save
from source_enclosure.format import identity,compact

PARENT=ROOT/'data/moe/results/full_source_conv98_20260920_v1/relocated'
PARENT_HASH='ade6c1b34db6fa4923c9c8f13e6a547771aa0aa723fa32027b77c93a93bb295d'
NATIVE_SECONDS=16.
CODE={'full_bounds/check.py':'bound_check.py','full_bounds/verify.py':'verify_bounds.py',
      'act/back_end/solver/lp_certificate.py':'act/back_end/solver/lp_certificate.py',
      'act/back_end/solver/sparse_lp_certificate.py':'act/back_end/solver/sparse_lp_certificate.py'}


def publish(path,obj):
    """Only a completely fsynced candidate becomes visible to the sealer."""
    temp=path.with_name(path.name+'.partial')
    with temp.open('xb') as stream:stream.write(compact(obj));stream.flush();os.fsync(stream.fileno())
    os.link(temp,path);temp.unlink()


def prepare(root,parent=PARENT,parent_hash=PARENT_HASH):
    start=time.monotonic()
    if sha(parent/'manifest.json')!=parent_hash:raise ValueError('new source parent identity')
    m=json.loads((parent/'manifest.json').read_bytes())
    for name,h in m['files'].items():
        p=(parent/name).resolve()
        if not p.is_relative_to(parent.resolve()) or sha(p)!=h:raise ValueError('new source parent changed')
    dst=root/'relocated';dst.mkdir();shutil.copytree(parent,dst/'source')
    for src,name in CODE.items():
        path=dst/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/src,path)
    save(root/'preparation.json',{'parent_manifest_sha256':parent_hash,'request':m['request'],'pair':m['pair'],
        'seconds':time.monotonic()-start,'scope':'Stored source copied; no old dual or bound imported.'})


def propose(root,native_seconds=NATIVE_SECONDS):
    start=time.monotonic();dst=root/'relocated';src=dst/'source'
    prep=json.loads((root/'preparation.json').read_bytes());base=json.loads((src/'lp_base.json').read_bytes())
    obligations=json.loads((src/'obligations.json').read_bytes())
    from full_source.obligations import materialize
    import numpy as np
    from scipy.optimize import linprog
    from scipy.optimize._highspy import _core
    from scipy.sparse import csr_matrix,vstack
    def matrix(m):return csr_matrix(([float(F(v)) for v in m['data']],m['indices'],m['indptr']),shape=m['shape'])
    commonA,commonE=matrix(base['A']),matrix(base['E']);common_rhs=np.asarray([float(F(v)) for v in base['b']])
    equality_rhs=np.asarray([float(F(v)) for v in base['h']]);conversion_seconds=time.monotonic()-start
    save(root/'solver_environment.json',{'scipy':importlib.metadata.version('scipy'),'numpy':importlib.metadata.version('numpy'),
        'bundled_highs':'.'.join(str(getattr(_core,'HIGHS_VERSION_'+k)) for k in ('MAJOR','MINOR','PATCH')),
        'method':'highs','options':{'time_limit':native_seconds,'threads':1},'base_conversion_seconds':conversion_seconds})
    calls=0
    for obligation in obligations['rows']:
        tick=time.monotonic();k=obligation['competitor'];lp=materialize(base,obligation);lp_id=identity(lp)
        a=vstack([commonA,matrix(obligation['A_extra'])],format='csr');b=np.concatenate([common_rhs,[float(F(v)) for v in obligation['b_extra']]])
        c=np.asarray([float(F(v)) for v in lp['c']]);bounds=[(float(F(x)),float(F(y))) for x,y in zip(lp['lower'],lp['upper'])]
        if not all(np.isfinite(v).all() for v in (a.data,commonE.data,b,equality_rhs,c,np.asarray(bounds))):raise ValueError('nonfinite numeric proposal input')
        save(root/f'entered_{k}.json',{'competitor':k,'lp_sha256':lp_id,'native_seconds':native_seconds,
              'seconds_since_worker_start':time.monotonic()-start,'assembly_seconds':time.monotonic()-tick})
        native_start=time.monotonic();calls+=1
        result=linprog(c,A_ub=a,b_ub=b,A_eq=commonE,b_eq=equality_rhs,bounds=bounds,method='highs',
                       options={'time_limit':float(native_seconds),'threads':1})
        native_elapsed=time.monotonic()-native_start
        candidate=None
        if result.success:
            yd=[min(0.,float(v)) for v in result.ineqlin.marginals];zd=[float(v) for v in result.eqlin.marginals]
            if len(yd)!=len(lp['b']) or len(zd)!=len(lp['h']) or any(not math.isfinite(v) for v in yd+zd):raise ValueError('invalid numeric multipliers')
            candidate={'lp_sha256':lp_id,'inequality_dual':yd,'equality_dual':zd}
        objective=float(result.fun)+float(F(lp['offset'])) if result.fun is not None and math.isfinite(float(result.fun)) else None
        primal=None
        if result.x is not None and len(result.x)==len(c) and np.isfinite(result.x).all():
            primal=[float(v) for v in result.x]
        record={'schema':'NEW_OUTPUT_DUAL_CANDIDATE_V1','request_id':identity(prep['request']),
            'parent_manifest_sha256':prep['parent_manifest_sha256'],'pair':prep['pair'],'competitor':k,
            'source_sha256':obligations['source_sha256'],'base_sha256':obligations['base_sha256'],'lp_sha256':lp_id,
            'candidate':candidate,'solver':{'status':int(result.status),'success':bool(result.success),
                'message':str(result.message),'iterations':int(result.nit) if result.nit is not None else None,
                'reported_objective_including_offset':objective,'native_limit_seconds':native_seconds,'native_seconds':native_elapsed},
            'approximate_primal_not_checked':primal,'elapsed_before_publication':time.monotonic()-tick}
        publish(dst/f'candidate_{k}.json',record)
        save(root/f'completed_{k}.json',{'competitor':k,'seconds':time.monotonic()-tick,'candidate_available':candidate is not None})
    save(root/'proposal.json',{'calls':calls,'seconds':time.monotonic()-start,'native_seconds_per_call':native_seconds,
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'old_LP_certificates_used':0})


def seal(root):
    start=time.monotonic();dst=root/'relocated';prep=json.loads((root/'preparation.json').read_bytes())
    obligations=json.loads((dst/'source/obligations.json').read_bytes());outcomes=[]
    for r in obligations['rows']:
        k=r['competitor'];name=f'candidate_{k}.json'
        if (dst/name).exists():json.loads((dst/name).read_bytes()) # malformed publication is ERROR, not missing
        outcomes.append({'competitor':k,'file':name if (dst/name).exists() else None})
    # .partial candidates survive in the raw root but are never proof inputs.
    required=list(CODE.values())+['source/manifest.json']
    parent=json.loads((dst/'source/manifest.json').read_bytes());required+=['source/'+n for n in parent['files']]
    required += [r['file'] for r in outcomes if r['file'] is not None]
    files={n:sha(dst/n) for n in required}
    save(dst/'manifest.json',{'schema':'FULL_SOURCE_DUAL_BUNDLE_V1','parent_manifest_sha256':prep['parent_manifest_sha256'],
        'request':prep['request'],'pair':prep['pair'],'outcomes':outcomes,'files':files})
    save(root/'sealed.json',{'manifest_sha256':sha(dst/'manifest.json'),'seconds':time.monotonic()-start,
        'bundle_bytes':sum((dst/n).stat().st_size for n in required)+(dst/'manifest.json').stat().st_size,
        'published_candidates':sum(r['file'] is not None for r in outcomes),'required_obligations':len(outcomes)})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('prepare','propose','seal'));p.add_argument('root',type=Path)
    a=p.parse_args();globals()[a.phase](a.root.resolve())
