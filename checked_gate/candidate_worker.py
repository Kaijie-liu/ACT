"""Single saved-LP proposal and independent complete-request replacement check."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

from checked_gate.bootstrap import ROOT, setup

PARENT = 'data/moe/results/checked_gate_saved_20260920_v1'
PARENT_HASH = 'b8ac608a4b917882ee186a566e925d3d02b7437b461d50cca57c7abdd32893b9'
ARCHIVE = 'act/pipeline/moe/results/request_lp_act_only_review_20260915_r1.json'
ARCHIVE_HASH = '56998876d4139928a42a2b16c515934c49e7702d9cd3a6af5686d3e7e0bb22a4'
OLD_ROOT = 'data/moe/results/request_lp_act_only_20260915_r1/seed1_4018'
REQUEST_ID = '94e5fde01384d365fda0c9ca40ddc7a08820e5e41aaacefcd38e7902aeff06ce'


def load(path, digest=None):
    raw=Path(path).read_bytes()
    if digest is not None and hashlib.sha256(raw).hexdigest()!=digest:
        raise ValueError('artifact identity mismatch: '+str(path))
    return json.loads(raw)


def save(path, value):
    raw=json.dumps(value,sort_keys=True,indent=2,allow_nan=False).encode()
    with Path(path).open('xb') as f: f.write(raw)
    return {'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}


def sources():
    parent=load(ROOT/PARENT/'result.json',PARENT_HASH)
    archive=load(ROOT/ARCHIVE,ARCHIVE_HASH)
    directory=ROOT/OLD_ROOT
    expected={k:v for k,v in archive['raw_hashes'].items() if k.startswith('seed1_4018/') and k.endswith('.json')}
    if parent['parent_dependencies']!=expected:
        raise ValueError('parent inventory drift')
    for name,digest in expected.items():
        load(directory.parent/name,digest)
    def read(ref):
        file=(directory/ref['file']).resolve()
        if not file.is_relative_to(directory.resolve()): raise ValueError('escaping proof reference')
        return load(file,ref['sha256'])
    m=load(directory/'manifest.json',expected['seed1_4018/manifest.json'])
    if m['request_id']!=REQUEST_ID: raise ValueError('request changed')
    record=load(ROOT/PARENT/'weighted_export.json',parent['artifacts']['weighted_export.json']['sha256'])
    return parent,m,read,record


def check_saved(root):
    from act.back_end.solver.lp_certificate import identity
    from checked_gate.replacement import check
    _,m,read,record=sources()
    candidate=load(root/'candidate.json')
    if (candidate['parent_result_sha256']!=PARENT_HASH or candidate['request_id']!=REQUEST_ID or
            candidate['candidate_calls']!=1 or candidate['native_limit_seconds']!=90):
        raise ValueError('proposal contract changed')
    values={k:load(root/k,candidate['artifacts'][k]['sha256']) for k in
            ('replacement.json','gate_proof.json','certificate.json')}
    if set(candidate['artifacts'])!=set(values): raise ValueError('candidate dependency inventory')
    return check(m,read,values['replacement.json'],values['gate_proof.json'],record,
                 values['certificate.json'],expected_request_id=REQUEST_ID,
                 expected_manifest_identity=identity(m))


def work(phase, root):
    start=time.monotonic()
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    setup(checker=phase!='propose')
    if phase=='validate':
        from checked_gate.review import review
        # review installs its own clean namespace; remove only our inert stubs.
        import sys
        for name in ('act','act.back_end','act.back_end.solver','act.pipeline','act.pipeline.moe'):
            sys.modules.pop(name,None)
        result=review(ROOT/PARENT,PARENT_HASH)
        save(root/'validation.json',result)
    elif phase=='propose':
        from act.back_end.solver.lp_certificate import identity
        from act.back_end.solver.sparse_lp_certificate import propose
        parent,m,read,record=sources()
        row=next(r for r in m['obligations'] if r['pair']==[3,5] and r['property_index']==1)
        replacement={'schema':'SINGLE_GATE_REPLACEMENT_V1','request_id':REQUEST_ID,
            'manifest_identity':identity(m),'pair':[3,5],'property_index':1}
        # Same gate arithmetic and endpoints. Only use canonical object hashes
        # for the new interface's externally checked support-certificate binding.
        gate=load(ROOT/PARENT/'gate_proof.json',parent['artifacts']['gate_proof.json']['sha256'])
        gate['context']['margin_lower_proof']=identity(read(m['proofs'][row['gate_lower']]['certificate']))
        gate['context']['margin_negative_upper_proof']=identity(read(m['proofs'][row['gate_upper']]['certificate']))
        save(root/'proposal_entered.json',{'calls':1,'native_limit_seconds':90,
                                         'lp_identity':identity(record['lp'])})
        tick=time.monotonic()
        certificate=propose(record['lp'],time_limit=90)
        proposal_seconds=time.monotonic()-tick
        artifacts={name:save(root/name,value) for name,value in
            (('replacement.json',replacement),('gate_proof.json',gate),('certificate.json',certificate))}
        save(root/'candidate.json',{'request_id':REQUEST_ID,'parent_result_sha256':PARENT_HASH,
            'candidate_calls':1,'native_limit_seconds':90,'proposal_inclusive_seconds':proposal_seconds,
            'artifacts':artifacts,'worker_seconds_before_publication':time.monotonic()-start})
    elif phase=='check':
        save(root/'checked.json',check_saved(root))
    else: raise ValueError('unknown phase')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('validate','propose','check')); p.add_argument('root',type=Path)
    a=p.parse_args(); work(a.phase,a.root.resolve())
