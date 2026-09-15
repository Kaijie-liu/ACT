"""New full-run authority, retaining old semantic and adapter identities."""
from pathlib import Path

from scripts.conv_three_arm_contract import (ROOT, ACT, ARMS, read, selection,
    SELECTION_HASH, PROTOCOL_HASH)
from scripts.conv_budget_smoke_v2 import identity as smoke_identity
from act.pipeline.moe.freeze_conv_three_arm import jobs
from act.pipeline.moe.experiment1 import _sha256

PROTOCOL=ROOT/'scripts/conv_full_v2_protocol.json'
DEFAULT=ROOT/'data/moe/results/conv_three_arm_full_20260915_v2'
FREEZE=ROOT/'act/pipeline/moe/results/conv_full_v2_freeze_review_20260915.json'
FILES=('scripts/conv_full_v2_contract.py','scripts/conv_full_v2_protocol.json',
       'scripts/conv_full_v2_worker.py','scripts/run_conv_full_v2.py',
       'scripts/audit_conv_full_v2.py','scripts/test_conv_full_v2.py','docs/conv_full_v2.md')


def identity():
    p=read(PROTOCOL)
    expected={'protocol':'CONV_THREE_ARM_FULL_V2','full_90_authorized':True,
        'selection_sha256':SELECTION_HASH,'semantic_protocol_sha256':PROTOCOL_HASH,
        'samples':30,'requests':90,'budget_seconds':300,'act_terminal_reserve_seconds_inside_budget':5,
        'multi_pair_tier1_fraction':.25,'arms':list(ARMS),'workers':1,'threads':1,'resume':False,'retry':False,
        'output':str(DEFAULT.relative_to(ROOT)),'freeze_review':str(FREEZE.relative_to(ROOT))}
    if any(p.get(k)!=v for k,v in expected.items()):raise ValueError('full protocol drift')
    return {'protocol':p,'protocol_sha256':_sha256(PROTOCOL),
        'sources':{f:_sha256(ROOT/f) for f in FILES},'smoke_execution':smoke_identity()}


def full_selection():
    v=selection()
    if (len(v['samples'])!=30 or v['full_jobs']!=jobs(v['samples']) or len(v['full_jobs'])!=90
            or len({s['dataset_index'] for s in v['samples']})!=30
            or {s['dataset_index'] for s in v['samples']}&set(v['excluded_indices'])):
        raise ValueError('full cohort/order differs')
    for sample in v['samples']:
        record=v['materialized_inputs'][str(sample['dataset_index'])]
        if _sha256(Path(record['path']))!=record['sha256']:raise ValueError('full materialized input drift')
    return v


def request_for(value,job,head,execution):
    if job not in value['full_jobs']:raise ValueError('not a registered full job')
    sample=value['samples'][job['rank']]
    if sample['dataset_index']!=job['dataset_index']:raise ValueError('rank/index mismatch')
    req={'protocol':value['protocol']['protocol'],'method':job['method'],'epsilon':2/255,
        'topology':{'num_experts':4,'top_k':2,'classes':10},'subject':value['subject'],
        'sample':sample,'tensors':value['materialized_inputs'][str(sample['dataset_index'])],
        'config':value['identities']['method_configs'].get(job['method']),'head':head,
        'full_execution':{'protocol':execution['protocol']['protocol'],'identity':execution}}
    if job['method']!='crown':req['execution_budget_contract']=execution['smoke_execution']['budget']
    return req


def validate_worker(root):
    root=root.resolve()
    if root.parent!=DEFAULT:raise ValueError('worker requires owning frozen full root')
    runtime=read(root.parent/'runtime.json');execution=identity();value=full_selection()
    if (runtime['schema']!='conv_three_arm_full_v2' or runtime['execution']!=execution
            or runtime['selection']!=value or runtime['smoke'] is not False
            or runtime['freeze_sha256']!=_sha256(FREEZE)):
        raise ValueError('owning runtime/selection drift')
    job=next((j for j in value['full_jobs'] if j['job_id']==root.name),None)
    if job is None:raise ValueError('unexpected directory/job')
    req=read(root/'request.json')
    if req!=request_for(value,job,runtime['git_head'],execution):raise ValueError('request differs')
    for name in ('package','budget_journal.jsonl','routes.json','external.json','terminal.json'):
        if (root/name).exists():raise ValueError('no resume/replacement')
    return req
