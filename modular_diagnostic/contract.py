"""Read-only selection, compatibility and immutable protocol; no solve here."""
from pathlib import Path
import time
from single_check_portable.execution import ROOT, read, save_new
from portable_proof.runtime import digest
from modular_supervised.flow import sources as component_sources
from modular_basis.engine import POLICY as ARITHMETIC
from modular_supervised.journal import JOURNAL_POLICY
from fidelity_supervised.native import OPTIONS
from fidelity_supervised.study import runtime, select, compatibility

FREEZE=ROOT/'docs/modular_diagnostic_v1_freeze.json'
REVIEW=ROOT/'docs/modular_diagnostic_v1_selection_review.json'
CONTROL_REVIEW=ROOT/'docs/modular_diagnostic_v1_controls_review.json'
OUTPUT=ROOT/'data/moe/results/modular_diagnostic_real_20260920_v1'
PRIOR=ROOT/'docs/primitive_diagnostic_v1_execution_results.json'
INTEGRATION=ROOT/'docs/modular_supervised_controls_attempt001.json'
INTEGRATION_REVIEW=ROOT/'docs/modular_supervised_v1_review.json'
POLICY={
    'requests':4,'indices':[220,222,230,232],
    'job_ids':['input220_p0','input222_p1','input230_p2','input232_p0'],
    'selection':'identical ordered original obligations; no new samples or properties',
    'total_seconds':300,'work_seconds':298,'proposal_seconds':218,'native_cap_seconds':10,
    'workers':1,'threads':1,'retry':False,'resume':False,'basis_attempts':1,
    'resource':{'minimum_ram_gib':16,'minimum_disk_gib':5,'maximum_load_per_core':.5,
                'wait_limit_seconds':86400,'poll_seconds':30},
    'arithmetic':dict(ARITHMETIC),'journal':dict(JOURNAL_POLICY),'native_options':dict(OPTIONS),
    'LP_changed':False,'feasibility_tolerance':0,'fallback':None,
    'stop':'ERROR stops; unresolved states continue; unstarted rows retained',
    'decision':'original LP exact primal check only; no new dual or network verdict',
    'comparison':'saved primitive V1 descriptive context; compare basis/system identity, not pivot trace or matched timing',
    'scope':'supplied-LP development diagnostic, historical upstream work excluded',
    'launch':'separate explicit execute-frozen action after clean pushed freeze',
}


def ref(path):
    path=Path(path).resolve()
    return {'path':str(path),'sha256':digest(path.read_bytes())}


def sources():
    result=component_sources()
    paths=list(Path(__file__).parent.glob('*.py'))+[ROOT/p for p in (
        'fidelity_supervised/study.py','fidelity_diagnostic_archive/structure.py',
        'sparse_diagnostic_archive/review.py','evidence_cohort/contract.py',
        'evidence_cohort/ownership.py','scripts/optional_evidence_dev_contract.py')]
    paths += [ROOT/'docs/modular_diagnostic_v1.md']
    result.update({str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in paths})
    return dict(sorted(result.items()))


def verify_sealed():
    from modular_supervised.controls import old
    old()
    c,r=read(INTEGRATION),read(INTEGRATION_REVIEW)
    if (c['status']!='PASS' or r['status']!='PASS' or r['issues'] or
            c['sources']!=component_sources() or r['sources']!=c['sources'] or
            r['controls_sha256']!=ref(INTEGRATION)['sha256']):
        raise ValueError('frozen integration gate')


def check_roster(jobs):
    if ([j['job_id'] for j in jobs]!=POLICY['job_ids'] or
            [j['dataset_index'] for j in jobs]!=POLICY['indices']):
        raise ValueError('ordered four-obligation roster')


def control_gate(controls):
    c=read(controls);r=read(CONTROL_REVIEW)
    if (c['status']!='PASS' or c['sources']!=sources() or r['status']!='PASS' or
            r['issues'] or r['sources']!=sources() or r['controls']!=ref(controls)):
        raise ValueError('diagnostic controls/fresh review required')


def freeze(controls):
    if any(p.exists() for p in (FREEZE,REVIEW,OUTPUT)):raise FileExistsError('new freeze only')
    begin=time.monotonic();verify_sealed();control_gate(controls)
    from evidence_cohort.run import resource,resource_ok
    resources=resource()
    if not resource_ok(resources):raise RuntimeError('read-only preflight resource gate')
    jobs=select();check_roster(jobs)
    value={'schema':'MODULAR_DIAGNOSTIC_FREEZE_V1','status':'FROZEN_NOT_EXECUTED',
        'policy':POLICY,'sources':sources(),'jobs':jobs,'compatibility':compatibility(jobs),
        'runtime':runtime(),'controls':ref(controls),'controls_review':ref(CONTROL_REVIEW),
        'integration':ref(INTEGRATION),'integration_review':ref(INTEGRATION_REVIEW),
        'prior_archive':ref(PRIOR),'output':str(OUTPUT),'resource':resources,
        'preparation_seconds':time.monotonic()-begin,'real_solver_calls':0,'real_reconstructions':0}
    save_new(FREEZE,value)
    return {'status':value['status'],'jobs':POLICY['job_ids'],'real_solver_calls':0}


def validate(value):
    if (value['schema']!='MODULAR_DIAGNOSTIC_FREEZE_V1' or value['status']!='FROZEN_NOT_EXECUTED'
            or value['policy']!=POLICY or value['sources']!=sources() or value['output']!=str(OUTPUT)
            or value['runtime']!=runtime()):raise ValueError('protocol/source/runtime drift')
    check_roster(value['jobs'])
    # Complete job identity, including source, pair and property, not just indices.
    if value['jobs']!=read(ROOT/'docs/fidelity_supervised_real_v2_freeze.json')['jobs']:
        raise ValueError('changed frozen obligation')
    for name,path in [('integration',INTEGRATION),('integration_review',INTEGRATION_REVIEW),
                      ('prior_archive',PRIOR),('controls_review',CONTROL_REVIEW)]:
        if value[name]!=ref(path):raise ValueError('evidence gate identity')
    controls=Path(value['controls']['path'])
    if value['controls']!=ref(controls):raise ValueError('controls changed')
    control_gate(controls)
    for job in value['jobs']:
        if job['export']!=ref(job['export']['path']):raise ValueError('LP source bytes changed')
    return value


def verify():
    verify_sealed()
    return validate(read(FREEZE))


def reconstruct():
    begin=time.monotonic();v=verify()
    if OUTPUT.exists():raise FileExistsError('selection review precedes execution')
    if v['jobs']!=select() or v['compatibility']!=compatibility(v['jobs']):
        raise ValueError('independent selection/compatibility reconstruction')
    result={'status':'PASS','issues':[],'freeze':ref(FREEZE),'sources':sources(),
        'jobs':POLICY['job_ids'],'output_absent':True,'seconds':time.monotonic()-begin,
        'real_solver_calls':0,'real_reconstructions':0,
        'scope':'read-only identity, original size and import preflight; new arithmetic efficacy unknown'}
    save_new(REVIEW,result)
    return result
