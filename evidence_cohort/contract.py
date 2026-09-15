"""Execution-only identity: frozen inputs and method implementations are immutable."""
from pathlib import Path
from portable_proof.runtime import digest
from scripts.optional_evidence_dev_contract import ROOT,ACT,read,save,git
from scripts.freeze_general_evidence import SELECTION,REVIEW,request_for,policy,jobs

DIRECTORY=ROOT/'evidence_cohort'
FREEZE=ROOT/'docs/general_evidence_execution_v1_freeze.json'
CONTROLS=ROOT/'docs/general_evidence_execution_v1_controls.json'
OUTPUT=ROOT/'data/moe/results/general_evidence_cohort_20260916_v1'
LAUNCH=ROOT/'data/moe/results/general_evidence_launch_20260916_v1'
SELECTION_SHA='db3043fb124703e8123e5326eda853dc0d67d45e9104ca316487a631603daea7'
ARMS=('matched','evidence','crown')
EXECUTION={
    'schema':'GENERAL_EVIDENCE_COHORT_EXECUTION_V1','samples':20,'requests':60,
    'arms':list(ARMS),'budget_seconds':300,'outer_work_seconds':298,
    'threads':1,'workers':1,'resume':False,'retry':False,
    'resource':{'minimum_ram_gib':16,'minimum_disk_gib':5,'maximum_load_per_core':.5,
                'wait_limit_seconds':86400,'poll_seconds':30},
    'audit_per_request_timeout_seconds':600,'final_audit_timeout_seconds':36000,
    'selection_sha256':SELECTION_SHA,'output':str(OUTPUT.relative_to(ROOT)),
    'acceptance':'unchanged method/evidence gates, independently replayed terminal; no late promotion',
    'audit_cost':'outside per-request budget, reported separately; cannot rescue a failed request',
    'full_run_authorized':True,
}


def hashes():
    paths=sorted(DIRECTORY.glob('*.py'))+[ROOT/'docs/general_evidence_execution_v1.md']
    return {str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in paths}


def selection():
    if digest(SELECTION.read_bytes())!=SELECTION_SHA:raise ValueError('selected cohort changed')
    v=read(SELECTION)
    if (v['jobs']!=jobs(v['samples']) or len(v['jobs'])!=60 or len(v['samples'])!=20
            or v['protocol']!=policy() or v['execution_started'] is not False):raise ValueError('selection semantics drift')
    for name,sha in v['source_sha256'].items():
        if digest((ROOT/name).read_bytes())!=sha:raise ValueError('frozen method source changed: '+name)
    if digest((ROOT/'docs/general_evidence_v1_controls.json').read_bytes())!=v['controls_sha256']:
        raise ValueError('method control record changed')
    if read(REVIEW)['status']!='PASS' or read(REVIEW)['selection_sha256']!=SELECTION_SHA:
        raise ValueError('selection review missing')
    for rec in v['materialized_inputs'].values():
        if digest(Path(rec['path']).read_bytes())!=rec['sha256']:raise ValueError('input file drift')
    if digest(Path(v['subject']['checkpoint']).read_bytes())!=v['subject']['checkpoint_sha256']:
        raise ValueError('checkpoint changed')
    return v


def verify_freeze():
    v=read(FREEZE)
    if v['execution']!=EXECUTION or v['sources']!=hashes():raise ValueError('execution source/protocol drift')
    if digest(CONTROLS.read_bytes())!=v['controls_sha256'] or read(CONTROLS)['status']!='PASS':
        raise ValueError('execution controls unavailable')
    if digest(REVIEW.read_bytes())!=v['selection_review_sha256']:raise ValueError('selection review drift')
    selection();return v


def freeze():
    if FREEZE.exists():raise FileExistsError('execution freeze immutable')
    control=read(CONTROLS)
    if control['status']!='PASS' or control['sources']!=hashes():raise ValueError('controls do not cover execution sources')
    selection()
    from scripts.freeze_general_evidence import audit as reconstruct
    rebuilt=reconstruct()
    if rebuilt['status']!='PASS':raise ValueError('fresh clean reconstruction failed')
    value={'execution':EXECUTION,'sources':hashes(),'controls_sha256':digest(CONTROLS.read_bytes()),
        'selection_review_sha256':digest(REVIEW.read_bytes()),'clean_reconstruction':rebuilt,
        'status':'FROZEN_NOT_EXECUTED','method_sources_unchanged':True}
    save(FREEZE,value);return value
