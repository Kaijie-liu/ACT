"""New execution identity only; previous interruption and scientific contract kept."""
import argparse
import copy
from pathlib import Path
import subprocess

from lp_sandwich.check import strict_json
from soplex_fidelity.io import save,sha
from soplex_execution.runtime import ROOT
from soplex_execution.supervisor import verify_freeze,FREEZE as OLD_PROTOCOL,ADDENDUM
from soplex_detached.run import PROTOCOL,FREEZE,OUTPUT,SOCKET,frozen


def prepare(raw):
    parent,old=verify_freeze();r=strict_json((raw/'controls.json').read_bytes())
    if r['status']!='PASS' or r['tests']!=7 or r['real_queries'] or r['native_queries']:raise ValueError('lifecycle controls')
    for name,digest in r['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('tested source drift')
    if OUTPUT.exists() or SOCKET.exists() or PROTOCOL.exists() or FREEZE.exists():raise FileExistsError('new identity only')
    interruption=ROOT/'docs/soplex_execution_interruption_20260920.json'
    i=strict_json(interruption.read_bytes());oldroot=ROOT/i['raw_root']
    if {p.name for p in oldroot.iterdir()}!=set(i['artifacts']):raise ValueError('previous run changed')
    for name,digest in i['artifacts'].items():
        if sha(oldroot/name)!=digest:raise ValueError('previous interruption evidence changed')
    controls=ROOT/'docs/soplex_detached_v2_controls.json';save(controls,r)
    proposal=copy.deepcopy(parent);proposal['output']=str(OUTPUT);save(PROTOCOL,proposal)
    bindings=dict(old['bindings'])
    paths=[OLD_PROTOCOL,ADDENDUM,PROTOCOL,controls,interruption,ROOT/'docs/soplex_detached_v2.md',Path('/usr/bin/tmux')]
    paths+=list(Path(__file__).parent.glob('*.py'))
    bindings.update({str(p):sha(p) for p in paths})
    new=dict(schema='SOPLEX_DETACHED_EXECUTION_FREEZE_V2',protocol_sha256=sha(PROTOCOL),
        original_scientific_protocol_sha256=sha(OLD_PROTOCOL),parent_execution_sha256=sha(ADDENDUM),
        authorized_by='User continued after explicit proposal for session-independent launch/new execution directory.',
        execution_ready=True,bindings=bindings,policy=parent['policy'],output=str(OUTPUT),socket=str(SOCKET),
        tmux_version=subprocess.check_output(['/usr/bin/tmux','-V'],text=True).strip(),
        change_scope='Process lifetime and interruption ledger only; same run_jobs/supervise/worker/candidate/checker/auditor.',
        retries=False,resume=False,raw_results_only_until_archival_commit=True)
    save(FREEZE,new);frozen()
    # Fresh receipt checks are separate from test runner's success flag.
    cases=['detach','killed','partial','completed','identity']
    for case in cases:
        path=raw/case
        if case=='detach':
            if strict_json((path/'guardian_terminal.json').read_bytes())['status']!='CONTROLLER_COMPLETE':raise ValueError('detached survival')
        else:
            v=strict_json((path/'interruption.json').read_bytes())
            if v['denominator']!=4 or any(row['upper_bound'] is not None for row in v['rows']):raise ValueError('postmortem promotion')
    save(ROOT/'docs/soplex_detached_v2_readiness.json',dict(status='PASS',issues=[],freeze_sha256=sha(FREEZE),
        tested_cases=cases,real_queries=0,native_queries=0,
        scope='Lifecycle controls and frozen dependency identities; not real LP outcomes.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('controls',type=Path);a=p.parse_args();prepare(a.controls);print('FROZEN AND REVIEWED')
