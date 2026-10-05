"""Recheck the terminal full source qualification without rerunning experiments."""
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent


def main():
    started=time.monotonic();run=EXP/'results/c97_once_power_20260913_v1'
    before=json.loads((run/'preregistered.json').read_text());after=json.loads((run/'exit.json').read_text())
    if (not after['all_declared_stages_passed'] or after['tests_count']!=2476
            or any(after[k]!=0 for k in ('tests_exit','preflight_exit','actual_worker_exit','restore_exit'))
            or after['source_drift'] or after['provenance_drift'] or before['provenance']!=_provenance(ROOT)):
        raise ValueError('complete terminal qualification or provenance differs')
    for name,sha in before['source_sha256'].items():
        if _sha256(EXP/name)!=sha:raise ValueError('frozen source changed: '+name)
    for name,sha in after['artifacts'].items():
        if _sha256(run/name)!=sha:raise ValueError('frozen artifact changed: '+name)
    records={n:json.loads((run/n).read_text()) for n in ('preflight/result.json','actual/result.json','restore_result.json')}
    if any(not r['completed'] for r in records.values()):raise ValueError('terminal stage lacks complete result')
    actual=records['actual/result.json'];restored=records['restore_result.json']
    if (actual['data']['physical_identity']!=restored['data']['physical_identity']
            or actual['data']['archive_sha256']!=_sha256(run/'actual/physical_hz.pickle')
            or restored['data']['archive_sha256']!=actual['data']['archive_sha256']):
        raise ValueError('complete actual and fresh restored source binding differs')
    scripts={'run_c97_once_power_supervisor_v1.py','c97_source_budget_worker_v1.py',
        'c97_actual_generator_worker_v1.py','c97_restore_worker_v1.py',
        'run_c96_word_row_supervisor_v1.py','c96_word_row_worker_v1.py'}
    live=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:argv=(path/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        if len(argv)>1 and Path(argv[1].decode(errors='replace')).name in scripts:live.append(int(path.name))
    if live:raise ValueError('registered qualification job remains live')
    result=dict(completed=True,tests=2476,unique_source_files=len(before['source_sha256']),
        artifacts=len(after['artifacts']),all_sources_and_artifacts_match=True,live_registered_processes=live,
        stage_result_sha256={n:_sha256(run/n) for n in records},exit_sha256=_sha256(run/'exit.json'),
        archive_sha256=actual['data']['archive_sha256'],physical_identity=actual['data']['physical_identity'],
        provenance=_provenance(ROOT),formal_gain=0,formal_baseline=1870,
        separate_E0=dict(cifar100=25,tinyimagenet=36),full_goal_status='active',wall_s=time.monotonic()-started)
    _atomic_exclusive_json(EXP/'C97_TERMINAL_INTEGRITY_20260913.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
