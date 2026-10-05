"""Fresh terminal source/artifact/provenance audit without rerunning experiments."""
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent


def main():
    started=time.monotonic();run=EXP/'results/c96_word_row_20260913_v1'
    before=json.loads((run/'preregistered.json').read_text());after=json.loads((run/'exit.json').read_text())
    if (not after['all_declared_stages_passed'] or after['tests_exit']!=0 or after['tests_count']!=2293
        or after['worker_exit']!=0 or after['source_drift'] or after['provenance_drift']
        or before['provenance']!=_provenance(ROOT)):raise ValueError('complete qualification/provenance differs')
    for name,sha in before['source_sha256'].items():
        if _sha256(EXP/name)!=sha:raise ValueError('frozen source changed: '+name)
    for name,sha in after['artifacts'].items():
        if _sha256(run/name)!=sha:raise ValueError('frozen artifact changed: '+name)
    live=[];scripts={'run_c95_word_filter_supervisor_v1.py','c95_word_filter_worker_v1.py','run_c94_raw_mask_plan_supervisor_v1.py','c94_raw_mask_plan_worker_v1.py','run_c93_packed_mask_supervisor_v1.py','c93_packed_mask_worker_v1.py','run_c96_word_row_supervisor_v1.py','c96_word_row_worker_v1.py',
        'run_c91_physical_circuit_supervisor_v1.py','c91_physical_circuit_worker_v1.py','c91_physical_restore_worker_v1.py'}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:argv=(p/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        if len(argv)>1 and Path(argv[1].decode(errors='replace')).name in scripts:live.append(int(p.name))
    if live:raise ValueError('registered experiment still live')
    result=dict(completed=True,tests=2293,unique_source_files=len(before['source_sha256']),
        artifacts=len(after['artifacts']),all_sources_and_artifacts_match=True,live_registered_processes=live,
        result_sha256=_sha256(run/'result.json'),exit_sha256=_sha256(run/'exit.json'),provenance=_provenance(ROOT),
        formal_gain=0,formal_baseline=1870,separate_E0=dict(cifar100=25,tinyimagenet=36),
        full_goal_status='active',wall_s=time.monotonic()-started)
    _atomic_exclusive_json(EXP/'C96_TERMINAL_INTEGRITY_20260913.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
