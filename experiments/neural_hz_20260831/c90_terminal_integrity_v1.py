"""Exclusive final frozen-source/result integrity record for C89 and C90."""
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent


def main():
    started=time.monotonic();records=[];sources={}
    for name,count in [('c89_quotient_budget_20260913_v1',2097),('c90_actual_circuit_proof_20260913_v1',2107)]:
        run=EXP/'results'/name;before=json.loads((run/'preregistered.json').read_text());after=json.loads((run/'exit.json').read_text())
        if (not after['all_declared_stages_passed'] or after['tests_exit']!=0 or after['tests_count']!=count
            or after['source_drift'] or after['provenance_drift'] or before['provenance']!=_provenance(ROOT)):
            raise ValueError('complete terminal qualification/provenance differs')
        for relative,sha in before['source_sha256'].items():
            if relative in sources and sources[relative]!=sha:raise ValueError('source manifests conflict')
            sources[relative]=sha
        for relative,sha in after['artifacts'].items():
            if _sha256(run/relative)!=sha:raise ValueError('result artifact changed: '+relative)
        records.append(dict(run=name,tests=count,source_files=len(before['source_sha256']),
            artifacts=len(after['artifacts']),result_sha256=_sha256(run/'result.json'),exit_sha256=_sha256(run/'exit.json')))
    for name,sha in sources.items():
        if _sha256(EXP/name)!=sha:raise ValueError('frozen source changed: '+name)
    script_names={'run_c89_quotient_budget_supervisor_v1.py','c89_quotient_budget_worker_v1.py',
        'run_c90_actual_circuit_proof_supervisor_v1.py','c90_actual_circuit_proof_worker_v1.py'}
    live=[]
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():continue
        try:argv=(directory/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        if len(argv)>1 and Path(argv[1].decode(errors='replace')).name in script_names:live.append(int(directory.name))
    if live:raise ValueError('registered target job still live')
    result=dict(completed=True,terminal_runs=records,unique_source_files=len(sources),
        all_sources_and_artifacts_match=True,live_registered_processes=live,provenance=_provenance(ROOT),
        historical_writes=False,formal_gain=0,formal_baseline=1870,separate_E0=dict(cifar100=25,tinyimagenet=36),
        wall_s=time.monotonic()-started)
    _atomic_exclusive_json(EXP/'C90_TERMINAL_INTEGRITY_20260913.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
