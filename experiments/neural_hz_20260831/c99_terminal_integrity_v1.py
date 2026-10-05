"""One terminal read-only provenance/source/artifact check; exclusive record."""
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent


def main():
    frozen={};artifacts={};exits={};reference=None
    for v,count in [('v1',2524),('v2',2527)]:
        directory=EXP/('results/c99_circuit_native_20260913_'+v)
        freeze=json.loads((directory/'preregistered.json').read_text())
        exit_record=json.loads((directory/'exit.json').read_text())
        if (not exit_record['all_declared_stages_passed'] or exit_record['tests_count']!=count
            or exit_record['tests_exit'] or exit_record['preflight_exit'] or exit_record['component_exit'] or exit_record['restore_exit']
            or exit_record['source_drift'] or exit_record['provenance_drift']):
            raise ValueError('full registered stages not terminal-successful: '+v)
        for name,sha in freeze['source_sha256'].items():
            if name in frozen and frozen[name]!=sha:raise ValueError('frozen source conflict')
            frozen[name]=sha
        for name,sha in exit_record['artifacts'].items():
            path=directory/name
            if _sha256(path)!=sha:raise ValueError('terminal artifact changed: '+str(path))
            artifacts[str(path.relative_to(EXP))]=sha
        exits[v]=_sha256(directory/'exit.json')
        if reference is not None and reference!=freeze['provenance']:raise ValueError('candidate provenance differs')
        reference=freeze['provenance']
    if any(_sha256(EXP/n)!=sha for n,sha in frozen.items()):raise ValueError('complete frozen source changed')
    if _provenance(ROOT)!=reference:raise ValueError('current production/branch provenance differs')
    process_names={'c99_native_worker_v1.py','c99_native_worker_v2.py',
        'run_c99_native_supervisor_v1.py','run_c99_native_supervisor_v2.py'}
    ps=subprocess.run(['ps','-eo','pid=,comm=,args='],check=True,capture_output=True,text=True).stdout
    active=[]
    for line in ps.splitlines():
        parts=line.split()
        if len(parts)>2 and parts[1].startswith('python'):
            if any(Path(arg).name in process_names or any(arg.endswith('.'+n[:-3]) for n in process_names)
                   for arg in parts[2:]):active.append(line)
    if active:raise ValueError('registered worker still live: '+repr(active))
    output=dict(schema='c99_terminal_integrity_v1',all_registered_processes_terminal=True,
        frozen_unique_sources=len(frozen),checked_artifacts=len(artifacts),
        complete_source_and_artifact_hashes_match=True,provenance=reference,exit_sha256=exits,
        frozen_source_sha256=frozen,artifact_sha256=artifacts,formal_gain=0,
        full_goal_complete=False,formal1870_and_all13_retention_unchanged=True)
    _atomic_exclusive_json(EXP/'C99_TERMINAL_INTEGRITY_20260913.json',output)
    print(json.dumps({k:v for k,v in output.items() if k not in ('frozen_source_sha256','artifact_sha256')}))


if __name__=='__main__':main()
