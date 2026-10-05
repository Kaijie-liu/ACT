"""Terminal source/artifact/provenance audit, including rejected and aborted arms."""
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent


def main():
    frozen={};artifacts={};exits={};reference=None;states={}
    for version,count in [('v1',2544),('v2',2548),('v3',2550),('v4',2550)]:
        directory=EXP/('results/c100_fresh_circuit_terminal_20260913_'+version)
        f=json.loads((directory/'preregistered.json').read_text())
        e=json.loads((directory/'exit.json').read_text())
        if e['tests_count']!=count or e['tests_exit'] or e['source_drift'] or e['provenance_drift']:
            raise ValueError('complete qualification/source outcome differs: '+version)
        if version in ('v1','v3'):
            p=json.loads((directory/'prepare_result.json').read_text())
            if p['completed'] or p['failure']['type']!='MemoryError' or e.get('fresh_terminal_started'):
                raise ValueError('recorded failed preparation differs')
            states[version]='PREPARATION_RSS_GATE_REJECTED'
        elif version=='v2':
            if e.get('fresh_terminal_started') or (directory/'prepare_result.json').exists():
                raise ValueError('operator-aborted v2 unexpectedly has a completed target')
            states[version]='OPERATOR_SIGINT_SUPERVISOR_EXIT130_PREPARATION_TERMINATED'
        else:
            p=json.loads((directory/'prepare_result.json').read_text())
            gate=json.loads((directory/'terminal_gate.json').read_text())
            events=[json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
            starts=[v for v in events if v['event']=='c100_ordinary_milp_start']
            if (not p['completed'] or not p['measurement']['measured_transient_gate']
                or not gate['terminal_gates_passed'] or not gate['physical_decrease']
                or not gate['final_native_ingestion']['passed'] or e.get('timeout_s')!=240
                or len(starts)!=1 or not starts[0]['first_base_call']
                or any(v['event']=='ordinary_milp_return' for v in events)
                or (directory/'native_witness_1.npz').exists()):
                raise ValueError('complete actual terminal/timeout outcome differs')
            states[version]='FRESH_NATIVE_LIVE_PASSED_BASE_MILP_NO_RETURN_WORKER_TIMEOUT240'
        if e['all_declared_stages_passed']:raise ValueError('failed whole target incorrectly admitted')
        for name,sha in f['source_sha256'].items():
            if name in frozen and frozen[name]!=sha:raise ValueError('source manifest conflict')
            frozen[name]=sha
        for name,sha in e['artifacts'].items():
            path=directory/name
            if _sha256(path)!=sha:raise ValueError('terminal output changed: '+str(path))
            artifacts[str(path.relative_to(EXP))]=sha
        exits[version]=_sha256(directory/'exit.json')
        if reference is not None and reference!=f['provenance']:raise ValueError('provenance conflict')
        reference=f['provenance']
    if any(_sha256(EXP/n)!=sha for n,sha in frozen.items()):raise ValueError('frozen source changed')
    if _provenance(ROOT)!=reference:raise ValueError('production/branch changed')
    names={f'{stem}_v{v}.py' for v in range(1,5) for stem in
        ('run_c100_terminal_supervisor','c100_prepare_worker','c100_changed_terminal_worker')}
    ps=subprocess.run(['ps','-eo','pid=,comm=,args='],capture_output=True,text=True,check=True).stdout
    active=[]
    for line in ps.splitlines():
        parts=line.split()
        if len(parts)>2 and parts[1].startswith('python') and any(Path(a).name in names for a in parts[2:]):
            active.append(line)
    if active:raise ValueError('registered C100 process still live: '+repr(active))
    record=dict(schema='c100_terminal_integrity_v1',all_registered_processes_terminal=True,
        outcomes=states,frozen_unique_sources=len(frozen),checked_artifacts=len(artifacts),
        complete_source_and_artifact_hashes_match=True,exit_sha256=exits,provenance=reference,
        frozen_source_sha256=frozen,artifact_sha256=artifacts,formal_gain=0,
        whole_target_passed=False,full_goal_complete=False,formal1870_all13_and_E0_unchanged=True)
    _atomic_exclusive_json(EXP/'C100_TERMINAL_INTEGRITY_20260913.json',record)
    print(json.dumps({k:v for k,v in record.items() if k not in ('frozen_source_sha256','artifact_sha256')}))


if __name__=='__main__':main()
