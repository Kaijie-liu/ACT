"""Offline compact proof transfer from independently completed C31 and C30.

Only the textual proof is loaded by the fresh native worker. No completed HZ
or full lineage array is a runtime generator input. This authenticates existing
complete proofs; it never promotes a score or substitutes a live source.
"""

import hashlib
import json
from pathlib import Path
import pickle
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from experiments.neural_hz_20260831.c24_closed_state_v1 import restore
from experiments.neural_hz_20260831.c31_prepared_report_audit_v1 import audit
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256

EXP=Path(__file__).resolve().parent
ANCHORS={
    'results/c31_prepared_generator_20260911_v1/closed_hz.pickle':'535b579e853f1ac5092e962e1c307644e89739da4cd63f3732593896649128f5',
    'results/c31_prepared_generator_20260911_v1/closed_proof.json':'cad0401a22b403114db73d32cae4ca949d7d8314106bfc39eaccdc71345f93d5',
    'results/c31_prepared_generator_20260911_v1/result.json':'c0d9a310c5d7a814f903ed48686d8db43bd7724a9c4e270d084a4b60a6c7a20a',
    'results/c31_prepared_generator_20260911_v1/exit.json':'3f76bfbbb78a488be55a36d79ff42b4b4d6ea16ae8d12a3a14c2acff36ffa8f1',
    'results/c30_first_write_20260911_v1/spliced_hz.pickle':'ed1099a692ef4dad2f0f6b1a04591dbc258cc2edc8083919ce86b465f208e145',
    'results/c30_first_write_20260911_v1/result.json':'7b0af27cfac7ae0826a740229d746287d626b97886d522a2bf47d4f289c5f5da',
    'results/c30_first_write_20260911_v1/exit.json':'33a4bd117a2279a7beea59036cd26661d330cbb1f006242db332c1a43ad6b560',
    'results/c25_live_relu_20260911_v1/relu78.pickle':'685e80ba9754fa821d3fa0486309a1572dcffecb6f40c83a67309f1bd3a5b9ba',
}


def build(*,enabled=False):
    if not enabled:return None
    if any(_sha256(EXP/n)!=sha for n,sha in ANCHORS.items()):
        raise ValueError('independent completed transfer inputs changed')
    reports=[]
    for directory in ('c31_prepared_generator_20260911_v1','c30_first_write_20260911_v1'):
        path=EXP/'results'/directory
        r=json.loads((path/'result.json').read_text());e=json.loads((path/'exit.json').read_text())
        if (not r['completed'] or e.get('worker_exit_code')!=0 or e.get('tests_exit_code')!=0
                or e.get('timeout_s') or e['source_drift'] or e['provenance_drift']):
            raise ValueError('incomplete/failed independent transfer source')
        if any(_sha256(EXP/n)!=sha for n,sha in r['source_sha256'].items()):
            raise ValueError('complete original transfer source/dependency drift')
        reports.append(r)
    with (EXP/'results/c31_prepared_generator_20260911_v1/closed_hz.pickle').open('rb') as f:new_saved=pickle.load(f)
    with (EXP/'results/c25_live_relu_20260911_v1/relu78.pickle').open('rb') as f:old_saved=pickle.load(f)
    with (EXP/'results/c30_first_write_20260911_v1/spliced_hz.pickle').open('rb') as f:spliced=pickle.load(f)
    new_raw=new_saved['proof_bytes'];new_sha=hashlib.sha256(new_raw).hexdigest()
    old_raw=old_saved['closed_proof_bytes'];old_sha=hashlib.sha256(old_raw).hexdigest()
    if new_sha!=ANCHORS['results/c31_prepared_generator_20260911_v1/closed_proof.json']:
        raise ValueError('new proof not independently bound')
    new=restore(new_saved['fields'],new_raw,expected_proof_sha256=new_sha)
    old=restore(old_saved['closed_fields'],old_raw,
        expected_proof_sha256='99a52f7adafb61275690f6993940a1eacb90b71c68b0140da7398d75526c41e6')
    pool=WorkPool(256_000_000)
    report_proof=audit(new,old,pool=pool)
    spliced['lineage'].validate()
    report=reports[1];proof=spliced['complete_UID_box_reconstruction_transfer']
    if (new_saved['identity']!=reports[0]['new_complete_source_proof']
            or report_proof!=reports[0]['complete_report_proof']
            or spliced['schema']!='c30_provisional_actual_spliced_HZ_v1'
            or spliced['closed_identity']!=old.fingerprint()
            or spliced['source_post_HZ_sha256']!=source_digest(old_saved['post_relu_hz'])
            or source_digest(spliced['hz'])!=report['actual_new_HZ_sha256']
            or spliced['lineage'].fingerprint()!=report['lineage_fingerprint']
            or proof!=report['complete_proof_transfer']
            or not report['all_matrix_RHS_output_bits_equal']
            or not proof['complete_transferred_incidence_equal']
            or not proof['independent_exact_source_and_box_proof_transferred']):
        raise ValueError('full C31/C25/C30 mathematical source relation changed')
    events=old_saved['phase_ownership']['events']
    event_sha=hashlib.sha256(memoryview(events).cast('B')).hexdigest()
    if event_sha!=old_saved['phase_ownership']['event_sha256']:
        raise ValueError('independent actual phase event binding changed')
    result=dict(schema='c32_independent_C31_C30_transfer_v1',completed=True,
        full_C31_source_math_and_report_checked=True,full_C30_HZ_UID_box_reconstruction_checked=True,
        all_pre_HZ_map_owner_UID_bits_equal=True,new_source_proof_sha256=new_sha,
        new_closed_identity=new.fingerprint(),old_closed_identity=old.fingerprint(),
        pre_HZ_sha256=source_digest(new.hz),actual_spliced_HZ_sha256=source_digest(spliced['hz']),
        semantic_lineage_sha256=spliced['lineage'].fingerprint(),actual_phase_events_sha256=event_sha,
        complete_unit_pairs=len(spliced['lineage'].columns),complete_lineage_slots=len(spliced['lineage'].eq_roots),
        independent_complete_splice_proof=proof,complete_report_transfer=report_proof,
        input_sha256=ANCHORS,transfer_diagnostic_work=pool.used,
        root_domain_base_feasibility_proved=False,new_native_execution_proved=False,formal_gain=0)
    return json.dumps(result,sort_keys=True,allow_nan=False).encode()


def main():
    import os
    import resource
    import time
    from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json
    directory=Path(sys.argv[1]).resolve()
    if directory!=EXP/'results/c32_live_splice_20260911_v1':raise ValueError('unregistered offline transfer target')
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    freeze=json.loads((directory/'preregistered.json').read_text());started=time.monotonic()
    result=dict(completed=False,formal_gain=0,new_native_execution_proved=False)
    try:
        if any(_sha256(EXP/n)!=sha for n,sha in freeze['source_sha256'].items()):raise ValueError('transfer source freeze drift')
        raw=build(enabled=True)
        with (directory/'transfer_proof.json').open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        result.update(completed=True,transfer_proof_sha256=hashlib.sha256(raw).hexdigest(),
            transfer_proof_bytes=len(raw),loaded_oracles_are_OFFLINE_only=True)
    except Exception as exc:result['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        result.update(wall_s=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        _atomic_exclusive_json(directory/'transfer_result.json',result);print(json.dumps(result),flush=True)
    if not result['completed']:raise SystemExit(1)


if __name__=='__main__':main()
