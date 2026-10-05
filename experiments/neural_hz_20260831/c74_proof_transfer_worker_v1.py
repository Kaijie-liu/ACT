"""Prepare text-only proof anchors and the complete new coupled bound."""
from pathlib import Path
import hashlib
import json
import pickle
import resource
import sys
import time
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.c41_owned_pickle_decode_v1 import load
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c65_physical_archive_v1 import check_restored
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout
from experiments.neural_hz_20260831.c70_native_proof_v1 import digest,entries,verify_inverse
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c74_live_runtime_v1 import extra_bound
from experiments.neural_hz_20260831.c74_native_binding_v1 import admit_source,admit_native
from experiments.neural_hz_20260831.c73_outer_query_v1 import GuardedLocalSpliceJournal
from experiments.neural_hz_20260831.c26_tagged_transplant_v1 import Plan
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect
from types import SimpleNamespace
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;RUN=EXP/'results/c74_native_binding_20260913_v1'
OLD=EXP/'results/c73_outer_query_20260913_v1'
ARCHIVE_SHA='dd96c133bd5e99835f4dcf5172c827b7f2b5fcf0762ec896754ca73f91d9c9a7'


def prepare(pool):
    done=json.loads((OLD/'component/result.json').read_text())
    inverse=json.loads((OLD/'restore/result.json').read_text())
    terminal=json.loads((OLD/'exit.json').read_text())
    if not (done['completed'] and inverse['completed'] and terminal['all_declared_stages_passed']
            and inverse['data']['inverse']['all_equations_exact']):
        raise ValueError('complete C73 source/native/inverse qualification required')
    with (OLD/'component/native.pickle').open('rb') as stream:
        saved,decoder=load(stream,expected_sha256=ARCHIVE_SHA,pool=pool,enabled=True)
    layout=numeric_layout(saved,pool)
    if layout.resident_entries>64_000_000:raise MemoryError('complete transfer input entries')
    pool.charge('c74_complete_independent_archive_proof',int(layout.resident_entries)+1024)
    source=check_restored(saved['source']);proof=json.loads(saved['proof_bytes'])
    if (saved['schema']!='c70_offline_native_component_archive_v1' or saved['full_LIVE_admission']
            or hashlib.sha256(saved['proof_bytes']).hexdigest()!=done['data']['proof_sha256']
            or source_digest(saved['hz'])!=proof['new_HZ_sha256']
            or digest(saved['packet'])!=proof['packet_identity']
            or digest(saved['journal'])!=proof['journal_identity']
            or saved['journal']['eq_roots'] is not saved['source']['fields']['eq_roots']
            or saved['journal']['eq_scales'] is not saved['source']['fields']['eq_scales']):
        raise ValueError('complete original component proof binding differs')
    p=saved['packet'];fields=saved['source']['fields'];h=fields['hz']
    raw_source=saved['source']['proof_bytes']
    transfer=dict(schema='c74_complete_source_native_transfer_v1',
        complete_C73_archive_authenticated=True,complete_independent_inverse_restored=True,
        archive_sha256=ARCHIVE_SHA,component_proof_sha256=saved['proof_sha256'],
        component_result_sha256=_sha256(OLD/'component/result.json'),
        restore_result_sha256=_sha256(OLD/'restore/result.json'),
        source_identity=source['physical_identity'],source_proof_sha256=saved['source']['proof_sha256'],
        packet_identity=proof['packet_identity'],new_HZ_sha256=proof['new_HZ_sha256'],
        journal_identity=proof['journal_identity'],event_sha256=hashlib.sha256(saved['events'].tobytes()).hexdigest(),
        packet_header={k:p[k] for k in ('schema','offline_only','fresh_native_execution','provenance')},
        complete_component_proof=proof,inverse=inverse['data']['inverse'],
        full_LIVE_admission=False,formal_gain=0)
    source_bound=json.loads((EXP/'results/c69_prepared_finite_20260913_v1/preflight/result.json').read_text())
    if not source_bound['completed']:raise ValueError('complete original source bound missing')
    b=source_bound['data']['bound']
    if (b['expected_identity_sha256']!=source['proof']['complete_source_boundary_sha256']
            or fields['report']['total_work_upper']>b['whole_work_upper']
            or fields['report']['largest_branch_work_upper']>b['branch_work_upper']):
        raise ValueError('complete source bound differs from independently proved construction')
    extra=extra_bound(h.n_out,h.Gc.nnz+h.Gb.nnz,len(p['eq_rhs']),
        p['eq_c'].nnz+p['le_c'].nnz,len(p['eq_rhs'])+len(p['le_rhs']))
    bound=dict(source_whole=b['whole_work_upper'],source_branch=b['branch_work_upper'],
        complete_C73_component_increment=done['coupled_increment'],runtime_extra_parts=extra,
        max_materialized_views=2,max_incremental_part_names=64,
        whole=b['whole_work_upper']+done['coupled_increment']+sum(extra.values()),
        branch=b['branch_work_upper']+done['coupled_increment']+sum(extra.values()),
        separate_paid_native_payload=done['paid_native_payload_work'],
        hash_authentication_separate_and_reported=True,full_CPU_work_256M_claim=False,
        actual_runtime_required=True,formal_gain=0)
    bound['fits']=bound['whole']<=256_000_000 and bound['branch']<=200_000_000
    if not bound['fits']:raise MemoryError('complete new native coupled bound exceeds unchanged caps: '+str(bound))
    for name,raw in [('source_proof.json',raw_source),
            ('transfer_proof.json',json.dumps(transfer,sort_keys=True,allow_nan=False,separators=(',',':')).encode())]:
        with (RUN/name).open('xb') as stream:stream.write(raw)
    _atomic_exclusive_json(RUN/'native_bound.json',bound)
    _atomic_exclusive_json(RUN/'live_inputs.json',dict(source_proof_sha256=_sha256(RUN/'source_proof.json'),
        transfer_proof_sha256=_sha256(RUN/'transfer_proof.json'),native_bound_sha256=_sha256(RUN/'native_bound.json')))
    return dict(complete_input=asdict(layout),decoder=decoder,bound=bound,
        complete_archived_numeric_inputs_retained=True,text_only_live_inputs=True,formal_gain=0)


def restore(pool):
    record=json.loads((RUN/'qualification.json').read_text())
    if not record['passed']:raise ValueError('actual native LIVE gate missing')
    with (RUN/'relu78.pickle').open('rb') as stream:
        saved,decoder=load(stream,expected_sha256=record['checkpoint_sha256'],pool=pool,enabled=True)
    # The checkpoint contains a subset of the proved complete LIVE union, plus
    # any new readonly backing explicitly copied by the authenticated decoder.
    envelope=int(record['whole_live_state']['resident_entries'])+decoder['copied_numeric_entries']
    pool.charge('c74_complete_restored_LIVE_traversal',envelope+1024)
    roots=collect(SimpleNamespace(),{'complete_native_checkpoint':saved});layout=roots.measure()
    if layout.resident_entries>min(64_000_000,envelope):raise MemoryError('complete restored LIVE entries')
    inp=json.loads((RUN/'live_inputs.json').read_text());f=saved['native_fields']
    source,_=admit_source(f['source_fields'],f['source_proof_bytes'],expected_sha256=inp['source_proof_sha256'],enabled=True)
    native,_=admit_native(enabled=True,source=source,hz=saved['post_relu_hz'],
        lineage=GuardedLocalSpliceJournal(**f['journal']),events=f['events'],
        actual_phase_image=f['actual_phase_image'],transfer_proof_bytes=f['transfer_proof_bytes'],
        expected_transfer_sha256=inp['transfer_proof_sha256'],construction_report=f['construction_report'])
    proof=json.loads(f['transfer_proof_bytes'])['complete_component_proof']
    plans=[Plan(**{**p,'tail':tuple(p['tail'])}) for p in proof['plans']]
    inverse=verify_inverse(source,native.hz,native.lineage,plans,pool=pool)
    if saved['hz_cache'][78] is not native.hz:raise ValueError('restored actual native cache sharing differs')
    return dict(complete_input=asdict(layout),decoder=decoder,inverse=inverse,
        complete_restored_native_bound=True,concrete_witness=False,formal_gain=0)


def main():
    stage=sys.argv[1]
    if stage not in ('prepare','restore'):raise ValueError('explicit qualification stage required')
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    freeze=json.loads((RUN/'preregistered.json').read_text())
    if any(_sha256(EXP/n)!=sha for n,sha in freeze['source_sha256'].items()):raise ValueError('frozen source drift')
    pool=WorkPool(256_000_000);started=time.monotonic();record=dict(completed=False,formal_gain=0)
    try:
        data,stats=measured(lambda:(prepare if stage=='prepare' else restore)(pool),
                            observe=lambda s:record.update(measurement=s))
        record.update(completed=True,data=data)
    except Exception as exc:record['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        record.update(wall_s=time.monotonic()-started,diagnostic_work=pool.used,diagnostic_parts=pool.parts)
        _atomic_exclusive_json(RUN/(stage+'_result.json'),record);print(json.dumps(record),flush=True)
    if not record['completed']:raise SystemExit(1)


if __name__=='__main__':main()
