"""C72 complete inverse-phase proof; unchanged C70 component/restore stages."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831 import c70_native_worker_v1 as core
from experiments.neural_hz_20260831.c72_inverse_phase_v1 import recover,native_digest
from experiments.neural_hz_20260831.c70_native_proof_v1 import digest,entries
from experiments.neural_hz_20260831.c24_closed_state_v1 import restore
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout
from experiments.neural_hz_20260831.c65_physical_archive_v1 import metadata
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256
EXP=Path(__file__).resolve().parent;RUN=EXP/'results/c72_inverse_phase_20260913_v2'
C31=EXP/'results/c31_prepared_generator_20260911_v1/closed_hz.pickle'
C30=EXP/'results/c30_first_write_20260911_v1/spliced_hz.pickle'
C31_SHA='535b579e853f1ac5092e962e1c307644e89739da4cd63f3732593896649128f5'
C30_SHA='ed1099a692ef4dad2f0f6b1a04591dbc258cc2edc8083919ce86b465f208e145'
C31_PROOF='cad0401a22b403114db73d32cae4ca949d7d8314106bfc39eaccdc71345f93d5'
TRANSFER=EXP/'results/c32_live_splice_20260911_v1/transfer_proof.json'
TRANSFER_SHA='bab5946186e350159087a9a5d512d8e01759591e0d4309d9d2998e2936b87e86'
POST_SHA='82df62f1233ca8f34b5163ee3fafa88ae0afc65a9f36fe5b1b7f3b022cca2367'


def recovered_phase(pool,result,emit):
    # Only an external full-byte provenance anchor, not a decoded C25 root or
    # an input to physical LIVE accounting. Complete C31/C30 inputs stay held.
    if _sha256(core.PHASE)!=core.PHASE_SHA or _sha256(TRANSFER)!=TRANSFER_SHA:
        raise ValueError('independent complete C25/C32 input anchors differ')
    transfer=json.loads(TRANSFER.read_bytes())
    if (not transfer['completed'] or not transfer['full_C31_source_math_and_report_checked']
            or not transfer['full_C30_HZ_UID_box_reconstruction_checked']
            or not transfer['all_pre_HZ_map_owner_UID_bits_equal']):
        raise ValueError('complete prior source/phase proof transfer required')
    source,source_decoder=core.authenticated(C31,C31_SHA,pool)
    saved,splice_decoder=core.authenticated(C30,C30_SHA,pool)
    portable=dict(saved);portable['lineage']=vars(saved['lineage'])
    inputs=dict(complete_C31=source,complete_C30=portable,complete_C32_text=transfer)
    layout=numeric_layout(inputs,pool)
    if layout.resident_entries>64_000_000:raise MemoryError('complete inverse-phase input entry cap')
    emit('complete_inverse_phase_inputs_loaded',dict(layout=asdict(layout)))
    source_layout=numeric_layout(source,pool)
    pool.charge('c72_complete_C31_restore_and_preservation',3*int(source_layout.resident_entries)+3072)
    c=restore(source['fields'],source['proof_bytes'],expected_proof_sha256=C31_PROOF)
    pool.charge('c72_complete_pre_and_C30_native_hashes',entries(c.hz)+2*entries(saved['hz'])+2048)
    pre_sha=source_digest(c.hz);new_sha=source_digest(saved['hz'])
    lineage=saved['lineage'];pool.charge('c72_complete_old_lineage_hash',
        2*sum(len(getattr(lineage,k)) for k in ('eq_roots','eq_scales','columns','retired','tails'))+256)
    line_sha=lineage.fingerprint()
    if (source['schema']!='c31_new_checked_prepared_Closed_v1'
            or saved['schema']!='c30_provisional_actual_spliced_HZ_v1'
            or transfer['new_closed_identity']!=c.seal or transfer['new_source_proof_sha256']!=C31_PROOF
            or saved['closed_identity']!=transfer['old_closed_identity']
            or pre_sha!=transfer['pre_HZ_sha256'] or new_sha!=transfer['actual_spliced_HZ_sha256']
            or line_sha!=transfer['semantic_lineage_sha256']
            or saved['source_post_HZ_sha256']!=POST_SHA
            or saved['complete_UID_box_reconstruction_transfer']!=transfer['independent_complete_splice_proof']
            or not saved['full_matrix_and_RHS_bit_identity']):
        raise ValueError('complete C31/C30/C25 actual source chain differs')
    packet,proof=recover(c,saved['hz'],lineage,provenance=dict(
        phase_archive_sha256=core.PHASE_SHA,source_proof_sha256=core.PHASE_PROOF,
        closed_identity=transfer['old_closed_identity'],phase_post_sha256=POST_SHA,
        phase_events_sha256=transfer['actual_phase_events_sha256'],
        recovered_from_complete_C31_sha256=C31_SHA,recovered_from_complete_C30_sha256=C30_SHA,
        independent_complete_transfer_sha256=TRANSFER_SHA),pool=pool,enabled=True)
    recovered=native_digest(c.hz,packet,pool=pool)
    if recovered!=POST_SHA:raise ValueError('ALL original native HZ bytes were not exactly recovered')
    proof['full_original_native_hash_still_required']=False
    proof['complete_original_native_HZ_sha256']=recovered
    emit('all_original_native_phase_bytes_recovered',proof)
    if c.fingerprint()!=c.seal or source_digest(saved['hz'])!=new_sha or lineage.fingerprint()!=line_sha:
        raise ValueError('complete original source/component mutated')
    total=numeric_layout(dict(inputs=inputs,packet=packet),pool)
    meta=metadata(dict(inputs=inputs,packet=packet),pool=pool)
    if total.resident_entries>64_000_000:raise MemoryError('complete inputs+phase packet entry gate')
    packet_layout=numeric_layout(packet,pool)
    pool.charge('c72_complete_recovered_packet_hash_and_archive',2*int(packet_layout.resident_entries)+1024)
    identity=digest(packet)
    with (RUN/'phase/packet.pickle').open('xb') as stream:pickle.dump(packet,stream,protocol=5)
    return dict(packet_sha256=_sha256(RUN/'phase/packet.pickle'),packet_identity=identity,
        packet_numeric_bytes=packet_layout.resident_bytes,packet_entries=packet_layout.resident_entries,
        complete_retained_inputs_plus_packet=asdict(total),metadata=meta,recovery_proof=proof,
        original_C25_file_authenticated_not_decoded=True,C70_full_C25_input_gate_claim=False,
        complete_C31_and_C30_inputs_retained_unchanged=True,
        source_decoder=source_decoder,splice_decoder=splice_decoder,
        new_EQ=len(packet['eq_rhs']),new_INEQ=len(packet['le_rhs']),
        fresh_native_execution=False,full_LIVE_admission=False)


if __name__=='__main__':
    core.RUN=RUN
    core.phase=recovered_phase
    core.main()
