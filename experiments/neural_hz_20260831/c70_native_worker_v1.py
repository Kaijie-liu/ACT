"""Exclusive, separately bounded C70 phase/component/restore workers."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import pickle
import resource
import sys
import time
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from experiments.neural_hz_20260831.c70_native_proof_v1 import extract, bind_phase, digest, entries, verify, verify_inverse
from experiments.neural_hz_20260831.c41_owned_pickle_decode_v1 import load
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c24_closed_state_v1 import restore
from experiments.neural_hz_20260831.c65_physical_archive_v1 import check_restored, fingerprint, metadata
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c10_fused_rows_v1 import WorkPool as CoupledPool
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import check_append, BranchPool
from experiments.neural_hz_20260831.c23_sparse_phase_overlay_v1 import Overlay, build as build_overlay
from experiments.neural_hz_20260831.c30_append_discovery_v1 import discover_append
from experiments.neural_hz_20260831.c30_first_write_v1 import splice_append
from experiments.neural_hz_20260831.c32_boundary_budget_v1 import WriterPool, remaining_pool, finish_local
from experiments.neural_hz_20260831.c68_local_splice_v1 import compile_journal, LocalSpliceJournal
from experiments.neural_hz_20260831.c62_local_equations_v1 import SCHEMA
from experiments.neural_hz_20260831.c26_tagged_transplant_v1 import Plan
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256, _atomic_exclusive_json

EXP = Path(__file__).resolve().parent
RUN = EXP / 'results/c70_native_proof_20260913_v1'
PHASE = EXP / 'results/c25_live_relu_20260911_v1/relu78.pickle'
PHASE_SHA = '685e80ba9754fa821d3fa0486309a1572dcffecb6f40c83a67309f1bd3a5b9ba'
PHASE_PROOF = '99a52f7adafb61275690f6993940a1eacb90b71c68b0140da7398d75526c41e6'
SOURCE = EXP / 'results/c69_prepared_finite_20260913_v1/actual/physical_hz.pickle'
SOURCE_SHA = 'acb6e5560503d42aa7b62fa4785f179f857471a9873b6d11189da88deea2db79'
SOURCE_PROOF = '6e0a339f08fd858ca45683a6e7cdf5a34b48e2012994889897d3290d33f0ea78'
SOURCE_ID = '9e8c3ddcbe09ebf2bdc97a87379beecebb659e3549ede4d10f802b53eba94349'


def authenticated(path, sha, pool):
    with path.open('rb') as stream:
        return load(stream, expected_sha256=sha, pool=pool, enabled=True)


def phase(pool, result, emit):
    saved, decoder = authenticated(PHASE, PHASE_SHA, pool)
    if saved['schema'] != 'c25_live_relu_checkpoint_v1' or not saved['whole_live_path_proved']:
        raise ValueError('completed actual native reference required')
    # Upper bound prepaid before traversal; all original checkpoint roots stay
    # strongly reachable through this worker, including network/caches/facts.
    pool.charge('c70_two_complete_C25_checkpoint_fingerprints', 128_000_000)
    before = collect(SimpleNamespace(), {'complete_checkpoint': saved})
    layout = before.measure()
    if layout.resident_entries > 64_000_000: raise MemoryError('complete C25 entry ceiling')
    emit('complete_original_phase_checkpoint', asdict(layout))
    source_layout = numeric_layout(saved['closed_fields'], pool)
    pool.charge('c70_original_Closed_full_restore_proof', int(source_layout.resident_entries) + 1024)
    closed = restore(saved['closed_fields'], saved['closed_proof_bytes'], expected_proof_sha256=PHASE_PROOF)
    pre, post, owned = closed.hz, saved['post_relu_hz'], saved['phase_ownership']
    pool.charge('c70_actual_native_post_full_hash', entries(post) + 1024)
    post_hash = source_digest(post)
    pool.charge('c70_original_phase_owner_validation', 8 * (len(closed.owners) + len(owned['events'])) + 256)
    overlay = Overlay(owned['base'], owned['events'], owned['old_uid_ceiling'])
    overlay.validate()
    if (owned['post_hz'] is not post or owned['base'] is not closed.owners
            or owned['post_sha256'] != post_hash
            or owned['event_sha256'] != hashlib.sha256(owned['events'].tobytes()).hexdigest()
            or overlay.old_uid_ceiling != closed.report['radix_uid_base'] + 16384):
        raise ValueError('complete actual phase/owner/source binding differs')
    nnz = sum(int(getattr(pre, k).nnz) for k in ('Ac', 'Ab', 'Auc', 'Aub'))
    pool.charge('c70_full_actual_phase_prefix_RHS_frame', 4 * nnz + 16 * (pre.n_eq + pre.n_ineq) + 512)
    check_append(pre, post)
    packet = extract(pre, post, old_n_cont=closed.old_n_cont, old_n_eq=closed.old_n_eq,
        logical_n_cont=closed.logical_n_cont, first_uid=overlay.old_uid_ceiling,
        provenance=dict(phase_archive_sha256=PHASE_SHA, source_proof_sha256=PHASE_PROOF,
            closed_identity=closed.seal, phase_post_sha256=post_hash,
            phase_events_sha256=owned['event_sha256']), pool=pool, enabled=True)
    packet_layout = numeric_layout(packet, pool)
    pool.charge('c70_complete_packet_hash_and_archive', 2 * int(packet_layout.resident_entries) + 1024)
    packet_hash = digest(packet)
    after = collect(SimpleNamespace(), {'complete_checkpoint': saved})
    if after.fingerprint != before.fingerprint: raise ValueError('complete original C25 checkpoint mutated')
    with (RUN / 'phase/packet.pickle').open('xb') as stream: pickle.dump(packet, stream, protocol=5)
    return dict(packet_sha256=_sha256(RUN / 'phase/packet.pickle'), packet_identity=packet_hash,
        packet_numeric_bytes=packet_layout.resident_bytes, packet_entries=packet_layout.resident_entries,
        complete_original_checkpoint=asdict(layout), complete_checkpoint_unchanged=True,
        decoder=decoder, old_n_cont=pre.n_cont, old_n_bin=pre.n_bin,
        new_n_cont=post.n_cont, new_n_bin=post.n_bin,
        new_EQ=len(packet['eq_rhs']), new_INEQ=len(packet['le_rhs']),
        fresh_native_execution=False, full_LIVE_admission=False)


def packet_input(pool):
    binding = json.loads((RUN / 'component/input_binding.json').read_text())
    if _sha256(RUN / 'phase/result.json') != binding['phase_result_sha256']:
        raise ValueError('completed phase result changed')
    record = json.loads((RUN / 'phase/result.json').read_text())
    if not record['completed']: raise ValueError('phase proof not complete')
    packet, _ = authenticated(RUN / 'phase/packet.pickle', record['data']['packet_sha256'], pool)
    layout = numeric_layout(packet, pool)
    pool.charge('c70_bind_entire_actual_packet', int(layout.resident_entries) + 1024)
    if (digest(packet) != record['data']['packet_identity']
            or packet['provenance']['phase_archive_sha256'] != PHASE_SHA
            or packet['provenance']['source_proof_sha256'] != PHASE_PROOF):
        raise ValueError('independently authenticated actual phase packet differs')
    return packet


def component(pool, result, emit):
    saved, decoder = authenticated(SOURCE, SOURCE_SHA, pool)
    layout = numeric_layout(saved, pool)
    pool.charge('c70_complete_C69_source_proof', int(layout.resident_entries) + 1024)
    proof = check_restored(saved)
    if saved['proof_sha256'] != SOURCE_PROOF or proof['physical_identity'] != SOURCE_ID:
        raise ValueError('complete independently expected C69 source proof differs')
    c = SimpleNamespace(**saved['fields'])  # Offline field interface, NEVER a Closed receipt.
    packet = packet_input(pool)
    view = bind_phase(c, packet, pool=pool, enabled=True)
    original_packet = digest(packet)
    g = c.report
    coupled = CoupledPool(g['total_work_upper'], g['largest_branch_work_upper'])
    result['coupled_source_base'] = dict(whole=coupled.whole_base, branch=coupled.branch_base)
    writer_pool = WriterPool(coupled, payload_cap=256_000_000)
    try:
        local = remaining_pool(coupled)
        first = packet['first_uid']
        overlay, event_report = build_overlay(c.owners,
            [(view.eq_c, first), (view.le_c, first + len(view.eq_rhs))],
            old_n_cont=c.old_n_cont, old_uid_ceiling=first, pool=local, enabled=True)
        finish_local(coupled, local, 'complete_actual_append_overlay')
        local = remaining_pool(coupled)
        plans, discovery = discover_append(c, view, overlay, pool=local, enabled=True)
        finish_local(coupled, local, 'all_actual_consumer_discovery')
        result.update(discovery=discovery, event_report=event_report)
        emit('complete_actual_population', discovery)
        if not plans: raise ValueError('complete actual population has no strict splice')
        new, writer = splice_append(view, plans, pool=writer_pool, enabled=True)
        local = remaining_pool(coupled)
        journal = compile_journal(c.eq_roots, c.eq_scales, plans, old_n_cont=c.old_n_cont,
            old_n_eq=c.old_n_eq, source_n_cont=c.hz.n_cont, source_schema=SCHEMA,
            pool=local, enabled=True)
        finish_local(coupled, local, 'complete_actual_local_splice_journal')
        result['writer'] = writer
        emit('complete_component_constructed', dict(plans=len(plans),
            whole=coupled.whole_base+coupled.used, branch=coupled.branch_base+coupled.used,
            native_payload_work=writer_pool.native.used))
        branch = BranchPool(pool)
        semantic = verify(c, view, overlay, plans, new, journal, pool=branch)
        result['semantic_proof'] = semantic
        result['semantic_branch_work'] = branch.used
        emit('complete_row_UID_owner_proof', semantic)
        pool.charge('c70_full_source_preservation_fingerprint', int(layout.resident_entries) + 1024)
        if fingerprint(saved['fields']) != SOURCE_ID or digest(packet) != original_packet:
            raise ValueError('original C69 source or actual packet mutated')
        pool.charge('c70_complete_new_HZ_fingerprint', entries(new) + 1024)
        hz_sha = source_digest(new)
        record = dict(schema='c70_offline_native_component_proof_v1', source_sha256=SOURCE_SHA,
            source_proof_sha256=SOURCE_PROOF, packet_identity=original_packet,
            new_HZ_sha256=hz_sha, journal_identity=digest(vars(journal)),
            plans=[asdict(p) for p in plans], semantic=semantic, full_LIVE_admission=False, formal_gain=0)
        raw = json.dumps(record, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()
        artifact = dict(schema='c70_offline_native_component_archive_v1', source=saved, packet=packet,
            hz=new, journal=vars(journal), events=overlay.events, proof_bytes=raw,
            proof_sha256=hashlib.sha256(raw).hexdigest(), full_LIVE_admission=False, formal_gain=0)
        complete = numeric_layout(artifact, pool); meta = metadata(artifact, pool=pool)
        if complete.resident_entries > 64_000_000: raise MemoryError('complete portable component entry cap')
        pool.charge('c70_complete_component_archive', int(complete.resident_entries) + 1024)
        with (RUN / 'component/native.pickle').open('xb') as stream: pickle.dump(artifact, stream, protocol=5)
        return dict(archive_sha256=_sha256(RUN/'component/native.pickle'), new_HZ_sha256=hz_sha,
            proof_sha256=artifact['proof_sha256'], portable_component=asdict(complete), metadata=meta,
            complete_original_inputs_preserved=True, decoder=decoder,
            original_generator_or_fresh_native_executed=False, full_LIVE_admission=False,
            missing_LIVE_runtime_view_binding_publication_cost=True)
    finally:
        result.update(coupled_increment=coupled.used, coupled_parts=coupled.parts,
            coupled_whole=coupled.whole_base+coupled.used, coupled_branch=coupled.branch_base+coupled.used,
            paid_native_payload_work=writer_pool.native.used, native_payload_parts=writer_pool.native.parts)


def fresh_restore(pool, result, emit):
    binding = json.loads((RUN / 'restore/input_binding.json').read_text())
    if _sha256(RUN / 'component/result.json') != binding['component_result_sha256']:
        raise ValueError('complete component result drift')
    done = json.loads((RUN / 'component/result.json').read_text())
    if not done['completed']: raise ValueError('component proof not complete')
    saved, decoder = authenticated(RUN / 'component/native.pickle', done['data']['archive_sha256'], pool)
    layout = numeric_layout(saved, pool)
    if layout.resident_entries > 64_000_000: raise MemoryError('complete restored component entry cap')
    pool.charge('c70_complete_restored_source_and_component', int(layout.resident_entries) + 1024)
    proof = json.loads(saved['proof_bytes'])
    source = check_restored(saved['source'])
    if (saved['schema'] != 'c70_offline_native_component_archive_v1' or saved['full_LIVE_admission']
            or source['physical_identity'] != SOURCE_ID
            or saved['source']['proof_sha256'] != SOURCE_PROOF
            or hashlib.sha256(saved['proof_bytes']).hexdigest() != done['data']['proof_sha256']
            or source_digest(saved['hz']) != proof['new_HZ_sha256']
            or digest(saved['packet']) != proof['packet_identity']
            or digest(saved['journal']) != proof['journal_identity']):
        raise ValueError('restored full source/component/journal proof binding differs')
    c = SimpleNamespace(**saved['source']['fields'])
    journal = LocalSpliceJournal(**saved['journal'])
    if journal.eq_roots is not c.eq_roots or journal.eq_scales is not c.eq_scales:
        raise ValueError('restored immutable source map sharing differs')
    plans = [Plan(**{**p, 'tail': tuple(p['tail'])}) for p in proof['plans']]
    inverse = verify_inverse(c, saved['hz'], journal, plans, pool=pool)
    emit('all_actual_unit_and_local_inverse_equations', inverse)
    return dict(inverse=inverse, complete_restored_component=asdict(layout), decoder=decoder,
        full_LIVE_admission=False, concrete_network_witness=False)


def main():
    stage = sys.argv[1]
    if stage not in ('phase', 'component', 'restore'): raise ValueError('explicit C70 stage required')
    resource.setrlimit(resource.RLIMIT_AS, (16*1024**3, 16*1024**3))
    freeze = json.loads((RUN/'preregistered.json').read_text())
    if any(_sha256(EXP/n) != sha for n, sha in freeze['source_sha256'].items()):
        raise ValueError('complete frozen source drift before execution')
    result = dict(completed=False, stage=stage, formal_gain=0, full_LIVE_admission=False)
    pool = WorkPool(256_000_000); start = time.monotonic()
    with (RUN/stage/'events.jsonl').open('x') as log:
        def emit(name, values):
            event = dict(event=name, elapsed_s=time.monotonic()-start, **values)
            log.write(json.dumps(event)+'\n'); log.flush(); print(json.dumps(event), flush=True)
        try:
            fn = {'phase': phase, 'component': component, 'restore': fresh_restore}[stage]
            data, stats = measured(lambda: fn(pool, result, emit), observe=lambda s: result.update(measurement=s))
            result.update(completed=True, data=data)
        except Exception as exc:
            result['failure'] = dict(type=type(exc).__name__, reason=str(exc))
        finally:
            result.update(wall_s=time.monotonic()-start, diagnostic_work=pool.used, diagnostic_parts=pool.parts)
            _atomic_exclusive_json(RUN/stage/'result.json', result)
            print(json.dumps(result), flush=True)
    if not result['completed']: raise SystemExit(1)


if __name__ == '__main__': main()
