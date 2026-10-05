"""Ordinary exact source/native custody; no old Closed or mutable alias maps."""
from dataclasses import asdict
import hashlib
import json
import pickle
import numpy as np
import pytest
from experiments.neural_hz_20260831.c74_native_binding_v1 import (
    admit_source, admit_native, phase_image, SourceState, NativeState)
from experiments.neural_hz_20260831.c74_live_runtime_v1 import installed, extra_bound
from experiments.neural_hz_20260831.c65_physical_archive_v1 import bind_proof, fingerprint
from experiments.neural_hz_20260831.c69_birth_emission_v1 import lift
from experiments.neural_hz_20260831.c65_full_source_audit_v1 import audit
from experiments.neural_hz_20260831.test_c62_physical_boundary_v2 import complete, old_fields
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c32_native_blocks_v1 import phase_blocks
from experiments.neural_hz_20260831.c23_sparse_phase_overlay_v1 import build
from experiments.neural_hz_20260831.c30_append_discovery_v1 import discover_append
from experiments.neural_hz_20260831.c30_first_write_v1 import splice_append
from experiments.neural_hz_20260831.c73_outer_query_v1 import compile_journal
from experiments.neural_hz_20260831.c62_local_equations_v1 import SCHEMA
from experiments.neural_hz_20260831.c70_native_proof_v1 import extract, digest, verify, verify_inverse
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from act.back_end.hybridz_tf import tf_mlp as mlp
from experiments.neural_hz_20260831 import c9_live_runtime_v1 as base


def pool(): return WorkPool(256_000_000)


def source(kind='conv_disjoint'):
    _, saved = complete(kind); legacy, _ = old_fields(saved)
    fresh = lift(saved['expression'],saved['keep'],enabled=True)
    proof = audit(saved,legacy,fresh,pool=pool(),enabled=True)
    identity, raw = bind_proof(fresh['fields'],proof,source_sha256='independent_fixture')
    state, report = admit_source(fresh['fields'],raw,
        expected_sha256=hashlib.sha256(raw).hexdigest(),enabled=True)
    return state, report, fresh


def native():
    s, _, fresh = source(); h = s.hz; n = h.n_out
    slots = [(h.n_cont+2*i,h.n_cont+2*i+1,h.n_bin+i) for i in range(n)]
    args = (h,np.full(n,-8.),np.full(n,8.),slots,h.n_cont+2*n,h.n_bin+n)
    old = mlp.sparse_hz_apply_relu_exact(*args)
    packet = extract(h,old,old_n_cont=s.old_n_cont,old_n_eq=s.old_n_eq,
        logical_n_cont=s.logical_n_cont,first_uid=s.report['radix_uid_base']+16384,
        provenance={'independent_fixture':True},pool=pool(),enabled=True)
    v = phase_blocks(*args,source_pre=h,enabled=True); first = packet['first_uid']
    overlay, _ = build(s.owners,[(v.eq_c,first),(v.le_c,first+len(v.eq_rhs))],
        old_n_cont=s.old_n_cont,old_uid_ceiling=first,pool=pool(),enabled=True)
    plans, _ = discover_append(s,v,overlay,pool=pool(),enabled=True)
    assert plans
    new, _ = splice_append(v,plans,pool=pool(),enabled=True)
    j = compile_journal(s.eq_roots,s.eq_scales,plans,old_n_cont=s.old_n_cont,
        old_n_eq=s.old_n_eq,source_n_cont=h.n_cont,source_schema=SCHEMA,pool=pool(),enabled=True)
    proof = verify(s,v,overlay,plans,new,j,pool=pool())
    inverse = verify_inverse(s,new,j,plans,pool=pool())
    header = {k:packet[k] for k in ('schema','offline_only','fresh_native_execution','provenance')}
    transfer = dict(schema='c74_complete_source_native_transfer_v1',
        complete_C73_archive_authenticated=True,complete_independent_inverse_restored=True,
        source_identity=fingerprint(s.original_fields),source_proof_sha256=s.expected_proof_sha256,
        packet_header=header,packet_identity=digest(packet),new_HZ_sha256=source_digest(new),
        journal_identity=digest(vars(j)),event_sha256=hashlib.sha256(overlay.events.tobytes()).hexdigest(),
        proof=proof,inverse=inverse,full_LIVE_admission=False,formal_gain=0)
    raw = json.dumps(transfer,sort_keys=True).encode()
    state, _ = admit_native(enabled=True,source=s,hz=new,lineage=j,events=overlay.events,
        actual_phase_image=phase_image(s,v,transfer),transfer_proof_bytes=raw,
        expected_transfer_sha256=hashlib.sha256(raw).hexdigest(),
        construction_report={'new_phase_binaries':n,'plans':[asdict(p) for p in plans]})
    return state, plans, fresh


@pytest.mark.parametrize('kind',['chain','shared','conv_disjoint'])
def test_fresh_complete_source_binds_without_substitution(kind):
    s, report, fresh = source(kind)
    assert type(s) is SourceState and s.hz is fresh['fields']['hz']
    assert s.eq_roots is fresh['fields']['eq_roots']
    assert report['identity'] == fingerprint(fresh['fields'])
    assert s.numeric_roots()['original_source_fields'] is fresh['fields']
    assert s.validate()['work'] > 0


@pytest.mark.parametrize('field',['eq_roots','eq_scales','owners','keep','b','Ab','frame'])
def test_full_original_source_mutation_rejected(field):
    s, _, _ = source()
    if field == 'frame': s.hz.frame_id += 1
    elif field in ('b','Ab'):
        a = getattr(s.hz,field); a = a.data if field == 'Ab' else a
        assert len(a); a[0] += .125
    elif field == 'keep': s.keep[0] = not s.keep[0]
    else: getattr(s,field)[0] += 1
    with pytest.raises(ValueError): s.validate()


def test_complete_fresh_phase_matches_full_oracle_and_exact_inverse():
    state, plans, fresh = native()
    assert type(state) is NativeState
    assert state.lineage.eq_roots is fresh['fields']['eq_roots']
    assert state.validate()['full_source_and_native_content_bound']
    roots = state.numeric_roots()
    assert roots['actual_phase_image'] is state.actual_phase_image
    assert roots['original_source_fields']['expression'] is fresh['fields']['expression']
    assert verify_inverse(state.source,state.hz,state.lineage,plans,pool=pool())['all_equations_exact']
    with pytest.raises(TypeError): pickle.dumps(state)


@pytest.mark.parametrize('part',['new_rhs','binary','journal','phase','events','report'])
def test_native_complete_bound_payload_change_rejected(part):
    state, _, _ = native()
    if part == 'new_rhs': state.hz.b[0] += .125
    elif part == 'binary': state.hz.Aub.data[0] += .125
    elif part == 'journal': state.lineage.offsets[0] += .125
    elif part == 'phase': state.actual_phase_image['eq_rhs'][0] += .125
    elif part == 'events':
        state.events = state.events.copy(); state.events[0] += np.uint64(1)
    else: state.construction_report['new_phase_binaries'] += 1
    with pytest.raises(ValueError): state.validate()


def test_default_off_and_live_hook_restoration():
    assert admit_source(None,None,expected_sha256=None) is None
    assert admit_native() is None
    original = (base.lift,base.value_view,mlp.sparse_hz_apply_relu_exact)
    with installed(): assert original == (base.lift,base.value_view,mlp.sparse_hz_apply_relu_exact)
    state, _, _ = native()
    with installed(enabled=True,source_bytes=state.source.proof_bytes,
        source_sha=state.source.expected_proof_sha256,
        transfer_bytes=state.transfer_proof_bytes,transfer_sha=state.expected_transfer_sha256):
        assert base.lift is lift
    assert original == (base.lift,base.value_view,mlp.sparse_hz_apply_relu_exact)


def test_original_work_caps_and_complete_runtime_bound():
    prices = extra_bound(200,200,200,1000,600)
    assert prices['value_views'] == 2*(64+32*200+8*200)
    assert prices['slot_disjointness'] == 16*200*(400).bit_length()
    assert sum(prices.values()) < 4_000_000
    with pytest.raises(MemoryError): WorkPool(0).charge('runtime',sum(prices.values()))


@pytest.mark.parametrize('name',['c74_live_worker_v1','c74_proof_transfer_worker_v1'])
def test_actual_worker_entrypoints_import(name):
    import importlib
    assert callable(importlib.import_module('experiments.neural_hz_20260831.'+name).main)
