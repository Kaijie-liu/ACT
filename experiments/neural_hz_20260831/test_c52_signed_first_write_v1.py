import copy
import numpy as np
import pytest
from experiments.neural_hz_20260831.c52_signed_first_write_v1 import build,reference,source_hash,state_hash,reconstruct
from experiments.neural_hz_20260831.c52_signed_first_write_audit_v1 import audit,comparison
from experiments.neural_hz_20260831.c52_neural_source_fixtures_v1 import fixture,check_points
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool


def test_default_off_never_reads_source():
    assert build(object()) is None


@pytest.mark.parametrize('kind',['chain','shared_add','conv_relu'])
def test_complete_ordinary_neural_sources_have_two_way_proof_and_whole_numeric_reduction(kind):
    program=fixture(kind);before_hash=source_hash(program)
    original=reference(program);compact=build(program,enabled=True)
    old_proof=audit(program,original);proof=audit(program,compact)
    measured=comparison(program,original,compact)
    assert proof['two_way_box_preserving_relation_proved'] and proof['all_predicates_and_outputs_exact']
    assert old_proof['eliminated_definitions']==0 and proof['eliminated_definitions']>=128
    assert all(measured[n] for n in ('strict_predicate_nnz_decrease','strict_numeric_bytes_decrease','strict_numeric_entries_decrease','strict_combined_reported_accounting_decrease'))
    assert compact['hz'].n_bin==program['nb']>=1 and compact['hz'].frame_id==program['frame_id']
    assert check_points(kind,program,original,compact)['feasible_full_inverse_points']>0
    assert source_hash(program)==before_hash
    assert compact['report']['first_write_no_removed_predicate_stored']


def test_shared_branch_coalescing_discovers_new_unit_relations_before_writing_them():
    program=fixture('shared_add');compact=build(program,enabled=True);proof=audit(program,compact)
    assert proof['new_unit_relations_exposed_by_shared_coalescing']==64
    assert proof['eliminated_definitions']==192 and compact['hz'].n_cont==3


def test_nonunit_and_directly_live_factors_stay_explicit():
    program=fixture('nonunit');original=reference(program);compact=build(program,enabled=True)
    proof=audit(program,compact)
    assert proof['eliminated_definitions']==128 and compact['hz'].n_cont==4
    assert program['nc']-1 in compact['global_ids'] and program['nc']-2 in compact['global_ids']
    assert check_points('nonunit',program,original,compact)['original_toy_network_outputs_exact']


def test_zero_hit_source_stays_an_exact_hz_and_gets_no_reduction_credit():
    program=fixture('zero_hit');original=reference(program);compact=build(program,enabled=True)
    proof=audit(program,compact);measurement=comparison(program,original,compact)
    assert proof['eliminated_definitions']==0 and compact['inverse'].size==0
    assert not measurement['strict_predicate_nnz_decrease']
    assert check_points('zero_hit',program,original,compact)['original_toy_network_outputs_exact']


@pytest.mark.parametrize('kind',['forward','uid','budget'])
def test_ordinary_source_or_budget_failure_never_changes_input(kind):
    program=fixture('chain',width=4)
    if kind=='forward':program['commands'][0]['cc'][0]=program['nc']-1
    elif kind=='uid':program['commands'][1]['uid']=program['commands'][0]['uid']
    before=source_hash(program)
    with pytest.raises((ValueError,MemoryError)):
        build(program,pool=WorkPool(0) if kind=='budget' else None,enabled=True)
    assert source_hash(program)==before


@pytest.mark.parametrize('kind',['coefficient','binary','inverse','uid','counter'])
def test_source_induction_rejects_a_resealed_wrong_result(kind):
    program=fixture('shared_add',width=8);compact=copy.deepcopy(build(program,enabled=True))
    if kind=='coefficient':compact['hz'].Ac.data[0]+=.25
    elif kind=='binary':compact['hz'].Aub.data[0]*=-1
    elif kind=='inverse':compact['inverse'][2]*=-1
    elif kind=='uid':compact['removed'][0,0]+=10000
    else:compact['report']['new_predicate_nnz']-=1
    compact['seal']=state_hash(compact)
    with pytest.raises(ValueError):audit(program,compact)


def test_inverse_uses_complete_original_frame_and_rejects_incomplete_box_points():
    program=fixture('chain',width=8);compact=build(program,enabled=True)
    point=reconstruct(compact,[0]*compact['hz'].n_cont)
    assert len(point)==program['nc'] and compact['hz'].n_bin==1
    with pytest.raises(ValueError):reconstruct(compact,[0])
    with pytest.raises(ValueError):reconstruct(compact,[2]*compact['hz'].n_cont)
