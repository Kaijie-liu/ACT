import copy
from fractions import Fraction as F
import pytest
from experiments.neural_hz_20260831.c54_scalar_fixtures_v1 import fixture
from experiments.neural_hz_20260831.c54_packed_scalar_hz_v2 import build as exact_build,state_hash as exact_hash
from experiments.neural_hz_20260831.c52_signed_first_write_v1 import reference
from experiments.neural_hz_20260831.c55_binary64_lift_v1 import build,state_hash,native,digit_rows
from experiments.neural_hz_20260831.c55_binary64_lift_audit_v1 import audit,check_points,derived
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool


def example(kind='chain'):
    program=fixture(kind);exact=exact_build(program,enabled=True)
    return program,exact,build(exact,enabled=True)


@pytest.mark.parametrize('kind',['chain','shared_add','conv_relu'])
def test_actual_emitted_equations_preserve_full_source_and_inverse(kind):
    program,exact,state=example(kind);proof=audit(program,exact,state)
    assert proof['every_new_box_proved_redundant']
    assert proof['all_auxiliary_equalities_checked']==state['n_cont']-exact['n_cont']
    assert state['n_bin']==program['nb'] and state['report']['predicate_nnz']<state['report']['original_predicate_nnz']
    points=check_points(program,reference(program),state)
    assert points['feasible_original_vectors']==(36 if kind=='conv_relu' else 6)
    assert points['benchmark_counterexamples']==0


def test_default_off_never_inspects_an_input():
    assert build(None) is None


@pytest.mark.parametrize('value',[(1,-20),(-12345,-16),(2**53-1,-53)])
def test_exact_binary64_literals(value):
    m,e=value
    assert F(native(value))==F(m)*F(2)**e


@pytest.mark.parametrize('change',['digit','consumer','rhs','inverse','uid','binary','box','frame','unsealed'])
def test_rejects_changed_actual_rows_and_provenance(change):
    program,exact,state=example();changed=copy.deepcopy(state);m=changed['csr'];row=changed['base_ne']
    if change=='digit':m['data'][int(m['indptr'][row])]*=2
    elif change=='consumer':m['data'][0]*=2
    elif change=='rhs':changed['rhs'][0]+=.125
    elif change=='inverse':changed['inverse'][3]^=1
    elif change=='uid':changed['eq_uids'][-1]=changed['eq_uids'][0]
    elif change=='binary':changed['n_bin']+=1;m['shape']=(m['shape'][0],m['shape'][1]+1)
    elif change=='box':m['data'][int(m['indptr'][row])]=-2.
    elif change=='frame':changed['frame_id']+=1
    else:m['data'][0]*=2
    if change!='unsealed':changed['seal']=state_hash(changed)
    with pytest.raises(ValueError):audit(program,exact,changed)


def test_work_overflow_rejects_without_changing_exact_source():
    program=fixture('chain');exact=exact_build(program,enabled=True);before=exact_hash(exact)
    with pytest.raises(MemoryError):build(exact,enabled=True,pool=WorkPool(1))
    assert exact_hash(exact)==before


def test_zero_hit_preserves_phase_domain_without_auxiliary():
    program,exact,state=example('zero_hit')
    assert state['report']['auxiliary_continuous']==0
    assert audit(program,exact,state)['all_auxiliary_equalities_checked']==0
    assert check_points(program,reference(program),state)['feasible_original_vectors']==6


@pytest.mark.parametrize('mantissa',[(1<<54)+3,(1<<106)+17,(1<<416)+19,(1<<511)+1])
def test_unsigned_digit_induction_and_no_long_float_cast(mantissa):
    value=(mantissa,-mantissa.bit_length());ratio=F(0)
    for digit,width in digit_rows(value):
        ratio=(ratio+digit)/(1<<width)
        assert 0<=ratio<1
    assert ratio==F(mantissa,1<<mantissa.bit_length())
    with pytest.raises(ValueError):native(value)
