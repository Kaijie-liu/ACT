import copy
from fractions import Fraction as F
import numpy as np
import pytest
from experiments.neural_hz_20260831.c54_exact_dyadic_v1 import source,multiply,add,canonical,Pool,unpack
from experiments.neural_hz_20260831.c54_packed_scalar_hz_v2 import build,state_hash
from experiments.neural_hz_20260831.c54_packed_scalar_hz_v2 import audit,comparison,reconstruct,check_points
from experiments.neural_hz_20260831.c54_scalar_fixtures_v1 import fixture
from experiments.neural_hz_20260831.c52_signed_first_write_v1 import reference,build as signed_build
from experiments.neural_hz_20260831.c52_neural_source_fixtures_v1 import row
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool


@pytest.mark.parametrize('a,b',[(.1,.2),(.7,.9),(-.7,.9),(.125,.8),(.6,-.6),(.3,.4)])
def test_integer_arithmetic_and_packed_pool_match_independent_fraction(a,b):
    work=WorkPool(256_000_000);product=multiply(source(a),source(b),work);total=add(source(a),source(b),work)
    pool=Pool(work);i=pool.intern(product);j=pool.intern(total)
    decoded=[F(m)*F(2)**e for m,e in unpack(pool.pack())]
    assert decoded[i]==F(a)*F(b) and decoded[j]==F(a)+F(b)
    if (a,b)==(.1,.2):assert decoded[i]!=F(a*b)


@pytest.mark.parametrize('kind',['chain','shared_add','conv_relu'])
def test_complete_general_scalar_source_relation_and_physical_accounting(kind):
    program=fixture(kind);old=reference(program);state=build(program,enabled=True)
    proof=audit(program,state);physical=comparison(program,old,state)
    assert proof['all_source_predicates_outputs_and_inverse_exact']
    assert proof['non_binary64_scalar_count']>0 and proof['non_binary64_matrix_scalar_count']>0
    assert proof['maximum_mantissa_bits']>53
    assert physical['strict_predicate_nnz_decrease']
    assert physical['strict_numeric_bytes_decrease'] and physical['strict_numeric_entries_decrease']
    assert physical['strict_combined_reported_accounting_decrease']
    assert check_points(kind,program,old,state)['feasible_full_inverse_points']==(36 if kind=='conv_relu' else 6)
    assert signed_build(program,enabled=True)['report']['eliminated']==0
    if kind=='shared_add':assert proof['coalescing_exposed_relations']==64


@pytest.mark.parametrize('guard',['nonredundant','offset','binary','output_live','nonpower_pivot'])
def test_uniform_nonadmitted_definitions_stay_explicit(guard):
    program=fixture('zero_hit');program['nc']=3
    command=dict(kind='def',uid=103,column=2,**row([(1,-.7),(2,1.)]))
    if guard=='nonredundant':command['cv'][0]=-1.25
    elif guard=='offset':command['rhs']=.1
    elif guard=='binary':command.update(bc=np.array([0],np.int64),bv=np.array([.1]))
    elif guard=='output_live':program['outputs']=[row([(2,.5)])]
    else:command['cv'][-1]=.75
    program['commands'].append(command)
    state=build(program,enabled=True)
    assert state['report']['eliminated']==0 and audit(program,state)['two_way_box_preserving_relation_proved']


@pytest.mark.parametrize('field',['limb','inverse','binary','uid','count'])
def test_resealed_incorrect_scalar_predicate_inverse_or_UID_is_rejected(field):
    program=fixture('chain');state=copy.deepcopy(build(program,enabled=True))
    if field=='limb':state['scalars']['limbs'][0]^=np.uint64(2)
    elif field=='inverse':state['inverse'][-2]&=np.uint64(0xffffffff00000000)
    elif field=='binary':
        from experiments.neural_hz_20260831.c54_scalar_hz_audit_v1 import scalars
        position=np.flatnonzero(state['csr']['indices']>=state['n_cont'])[0]
        state['csr']['coefficients'][position]=scalars(state).index(F(1))
    elif field=='uid':state['removed'][0,0]+=1
    else:state['report']['eliminated']+=1
    state['seal']=state_hash(state)
    with pytest.raises(ValueError):audit(program,state)


def test_default_off_does_not_access_source():
    assert build(object()) is None


def test_zero_hit_remains_exact_without_reduction_credit():
    program=fixture('zero_hit');state=build(program,enabled=True)
    assert audit(program,state)['eliminated']==0
    assert not comparison(program,reference(program),state)['strict_predicate_nnz_decrease']
    assert check_points('zero_hit',program,reference(program),state)['feasible_full_inverse_points']==6


def test_entry_arithmetic_and_work_limits_fail_closed():
    with pytest.raises(MemoryError):canonical((1<<513)-1,-513)
    with pytest.raises(ValueError):source(2.**-21)
    with pytest.raises(MemoryError):build(fixture('chain'),enabled=True,pool=WorkPool(0))


def test_inverse_requires_complete_in_box_point():
    state=build(fixture('chain'),enabled=True)
    with pytest.raises(ValueError):reconstruct(state,[0])
    with pytest.raises(ValueError):reconstruct(state,[2]*state['n_cont'])


@pytest.mark.parametrize('bad',['partition','inverse_high','domain'])
def test_joint_row_discrete_domain_and_inverse_word_ranges(bad):
    program=fixture('chain');state=build(program,enabled=True)
    assert state['report']['unshared_split_numeric_buffers_retired']==23
    if bad=='partition':state['n_eq']+=1;state['n_ineq']-=1
    elif bad=='inverse_high':state['inverse'][0]|=np.uint64(0xffffffff00000000)
    else:state['csr']['indices'][-1]=state['n_cont']+state['n_bin']
    state['seal']=state_hash(state)
    with pytest.raises(ValueError):audit(program,state)
