from fractions import Fraction as F
import numpy as np
import pytest
import scipy.sparse as sp
from act.back_end.hybridz_tf import tf_cnn as cnn
from experiments.neural_hz_20260831.c8_dyadic_balance_v1 import box_exponent
from experiments.neural_hz_20260831.c9_integrated_suffix_v1 import lift
from experiments.neural_hz_20260831.c9_integrated_suffix_audit_v1 import audit,proxy
from experiments.neural_hz_20260831.c9_radix_predicate_audit_v1 import recover_row
from experiments.neural_hz_20260831.c53_source_normalization_v1 import assess,strict_envelope,scalar_identity
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.test_c5_live_value_contraction_v1 import source
from experiments.neural_hz_20260831.test_c6_support_affine_plan_v1 import fixture as conv_fixture


@pytest.mark.parametrize('values,powers',[([1.],[0]),([-1.],[0]),([.5],[2]),([.75,.25],[1,1]),([1.,1.,1.],[0,0,0]),([-.125,.375],[0,3])])
def test_strict_integer_envelope_against_exact_fraction(values,powers):
    values=np.array(values);powers=np.array(powers)
    unit=strict_envelope(values,powers)
    assert unit==box_exponent(values,powers)
    assert sum(abs(F(float(v)))*F(2)**int(p) for v,p in zip(values,powers))<F(2)**unit
    if len(values)==1 and abs(values[0]) in (1.,.5):
        sign,shift=scalar_identity(values[0],powers[0],unit)
        assert shift<0 and sign*F(2)**shift==F(float(values[0]))*F(2)**int(powers[0])/F(2)**unit


def fixture(kind='chain'):
    if kind=='conv':expr,_=conv_fixture()
    else:
        s=source(8);identity=sp.eye(8,format='csr');negative=sp.diags(np.full(8,-.5),format='csr')
        term=cnn.SparseHZAffineTerm(s,(identity,negative,identity))
        expr=cnn.SparseHZAffineExpr((term,term) if kind=='shared' else (term,),np.zeros(8),8,s.frame_id)
    candidate=lift(expr,np.ones(expr.n_out,bool),enabled=True)
    saved=dict(definition_graph=candidate.nodes,expression=expr,hz=candidate.hz,
        old_n_cont=candidate.old_n_cont,logical_n_cont=candidate.logical_n_cont,
        old_n_bin=candidate.old_n_bin,root=candidate.root,keep=candidate.keep)
    return candidate,saved


@pytest.mark.parametrize('kind',['chain','shared','conv'])
def test_complete_original_source_census_and_fraction_rows(kind):
    candidate,saved=fixture(kind);before=source_digest(candidate.hz)
    assert audit(candidate)['all_original_coefficients_exact']
    result=assess(saved,pool=WorkPool(256_000_000),enabled=True)
    assert result['current_C52_signed_unit_population']==0
    assert result['totals']['MAIN_rows']==candidate.logical_n_cont-candidate.old_n_cont
    assert result['potential_scale_aware_singleton_only_lower_bound']>0
    for i in range(candidate.old_n_eq,len(candidate.eq_roots)):
        cc,bc,rhs,_=recover_row(proxy(candidate),i)
        slot=candidate.old_n_cont+i-candidate.old_n_eq;pivot=F(cc.pop(slot))
        assert sum(abs(F(v)) for v in cc.values())+sum(abs(F(v)) for v in bc.values())+abs(F(rhs))<pivot
    assert source_digest(candidate.hz)==before
    assert not result['actual_forwarding_or_new_HZ_constructed'] and result['formal_gain']==0


@pytest.mark.parametrize('bad',['slot','exponent','output'])
def test_original_source_binding_rejects_ordinary_mismatch(bad):
    _,saved=fixture();node=saved['definition_graph'][0];row=np.flatnonzero(node['needed'])[0]
    if bad=='slot':node['slots'][row]+=1
    elif bad=='exponent':node['exponents'][row]+=1
    else:saved['hz'].Gc.data[0]*=2
    with pytest.raises(ValueError):assess(saved,pool=WorkPool(256_000_000),enabled=True)


def test_default_off_has_no_source_access():
    assert assess(object(),pool=object()) is None


def test_unchanged_budget_rejects_before_source_scan():
    _,saved=fixture()
    with pytest.raises(MemoryError):assess(saved,pool=WorkPool(0),enabled=True)
