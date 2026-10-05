"""Sufficient statistics, local inverse and complete ordinary physical rows."""
from fractions import Fraction as F
import numpy as np
import pytest
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c57_scalar_consumer_probe_v2 import native_word,fraction,in_window
from experiments.neural_hz_20260831.c61_retained_boundary_v1 import optimize
from experiments.neural_hz_20260831.test_c61_retained_boundary_v1 import data,binary64_mantissa_fits
from experiments.neural_hz_20260831.c62_precision_plan_v1 import odd_significands,choose,plan
from experiments.neural_hz_20260831.c62_local_equations_v1 import SCHEMA,encode,decode,reconstruct
from experiments.neural_hz_20260831.c62_physical_quotient_v1 import prepare,emit
from experiments.neural_hz_20260831.test_c53_logical_singleton_census_v2 import complete
from experiments.neural_hz_20260831.c31_prepared_emission_v1 import lift
from experiments.neural_hz_20260831.c24_closed_state_v1 import close,export
from experiments.neural_hz_20260831.c17_ownership_audit_v1 import actual_words


def test_exact_odd_significands_match_independent_native_integer_ratios():
    values=np.ldexp((2**23+2*np.arange(128)+1).astype(np.float64),-24)
    values=np.r_[values,-values,np.ldexp(values,3),np.ldexp(values,-4),[.5,.75,-.125]]
    assert odd_significands(values).tolist()==[abs(native_word(v)[0]) for v in values]


@pytest.mark.parametrize('kind',['chain','branch','mixed','halves'])
def test_maximum_statistic_and_leaf_recurrence_equal_independent_C61(kind):
    parents,ratios,external,states,keep,remove=data(kind)
    local={v:native_word(float(r)) for v,r in ratios.items()}
    maxima={v:max((abs(native_word(float(c))[0]) for c in cs),default=0) for v,cs in external.items()}
    selected,anchors,weights,stats=choose(parents,local,maxima,WorkPool(256_000_000))
    total,expected,expected_anchors,_=optimize(parents,states,keep,remove,WorkPool(256_000_000))
    assert selected==expected and anchors==expected_anchors and stats['optimum']==total
    for v in states:
        for a,w in states[v].items():
            sufficient=not maxima[v] or abs(w[0])*maxima[v]<2**53
            assert sufficient==all(binary64_mantissa_fits(c*fraction(w)) for c in external[v])


@pytest.mark.parametrize('r',[F(3,4),F(-7,16),F(2**23+3,2**61),F(1,2)])
def test_native_local_equation_words_and_exact_chained_inverse(r):
    word=native_word(float(r));tag,num=encode(0,word)
    assert decode(tag,num,column=1,n_cont=3,schema=SCHEMA)==(0,word)
    assert in_window(native_word(num))
    tag2,num2=encode(1,word)
    roots=np.array([tag,tag2],np.int64);scales=np.array([num,num2],np.float64).view(np.int64)
    got=reconstruct([F(1,3),F(0),F(0)],roots,scales,old_n_cont=1,old_n_eq=0,n_cont=3,schema=SCHEMA)
    assert got==[F(1,3),r/3,r*r/3] and max(map(abs,got))<=1
    with pytest.raises(ValueError):decode(tag,num,column=1,n_cont=3,schema='legacy_alias')


def old_fields(saved):
    expr=saved['expression'];draft=lift(expr,saved['keep'],enabled=True,frame_widths=(saved['old_n_cont'],saved['old_n_bin']))
    maps={k:saved[k] for k in ('eq_roots','eq_scales','ineq_roots','ineq_scales','def_rows')}
    checked,proof=close(draft,saved['hz'],maps,enabled=True)
    fields,raw=export(checked)
    return fields,raw


def residual(hz,cm,bm,rhs,row,x,binary):
    m=getattr(hz,cm);b=getattr(hz,bm)
    start,stop=map(int,m.indptr[row:row+2]);bs,be=map(int,b.indptr[row:row+2])
    return (sum((F(float(a))*x[int(c)] for c,a in zip(m.indices[start:stop],m.data[start:stop])),F(0))
            +sum((F(float(a))*binary[int(c)] for c,a in zip(b.indices[bs:be],b.data[bs:be])),F(0))-F(float(getattr(hz,rhs)[row])))


@pytest.mark.parametrize('kind',['chain','shared','conv'])
def test_complete_physical_source_inverse_and_independent_full_owner_incidence(kind):
    _,saved=complete(kind);legacy,raw=old_fields(saved);pool=WorkPool(256_000_000)
    selected=plan(saved,pool=pool,enabled=True)
    prepared=prepare(saved,legacy,selected,pool=pool,enabled=True)
    state,proof=emit(legacy,prepared,pool=pool,enabled=True);fields=state['fields'];hz=fields['hz']
    vals=[F((i%5)-2,5) for i in range(hz.n_cont)];binary=[F(-1 if i%2 else 1) for i in range(hz.n_bin)]
    full=reconstruct(vals,fields['eq_roots'],fields['eq_scales'],old_n_cont=saved['old_n_cont'],old_n_eq=saved['old_n_eq'],n_cont=hz.n_cont,schema=state['lineage_schema'])
    for cm,bm,rhs in [('Ac','Ab','b'),('Auc','Aub','ub')]:
        p=prepared['parts'][cm]
        for original_row,target in enumerate(p['mapping']):
            a=residual(saved['hz'],cm,bm,rhs,original_row,full,binary)
            if target<0:assert a==0
            else:assert residual(hz,cm,bm,rhs,int(target),vals,binary)==a*F(2)**int(p['qrows'][original_row])
    eq=prepared['parts']['Ac'];le=prepared['parts']['Auc']
    expected=actual_words(hz,fields['old_n_cont'],fields['logical_n_cont'],eq['uids'][eq['keep']],le['uids'][le['keep']])
    assert np.array_equal(expected,fields['owners'])
    assert proof['actual_predicate_nnz']==proof['original_predicate_nnz']-2*proof['local_equations_checked']
    assert not proof['source_first_or_native_admission'] and hz.n_bin==saved['hz'].n_bin


def test_default_off_and_prepaid_limit():
    assert plan(None,pool=None) is None and prepare(None,None,None,pool=None) is None and emit(None,None,pool=None) is None
    with pytest.raises(MemoryError):choose({1:0},{1:(3,-2)},{1:1},WorkPool(0))
