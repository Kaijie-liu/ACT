"""Complete ordinary original-native byte recovery, not subset equality."""
import pickle
import numpy as np
import pytest
from experiments.neural_hz_20260831.c72_inverse_phase_v1 import recover,native_digest
from experiments.neural_hz_20260831.c70_native_proof_v1 import same_matrix,equal
from experiments.neural_hz_20260831.c26_tagged_transplant_v1 import compile_lineage
from experiments.neural_hz_20260831.c30_append_discovery_v1 import discover_append
from experiments.neural_hz_20260831.c30_first_write_v1 import splice_append
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.test_c28_consumer_discovery_v1 import source
from experiments.neural_hz_20260831.test_c30_first_write_v1 import view_from


def pool():return WorkPool(256_000_000)


def setup(**kw):
    c,h,o,*_=source(**kw);view=view_from(c,h)
    plans,_=discover_append(c,view,o,pool=pool(),enabled=True)
    new,_=splice_append(view,plans,pool=pool(),enabled=True)
    lineage=compile_lineage(c.eq_roots,c.eq_scales,plans,old_n_cont=c.old_n_cont,
        old_n_eq=c.old_n_eq,pool=pool(),enabled=True)
    return c,h,new,lineage


@pytest.mark.parametrize('mixed',[False,True])
@pytest.mark.parametrize('subtract',[False,True])
@pytest.mark.parametrize('phase',[False,True])
def test_every_original_native_array_recovers_exactly(mixed,subtract,phase):
    c,h,new,lineage=setup(mixed=mixed,subtract=subtract,phase=phase)
    original=source_digest(h);pre=source_digest(c.hz);oldnew=source_digest(new)
    p,report=recover(c,new,lineage,provenance={'fixture':True},pool=pool(),enabled=True)
    p=pickle.loads(pickle.dumps(p,protocol=5))
    assert native_digest(c.hz,p,pool=pool())==original
    for k,name in [('eq_c','Ac'),('eq_b','Ab'),('le_c','Auc'),('le_b','Aub')]:
        start=c.hz.n_ineq if name.startswith('Au') else c.hz.n_eq
        assert same_matrix(p[k],getattr(h,name)[start:])
    assert equal(p['eq_rhs'],h.b[c.hz.n_eq:]) and equal(p['le_rhs'],h.ub[c.hz.n_ineq:])
    assert source_digest(c.hz)==pre and source_digest(new)==oldnew
    assert report['all_original_phase_rows']==h.n_eq+h.n_ineq-c.hz.n_eq-c.hz.n_ineq


@pytest.mark.parametrize('old',[0,6,12])
def test_all_phase_rows_with_old_and_new_selected_consumers(old):
    c,h,new,j=setup(phase=True,old_consumers=old)
    p,report=recover(c,new,j,provenance={},pool=pool(),enabled=True)
    assert report['changed_phase_rows_inverted']==12-old
    assert native_digest(c.hz,p,pool=pool())==source_digest(h)


@pytest.mark.parametrize('key',['eq_c','le_b','Gc'])
def test_entire_native_digest_detects_each_phase_component_change(key):
    c,h,new,j=setup(phase=True,mixed=True)
    p,_=recover(c,new,j,provenance={},pool=pool(),enabled=True)
    assert p[key].nnz
    p[key].data[0]+=.125
    assert native_digest(c.hz,p,pool=pool())!=source_digest(h)


def test_default_off_and_resource_rejection():
    assert recover(object(),object(),object(),provenance=object(),pool=object()) is None
    c,h,new,j=setup()
    with pytest.raises(MemoryError):recover(c,new,j,provenance={},pool=WorkPool(0),enabled=True)
