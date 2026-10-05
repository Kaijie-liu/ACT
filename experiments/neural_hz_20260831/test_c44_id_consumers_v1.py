from types import SimpleNamespace
import pytest
from act.back_end.core import Layer
from act.pipeline.verification.batchnorm_graph import _validate_variable_flow
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c44_var_ids_v1 import paid,span
from experiments.neural_hz_20260831.c44_id_consumers_v1 import fixture,evidence,diagnostic_python_closure
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect


@pytest.mark.parametrize('consumers',[1,2,4])
def test_actual_Net_Layer_ConSet_values_and_outer_aliases(consumers):
    pool=WorkPool(256_000_000)
    with paid(pool):
        old=fixture(512,consumers,compact=False,pool=pool)
        new=fixture(512,consumers,compact=True,pool=pool)
        a=evidence(old,pool);b=evidence(new,pool)
        assert a['ordered_sequence_sha256']==b['ordered_sequence_sha256']
        assert a['outer_alias_matrix']==b['outer_alias_matrix']
        assert a['baseline_scalar_shared_with_layer'] and not b['baseline_scalar_shared_with_layer']
        if consumers>1:
            assert a['repeated_consumer_scalar_identity'] and not b['repeated_consumer_scalar_identity']
        am=diagnostic_python_closure(old,pool);bm=diagnostic_python_closure(new,pool)
        assert not am['full_HZ_LIVE_gate_proved'] and not bm['full_HZ_LIVE_gate_proved']
        # Enough downstream tuples make implicit scalar reallocation observable.
        if consumers==4:assert bm['unique_python_shallow_bytes']>am['unique_python_shallow_bytes']


def test_residual_binary_operand_guard_is_not_relaxed():
    pool=WorkPool(256_000_000)
    with paid(pool):
        x=span(1000,8,enabled=True);y=span(1008,8,enabled=True);z=span(1016,8,enabled=True)
        layers=[Layer(0,'INPUT',{'shape':(1,8),'dtype':'torch.float64'},[],x),Layer(1,'RELU',{},x.copy(),y),
            Layer(2,'ADD',{'x_vars':x,'y_vars':y},x+y,z)]
        preds={0:[],1:[0],2:[0,1]}
        with pytest.raises(ValueError,match='missing ordered binary operands'):
            _validate_variable_flow(layers,preds)
        # Ordinary explicitly materialized parameters still satisfy OLD guard;
        # this is only a diagnostic control, not an uncharged runtime adapter.
        layers[2].params.update(x_vars=list(x),y_vars=list(y))
        _validate_variable_flow(layers,preds)


def test_existing_full_LIVE_collector_still_rejects_unregistered_sequence():
    pool=WorkPool(256_000_000)
    with paid(pool):
        ids=span(1000,8,enabled=True)
        with pytest.raises(ValueError,match='unknown live root'):
            collect(SimpleNamespace(),{'unregistered_IDs':ids})


@pytest.mark.parametrize('bad',['width','consumer','mode','unknown','field','budget'])
def test_diagnostic_domain_and_opaque_roots_fail_closed(bad):
    pool=WorkPool(0 if bad=='budget' else 256_000_000)
    with paid(pool),pytest.raises((ValueError,MemoryError)):
        if bad in ('width','consumer','mode','budget'):
            fixture(65537 if bad=='width' else 8,3 if bad=='consumer' else 1,
                compact='yes' if bad=='mode' else True,pool=pool)
        elif bad=='unknown':diagnostic_python_closure({'opaque':object()},pool)
        else:
            ids=span(1000,8,enabled=True);object.__setattr__(ids,'hidden',[])
            diagnostic_python_closure(ids,pool)
