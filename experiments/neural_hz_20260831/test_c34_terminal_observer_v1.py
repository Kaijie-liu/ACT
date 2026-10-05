from types import SimpleNamespace
import numpy as np
import pytest
from act.back_end.solver import solver_hz as backend
from experiments.neural_hz_20260831.c34_terminal_observer_v1 import installed
from experiments.neural_hz_20260831.c9_live_runtime_v1 import SelectedRejected
from experiments.neural_hz_20260831.test_c34_terminal_binding_v1 import fixture


def test_default_off_preserves_every_hook():
    old=(backend.milp,backend._lower_hz_milp,backend.HZSolver._recover_input)
    with installed(*([object()]*5)):assert old==(backend.milp,backend._lower_hz_milp,backend.HZSolver._recover_input)


@pytest.mark.parametrize('status',[1,2])
def test_ordinary_base_failure_stays_unknown_no_property_or_extra_solve(monkeypatch,status):
    state,final,inp,spec,kw,raw,sha=fixture(monkeypatch)
    model=backend._lower_hz_milp(final);proof=dict(native_lowered_n_cont=model.n_cont,native_lowered_n_bin=model.n_bin)
    calls=[];events=[]
    def solve(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status=status,message='fixture',success=False,x=None,mip_node_count=0)
    monkeypatch.setattr(backend,'milp',solve)
    with installed(state['lifted'],final,inp,kw['input_shape'],proof,enabled=True,emit=events.append,
            on_point=lambda *a:pytest.fail('base failure manufactured witness')):
        result=backend.HZSolver().evaluate_spec(final,spec,input_hz=inp,**kw)
    assert len(calls)==1 and calls[0]['options']['presolve'] is True
    assert 0<calls[0]['options']['time_limit']<=45.
    assert np.count_nonzero(calls[0]['c'])==0
    assert result[0].status.name=='UNKNOWN'
    assert result[0].metadata['reason'] in ('empty_hz','base_unknown')
    assert [e['event'] for e in events]==['c34_actual_ordinary_model_bound','c34_ordinary_milp_start','ordinary_milp_return']


def test_exact_recover_is_one_call_and_hooks_restore(monkeypatch):
    state,final,inp,spec,kw,raw,sha=fixture(monkeypatch)
    m=backend._lower_hz_milp(final);proof=dict(native_lowered_n_cont=m.n_cont,native_lowered_n_bin=m.n_bin)
    old=(backend._lower_hz_milp,backend.HZSolver._recover_input);events=[];points=[]
    with installed(state['lifted'],final,inp,kw['input_shape'],proof,enabled=True,emit=events.append,on_point=lambda m,x:points.append(x.copy())):
        model=backend._lower_hz_milp(final,prune_unused=True,coalesce_rows=True,project_inactive_cont=False,fix_implied_binary=False)
        result=backend.HZSolver._recover_input(model,np.zeros(model.n_var),inp,kw['input_shape'],0)
        assert result is not None and len(points)==1
        with pytest.raises(SelectedRejected):backend.HZSolver._recover_input(model,np.zeros(model.n_var),inp,kw['input_shape'],0)
    assert old==(backend._lower_hz_milp,backend.HZSolver._recover_input)
    assert events[-1]['event']=='c34_exact_witness_reconstruction_passed'


def test_failed_exact_recover_aborts_without_solver_rescue(monkeypatch):
    state,final,inp,spec,kw,raw,sha=fixture(monkeypatch)
    m=backend._lower_hz_milp(final);proof=dict(native_lowered_n_cont=m.n_cont,native_lowered_n_bin=m.n_bin)
    old=(backend._lower_hz_milp,backend.HZSolver._recover_input);events=[]
    with pytest.raises(SelectedRejected):
        with installed(state['lifted'],final,inp,kw['input_shape'],proof,enabled=True,emit=events.append):
            m=backend._lower_hz_milp(final,prune_unused=True,coalesce_rows=True,project_inactive_cont=False,fix_implied_binary=False)
            point=np.zeros(m.n_var);point[0]=2.
            backend.HZSolver._recover_input(m,point,inp,kw['input_shape'],0)
    assert old==(backend._lower_hz_milp,backend.HZSolver._recover_input)
    assert events[-1]['event']=='c34_exact_witness_reconstruction_rejected'
