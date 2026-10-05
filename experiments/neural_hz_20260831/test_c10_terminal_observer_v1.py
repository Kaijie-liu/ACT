from types import SimpleNamespace
import numpy as np
import pytest

from experiments.neural_hz_20260831.c10_terminal_observer_v1 import observed, backend, scalar


def test_identical_call_objects_options_and_return_without_extra_solves(monkeypatch):
    calls, events = [], []
    c, integrality, constraints = np.zeros(3), np.ones(3), object()
    options = {'time_limit': 45., 'presolve': True, 'mip_rel_gap': 0.}
    result = SimpleNamespace(status=1, message='time limit', success=False, x=None)
    def original(*args, **kwargs):
        assert kwargs['c'] is c and kwargs['integrality'] is integrality
        assert kwargs['constraints'] is constraints and kwargs['options'] is options
        calls.append(1)
        return result
    monkeypatch.setattr(backend, 'milp', original)
    with observed(events.append):
        assert backend.milp(c=c, integrality=integrality, constraints=constraints, options=options) is result
    assert calls == [1] and backend.milp is original
    assert events[0]['status'] == 1 and not events[0]['incumbent_present']
    assert options == {'time_limit': 45., 'presolve': True, 'mip_rel_gap': 0.}


@pytest.mark.parametrize('accepted', [True, False])
def test_validator_exact_delegation_and_no_extra_validation(monkeypatch, accepted):
    calls, events = [], []
    x = np.zeros(2)
    def original(*args):
        assert args[0] is x and args[1] == 1e-7
        calls.append(1)
        return accepted
    monkeypatch.setattr(backend, '_valid_milp_point', original)
    with observed(events.append):
        assert backend._valid_milp_point(x, 1e-7) is accepted
    assert calls == [1] and backend._valid_milp_point is original
    assert events == [{'event': 'ordinary_point_validation', 'accepted': accepted, 'tolerance': 1e-7}]


def test_solver_exception_propagates_unchanged_and_hooks_restore(monkeypatch):
    error, events = RuntimeError('fixture'), []
    def original(*args, **kwargs):
        raise error
    monkeypatch.setattr(backend, 'milp', original)
    validator = backend._valid_milp_point
    with pytest.raises(RuntimeError) as got:
        with observed(events.append):
            backend.milp()
    assert got.value is error and backend.milp is original and backend._valid_milp_point is validator
    assert events[0]['event'] == 'ordinary_milp_exception'


def test_nonfinite_diagnostic_scalars_are_not_json_nan():
    assert scalar(None) is None and scalar(np.nan) is None and scalar(np.inf) is None
    assert scalar(4) == 4
