# SPDX-License-Identifier: AGPL-3.0-or-later
"""One ordinary nonconvex fixture proves observation before the actual target."""
import json
import sys

import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import block_diag

from act.back_end.solver import solver_hz as backend
from experiments.neural_hz_20260831 import c108_ordinary_phase_v2 as observer
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured


def test_ordinary_nonconvex_phase_observation():
    """All phases, original nonconvex problem, argument/result identities, cleanup."""
    blocks = 64
    A = block_diag([np.array([[1., -1., -.5], [1., 1., 0.]])]*blocks, format='csr')
    lower = np.tile([.125, -.25], blocks)
    upper = np.tile([.125, .75], blocks)
    c = np.tile([1., 2., -.5], blocks)
    integral = np.tile([0, 0, 1], blocks)
    bounds = Bounds(np.tile([-1., -1., 0.], blocks), np.ones(blocks*3))
    constraint = LinearConstraint(A, lower, upper)
    arrays = [c, integral, bounds.lb, bounds.ub, A.data, A.indices, A.indptr,
              constraint.lb, constraint.ub]
    before = [a.copy() for a in arrays]
    options = dict(presolve=True, time_limit=45., mip_rel_gap=0., disp=False)
    plain, plain_stats = measured(lambda: milp(c, integrality=integral, bounds=bounds,
        constraints=constraint, options=dict(options)), observe=lambda _: None)
    events = []
    original = backend.milp
    identities = {}
    observed_options = dict(options)

    def delegate(*args, **kwargs):
        assert args == ()
        assert kwargs['c'] is c and kwargs['integrality'] is integral
        assert kwargs['bounds'] is bounds and kwargs['constraints'] is constraint
        assert kwargs['options'] is observed_options
        result = original(*args, **kwargs)
        identities['result'] = result
        return result

    def execute():
        backend.milp = delegate
        try:
            with observer.installed(pool=WorkPool(256_000_000), emit=events.append,
                                    enabled=True) as report:
                result = backend.milp(c=c, integrality=integral, bounds=bounds,
                                     constraints=constraint, options=observed_options)
            assert backend.milp is delegate
            return result, report
        finally:
            backend.milp = original

    (result, report), observed_stats = measured(execute, observe=lambda _: None)
    assert result is identities['result']
    assert plain.status == result.status == 0
    assert result.fun == plain.fun
    np.testing.assert_array_equal(result.x, plain.x)
    np.testing.assert_array_equal(A@result.x >= lower-1e-12, True)
    np.testing.assert_array_equal(A@result.x <= upper+1e-12, True)
    assert np.all(result.x >= bounds.lb) and np.all(result.x <= bounds.ub)
    np.testing.assert_array_equal(result.x[integral == 1], np.round(result.x[integral == 1]))
    for current, saved in zip(arrays, before):
        np.testing.assert_array_equal(current, saved)
    # Original SciPy removes disp; the observer must not change that behavior.
    assert observed_options == {k:v for k,v in options.items() if k != 'disp'}
    phases = [e['phase'] for e in events if e['event'] == 'c108_ordinary_phase']
    print(json.dumps(dict(event='c108_fixture_raw_phases', phases=events, report=report)))
    assert phases == observer.EXPECTED
    assert report['restored'] and report['callbacks'] <= 512 and report['events'] == 14
    assert sys.monitoring.get_tool(observer.TOOL) is None
    assert sys.monitoring.get_events(observer.TOOL) == 0
    assert all(sys.monitoring.get_local_events(observer.TOOL, code) == 0
               for code in observer.BOUNDARIES)
    assert plain_stats['measured_transient_gate'] and observed_stats['measured_transient_gate']
    print(json.dumps(dict(event='c108_fixture_passed', blocks=blocks, continuous=128,
        binary=64, equalities=64, inequalities=64, phases=events, report=report,
        plain_measurement=plain_stats, observed_measurement=observed_stats,
        original_arguments_and_result_identity=True, input_arrays_unchanged=True)))

