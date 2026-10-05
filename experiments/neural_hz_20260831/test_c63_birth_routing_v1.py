"""Complete generic-incidence oracle for the new birth-block routing."""
import numpy as np
import pytest
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c62_precision_plan_v1 import plan as generic
from experiments.neural_hz_20260831.c63_precision_plan_v1 import plan
from experiments.neural_hz_20260831.c63_birth_blocks_v1 import route_rows
from experiments.neural_hz_20260831.test_c62_physical_boundary_v2 import complete


def compare(saved):
    left=generic(saved,pool=WorkPool(256_000_000),enabled=True)
    right=plan(saved,pool=WorkPool(256_000_000),enabled=True)
    for key in ('selected','roots','erased'):
        assert np.array_equal(left[key],right[key])
    for key in ('weights','parents','local','tags','numerators'):
        assert left[key]==right[key]
    for name in ('Ac','Auc','Gc'):assert np.array_equal(left['raw_hits'][name],right['raw_hits'][name])
    a=left['report'];b=dict(right['report']);route=b.pop('birth_routing')
    assert a==b and route['complete_path_slot_output_binding']
    routed,_=route_rows(saved,np.array(sorted(left['parents']),np.int64),pool=WorkPool(256_000_000),enabled=True)
    own=np.zeros(saved['hz'].n_eq,bool)
    for v in left['parents']:own[saved['eq_roots'][saved['old_n_eq']+v-saved['old_n_cont']]]=True
    assert not np.any(left['raw_hits']['Ac'] & ~own & ~routed)
    return right


@pytest.mark.parametrize('kind',['chain','shared','conv','conv_disjoint'])
def test_complete_birth_route_equals_generic_incidence_and_boundary_solution(kind):
    _,saved=complete(kind);compare(saved)


def test_default_off_and_unchanged_prepaid_budget():
    assert plan(None,pool=None) is None and route_rows(None,None,pool=None) is None
    _,saved=complete('chain')
    with pytest.raises(MemoryError):plan(saved,pool=WorkPool(0),enabled=True)

