"""Source-derived bound versus the actually executed frozen small generator."""
import pytest
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c62_precision_plan_v1 import plan
from experiments.neural_hz_20260831.c69_source_budget_v1 import bound
from experiments.neural_hz_20260831.c69_birth_emission_v1 import lift
from experiments.neural_hz_20260831.test_c62_physical_boundary_v2 import complete,old_fields


@pytest.mark.parametrize('kind',['chain','shared','conv_disjoint'])
def test_complete_bound_dominates_every_actual_fresh_generator_charge(kind):
    _,saved=complete(kind);legacy,_=old_fields(saved);pool=WorkPool(256_000_000)
    reference=plan(saved,pool=pool,enabled=True);report=bound(saved,legacy,reference,pool=pool,enabled=True)
    candidate=lift(saved['expression'],saved['keep'],enabled=True)['fields']['report']
    actual=candidate['alias_quotient']
    assert candidate['whole_base_work']==report['whole_base_work'] and candidate['branch_base_work']==report['branch_base_work']
    assert candidate['total_work_upper']<=report['whole_work_upper']
    assert set(actual['work_parts'])<=set(report['work_parts'])
    for key,value in actual['work_parts'].items():assert value<=report['work_parts'][key],key
    assert actual['optimum']==report['selected'] and actual['new_predicate_nnz']==report['expected_predicate_nnz']
    assert actual['rewritten_rows']==report['rewritten_rows'] and actual['row_gauges']==report['row_gauges']
    assert actual['identity_sha256']==report['expected_identity_sha256']


def test_default_off_and_no_unpaid_source_budget_walk():
    assert bound(None,None,None,pool=None) is None
    _,saved=complete('chain');legacy,_=old_fields(saved);reference=plan(saved,pool=WorkPool(256_000_000),enabled=True)
    with pytest.raises(MemoryError):bound(saved,legacy,reference,pool=WorkPool(0),enabled=True)
