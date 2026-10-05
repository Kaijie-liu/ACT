"""Complete C69 source bound with two actually removed repeated comparisons."""
from experiments.neural_hz_20260831.c69_source_budget_v1 import bound as previous_bound


def bound(saved,legacy,reference,*,pool,enabled=False):
    if not enabled:return None
    result=previous_bound(saved,legacy,reference,pool=pool,enabled=True)
    pool.charge('c97_complete_original_power_inventory',1024)
    original=result['original_logical_coefficient_inventory']
    rows=legacy['report']['prepared_encoding']['logical_input_rows']
    if original!=legacy['report']['prepared_encoding']['logical_input_coefficients']:
        raise ValueError('original source logical power population differs')
    result['work_parts']['c97_once_power_producer_binding']=128
    result['work_parts']['c97_actual_logical_power_counter']=rows
    result['whole_base_work']-=2*original
    result['coupled_extra_upper']+=128+rows
    result['whole_work_upper']+=128+rows-2*original
    result['branch_work_upper']+=128+rows
    result['whole_headroom']=256_000_000-result['whole_work_upper']
    result['branch_headroom']=200_000_000-result['branch_work_upper']
    result.update(schema='c97_complete_once_power_construction_bound_v1',
        original_once_checked_power_inventory=original,
        original_logical_power_counter_rows=rows,
        removed_duplicate_comparisons_per_original_coefficient=2,
        power_RHS_extra_radix_and_native_credits=0,
        work_caps_fit=result['whole_headroom']>=0 and result['branch_headroom']>=0)
    return result
