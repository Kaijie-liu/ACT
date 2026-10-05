"""Complete C67 source bound minus one genuinely absent coefficient scan."""
from experiments.neural_hz_20260831.c67_source_budget_v1 import bound as previous_bound


def bound(saved, legacy, reference, *, pool, enabled=False):
    if not enabled:
        return None
    result = previous_bound(saved, legacy, reference, pool=pool, enabled=True)
    pool.charge('c69_independent_original_coefficient_inventory', 1024)
    report = legacy['report']
    original = int(report['logical_coefficient_credit_preflight'])
    emitted = report['prepared_encoding']
    if (original != emitted['logical_input_coefficients']
            or emitted['actual_omitted_post_magnitude_coefficients'] < original):
        raise ValueError('complete bound original logical inventory differs')
    # The externally checked C31 graph/report fixes this ORIGINAL inventory;
    # no new target result, normalized nnz or extra radix coefficient is used.
    result['work_parts']['c69_finite_inverse_producer_binding'] = 128
    result['whole_base_work'] -= original
    result['coupled_extra_upper'] += 128
    result['whole_work_upper'] += 128 - original
    result['branch_work_upper'] += 128
    result['whole_headroom'] = 256_000_000 - result['whole_work_upper']
    result['branch_headroom'] = 200_000_000 - result['branch_work_upper']
    result.update(schema='c69_complete_prepared_finite_construction_bound_v1',
        original_logical_coefficient_inventory=original,
        one_removed_finite_scan_per_original_coefficient=True,
        RHS_extra_radix_and_native_credits=0,
        original_conservative_branch_encoding_price_unchanged=True,
        work_caps_fit=result['whole_headroom'] >= 0 and result['branch_headroom'] >= 0)
    return result
