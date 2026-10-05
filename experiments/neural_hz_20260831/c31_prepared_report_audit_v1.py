"""Full report-field comparison against independently checked original C24.

This is a diagnostic cost/identity proof, never a replacement for the complete
new original-DAG/box/alias/ownership/UID close() proof. Old receipts cannot
bind the new report. No old report is returned as the new generation result.
"""

import copy
import json
import numpy as np
from experiments.neural_hz_20260831.c5_zero_suffix_audit_v1 import zero_transfer_reference
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest


def audit(candidate,original,*,pool,max_work=256_000_000,max_branch_work=200_000_000):
    candidate.validate();original.validate()
    if (type(max_work) is not int or type(max_branch_work) is not int
            or not 0<=max_work<=256_000_000 or not 0<=max_branch_work<=200_000_000):
        raise ValueError('invalid original fixed report ceilings')
    old=original.report
    pool.charge('independent_complete_prepared_report_fields',512+64*len(old['node_counts']))
    base=zero_transfer_reference(candidate.expression)
    old_coefficients=sum(int(getattr(base,k).nnz) for k in ('Ac','Ab','Auc','Aub'))
    if old['old_predicate_work_upper']!=12*(old_coefficients+base.n_eq+base.n_ineq):
        raise ValueError('independent original predicate work/source geometry differs')
    expected=copy.deepcopy(old)
    main_credit=0
    for item in expected['node_counts']:
        coefficients=item['continuous_edges']+item['binary_edges']+item['auxiliaries']
        item['encoding_work_upper']-=coefficients;main_credit+=coefficients
    credit=old_coefficients+main_credit
    logical=base.n_eq+base.n_ineq+old['auxiliaries']
    physical=old['n_eq']+old['n_ineq']+old['alias_quotient']['selected_aliases']
    attempts=logical+old['radix_auxiliaries']+old['packed_logical_rows']
    if physical!=logical+old['radix_auxiliaries']:
        raise ValueError('complete physical/logical/radix report topology differs')
    extra_parts={'prepared_row_decision_and_head':16*attempts,
        'prepared_logical_input_counters':4*logical,'prepared_head_metadata_retirement':16,
        'prepared_report_accounting':8*len(old['node_counts'])+32}
    extra=sum(extra_parts.values())
    expected.update(encoding_operations_per_entry=None,
        encoding_price_rule='C24_original_prices_minus_one_proved_post_abs_per_logical_coefficient',
        logical_coefficient_credit_preflight=credit,original_branch_encoding_price_retained=True,
        old_predicate_work_upper=old['old_predicate_work_upper']-old_coefficients,
        affine_work_upper=old['affine_work_upper']-main_credit,
        whole_base_work=old['whole_base_work']-credit,
        total_work_upper=old['total_work_upper']-credit+extra,
        largest_branch_work_upper=old['largest_branch_work_upper']+extra)
    expected['prepared_encoding']=dict(schema='c31_actual_prepared_owned_encoding_v1',
        logical_input_rows=logical,logical_input_coefficients=credit,actual_preparation_attempts=attempts,
        physical_rows_before_alias=physical,
        actual_omitted_post_magnitude_coefficients=old['alias_quotient']['old_predicate_nnz'],
        credited_logical_coefficients=credit,
        extra_radix_coefficients_not_credited=old['alias_quotient']['old_predicate_nnz']-credit,
        discarded_head_entries=physical,head_metadata_retained_after_alias=False,
        direct_store_canonical_UID_hooks=True,default_off=True,formal_gain=0)
    alias=expected['alias_quotient']
    if old['alias_quotient']['coupled_extra_capacity']!=min(max_work-old['whole_base_work'],max_branch_work-old['branch_base_work']):
        raise ValueError('original report cap/configuration not independently bound')
    alias['coupled_extra_capacity']=min(max_work-expected['whole_base_work'],max_branch_work-expected['branch_base_work'])
    alias['coupled_extra_work']+=extra
    if any(k in alias['work_parts'] for k in extra_parts):raise ValueError('new operations already in old report')
    alias['work_parts'].update(extra_parts)
    if json.dumps(candidate.report,sort_keys=True,allow_nan=False)!=json.dumps(expected,sort_keys=True,allow_nan=False):
        differences=[k for k in set(expected)|set(candidate.report) if expected.get(k)!=candidate.report.get(k)]
        raise ValueError('complete new report differs from proved source/counter model: '+str(sorted(differences)))
    names=('keep','eq_roots','eq_scales','ineq_roots','ineq_scales','def_rows','owners','uid_slabs')
    pool.charge('independent_all_prepared_map_and_owner_bits',4*sum(getattr(candidate,k).size for k in names))
    for k in names:
        a,b=getattr(candidate,k),getattr(original,k)
        if a.shape!=b.shape or a.dtype!=b.dtype or not np.array_equal(a.view(np.uint8),b.view(np.uint8)):
            raise ValueError('new complete original map/owner/UID changed: '+k)
    if source_digest(candidate.hz)!=source_digest(original.hz):
        raise ValueError('complete prepared HZ differs from independently checked original')
    return dict(schema='c31_complete_prepared_report_transfer_v1',all_report_fields_checked=True,
        all_HZ_map_owner_UID_bits_equal=True,logical_coefficients=credit,logical_rows=logical,
        actual_preparation_attempts=attempts,physical_emitted_rows=physical,
        removed_logical_post_magnitude_operations=credit,new_extra_operations=extra,
        new_extra_parts=extra_parts,old_whole_work=old['total_work_upper'],
        new_whole_work=expected['total_work_upper'],new_branch_work=expected['largest_branch_work_upper'],
        unchanged_old_tariffs_and_conservative_branch=True,
        old_receipt_cannot_bind_new_report=True,full_new_source_proof_still_required=True,formal_gain=0)
