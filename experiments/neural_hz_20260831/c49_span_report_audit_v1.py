"""Independent complete C31->C49 report/model/bit proof on original source.

The reference scans EVERY original row and all product-input images. This
is an offline diagnostic, not a replacement for the NEW full close checker.
"""
import copy
import json
import numpy as np
from experiments.neural_hz_20260831.c48_alias_span_census_v1 import assess
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest


def audit(candidate,original,original_hz,maps,*,pool,enabled=False):
    if not enabled:return None
    candidate.validate();original.validate()
    before=candidate.fingerprint();old_before=original.fingerprint()
    if set(maps)!=set(('eq_roots','eq_scales','ineq_roots','ineq_scales','def_rows')):
        raise ValueError('complete independent original oracle maps required')
    pool.charge('c49_complete_report_model_and_fields',1024+64*len(original.report['node_counts']))
    check=assess(original_hz,old_nc=original.old_n_cont,logical_nc=original.logical_n_cont,
        old_eq=original.old_n_eq,eq_roots=maps['eq_roots'],def_rows=maps['def_rows'],
        output_slots=original_hz.Gc.indices,pool=pool,enabled=True)
    expected=copy.deepcopy(original.report);alias=expected['alias_quotient']
    old_scan=check['original_C31_incidence_scan_work']
    if (alias['work_parts']['continuous_incidence_scan']!=old_scan
            or alias['owned_continuous_coefficients_inspected']!=old_scan
            or alias['local_aliases']!=check['local_aliases']
            or alias['alias_products_checked']!=check['all_original_hits']
            or alias['product_certification']['rows']!=check['all_original_hit_rows']):
        raise ValueError('independent C31 population differs from full original oracle')
    retained=old_scan-check['routing']['full_row_coefficient_scans_removed']
    extra={'c49_private_source_boundary_and_complete_routing_report':512,
        **{k:v for k,v in check['new_routing_parts'].items() if v}}
    if any(k in alias['work_parts'] for k in extra) or 'source_span_routing' in alias:
        raise ValueError('reference is not the independently checked original C31 generator')
    delta=sum(extra.values())+retained-old_scan
    alias['work_parts']['continuous_incidence_scan']=retained
    alias['work_parts'].update(extra)
    alias['coupled_extra_work']+=delta
    alias['owned_continuous_coefficients_inspected']=retained
    alias['original_continuous_coefficient_population']=old_scan
    alias['source_span_routing']=check['routing']
    expected['total_work_upper']+=delta
    expected['largest_branch_work_upper']+=delta
    expected['source_span_index_transient_reserve_bytes']=13*(original.logical_n_cont+16384)+4
    if json.dumps(candidate.report,sort_keys=True,allow_nan=False)!=json.dumps(expected,sort_keys=True,allow_nan=False):
        keys=[k for k in set(expected)|set(candidate.report) if expected.get(k)!=candidate.report.get(k)]
        raise ValueError('complete new routing report differs from independently checked original: '+str(sorted(keys)))
    names=('keep','eq_roots','eq_scales','ineq_roots','ineq_scales','def_rows','owners','uid_slabs')
    pool.charge('c49_complete_map_owner_UID_bit_comparison',4*sum(getattr(candidate,k).size for k in names))
    for k in names:
        a,b=getattr(candidate,k),getattr(original,k)
        if a.shape!=b.shape or a.dtype!=b.dtype or not np.array_equal(a.view(np.uint8),b.view(np.uint8)):
            raise ValueError('complete new routing changed original coordinate/owner/UID bits: '+k)
    if source_digest(candidate.hz)!=source_digest(original.hz):
        raise ValueError('complete new generator HZ differs from independently checked C31')
    if candidate.fingerprint()!=before or original.fingerprint()!=old_before:
        raise ValueError('source state changed during independent routing report proof')
    return dict(schema='c49_complete_original_source_routing_report_proof_v1',
        all_report_fields_checked=True,all_HZ_map_owner_UID_bits_equal=True,
        full_original_census=check,extra_parts=extra,complete_work_delta=delta,
        old_whole_work=original.report['total_work_upper'],new_whole_work=expected['total_work_upper'],
        old_branch_work=original.report['largest_branch_work_upper'],
        new_branch_work=expected['largest_branch_work_upper'],strict_component_payment=delta<0,
        old_receipt_cannot_bind_new_report=True,new_full_source_proof_still_required=True,
        native_or_full_LIVE_payment_proved=False,formal_gain=0)
