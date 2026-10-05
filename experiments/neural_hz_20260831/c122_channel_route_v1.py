"""Demand-based channel routing: one conditional mixed direct/F4 packet.

For each channel, the fixed-overhead byte increment is w = 12*q + 88*a,
where q = l + a + b - d.  Select ALL and only w < 0 channels once.  Then
tighten the same set's component overhead; no second selector is permitted.
The coefficients are not inspected or transformed here.  Dense original
kernels, injective surviving coordinates, native exactness, full HZ physical
costs and authenticated shared old reserves remain separate prerequisites.

The complete prepaid tariff is 4096 + 1024*(C+K) + 16*(7*C).  The first part
covers headers, every 16/36-bit support operation, Python integer arithmetic,
second-pass selection/active-component scan, reports and boolean evidence;
the last part covers the complete owned int64 statistics table.  For all 52
C=K=128 tiles this is 14,589,952 units, below the declared 16M census budget.
Input access and all evidence allocations occur only after this prepayment.
This tariff is not a fee for source construction, coefficient transforms,
native proof, full retained-root accounting or evidence serialization.
"""

import numpy as np

from experiments.neural_hz_20260831.c121_f4_mask_cost_v1 import _INPUT, _OUTPUT, _PATCH


STATS_FIELDS = ('d', 'a', 'l', 'b', 'q', 'w', 'e')
_I64 = np.iinfo(np.int64)
_DIM_LIMIT = int(np.iinfo(np.int32).max)


def work_fee(channels, filters):
    """Exact complete call tariff, based only on validated ordinary shapes."""
    if (type(channels) is not int or type(filters) is not int
            or not 1 <= channels <= _DIM_LIMIT
            or not 1 <= filters <= _DIM_LIMIT):
        raise ValueError('positive ordinary int32-sized channel/filter counts required')
    return 4096 + 1024*(channels+filters) + 16*len(STATS_FIELDS)*channels


def _bill(*, direct, output_rows, nv, nm, input_terms, connections,
          output_terms, residual_terms, route_entries, noop):
    """Declared PACKET upper only; the routing bool array is an extra cost."""
    if noop:
        old_bytes = None if direct is None else 12*direct+16*output_rows+8
        return dict(
            kept_v=0, kept_m=0, new_factors=0, output_rows=output_rows,
            rows=0, nnz_upper=direct, direct_nnz=direct, delta_nnz_upper=0,
            nnz_saving_lower=0, byte_delta_upper=0, byte_saving_lower=0,
            declared_old_bytes=old_bytes, declared_new_bytes_upper=old_bytes,
            entry_delta_upper=0, new_emission_work_upper=0,
            v_nnz_upper=0, m_nnz_upper=0, output_terms_upper=0,
            residual_direct_terms=residual_terms,
            byte_saving_after_route_mask_lower=0,
            entry_delta_with_route_mask_upper=0,
            required_auxiliary_headroom=0, required_emission_headroom=0,
            required_positive_packet_entry_headroom=0, literal_noop=True)
    if direct is None:
        raise ValueError('nonempty route requires an authenticated conditional direct count')
    aux = nv+nm
    rows = aux+output_rows
    vnnz = input_terms+nv
    mnnz = connections+nm
    nnz = output_rows+residual_terms+vnnz+mnnz+output_terms
    delta = nnz-direct
    byte_delta = 12*delta+88*aux+16*output_rows+64
    entry_delta = 2*delta+13*aux+3*output_rows+8
    emission = 16*nnz+80*rows
    old_bytes = 12*direct+16*output_rows+8
    return dict(
        kept_v=nv, kept_m=nm, new_factors=aux, output_rows=output_rows,
        rows=rows, nnz_upper=nnz, direct_nnz=direct, delta_nnz_upper=delta,
        nnz_saving_lower=-delta, byte_delta_upper=byte_delta,
        byte_saving_lower=-byte_delta, declared_old_bytes=old_bytes,
        declared_new_bytes_upper=old_bytes+byte_delta,
        entry_delta_upper=entry_delta, new_emission_work_upper=emission,
        v_nnz_upper=vnnz, m_nnz_upper=mnnz, output_terms_upper=output_terms,
        residual_direct_terms=residual_terms,
        byte_saving_after_route_mask_lower=-byte_delta-route_entries,
        entry_delta_with_route_mask_upper=entry_delta+route_entries,
        required_auxiliary_headroom=aux, required_emission_headroom=emission,
        required_positive_packet_entry_headroom=max(0, entry_delta+route_entries),
        literal_noop=False)


def route(input_masks, output_masks, direct_nnz, *, pool, enabled=False):
    """Return (JSON-safe report, newly owned numeric evidence), or disabled None.

    A supplied direct_nnz must equal all conditional dense original pair
    occurrences PLUS every original output pivot.  Passing None explicitly
    leaves that original-source condition unknown: statistics can be inspected
    but no channel is selected.  Neither form authenticates actual native HZ
    source coordinates or admits a route against unknown global reserves.
    """
    if not enabled:
        return None
    # Inspect only array headers before the complete scan/allocation charge.
    if (type(input_masks) is not np.ndarray or type(output_masks) is not np.ndarray
            or input_masks.ndim != 1 or output_masks.ndim != 1
            or input_masks.dtype != np.dtype(np.uint64)
            or output_masks.dtype != np.dtype(np.uint16)):
        raise ValueError('complete uint64[C]/uint16[K] ndarray footprints required')
    channels, filters = len(input_masks), len(output_masks)
    fee = work_fee(channels, filters)
    if direct_nnz is not None and (type(direct_nnz) is not int or direct_nnz < 0):
        raise ValueError('nonnegative exact conditional direct count or explicit unknown required')
    pool.charge('c122_complete_channel_route_headers_masks_counts_and_evidence', fee)

    output_rows = 0
    position_uses = [0]*16
    component_filters = [0]*36
    component_uses = [0]*36
    for raw in output_masks:
        mask = int(raw)
        output_rows += mask.bit_count()
        for position in range(16):
            position_uses[position] += (mask >> position) & 1
        for component, support in enumerate(_OUTPUT):
            uses = (mask & support).bit_count()
            component_filters[component] += int(uses != 0)
            component_uses[component] += uses

    stats = np.empty((channels, len(STATS_FIELDS)), dtype=np.int64)
    selected = np.zeros(channels, dtype=bool)
    active_components = np.zeros(36, dtype=bool)
    all_direct_terms = 0
    for channel, raw in enumerate(input_masks):
        mask = int(raw)
        if mask >> 36:
            raise ValueError('input footprint extends beyond 6x6')
        d = sum((mask & patch).bit_count()*position_uses[position]
                for position, patch in enumerate(_PATCH))
        a = l = b = 0
        for component, support in enumerate(_INPUT):
            terms = (mask & support).bit_count()
            if terms and component_filters[component]:
                a += 1
                l += terms
                b += component_filters[component]
        q = l+a+b-d
        w = 12*q+88*a
        e = 2*q+13*a
        values = (d, a, l, b, q, w, e)
        # Arithmetic above is Python integer arithmetic, never int64 wraparound.
        if any(value < int(_I64.min) or value > int(_I64.max) for value in values):
            raise ValueError('conditional per-channel counts exceed exact int64 evidence')
        stats[channel] = values
        all_direct_terms += d
    conditional_direct = output_rows+all_direct_terms
    if direct_nnz is not None and direct_nnz != conditional_direct:
        raise ValueError('conditional direct count differs from all original dense pairs and pivots')

    selected_count = nv = input_terms = connections = selected_direct = 0
    if direct_nnz is not None:
        for channel in range(channels):
            if int(stats[channel, 5]) >= 0:
                continue
            selected[channel] = True
            selected_count += 1
            selected_direct += int(stats[channel, 0])
            nv += int(stats[channel, 1])
            input_terms += int(stats[channel, 2])
            connections += int(stats[channel, 3])
            mask = int(input_masks[channel])
            for component, support in enumerate(_INPUT):
                if component_filters[component] and mask & support:
                    active_components[component] = True

    noop = selected_count == 0
    route_entries = channels if not noop else 0
    residual_terms = all_direct_terms-selected_direct
    shared = dict(direct=direct_nnz, output_rows=output_rows, nv=nv,
                  input_terms=input_terms, connections=connections,
                  residual_terms=residual_terms, route_entries=route_entries,
                  noop=noop)
    fixed_bill = _bill(nm=sum(component_filters), output_terms=sum(component_uses), **shared)
    tight_bill = _bill(
        nm=sum(component_filters[t] for t in range(36) if bool(active_components[t])),
        output_terms=sum(component_uses[t] for t in range(36) if bool(active_components[t])),
        **shared)
    candidate = bool(not noop and tight_bill['nnz_saving_lower'] > 0
                     and tight_bill['byte_saving_after_route_mask_lower'] > 0
                     and tight_bill['entry_delta_with_route_mask_upper'] <= 0)
    report = dict(
        channels=channels, filters=filters, channel_fields=list(STATS_FIELDS),
        selected_channels=selected_count, residual_channels=channels-selected_count,
        active_components=int(np.count_nonzero(active_components)),
        conditional_dense_direct_nnz=conditional_direct, direct_nnz=direct_nnz,
        fixed_bill=fixed_bill, tight_bill=tight_bill,
        route_mask_bytes=route_entries, route_mask_entries=route_entries,
        route_mask_cost_is_additional_to_packet_bill=True,
        evidence_arrays_are_diagnostic_and_require_separate_retained_root_accounting=True,
        conditional_candidate=candidate, topology_qualified=candidate,
        reason=('conditional_packet_candidate' if candidate else
                'original_direct_count_unknown' if direct_nnz is None else
                'empty_selection_literal_noop' if noop else 'not_certified_by_upper_bound'),
        literal_noop=noop, prepaid_work=fee,
        selector='all_channels_with_12q_plus_88a_strictly_negative',
        active_component_tightening_does_not_change_selection=True,
        full_residual_direct_tail_included=True,
        packet_bill_only=True, actual_whole_HZ_physical_reduction_unproved=True,
        original_dense_kernel_unproved=True, original_coordinate_survival_unproved=True,
        raw_direct_count_requires_injective_original_ids=True,
        transformed_kernel_density_unproved=True,
        transformed_all_nonzero_used_only_for_upper_bound=True,
        native_exactness_unproved=True, actual_source_generation_unproved=True,
        actual_global_reserves_unbound=True, no_cross_tile_selector=True,
        numeric_admission=False, actual_global_admission=False, formal_gain=0)
    evidence = dict(channel_stats=stats, selected=selected, active_components=active_components)
    return report, evidence
