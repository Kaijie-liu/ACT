"""Complete raw-DAG F4 mask/cost census, without numerical transforms or admission.

All aligned output origins, including borders and empty tiles, are retained.
The all-nonzero transformed-kernel assumption gives an UPPER packet cost, not
an actual circuit or a whole-SparseHZ physical cost.  Original coordinate
survival, coalescence, native coefficients, old reserves and full source costs
remain unproved.  No existing reserve is silently initialized to zero.
"""
from collections import Counter
import numpy as np

from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831.c119_denominator_f4_v1 import AT, BT


_INPUT = tuple(sum(1 << (6*i+j) for i in range(6) for j in range(6)
                   if BT[t][i]*BT[u][j]) for t in range(6) for u in range(6))
_OUTPUT = tuple(sum(1 << (4*i+j) for i in range(4) for j in range(4)
                    if AT[i][t]*AT[j][u]) for t in range(6) for u in range(6))
_PATCH = tuple(sum(1 << (6*(i+a)+j+b) for a in range(3) for b in range(3))
               for i in range(4) for j in range(4))
BILL_FIELDS = (
    'direct_nnz', 'nnz_upper', 'new_factors', 'kept_v', 'kept_m',
    'output_rows', 'rows', 'nnz_saving_lower', 'byte_saving_lower',
    'entry_delta_upper', 'new_emission_work_upper',
    'required_auxiliary_headroom', 'required_emission_headroom',
    'required_positive_packet_entry_headroom')


def footprints(parent_mask, output_mask, padding, *, pool, enabled=False):
    """Pack every 6x6 input/4x4 output footprint, with no width-64 restriction."""
    if not enabled:
        return None
    parent, output = np.asarray(parent_mask), np.asarray(output_mask)
    if (parent.dtype != np.dtype(bool) or output.dtype != np.dtype(bool)
            or parent.ndim != 3 or output.ndim != 3
            or any(not 1 <= int(v) <= np.iinfo(np.int32).max
                   for v in (*parent.shape, *output.shape))
            or not isinstance(padding, tuple) or len(padding) != 2
            or any(type(v) is not int or v < 0 for v in padding)):
        raise ValueError('complete ordinary boolean masks and nonnegative integer padding required')
    channels, height, width = map(int, parent.shape)
    filters, outheight, outwidth = map(int, output.shape)
    py, px = padding
    if outheight != height+2*py-2 or outwidth != width+2*px-2:
        raise ValueError('complete original stride1 padded geometry differs')
    tiles = ((outheight+3)//4)*((outwidth+3)//4)
    # Includes every input/output bit read, cast, shift and OR, plus all mask
    # and position allocation.  Charge before inspecting mask coefficients.
    pool.charge('c121_complete_f4_footprint_extraction',
                tiles*(256+160*channels+80*filters))
    inputs = np.zeros((tiles, channels), dtype=np.uint64)
    outputs = np.zeros((tiles, filters), dtype=np.uint16)
    positions = np.empty((tiles, 2), dtype=np.int32)
    tile = 0
    for y in range(0, outheight, 4):
        for x in range(0, outwidth, 4):
            positions[tile] = y, x
            for i in range(6):
                sy = y-py+i
                if not 0 <= sy < height:
                    continue
                for j in range(6):
                    sx = x-px+j
                    if 0 <= sx < width:
                        inputs[tile] |= parent[:, sy, sx].astype(np.uint64) << np.uint64(6*i+j)
            for i in range(min(4, outheight-y)):
                for j in range(min(4, outwidth-x)):
                    outputs[tile] |= output[:, y+i, x+j].astype(np.uint16) << np.uint16(4*i+j)
            tile += 1
    return dict(input_masks=inputs, output_masks=outputs, positions=positions)


def _histograms(input_masks, output_masks, *, pool, header, fixed):
    inputs, outputs = np.asarray(input_masks), np.asarray(output_masks)
    if (inputs.ndim != 1 or outputs.ndim != 1
            or inputs.dtype != np.dtype(np.uint64)
            or outputs.dtype != np.dtype(np.uint16)
            or not len(inputs) or not len(outputs)):
        raise ValueError('complete nonempty uint64/uint16 F4 footprints required')
    pool.charge(header, fixed+12*(len(inputs)+len(outputs)))
    if np.any(inputs >> np.uint64(36)):
        raise ValueError('input footprint extends beyond 6x6')
    return Counter(map(int, inputs)), Counter(map(int, outputs))


def dense_direct(input_masks, output_masks, *, pool, enabled=False):
    """Count all raw original pairs plus pivots, CONDITIONAL on dense kernel/IDs.

    Every original kernel coefficient must separately be proved nonzero, and
    raw input/output coordinates must survive injectively in the actual source.
    This does not inspect or prove either precondition by itself.
    """
    if not enabled:
        return None
    inputs, outputs = _histograms(
        input_masks, output_masks, pool=pool,
        header='c121_complete_dense_direct_histogram_headers', fixed=128)
    pool.charge('c121_complete_original_pair_class_count',
                4*16*(len(inputs)+len(outputs)))
    direct = sum(mask.bit_count()*count for mask, count in outputs.items())
    for position, patch in enumerate(_PATCH):
        used_outputs = sum(count for mask, count in outputs.items() if (mask >> position) & 1)
        used_inputs = sum((mask & patch).bit_count()*count for mask, count in inputs.items())
        direct += used_outputs*used_inputs
    return int(direct)


def upper_bill(input_masks, output_masks, direct_nnz, *, pool, enabled=False):
    """All-retained C120 topology upper bound and declared PACKET bill only.

    V rows are retained whenever a raw nonzero form has any demanded M use;
    M rows are retained whenever active channels and output users exist.
    Cancellation or zero transformed coefficients can only remove these rows
    or terms.  A losing bound is inconclusive, never proof F4 cannot win.
    """
    if not enabled:
        return None
    inputs, outputs = _histograms(
        input_masks, output_masks, pool=pool,
        header='c121_complete_retained_f4_histogram_headers', fixed=512)
    if direct_nnz is not None and type(direct_nnz) is not int:
        raise ValueError('exact conditional direct count or explicit unknown required')
    pool.charge('c121_complete_36_component_observed_class_counts',
                8*36*(len(inputs)+len(outputs)))
    output_rows = sum(mask.bit_count()*count for mask, count in outputs.items())
    if direct_nnz is not None and direct_nnz < output_rows:
        raise ValueError('conditional direct count omits original output pivots')
    nv = nm = vnnz = mnnz = output_terms = 0
    for input_support, output_support in zip(_INPUT, _OUTPUT, strict=True):
        active_channels = input_terms = 0
        for mask, count in inputs.items():
            terms = (mask & input_support).bit_count()
            active_channels += count*bool(terms)
            input_terms += count*terms
        active_filters = uses = 0
        for mask, count in outputs.items():
            needed = (mask & output_support).bit_count()
            active_filters += count*bool(needed)
            uses += count*needed
        if active_channels and active_filters:
            nv += active_channels
            vnnz += input_terms+active_channels
            nm += active_filters
            mnnz += active_filters*(active_channels+1)
            output_terms += uses
    nnz = vnnz+mnnz+output_terms+output_rows
    aux = nv+nm
    rows = aux+output_rows
    new_bytes = 12*nnz+88*aux+32*output_rows+72
    old_bytes = None if direct_nnz is None else 12*direct_nnz+16*output_rows+8
    nnz_saving = None if direct_nnz is None else direct_nnz-nnz
    byte_saving = None if old_bytes is None else old_bytes-new_bytes
    entry_delta = None if direct_nnz is None else 2*(nnz-direct_nnz)+13*aux+3*output_rows+8
    emission = 64*rows+16*(nnz+rows)
    qualified = bool(direct_nnz is not None and nnz_saving > 0
                     and byte_saving > 0 and entry_delta <= 0)
    bill = dict(
        new_factors=int(aux), rows=int(rows), output_rows=int(output_rows),
        nnz_upper=int(nnz), direct_nnz=direct_nnz, nnz_saving_lower=nnz_saving,
        declared_old_bytes=old_bytes, declared_new_bytes_upper=int(new_bytes),
        byte_saving_lower=byte_saving, entry_delta_upper=entry_delta,
        new_emission_work_upper=int(emission), kept_v=int(nv), kept_m=int(nm),
        v_nnz_upper=int(vnnz), m_nnz_upper=int(mnnz), output_terms_upper=int(output_terms),
        required_auxiliary_headroom=int(aux), required_emission_headroom=int(emission),
        required_positive_packet_entry_headroom=None if entry_delta is None else max(0, entry_delta))
    return dict(
        topology_qualified=qualified,
        reason='conditional_packet_candidate' if qualified else
               'original_direct_count_unknown' if direct_nnz is None else
               'not_certified_by_upper_bound',
        bill=bill, packet_bill_only=True,
        actual_whole_HZ_physical_reduction_unproved=True,
        original_coordinate_survival_unproved=True,
        raw_direct_count_requires_injective_original_ids=True,
        transformed_kernel_density_unproved=True,
        transformed_all_nonzero_used_only_for_upper_bound=True,
        actual_global_reserves_unbound=True,
        numeric_admission=False, actual_global_admission=False, formal_gain=0)


def census(nodes, *, pool, enabled=False):
    """Inspect every actual DAG node and every eligible active operator tile.

    Returns a JSON-safe report and newly owned numeric footprints/bill tables.
    Zero-demand eligible operators remain in the eligibility record; all tiles
    of every active operator remain in evidence even when their demand is zero.
    There is deliberately no selection without authenticated actual old source
    reserves.  Node indices identify evidence, never an eligibility menu.
    """
    if not enabled:
        return None
    if not isinstance(nodes, (list, tuple)):
        raise ValueError('complete topologically ordered DAG nodes required')
    pool.charge('c121_complete_DAG_geometry_and_eligibility_headers', 128*len(nodes))
    records, operators, eligibility, evidence = [], [], [], {}
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ValueError('complete DAG node dictionary required')
        if node.get('kind') != 'op':
            eligibility.append(dict(node=index, eligible=False, reason='not_operator'))
            continue
        op = node.get('op')
        if (type(op) is not ImplicitConv2DOp or op._stride != (1, 1)
                or op._dilation != (1, 1) or op._groups != 1
                or op._kernel.shape[-2:] != (3, 3)
                or op.input_shape[0] != 1 or op.output_shape[0] != 1):
            eligibility.append(dict(node=index, eligible=False, reason='outside_F4_operator_structure'))
            continue
        parents = node.get('parents')
        if (not isinstance(parents, (list, tuple)) or len(parents) != 1
                or type(parents[0]) is not int or not 0 <= parents[0] < index):
            raise ValueError('eligible original operator must have one earlier DAG parent')
        parent = nodes[parents[0]]
        channels, height, width = map(int, op.input_shape[1:])
        filters, outheight, outwidth = map(int, op.output_shape[1:])
        wanted, parent_wanted = np.asarray(node.get('needed')), np.asarray(parent.get('needed'))
        if (wanted.dtype != np.dtype(bool) or parent_wanted.dtype != np.dtype(bool)
                or wanted.shape != (filters*outheight*outwidth,)
                or parent_wanted.shape != (channels*height*width,)
                or node.get('width') != wanted.size or parent.get('width') != parent_wanted.size):
            raise ValueError('complete actual original boolean demand maps required')
        pool.charge('c121_complete_eligible_output_demand_scan', 3*int(wanted.size))
        needed_outputs = int(np.count_nonzero(wanted))
        operator = dict(node=index, parent=parents[0], input_shape=[channels, height, width],
                        output_shape=[filters, outheight, outwidth],
                        padding=list(map(int, op._padding)), needed_outputs=needed_outputs,
                        kernel_scanned=False, tiles=0, active=bool(needed_outputs))
        operators.append(operator)
        eligibility.append(dict(node=index, eligible=True, active=bool(needed_outputs),
                                reason='active_F4_structure' if needed_outputs else 'zero_output_demand'))
        if not needed_outputs:
            continue
        if op._row_mask is not None:
            row_mask = np.asarray(op._row_mask)
            if row_mask.dtype != np.dtype(bool) or row_mask.shape != wanted.shape:
                raise ValueError('complete original operator row mask required')
            pool.charge('c121_complete_original_row_mask_demand_check', 3*int(wanted.size))
            if np.any(wanted & ~row_mask):
                raise ValueError('original demanded output contradicts disabled operator row')
        operator['original_row_mask_demand_compatible'] = True
        kernel = np.asarray(op._kernel)
        if (kernel.shape != (filters, channels, 3, 3)
                or kernel.dtype.kind not in 'iuf' or kernel.dtype.itemsize > 8):
            raise ValueError('complete ordinary original native 3x3 kernel required')
        pool.charge('c121_complete_original_kernel_finiteness_and_density', 128+4*int(kernel.size))
        if not np.all(np.isfinite(kernel)):
            raise ValueError('nonfinite original kernel is not a conditional dense source')
        original_nonzero = int(np.count_nonzero(kernel))
        dense = original_nonzero == int(kernel.size)
        pool.charge('c121_complete_active_parent_demand_scan', 2*int(parent_wanted.size))
        operator.update(kernel_scanned=True, original_weights=int(kernel.size),
                        original_nonzero=original_nonzero, original_kernel_all_nonzero=dense,
                        needed_parent_coordinates=int(np.count_nonzero(parent_wanted)))
        masks = footprints(parent_wanted.reshape(channels, height, width),
                           wanted.reshape(filters, outheight, outwidth),
                           tuple(map(int, op._padding)), pool=pool, enabled=True)
        tile_count = len(masks['positions'])
        pool.charge('c121_complete_tile_numeric_evidence',
                    128+16*tile_count*(len(BILL_FIELDS)+2))
        masks['bills'] = np.empty((tile_count, len(BILL_FIELDS)), dtype=np.int64)
        masks['direct_known'] = np.full(tile_count, dense, dtype=bool)
        masks['topology_candidates'] = np.zeros(tile_count, dtype=bool)
        operator['tiles'] = tile_count
        evidence[index] = masks
        for tile, (y, x) in enumerate(masks['positions'].tolist()):
            inputs, outputs = masks['input_masks'][tile], masks['output_masks'][tile]
            direct = dense_direct(inputs, outputs, pool=pool, enabled=True) if dense else None
            cost = upper_bill(inputs, outputs, direct, pool=pool, enabled=True)
            for column, name in enumerate(BILL_FIELDS):
                value = cost['bill'][name]
                masks['bills'][tile, column] = -1 if value is None else int(value)
            masks['topology_candidates'][tile] = cost['topology_qualified']
            records.append(dict(node=index, y=y, x=x, tile=tile, cost=cost,
                                original_kernel_all_nonzero=dense,
                                count_scope='conditional_raw_DAG_packet_not_current_quotient_or_whole_HZ'))
    report = dict(
        nodes_scanned=len(nodes), eligibility=eligibility, operators=operators,
        eligible_operators=len(operators),
        active_operators=sum(int(operator['active']) for operator in operators),
        total_tiles=len(records),
        topology_candidates=sum(int(record['cost']['topology_qualified']) for record in records),
        zero_demand_tiles=sum(int(record['cost']['bill']['output_rows'] == 0) for record in records),
        records=records, bill_fields=list(BILL_FIELDS), unknown_bill_sentinel=-1,
        unknown_sentinel_disambiguated_by_direct_known=True,
        selected_positions=[], no_selection_without_authenticated_old_reserves=True,
        all_structurally_eligible_nodes_inspected=True,
        all_active_operator_origins_including_borders_and_no_hit_tiles_retained=True,
        complete_original_kernel_scan_for_each_active_operator=True,
        no_kernel_transform_or_numeric_circuit_constructed=True,
        packet_bill_only=True, actual_whole_HZ_physical_reduction_unproved=True,
        original_coordinate_survival_unproved=True, actual_global_reserves_unbound=True,
        numeric_admission=False, actual_global_admission=False, formal_gain=0)
    return report, evidence
