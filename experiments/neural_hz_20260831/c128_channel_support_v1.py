"""Default-off exact channel factorization of dense Conv support/ownership.

This changes integer incidence propagation, not numerical convolution or HZ
rows. All old public compute/owners validation, source digests, cache keys,
zero-input handling and CSR traversal are inherited without alteration.

For each fresh nonzero Conv traversal, before inspecting kernel values pay
I=1024+4W+16(KH+KW). The 4W pays full finiteness/density scans and temporary
Boolean values; the axis term pays clipped integer range construction. Dense
execution then prepays 8(NI+NO)+16E. The first term covers complete input range
scans, reductions, masking, allocation, broadcast and stores. Each of E valid
batch/group/stencil incidences has one exact scalar gather/add/store, with its
loop/index operations covered by the composite16 tariff. Invalid coordinates
are excluded using scalar axis intervals, not unpaid full spatial scans.

Non-dense traversal retains the entire old V2 fee in addition to I. Existing
operator/mask authentication traffic is unchanged and is not repriced here.
No transform/source receipt, persistent workspace or new cache is introduced.
"""

import math

import numpy as np

from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831.c24_dense_ownership_v1 import DenseOwnerEngine


INT64_MAX = (1 << 63) - 1
ENTRY_CAP = 64_000_000
BYTE_CAP = 1 << 30
HEADER_FEE = 1024
KERNEL_INSPECTION_FEE = 4
AXIS_FEE = 16
FULL_VECTOR_FEE = 8
INCIDENCE_FEE = 16


def _integer(value, *, minimum):
    if type(value) is not int or not minimum <= value <= INT64_MAX:
        raise ValueError('canonical bounded convolution integer header required')
    return value


def _tuple_header(value, size, *, minimum):
    if type(value) is not tuple or len(value) != size:
        raise ValueError('canonical convolution tuple header required')
    return tuple(_integer(v, minimum=minimum) for v in value)


def _headers(op):
    """Only fixed-size metadata work; no kernel/mask value or spatial scan."""
    if type(op) is not ImplicitConv2DOp:
        raise ValueError('fee plan requires an exact implicit Conv operator')
    ins = _tuple_header(op.input_shape, 4, minimum=1)
    outs = _tuple_header(op.output_shape, 4, minimum=0)
    stride = _tuple_header(op._stride, 2, minimum=1)
    padding = _tuple_header(op._padding, 2, minimum=0)
    dilation = _tuple_header(op._dilation, 2, minimum=1)
    groups = _integer(op._groups, minimum=1)
    kernel = op._kernel
    if (type(kernel) is not np.ndarray or kernel.ndim != 4
            or kernel.dtype.kind not in 'biuf'):
        raise ValueError('complete real numeric kernel required')
    ks = tuple(int(v) for v in kernel.shape)
    if min(ks) <= 0:
        raise ValueError('empty kernel dimensions are not canonical')
    batch, ci, hi, wi = ins
    out_batch, co, ho, wo = outs
    ko, cig, kh, kw = ks
    if (out_batch != batch or co <= 0 or ko != co or ci != cig * groups
            or co % groups):
        raise ValueError('inconsistent complete grouped convolution geometry')
    expected_h = (hi + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
    expected_w = (wi + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
    if (ho, wo) != (expected_h, expected_w):
        raise ValueError('output shape differs from original convolution geometry')
    ni, no, weights = math.prod(ins), math.prod(outs), math.prod(ks)
    if max(ni, no, weights) > ENTRY_CAP:
        raise MemoryError('complete convolution header exceeds numeric entry cap')
    row_mask = op._row_mask
    if (row_mask is not None and
            (type(row_mask) is not np.ndarray or row_mask.dtype != np.dtype(bool)
             or row_mask.shape != (no,))):
        raise ValueError('complete channel-dependent Boolean output row mask required')
    # This is deliberately a conservative bound for simultaneously live input
    # views/copies, masked values, group reductions, accumulators and the owned
    # final result, plus density temporaries. It does not count caller data as
    # free memory or claim a complete process/metadata occupancy measurement.
    workspace_entries = 3 * (ni + no) + 2 * weights
    workspace_bytes = 24 * (ni + no) + 2 * weights
    return dict(input_shape=ins, output_shape=outs, kernel_shape=ks,
        stride=stride, padding=padding, dilation=dilation, groups=groups,
        input_entries=ni, output_entries=no, kernel_entries=weights,
        input_channels_per_group=cig, output_channels_per_group=co // groups,
        inspection_work=HEADER_FEE + KERNEL_INSPECTION_FEE * weights
                        + AXIS_FEE * (kh + kw),
        workspace_entries_upper=workspace_entries,
        workspace_bytes_upper=workspace_bytes)


def _axis_ranges(length, outputs, kernel, stride, padding, dilation):
    """One exact clipped half-open output interval per kernel offset."""
    intervals = []
    visits = 0
    for offset in range(kernel):
        start = min(outputs, max(0, (padding - offset * dilation + stride - 1) // stride))
        stop = max(start, min(outputs, (length - 1 + padding - offset * dilation) // stride + 1))
        intervals.append((start, stop))
        visits += stop - start
    return tuple(intervals), visits


def _finish_plan(header):
    batch, _, hi, wi = header['input_shape']
    _, _, ho, wo = header['output_shape']
    _, _, kh, kw = header['kernel_shape']
    stride, padding, dilation = (header[key] for key in ('stride', 'padding', 'dilation'))
    hr, nh = _axis_ranges(hi, ho, kh, stride[0], padding[0], dilation[0])
    wr, nw = _axis_ranges(wi, wo, kw, stride[1], padding[1], dilation[1])
    incidences = batch * header['groups'] * nh * nw
    dense = FULL_VECTOR_FEE * (header['input_entries'] + header['output_entries']) + INCIDENCE_FEE * incidences
    return dict(header, height_ranges=hr, width_ranges=wr,
        geometric_incidences=incidences, dense_work=dense,
        total_work=header['inspection_work'] + dense,
        complete_kernel_density_not_observed=True,
        fallback_old_traversal_work_not_included=True)


def fee_plan(op):
    """Metadata-only fresh dense-call bound, never a density/cache certificate.

    No source values, masks or coefficient arrays are read or materialized.
    The caller of this descriptive API owns its scalar planning work. Actual
    engine execution independently prepays before constructing its intervals.
    """
    return _finish_plan(_headers(op))


class FactorizedOwnerEngine(DenseOwnerEngine):
    """Uniform structural fast path, explicitly disabled by default."""

    def __init__(self, max_visits=256_000_000, *, enabled=False):
        if type(enabled) is not bool:
            raise ValueError('enabled must be an explicit Boolean')
        super().__init__(max_visits)
        self.enabled = enabled
        self.channel_stats = dict(dense_calls=0, fallback_calls=0,
            inspection_work=0, dense_work=0, geometric_incidences=0,
            workspace_entries_upper=0, workspace_bytes_upper=0,
            new_persistent_workspaces=0, source_or_LIVE_admitted=False,
            formal_gain=0)

    def _conv_counts(self, op, mask, transpose):
        if not self.enabled:
            return super()._conv_counts(op, mask, transpose)
        header = _headers(op)
        self.charge(header['inspection_work'])
        self.channel_stats['inspection_work'] += header['inspection_work']
        if (self.cache_bytes + 2 * header['kernel_entries'] > BYTE_CAP
                or self.cache_bytes // 8 + 2 * header['kernel_entries'] > ENTRY_CAP):
            raise MemoryError('complete density workspace/cache preallocation exceeds unchanged caps')
        # Charge before ALL new value scans, including rejected nonfinite data.
        if not np.isfinite(op._kernel).all():
            raise ValueError('complete finite kernel required for structural dispatch')
        dense = int(np.count_nonzero(op._kernel)) == header['kernel_entries']
        if not dense:
            # No old operation is discounted, even if this fallback is dearer.
            result = super()._conv_counts(op, mask, transpose)
            self.channel_stats['fallback_calls'] += 1
            return result
        plan = _finish_plan(header)
        self.charge(plan['dense_work'])
        self.channel_stats['dense_work'] += plan['dense_work']
        if (self.cache_bytes // 8 + plan['workspace_entries_upper'] > ENTRY_CAP
                or self.cache_bytes + plan['workspace_bytes_upper'] > BYTE_CAP):
            raise MemoryError('factorized complete workspace/cache preallocation exceeds unchanged caps')
        # Inherited public compute validates ordinary masks; inherited owners
        # supplies larger canonical packed labels. Guard both before a cast,
        # sum or add, including direct private-method misuse in isolated tests.
        expected = plan['output_entries'] if transpose else plan['input_entries']
        if (type(mask) is not np.ndarray or mask.dtype.kind not in 'biu'
                or mask.shape != (expected,)):
            raise ValueError('complete flat nonnegative integer values required')
        minimum = int(mask.min(initial=0))
        maximum = int(mask.max(initial=0))
        if minimum < 0 or maximum > INT64_MAX:
            raise ValueError('input value is outside nonnegative signed-int64 domain')
        batch, _, hi, wi = plan['input_shape']
        _, _, ho, wo = plan['output_shape']
        _, cig, khn, kwn = plan['kernel_shape']
        cog, groups = plan['output_channels_per_group'], plan['groups']
        channels = cog if transpose else cig
        source_spatial = ho * wo if transpose else hi * wi
        # Positive stride/dilation give at most one offset for any fixed
        # input/output pair. Thus this bounds every final AND partial sum;
        # the separate channels bound covers the initial full group reduction.
        summands = max(channels, min(channels * khn * kwn, channels * source_spatial))
        if maximum * summands > INT64_MAX:
            raise ValueError('factorized nonnegative integer accumulation could overflow')
        selected = mask.reshape(plan['output_shape'] if transpose else plan['input_shape'])
        row_mask = None if op._row_mask is None else op._row_mask.reshape(plan['output_shape'])
        result = np.zeros(plan['input_entries'] if transpose else plan['output_entries'], dtype=np.int64)
        if transpose:
            if row_mask is not None:
                selected = selected * row_mask
            reduced = selected.reshape(batch, groups, cog, ho, wo).sum(axis=2, dtype=np.int64)
            spatial = np.zeros((batch, groups, hi, wi), dtype=np.int64)
        else:
            reduced = selected.reshape(batch, groups, cig, hi, wi).sum(axis=2, dtype=np.int64)
            spatial = np.zeros((batch, groups, ho, wo), dtype=np.int64)
        stride, padding, dilation = (plan[key] for key in ('stride', 'padding', 'dilation'))
        # Filter once under the paid axis term. Otherwise B*G empty-offset
        # loops could remain when E=0; no incidence tariff may hide that work.
        heights = tuple((k, lo, hi_) for k, (lo, hi_) in enumerate(plan['height_ranges']) if lo < hi_)
        widths = tuple((k, lo, hi_) for k, (lo, hi_) in enumerate(plan['width_ranges']) if lo < hi_)
        if heights and widths:
            for b in range(batch):
                for g in range(groups):
                    for kh, oh_start, oh_stop in heights:
                        for kw, ow_start, ow_stop in widths:
                            for oh in range(oh_start, oh_stop):
                                ih = oh * stride[0] - padding[0] + kh * dilation[0]
                                for ow in range(ow_start, ow_stop):
                                    iw = ow * stride[1] - padding[1] + kw * dilation[1]
                                    if transpose:
                                        spatial[b, g, ih, iw] += reduced[b, g, oh, ow]
                                    else:
                                        spatial[b, g, oh, ow] += reduced[b, g, ih, iw]
        if transpose:
            result.reshape(batch, groups, cig, hi, wi)[:] = spatial[:, :, None, :, :]
        else:
            output = result.reshape(batch, groups, cog, ho, wo)
            output[:] = spatial[:, :, None, :, :]
            if row_mask is not None:
                np.multiply(result, op._row_mask, out=result)
        self.channel_stats['dense_calls'] += 1
        self.channel_stats['geometric_incidences'] += plan['geometric_incidences']
        for key in ('workspace_entries_upper', 'workspace_bytes_upper'):
            self.channel_stats[key] = max(self.channel_stats[key], plan[key])
        # The inherited caller marks this owned flat array read-only and applies
        # its unchanged final cache cap; none of the workspaces escape here.
        return result
