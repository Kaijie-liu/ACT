"""Same support identity; omit zero integer products inside the proof engine."""

import numpy as np

from experiments.neural_hz_20260831 import c6_support_affine_plan_v1 as v1

_OriginalEngine = v1.SupportEngine


class SupportEngine(_OriginalEngine):
    def _conv_counts(self, op, mask, transpose):
        batch, ci, hi, wi = op.input_shape
        _, co, ho, wo = op.output_shape
        _, cig, khn, kwn = op._kernel.shape
        cog = co // op._groups
        counts = np.zeros(op.input_shape if transpose else op.output_shape, dtype=np.int64)
        selected = mask.reshape(op.output_shape if transpose else op.input_shape)
        out_mask = None if op._row_mask is None else op._row_mask.reshape(op.output_shape)
        for kh in range(khn):
            for kw in range(kwn):
                for oh in range(ho):
                    ih = oh * op._stride[0] - op._padding[0] + kh * op._dilation[0]
                    if not 0 <= ih < hi:
                        continue
                    for ow in range(wo):
                        iw = ow * op._stride[1] - op._padding[1] + kw * op._dilation[1]
                        if not 0 <= iw < wi:
                            continue
                        for b in range(batch):
                            for g in range(op._groups):
                                ic, oc = slice(g * cig, (g + 1) * cig), slice(g * cog, (g + 1) * cog)
                                if transpose:
                                    values = selected[b, oc, oh, ow]
                                    enabled = values != 0
                                    if out_mask is not None:
                                        enabled = enabled & out_mask[b, oc, oh, ow]
                                    active = np.flatnonzero(enabled)
                                    if not active.size:
                                        continue
                                    weights = (op._kernel[g * cog + active, :, kh, kw] != 0).astype(np.int64)
                                    self.charge(int(cig * active.size))
                                    counts[b, ic, ih, iw] += weights.T @ values[active].astype(np.int64)
                                else:
                                    values = selected[b, ic, ih, iw]
                                    active = np.flatnonzero(values != 0)
                                    outputs = np.arange(cog) if out_mask is None else np.flatnonzero(out_mask[b, oc, oh, ow])
                                    if not active.size or not outputs.size:
                                        continue
                                    weights = (op._kernel[g * cog + outputs[:, None], active[None, :], kh, kw] != 0).astype(np.int64)
                                    self.charge(int(active.size * outputs.size))
                                    counts[b, g * cog + outputs, oh, ow] += weights @ values[active].astype(np.int64)
        return counts.reshape(-1)


def plan(expr, keep_rows, **kwargs):
    # Single-threaded research worker only, not a production dispatch patch.
    previous = v1.SupportEngine
    if previous is not _OriginalEngine:
        raise ValueError('nested or conflicting support planner binding')
    v1.SupportEngine = SupportEngine
    try:
        return v1.plan(expr, keep_rows, **kwargs)
    finally:
        v1.SupportEngine = previous
