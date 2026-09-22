"""Conservative Conv2d CSR support-union planning; no numeric operator change.

For each batch/group/output spatial site, the union of input factor indices
over its receptive field contains the support of EVERY output channel in that
group. Count that union once and multiply by channels/group. Stored zeros and
zero weights are deliberately included. No weighted floating-point sum, sparse
matrix product, or dense HZ is used to estimate a count.
"""
import time

import numpy as np

from .sparse_budget import MATRICES, estimate, retained_bound

INDEX_CHUNK = 4096


def scratch_bytes(hz):
    # One reused bool marker vector (8x reserve), bounded index scatter chunks,
    # and fixed Python metadata. Never a rows x factors temporary.
    return 8*max(hz.n_cont, hz.n_bin) + 16*INDEX_CHUNK + 4096


def _pair(value):
    if isinstance(value, int):
        value = (value, value)
    if len(value) != 2 or any(int(v) != v for v in value):
        raise ValueError('integer spatial pair required')
    return tuple(int(v) for v in value)


def geometry(layer, hz, out_dim):
    if layer.kind.upper() != 'CONV2D':
        raise ValueError('support union is Conv2d only')
    shape = tuple(int(v) for v in layer.params['input_shape'])
    if len(shape) not in (3, 4) or min(shape) <= 0:
        raise ValueError('positive CHW/NCHW input shape required')
    c, h, w = shape[-3:]
    oc, icg, kh, kw = (int(v) for v in layer.params['weight'].shape)
    groups = int(layer.params.get('groups', 1))
    sh, sw = _pair(layer.params.get('stride', 1))
    ph, pw = _pair(layer.params.get('padding', 0))
    dh, dw = _pair(layer.params.get('dilation', 1))
    if (min(oc, icg, kh, kw, groups, sh, sw, dh, dw) <= 0 or min(ph, pw) < 0 or
            c != groups*icg or oc % groups or hz.n_out % (c*h*w)):
        raise ValueError('invalid grouped convolution geometry')
    batch = hz.n_out // (c*h*w)
    oh = (h+2*ph-dh*(kh-1)-1)//sh+1
    ow = (w+2*pw-dw*(kw-1)-1)//sw+1
    if batch <= 0 or min(oh, ow) <= 0 or batch*oc*oh*ow != out_dim:
        raise ValueError('convolution output width mismatch')
    return batch, c, h, w, oc, icg, kh, kw, groups, sh, sw, ph, pw, dh, dw, oh, ow


def conv_support_plan(layer, hz, out_dim):
    """Call only AFTER scratch_bytes has passed the caller's resource gate.

    No persistent plan cache: counts are recomputed on this exact input HZ.
    An exception/termination yields no partial usable plan.
    """
    start = time.monotonic()
    b, c, h, w, oc, icg, kh, kw, groups, sh, sw, ph, pw, dh, dw, oh, ow = geometry(layer, hz, out_dim)
    marker = np.zeros(max(hz.n_cont, hz.n_bin), dtype=np.bool_)
    totals = []
    visited = 0
    for matrix in (hz.Gc, hz.Gb):
        if matrix.shape[0] != hz.n_out:
            raise ValueError('generator row identity mismatch')
        seen = marker[:matrix.shape[1]]
        for first in range(0, matrix.indices.size, INDEX_CHUNK):
            chunk = matrix.indices[first:first+INDEX_CHUNK]
            if chunk.size and (chunk.min() < 0 or chunk.max() >= seen.size):
                raise ValueError('factor index outside bound')
        total = 0
        for batch in range(b):
            for group in range(groups):
                for y in range(oh):
                    for x in range(ow):
                        seen.fill(False)
                        for channel in range(group*icg, (group+1)*icg):
                            for ky in range(kh):
                                iy = y*sh-ph+ky*dh
                                if not 0 <= iy < h:
                                    continue
                                for kx in range(kw):
                                    ix = x*sw-pw+kx*dw
                                    if not 0 <= ix < w:
                                        continue
                                    row = ((batch*c+channel)*h+iy)*w+ix
                                    lo, hi = int(matrix.indptr[row]), int(matrix.indptr[row+1])
                                    for first in range(lo, hi, INDEX_CHUNK):
                                        cols = matrix.indices[first:min(first+INDEX_CHUNK, hi)]
                                        seen[cols] = True
                                    visited += hi-lo
                        total += int(np.count_nonzero(seen))*(oc//groups)
        totals.append(total)
    del marker, seen
    coarse = estimate(layer, hz, out_dim)
    nnz = sum(totals) + sum(int(getattr(hz, name).nnz) for name in MATRICES[2:])
    if nnz > coarse['nnz_upper_bound']:
        raise ValueError('support union exceeded independent coarse upper bound')
    retained = retained_bound(nnz, int(out_dim), hz.n_eq, hz.n_ineq)
    # EXACTLY the old reserve formula; only the integer retained-nnz bound changes.
    work = 8*retained + 160*coarse['operator_nnz_upper_bound'] + 16*(hz.n_cont+hz.n_bin)
    return {**coarse, 'estimated_retained_bytes': retained, 'workspace_reserve_bytes': work,
            'nnz_upper_bound': nnz, 'planner': 'conv_support_union_v1',
            'coarse_nnz_upper_bound': coarse['nnz_upper_bound'],
            'coarse_workspace_reserve_bytes': coarse['workspace_reserve_bytes'],
            'continuous_nnz_upper_bound': totals[0], 'binary_nnz_upper_bound': totals[1],
            'index_visits': visited, 'planner_scratch_reserve_bytes': scratch_bytes(hz),
            'planner_seconds': time.monotonic()-start}
