"""Default-off actual native phase blocks, without any old-predicate padding.

Coordinates already carry global identities. Active value COO rows need no
column padding before new triplets with the final frame width are assembled.
No full padded/post HZ or complete concatenated predicate CSR is created.
"""

import numpy as np
from act.back_end.solver.solver_hz import SparseHZono
from act.back_end.hybridz_tf.tf_mlp import _sparse_triplets
from experiments.neural_hz_20260831.c30_first_write_v1 import AppendView


def phase_blocks(
    hz: SparseHZono,
    lb,
    ub,
    slots,
    n_cont: int,
    n_bin: int,
    *, source_pre, enabled=False,
):
    """The same exact native phase triplets, before whole-predicate assembly."""
    if not enabled: return None
    if source_pre.frame_id != hz.frame_id or not source_pre.exact or not hz.exact:
        raise ValueError("native phase source frame/exactness changed")
    lb = np.asarray(lb, dtype=np.float64).reshape(-1)
    ub = np.asarray(ub, dtype=np.float64).reshape(-1)
    active_idx = np.flatnonzero(lb >= 0.0).astype(np.int64)
    unstable_idx = np.flatnonzero((lb < 0.0) & (ub > 0.0)).astype(np.int64)
    k = int(unstable_idx.size)
    if len(slots) != k:
        raise ValueError("sparse ReLU slot count mismatch")

    out_c = np.zeros(hz.n_out, dtype=np.float64)
    gc_parts = []
    gb_parts = []
    if active_idx.size:
        out_c[active_idx] = hz.c[active_idx]
        active_gc = hz.Gc[active_idx].tocoo()
        gc_parts.append(
            (active_idx[active_gc.row], active_gc.col, active_gc.data)
        )
        active_gb = hz.Gb[active_idx].tocoo()
        gb_parts.append(
            (active_idx[active_gb.row], active_gb.col, active_gb.data)
        )

    eq_c_parts = []
    eq_b_parts = []
    ineq_c_parts = []
    ineq_b_parts = []
    if k:
        slot_array = np.asarray(slots, dtype=np.int64)
        xi1_cols = slot_array[:, 0]
        xi2_cols = slot_array[:, 1]
        z_cols = slot_array[:, 2]
        rows = np.arange(k, dtype=np.int64)
        alpha = lb[unstable_idx]
        beta = ub[unstable_idx]

        out_c[unstable_idx] = beta / 2.0
        gc_parts.append((unstable_idx, xi2_cols, -beta / 2.0))

        eq_c_parts.extend(
            [
                (rows, xi1_cols, alpha / 2.0),
                (rows, xi2_cols, -beta / 2.0),
            ]
        )
        pre_gc = hz.Gc[unstable_idx].tocoo()
        eq_c_parts.append((pre_gc.row, pre_gc.col, -pre_gc.data))
        eq_b_parts.append((rows, z_cols, alpha / 2.0))
        pre_gb = hz.Gb[unstable_idx].tocoo()
        eq_b_parts.append((pre_gb.row, pre_gb.col, -pre_gb.data))

        ineq_c_parts.extend(
            [
                (rows, xi1_cols, -np.ones(k, dtype=np.float64)),
                (k + rows, xi2_cols, -np.ones(k, dtype=np.float64)),
            ]
        )
        ineq_b_parts.extend(
            [
                (rows, z_cols, -np.ones(k, dtype=np.float64)),
                (k + rows, z_cols, np.ones(k, dtype=np.float64)),
            ]
        )
        eq_rhs = hz.c[unstable_idx] - beta / 2.0
    else:
        eq_rhs = np.zeros(0, dtype=np.float64)

    out_Gc = _sparse_triplets(gc_parts, (hz.n_out, n_cont))
    out_Gb = _sparse_triplets(gb_parts, (hz.n_out, n_bin))
    eq_Ac = _sparse_triplets(eq_c_parts, (k, n_cont))
    eq_Ab = _sparse_triplets(eq_b_parts, (k, n_bin))
    ineq_Ac = _sparse_triplets(ineq_c_parts, (2 * k, n_cont))
    ineq_Ab = _sparse_triplets(ineq_b_parts, (2 * k, n_bin))
    return AppendView(source_pre, eq_Ac, eq_Ab, eq_rhs, ineq_Ac, ineq_Ab,
        np.zeros(2*k, dtype=np.float64), out_c, out_Gc, out_Gb)

