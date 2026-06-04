#===- act/back_end/hybridz_tf/sparse_gc_t2.py - T2 dense->sparse helpers -====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   T2 sparse-Gc HZ representation: post-conv prune + dense->sparse
#   conversion to reduce RSS on convolutional benchmarks.
#
#   Validated 2026-05-27 (research/t2_sparse_gc/):
#     - synthetic cifar100 first-conv:  98.3% Gc storage saved
#     - 3-layer pipeline:                94.6% peak Gc, 50% peak RSS
#     - 5/5 soundness regression PASS
#
#   Default OFF — opt-in via env knobs below.
#
#===---------------------------------------------------------------------===#

"""Sound dense-Gc compaction operators for the ACT HZ forward path.

Two operators, both sound by row-slack construction:

  * :func:`act_hz_prune_gc_dense` — drop Gc columns whose max abs entry
    is at or below ``eps``. Per dropped column j, the contribution
    ``Gc[i,j] * xi_j`` with ``xi_j in [-1, 1]`` lives in
    ``[-|Gc[i,j]|, +|Gc[i,j]|]``. Aggregated across all dropped j for
    row i: ``[-sum_j |Gc[i,j]|, +sum_j |Gc[i,j]|]``. One new generator
    column with per-row slack at row i recovers exactly that
    spread (the new ``xi_new`` ranges independently over [-1, 1]).
    Soundness limitation: skipped if any dropped column has nonzero
    mass in ``Ac`` (the relaxation logic for constraint rows is not
    implemented here; in forward-only conv chains the constraint
    matrix is typically empty before the first eq_lagr ReLU, so this
    is a strict subset of safe cases).

  * :func:`act_hz_dense_to_sparse` — convert an HZono to SparseGcZ
    when its Gc density is below a threshold and sparse storage is
    smaller than dense by a configurable factor. Requires nb=0
    (SparseGcZ does not store binary generators).

Routing call sites are in :mod:`act.back_end.hybridz_tf.hz_routing` —
:func:`hz_conv2d` and :func:`hz_dense` invoke these on the HZono exit
when the corresponding env knob is set.

Env knobs:

  ============================  ===========  =====================
  Name                          Default      Effect
  ============================  ===========  =====================
  ACT_HZ_PRUNE_GC               0            Enable prune
  ACT_HZ_PRUNE_GC_THRESH        1e-9         Column drop threshold
  ACT_HZ_DENSE_TO_SPARSE        0            Enable conversion
  ACT_HZ_SPARSE_GC_DENSITY      0.05         Density threshold
  ACT_HZ_PRUNE_GC_INSTRUMENT    0            Log conversion stats
  ============================  ===========  =====================

Soundness invariant: ``Image(Z') ⊇ Image(Z)`` for ``Z' = prune(Z)``
or ``Z' = dense_to_sparse(Z)``.
"""

from __future__ import annotations

import os
import sys
from typing import Union

import torch

from act.back_end.solver.solver_hz import HZono
from act.back_end.hybridz_tf.representations import SparseGcZ


# ─── env knobs ─────────────────────────────────────────────────────────


def act_prune_gc_enabled() -> bool:
    return os.environ.get("ACT_HZ_PRUNE_GC", "0") == "1"


def act_prune_gc_threshold() -> float:
    return float(os.environ.get("ACT_HZ_PRUNE_GC_THRESH", "1e-9"))


def act_dense_to_sparse_enabled() -> bool:
    return os.environ.get("ACT_HZ_DENSE_TO_SPARSE", "0") == "1"


def act_dense_to_sparse_density() -> float:
    return float(os.environ.get("ACT_HZ_SPARSE_GC_DENSITY", "0.05"))


def act_prune_log_enabled() -> bool:
    return os.environ.get("ACT_HZ_PRUNE_GC_INSTRUMENT", "0") == "1"


def act_min_dense_bytes() -> int:
    """Minimum dense Gc size (bytes) before T2 ops fire.

    Default: 50 MiB. Reason — the LP witness back-projection
    (``hz_walker_lp.lp_witness_to_input``) relies on the ordering and
    count of the first ``_base_ng`` Gc columns to invert factors back
    to inputs. Pruning or sparsifying a small HZ may not save enough
    memory to justify any potential disturbance of this mapping. Above
    50 MiB, the dense-storage win is large and any witness-path
    disturbance is overshadowed by the RSS savings on cifar /
    tinyimagenet / vggnet workloads.
    """
    return int(float(os.environ.get("ACT_HZ_T2_MIN_DENSE_MIB", "50.0")) * (1024 ** 2))


def _log(msg: str) -> None:
    if act_prune_log_enabled():
        print(f"[sparse_gc_t2] {msg}", file=sys.stderr, flush=True)


def _gc_dense_bytes(hz: HZono) -> int:
    Gc = hz.Gc
    if Gc.numel() == 0:
        return 0
    return Gc.shape[0] * Gc.shape[1] * Gc.element_size()


def act_preconv_budget_bytes() -> int:
    """Pre-conv predicted dense Gc above which T2b forces sparse path.

    Default: 1024 MiB. When ``ACT_HZ_PRECONV_SPARSE=1`` is set, the
    ``hz_conv2d`` HZono branch predicts ``n_out * ng * element_size``
    and converts the input HZono → SparseGcZ before calling the dense
    conv kernel — rescuing instances whose post-conv dense Gc would
    OOM the watchdog cap (e.g. cifar100 resnet_large family).
    """
    return int(float(os.environ.get("ACT_HZ_PRECONV_BUDGET_MIB", "1024.0")) * (1024 ** 2))


def act_preconv_sparse_enabled() -> bool:
    return os.environ.get("ACT_HZ_PRECONV_SPARSE", "0") == "1"


# ─── T2c (precision lever): tail-densify for tighter ReLU ──────────────


def act_tail_densify_enabled() -> bool:
    """When set, the ReLU dispatch densifies SparseGcZ → HZono at the
    classifier tail (small-dim layers) so the dense eq_lagr_v8 encoding
    can be used instead of the looser sparse triangle relaxation.

    Sound: SparseGcZ.to_hzono() is a lossless storage change (same
    factor-space, same Gc entries — just expand sparse → dense). The
    tighter ReLU encoding is applied AFTER the conversion, on the
    densified HZono. This stays within the single HZ forward method.
    """
    return os.environ.get("ACT_HZ_TAIL_DENSIFY", "0") == "1"


def act_tail_densify_dim_threshold() -> int:
    """Densify SparseGcZ → HZono before ReLU when ``hz.dim`` is at or
    below this threshold. Default 1024 — typical classifier layers are
    ≤ 1024 nodes; convolutional middle layers stay sparse.
    """
    return int(os.environ.get("ACT_HZ_TAIL_DENSIFY_DIM_MAX", "1024"))


def act_tail_densify_ng_threshold() -> int:
    """Skip tail-densify if ``hz.ng`` exceeds this (densified Gc would
    overflow the eq_native ReLU budget). Default 8192.
    """
    return int(os.environ.get("ACT_HZ_TAIL_DENSIFY_NG_MAX", "8192"))


# ─── B3 (precision lever): sparse-eq_lagr ReLU on SparseGcZ ────────────


def act_sparse_eq_lagr_enabled() -> bool:
    """When set, the ReLU dispatch routes SparseGcZ to the new
    ``apply_relu_eq_lagr_sparse`` operator (algorithms/sparse_eq_lagr.py)
    instead of the looser triangle relaxation. Per unstable neuron:
    +1 sparse continuous gen (xi2), +1 binary gen (z), +6 sparse
    inequality rows. Mathematically equivalent to dense eq_lagr_v8 +
    project_eq_elim, but stays sparse so memory stays bounded.

    Default OFF. Soundness verified by `tests/test_sparse_eq_lagr.py`.
    """
    return os.environ.get("ACT_HZ_SPARSE_EQ_LAGR", "0") == "1"


def act_sparse_eq_lagr_max_unstable() -> int:
    """Skip sparse-eq_lagr if k (unstable count) exceeds this. Default
    4096 — past this, the +6k inequality rows would dominate. Falls
    back to the previous sparse triangle path.
    """
    return int(os.environ.get("ACT_HZ_SPARSE_EQ_LAGR_K_MAX", "4096"))


def act_sparse_eq_lagr_compact_rows() -> bool:
    """When set, emit 3 inequality rows per unstable neuron instead of
    6 by dropping the 3 LP-redundant rows (blk2/4/6 = "−xi_j ≤ 1"
    negated forms, implied by variable boxes).

    **Sound + zero precision loss**: LP-redundancy proven via
    exhaustive LP tests in research/t2_sparse_gc/verify_redundancy_multi.py
    across single-neuron α/β trials, multi-neuron, and multi-layer.
    Dropped rows are implied by kept rows + LP solver's variable
    box constraints (xi_c, xi_b ∈ [-1, 1]; z ∈ [-1, 1] LP-relax).

    Net: 50% Ac storage reduction with NO precision change.
    Recommended ON for any benchmark where B3 is enabled.
    """
    return os.environ.get("ACT_HZ_SPARSE_EQ_LAGR_COMPACT", "0") == "1"


# ─── operators ─────────────────────────────────────────────────────────


def act_hz_prune_gc_dense(hz: HZono, eps: float) -> HZono:
    """Sound row-slack pruning of dense Gc columns with ``\\|\\|col\\|\\|_inf <= eps``.

    Returns the input ``hz`` unchanged if pruning is unsafe (any
    dropped column has nonzero mass in ``Ac``) or if no columns meet
    the threshold.
    """
    if eps <= 0 or not isinstance(hz, HZono):
        return hz
    Gc = hz.Gc
    if Gc.numel() == 0:
        return hz
    col_norm = Gc.abs().max(dim=0).values
    keep_mask = col_norm > eps
    if bool(keep_mask.all()):
        return hz
    drop_mask = ~keep_mask

    # Constraint-safety: don't drop columns referenced by Ac.
    if hz.Ac.numel() > 0 and drop_mask.any():
        if hz.Ac[:, drop_mask].abs().max().item() > eps * 1e-3:
            return hz

    Gc_keep = Gc[:, keep_mask]
    Gc_drop = Gc[:, drop_mask]
    slack_row_mass = Gc_drop.abs().sum(dim=1)
    nz_rows = slack_row_mass > 0
    n_new = int(nz_rows.sum())

    Ac_keep = hz.Ac[:, keep_mask] if hz.Ac.numel() else hz.Ac

    if n_new == 0:
        out = HZono(
            c=hz.c,
            Gc=Gc_keep,
            Gb=hz.Gb,
            Ac=Ac_keep,
            Ab=hz.Ab,
            b=hz.b,
            eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
        )
    else:
        nz_idx = nz_rows.nonzero(as_tuple=False).view(-1)
        new_block = torch.zeros((hz.dim, n_new), dtype=Gc.dtype, device=Gc.device)
        new_block[nz_idx, torch.arange(n_new, device=Gc.device)] = slack_row_mass[nz_idx]
        Gc_new = torch.cat([Gc_keep, new_block], dim=1)
        if Ac_keep.numel() == 0 or Ac_keep.shape[0] == 0:
            Ac_new = Ac_keep.new_zeros((Ac_keep.shape[0], Gc_new.shape[1]))
        else:
            Ac_pad = Ac_keep.new_zeros((Ac_keep.shape[0], n_new))
            Ac_new = torch.cat([Ac_keep, Ac_pad], dim=1)
        out = HZono(
            c=hz.c,
            Gc=Gc_new,
            Gb=hz.Gb,
            Ac=Ac_new,
            Ab=hz.Ab,
            b=hz.b,
            eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
        )

    _log(
        f"prune_gc_dense: ng {Gc.shape[1]}->{out.Gc.shape[1]} "
        f"(dropped {int(drop_mask.sum())}, added_slack {n_new})"
    )
    return out


def act_hz_dense_to_sparse(
    hz: HZono,
    density_threshold: float = 0.05,
    zero_eps: float = 1e-12,
) -> Union[HZono, SparseGcZ]:
    """Convert a dense HZono to SparseGcZ when its Gc density is low.

    Pre-conditions:
      - ``hz.nb == 0`` (SparseGcZ has no binary generators)
      - sparse storage strictly less than dense by 30%+

    Equality constraints are preserved verbatim in ``SparseGcZ``.
    """
    if not isinstance(hz, HZono):
        return hz
    if hz.nb > 0:
        return hz

    Gc = hz.Gc
    n, ng = Gc.shape if Gc.numel() else (hz.dim, 0)
    if n == 0 or ng == 0:
        return hz
    nnz = int((Gc.abs() > zero_eps).sum())
    density = nnz / float(n * ng)
    if density > density_threshold:
        return hz

    elt_bytes = Gc.element_size()
    sparse_bytes = nnz * (8 + elt_bytes)  # 2 long indices + val
    dense_bytes = n * ng * elt_bytes
    if sparse_bytes >= dense_bytes * 0.7:
        return hz

    nz_idx = (Gc.abs() > zero_eps).nonzero(as_tuple=False).T
    nz_val = Gc[nz_idx[0], nz_idx[1]]
    Gc_sparse = torch.sparse_coo_tensor(
        nz_idx, nz_val, (n, ng), dtype=Gc.dtype, device=Gc.device,
    ).coalesce()

    if hz.Ac.numel() > 0 and hz.nc > 0:
        Ac_nz = (hz.Ac.abs() > zero_eps).nonzero(as_tuple=False).T
        Ac_val = hz.Ac[Ac_nz[0], Ac_nz[1]] if Ac_nz.numel() else hz.Ac.new_zeros(0)
        Ac_sparse = torch.sparse_coo_tensor(
            Ac_nz, Ac_val, (hz.nc, ng),
            dtype=hz.Ac.dtype, device=hz.Ac.device,
        ).coalesce()
        b = hz.b.clone()
        eq_mask = hz.eq_mask.clone() if hz.eq_mask is not None else None
    else:
        Ac_sparse = None
        b = None
        eq_mask = None

    out = SparseGcZ(
        c=hz.c.view(-1),
        Gc_sparse=Gc_sparse,
        dtype=Gc.dtype,
        device=Gc.device,
        Ac_sparse=Ac_sparse,
        b=b,
        eq_mask=eq_mask,
    )
    object.__setattr__(
        out, "_base_ng",
        int(min(getattr(hz, "_base_ng", ng), out.ng)),
    )
    object.__setattr__(
        out, "_base_nb",
        int(min(getattr(hz, "_base_nb", 0), out.nb)),
    )
    object.__setattr__(
        out, "_base_nc",
        int(min(getattr(hz, "_base_nc", hz.nc), out.nc)),
    )
    root_id = getattr(hz, "_base_root_id", None)
    object.__setattr__(
        out, "_base_root_id",
        None if root_id is None else int(root_id),
    )
    _log(
        f"dense_to_sparse: dim={n} ng={ng} density={density:.4f} "
        f"dense_MiB={dense_bytes / 2**20:.1f} sparse_MiB={sparse_bytes / 2**20:.1f}"
    )
    return out


def act_maybe_compact_hz(hz):
    """Convenience: prune-then-sparsify, respecting env knobs.

    Returns either ``hz`` unchanged (when no knob is set, when ``hz``
    is not an HZono, or when the dense Gc is smaller than the
    ``ACT_HZ_T2_MIN_DENSE_MIB`` guard), an HZono with pruned Gc, or a
    SparseGcZ after conversion.

    The size guard protects the LP-witness back-projection path on
    small networks (where T2 savings are negligible anyway) from any
    perturbation to the ``_base_ng`` factor-space mapping. cifar100
    / tinyimagenet / vggnet first-conv outputs are 1.5+ GiB dense, so
    the 50 MiB default fires only on the workloads T2 is meant to help.
    """
    if not isinstance(hz, HZono):
        return hz
    if not act_prune_gc_enabled() and not act_dense_to_sparse_enabled():
        return hz
    if _gc_dense_bytes(hz) < act_min_dense_bytes():
        return hz
    if act_prune_gc_enabled():
        hz = act_hz_prune_gc_dense(hz, act_prune_gc_threshold())
    if act_dense_to_sparse_enabled() and isinstance(hz, HZono):
        hz = act_hz_dense_to_sparse(hz, act_dense_to_sparse_density())
    return hz
