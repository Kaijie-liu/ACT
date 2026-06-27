# ===- act/back_end/solver/sparse_hz.py - Sparse Hybrid-Zonotope ---------===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#
"""Sparse CSR representation backend for the Hybrid Zonotope domain.

``SparseHZono`` is an independent concrete representation and propagation
carrier for the same Hybrid Zonotope abstract domain represented densely by
``act.back_end.solver.solver_hz.HZono``.  It is not a separate abstract domain.
It stores the exact 6-tuple

    z = c + Gc xi_c + Gb xi_b
    Ac xi_c + Ab xi_b == b
    Auc xi_c + Aub xi_b <= ub

in scipy CSR form.  It deliberately contains no benchmark loader, sampling,
commercial-solver diagnostic path, or per-instance rescue logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp

from act.back_end.solver.solver_hz import HZono, hz_split_constraints


def _as_csr(mat, *, shape: Optional[Tuple[int, int]] = None) -> sp.csr_matrix:
    if mat is None:
        if shape is None:
            raise ValueError("shape is required when mat is None")
        return sp.csr_matrix(shape, dtype=np.float64)
    if sp.issparse(mat):
        out = mat.tocsr().astype(np.float64)
    else:
        out = sp.csr_matrix(np.asarray(mat, dtype=np.float64))
    if shape is not None and out.shape != shape:
        raise ValueError(f"sparse matrix shape mismatch: got {out.shape}, expected {shape}")
    return out


def _torch_to_csr(t) -> sp.csr_matrix:
    shape = tuple(int(x) for x in t.shape)
    if t.numel() == 0:
        return sp.csr_matrix(shape, dtype=np.float64)
    return sp.csr_matrix(t.detach().cpu().double().numpy(), dtype=np.float64)


def _torch_to_np(t) -> np.ndarray:
    return t.detach().cpu().double().numpy().reshape(-1).astype(np.float64, copy=False)


@dataclass
class SparseHZono:
    """Sparse CSR representation of the Hybrid Zonotope domain.

    This is a native sparse propagation backend, not a view of ``HZono`` and
    not a second abstract domain.  The dense ``HZono`` and sparse
    ``SparseHZono`` representations lower to the same verdict layer.

    Continuous variables use ``xi_c in [-1, 1]``.  Binary variables use
    ``xi_b in {-1, 1}``.  Inequality rows are optional; absent rows are exposed
    to solver code as zero-row CSR blocks.
    """

    c: np.ndarray
    Gc: sp.csr_matrix
    Gb: sp.csr_matrix
    Ac: sp.csr_matrix
    Ab: sp.csr_matrix
    b: np.ndarray
    Auc: Optional[sp.csr_matrix] = None
    Aub: Optional[sp.csr_matrix] = None
    ub: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.c = np.asarray(self.c, dtype=np.float64).reshape(-1)
        self.b = np.asarray(self.b, dtype=np.float64).reshape(-1)
        self.Gc = _as_csr(self.Gc)
        self.Gb = _as_csr(self.Gb)
        self.Ac = _as_csr(self.Ac)
        self.Ab = _as_csr(self.Ab)

        n_out = int(self.c.size)
        n_cont = int(self.Gc.shape[1])
        n_bin = int(self.Gb.shape[1])
        if self.Gc.shape[0] != n_out or self.Gb.shape[0] != n_out:
            raise ValueError(
                "SparseHZono value shape mismatch: "
                f"c={n_out}, Gc={self.Gc.shape}, Gb={self.Gb.shape}"
            )
        if self.Ac.shape[1] != n_cont or self.Ab.shape[1] != n_bin:
            raise ValueError(
                "SparseHZono equality column mismatch: "
                f"Gc_cols={n_cont}, Gb_cols={n_bin}, Ac={self.Ac.shape}, Ab={self.Ab.shape}"
            )
        if self.Ac.shape[0] != self.Ab.shape[0] or self.Ac.shape[0] != self.b.size:
            raise ValueError(
                "SparseHZono equality row mismatch: "
                f"Ac={self.Ac.shape}, Ab={self.Ab.shape}, b={self.b.size}"
            )

        has_upper = self.Auc is not None or self.Aub is not None or self.ub is not None
        if has_upper:
            if self.Auc is None or self.Aub is None or self.ub is None:
                raise ValueError("upper constraints require Auc, Aub, and ub together")
            self.Auc = _as_csr(self.Auc, shape=(self.Auc.shape[0], n_cont))
            self.Aub = _as_csr(self.Aub, shape=(self.Aub.shape[0], n_bin))
            self.ub = np.asarray(self.ub, dtype=np.float64).reshape(-1)
            if self.Auc.shape[0] != self.Aub.shape[0] or self.Auc.shape[0] != self.ub.size:
                raise ValueError(
                    "SparseHZono upper row mismatch: "
                    f"Auc={self.Auc.shape}, Aub={self.Aub.shape}, ub={self.ub.size}"
                )

    @classmethod
    def from_dense_hz(cls, hz: HZono) -> "SparseHZono":
        """Convert a dense torch-backed ``HZono`` to CSR form."""

        (Ace, Abe, be), (Acl, Abl, bl) = hz_split_constraints(hz)
        out = cls(
            c=_torch_to_np(hz.c),
            Gc=_torch_to_csr(hz.Gc),
            Gb=_torch_to_csr(hz.Gb),
            Ac=_torch_to_csr(Ace),
            Ab=_torch_to_csr(Abe),
            b=_torch_to_np(be),
            Auc=_torch_to_csr(Acl),
            Aub=_torch_to_csr(Abl),
            ub=_torch_to_np(bl),
        )
        if getattr(hz, "_solver_known_nonempty", False):
            setattr(out, "_solver_known_nonempty", True)
            setattr(
                out,
                "_solver_known_nonempty_reason",
                getattr(hz, "_solver_known_nonempty_reason", "dense_conversion"),
            )
        return out

    @property
    def n_out(self) -> int:
        return int(self.c.size)

    @property
    def n_cont(self) -> int:
        return int(self.Gc.shape[1])

    @property
    def n_bin(self) -> int:
        return int(self.Gb.shape[1])

    @property
    def n_eq(self) -> int:
        return int(self.Ac.shape[0])

    @property
    def n_ub(self) -> int:
        return 0 if self.Auc is None else int(self.Auc.shape[0])

    @property
    def value_nnz(self) -> int:
        return int(self.Gc.nnz + self.Gb.nnz)

    @property
    def eq_nnz(self) -> int:
        return int(self.Ac.nnz + self.Ab.nnz)

    @property
    def ub_nnz(self) -> int:
        if self.Auc is None or self.Aub is None:
            return 0
        return int(self.Auc.nnz + self.Aub.nnz)

    @property
    def constraint_nnz(self) -> int:
        return int(self.eq_nnz + self.ub_nnz)

    def solver_tuple(self):
        """Return arrays in the layout consumed by ``solver_hz_verdict``."""

        Auc = self.Auc
        Aub = self.Aub
        ub = self.ub
        if Auc is None or Aub is None or ub is None:
            Auc = sp.csr_matrix((0, self.n_cont), dtype=np.float64)
            Aub = sp.csr_matrix((0, self.n_bin), dtype=np.float64)
            ub = np.zeros(0, dtype=np.float64)
        return (
            self.c,
            self.Gc,
            self.Gb,
            self.Ac,
            self.Ab,
            self.b,
            Auc,
            Aub,
            ub,
        )


__all__ = ["SparseHZono"]
