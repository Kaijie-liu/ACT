"""Batched PDHG (Chambolle-Pock) for  max g_i^T w  s.t.  A w <= b, w in [-1,1]^K.

Returns weak-duality bounds  c_i + y^T b + ||g_i - A^T y||_1  at the best dual
iterate (valid for every y >= 0, independent of convergence) and the last
primal iterate.  Rows are normalised (an equivalent LP); step sizes from a
power-iteration estimate of ||A||_2.
"""
import torch


def spectral_norm(A: torch.Tensor, iters: int = 30) -> float:
    v = torch.randn(A.shape[1], device=A.device, dtype=A.dtype)
    v /= v.norm()
    for _ in range(iters):
        u = A @ v; v = A.t() @ u; nv = v.norm()
        if nv == 0:
            return 1.0
        v /= nv
    return float((A @ v).norm()) * 1.01 + 1e-12


def pdhg_upper(g: torch.Tensor, c: torch.Tensor, A: torch.Tensor, b: torch.Tensor, iters: int = 2000,
               eval_every: int = 50, restart_every: int = 0, y0=None):
    m, K = g.shape; R = A.shape[0]
    if R == 0:
        return c.double() + g.double().abs().sum(1), torch.sign(g), g.new_zeros((m, 0))
    rn = A.norm(dim=1).clamp(min=1e-30)
    An = A / rn.unsqueeze(1); bn = b / rn
    L = spectral_norm(An)
    tau = sigma = 0.95 / L
    w = torch.zeros((m, K), device=g.device, dtype=g.dtype)
    y = torch.zeros((m, R), device=g.device, dtype=g.dtype) if y0 is None else (y0 * rn).clone()
    A64 = A.double(); b64 = b.double(); g64 = g.double(); c64 = c.double()
    best = c64 + g64.abs().sum(1); best_y = torch.zeros((m, R), device=g.device, dtype=g.dtype)
    for it in range(1, iters + 1):
        w_new = (w + tau * (g - y @ An)).clamp_(-1.0, 1.0)
        y = (y + sigma * ((2 * w_new - w) @ An.t() - bn)).clamp_(min=0.0)
        w = w_new
        if it % eval_every == 0 or it == iters:
            yo = (y / rn).double()            # multipliers of the original rows
            val = c64 + yo @ b64 + (g64 - yo @ A64).abs().sum(1)
            bt = val < best
            best = torch.where(bt, val, best)
            best_y[bt] = (y / rn)[bt]
    return best, w, best_y
