"""N002 probe: projection-aligned HZ with GPU-batched LP bound tightening.

Diagnostic only (float32 optimisation, float64 bound re-evaluation, no
directed-rounding pad yet).  Not a verifier and not a verdict source.

State.  All latent continuous factors w in [-1, 1]^K (input factors first,
then one eta per unstable ReLU in creation order) and, implicitly, one binary
phase per unstable ReLU.  Every tensor value x has an exact affine form
x = c + G^T w over the retained factors (the HZ value map).

Projection-aligned exact ReLU (for an unstable pre-activation x with valid
bounds l < 0 < u over the exact set):

    y = lam x + mu + mu eta,   lam = u/(u-l),  mu = -lam l / 2,
    rows (LP relaxation):  -y <= 0,   x - y <= 0,
    rows (MILP only):       y <= u d, y <= x - l (1 - d),  d in {0,1}.

The four rows plus eta in [-1,1] encode y = ReLU(x) exactly; projecting out
d gives the triangle, whose upper facet is already implied by eta <= 1, so the
LP relaxation keeps only the two lower-facet rows.  The constraint-erased
shadow of the value map is the DeepZ ReLU image.

LP bound query.  For an affine objective f(w) = c + g^T w,

    max f  s.t.  A w <= b, w in [-1,1]^K
       <=  c + nu^T b + || g - A^T nu ||_1     for every nu >= 0   (weak duality).

The multipliers are chosen by projected Adam on GPU, batched over all
objectives of a layer.  Every evaluated nu yields a valid bound; the best one
is kept.  nu = 0 recovers the shadow bound, so the LP bound is never looser
than the aligned shadow.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from gpu_shadow import Graph, _bn_params, _conv_params


@dataclasses.dataclass
class AZ:
    """Affine form over the global factor space: value = c + sum_k G[k] w_k."""

    c: torch.Tensor          # [*S]
    G: torch.Tensor          # [K_t, *S]  (K_t <= current global K, zero padded on demand)

    @property
    def k(self) -> int:
        return int(self.G.shape[0])

    def pad_to(self, k: int) -> "AZ":
        if k == self.k:
            return self
        pad = torch.zeros((k - self.k,) + tuple(self.G.shape[1:]), dtype=self.G.dtype, device=self.G.device)
        return AZ(self.c, torch.cat([self.G, pad], dim=0))


class RowStore:
    """LP relaxation rows  A w <= b  (dense, columns = global factors)."""

    def __init__(self, device, dtype):
        self.blocks: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.device, self.dtype = device, dtype

    def add(self, A: torch.Tensor, b: torch.Tensor):
        self.blocks.append((A, b))

    def dense(self, K: int):
        if not self.blocks:
            return (torch.zeros((0, K), device=self.device, dtype=self.dtype),
                    torch.zeros((0,), device=self.device, dtype=self.dtype))
        As, bs = [], []
        for A, b in self.blocks:
            if A.shape[1] < K:
                A = torch.cat([A, torch.zeros((A.shape[0], K - A.shape[1]), device=A.device, dtype=A.dtype)], 1)
            As.append(A); bs.append(b)
        return torch.cat(As, 0), torch.cat(bs, 0)

    @property
    def n_rows(self) -> int:
        return sum(int(b.numel()) for _, b in self.blocks)


def lp_upper_bounds(Gobj: torch.Tensor, cobj: torch.Tensor, A: torch.Tensor, b: torch.Tensor,
                    iters: int, lr: float, eval_dtype=torch.float64,
                    chunk: int = 2048) -> torch.Tensor:
    """Sound (up to float rounding) upper bounds of c_i + g_i^T w over {Aw<=b, |w|<=1}.

    Gobj: [n, K] objective coefficients, cobj: [n].  Returns [n] bounds.
    """
    n, K = Gobj.shape
    R = A.shape[0]
    out = torch.empty(n, device=Gobj.device, dtype=eval_dtype)
    A64 = A.to(eval_dtype); b64 = b.to(eval_dtype)
    for s in range(0, n, chunk):
        g = Gobj[s:s + chunk]
        c = cobj[s:s + chunk]
        m = g.shape[0]
        shadow = c.to(eval_dtype) + g.to(eval_dtype).abs().sum(1)
        if R == 0 or iters == 0:
            out[s:s + chunk] = shadow
            continue
        nu = torch.zeros((m, R), device=g.device, dtype=g.dtype, requires_grad=False)
        best = shadow.clone()
        best_nu = torch.zeros_like(nu)
        m1 = torch.zeros_like(nu); m2 = torch.zeros_like(nu)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for it in range(1, iters + 1):
            resid = g - nu @ A                     # [m, K]
            val = c + nu @ b + resid.abs().sum(1)  # dual objective (float32)
            grad = b.unsqueeze(0) - torch.sign(resid) @ A.t()   # d val / d nu
            m1.mul_(b1).add_(grad, alpha=1 - b1)
            m2.mul_(b2).addcmul_(grad, grad, value=1 - b2)
            step = lr * (m1 / (1 - b1 ** it)) / ((m2 / (1 - b2 ** it)).sqrt() + eps)
            nu = (nu - step).clamp_(min=0.0)
            if it % 25 == 0 or it == iters:
                nu64 = nu.to(eval_dtype)
                v64 = c.to(eval_dtype) + nu64 @ b64 + (g.to(eval_dtype) - nu64 @ A64).abs().sum(1)
                better = v64 < best
                best = torch.where(better, v64, best)
                best_nu[better] = nu[better]
        out[s:s + chunk] = best
    return out


@dataclasses.dataclass
class LayerLog:
    name: str
    n: int
    unstable_shadow: int
    unstable_lp: int
    wall_s: float


def run(graph: Graph, lb: torch.Tensor, ub: torch.Tensor, iters: int = 200, lr: float = 0.05,
        tighten: bool = True, log: Optional[List[LayerLog]] = None):
    dev, dt = lb.device, lb.dtype
    c0 = (lb + ub) / 2
    r0 = ((ub - lb) / 2).reshape(-1)
    nz = torch.nonzero(r0 > 0).reshape(-1)
    K = int(nz.numel())
    G0 = torch.zeros((K, r0.numel()), device=dev, dtype=dt)
    G0[torch.arange(K, device=dev), nz] = r0[nz]
    env: Dict[str, AZ] = {graph.input_name: AZ(c0.unsqueeze(0), G0.reshape((K, 1) + tuple(lb.shape)))}
    rows = RowStore(dev, dt)
    for node in graph.nodes:
        op = node.op_type
        x = env.get(node.input[0])
        if op == "Conv":
            p = _conv_params(node, graph)
            cc = F.conv2d(x.c, p["weight"], p["bias"], p["stride"], p["padding"], p["dilation"], p["groups"])
            outs = []
            Gx = x.G.reshape((x.k,) + tuple(x.c.shape[1:]))
            for s in range(0, x.k, 1024):
                outs.append(F.conv2d(Gx[s:s + 1024], p["weight"], None, p["stride"], p["padding"], p["dilation"], p["groups"]))
            env[node.output[0]] = AZ(cc, torch.cat(outs, 0).unsqueeze(1))
        elif op == "BatchNormalization":
            scale, shift = _bn_params(node, graph)
            shp = (1, -1, 1, 1)
            env[node.output[0]] = AZ(x.c * scale.view(shp) + shift.view(shp), x.G * scale.view((1,) + shp))
        elif op == "Add":
            y = env[node.input[1]]
            k = max(x.k, y.k)
            xa, ya = x.pad_to(k), y.pad_to(k)
            env[node.output[0]] = AZ(xa.c + ya.c, xa.G + ya.G)
        elif op == "Flatten":
            env[node.output[0]] = AZ(x.c.reshape(1, -1), x.G.reshape(x.k, 1, -1))
        elif op == "Gemm":
            a = Graph.attrs(node)
            W = graph.inits[node.input[1]]
            bb = graph.inits[node.input[2]] if len(node.input) > 2 else None
            if int(a.get("transB", 0)):
                W = W.t()
            env[node.output[0]] = AZ(x.c @ W + (bb if bb is not None else 0), x.G @ W)
        elif op == "Relu":
            t0 = time.time()
            x = x.pad_to(K)
            flat_c = x.c.reshape(-1)
            flat_G = x.G.reshape(K, -1)              # [K, n]
            rad = flat_G.abs().sum(0)
            l = flat_c - rad; u = flat_c + rad
            unst = (l < 0) & (u > 0)
            n_sh = int(unst.sum())
            if tighten and n_sh and rows.n_rows:
                idx = torch.nonzero(unst).reshape(-1)
                A, b = rows.dense(K)
                gobj = flat_G[:, idx].t().contiguous()       # [m, K]
                cobj = flat_c[idx]
                ub_lp = lp_upper_bounds(gobj, cobj, A, b, iters, lr).to(dt)
                lb_lp = -lp_upper_bounds(-gobj, -cobj, A, b, iters, lr).to(dt)
                u[idx] = torch.minimum(u[idx], ub_lp)
                l[idx] = torch.maximum(l[idx], lb_lp)
            neg = u <= 0; pos = l >= 0; unst = ~(neg | pos)
            ku = int(unst.sum())
            lam = torch.where(pos, torch.ones_like(u), torch.zeros_like(u))
            lam_u = torch.where(unst, u / (u - l), torch.zeros_like(u))
            lam = torch.where(unst, lam_u, lam)
            mu = torch.where(unst, -lam_u * l / 2, torch.zeros_like(u))
            yc = flat_c * lam + mu
            yG = flat_G * lam.unsqueeze(0)
            idx = torch.nonzero(unst).reshape(-1)
            eta = torch.zeros((ku, flat_c.numel()), device=dev, dtype=dt)
            eta[torch.arange(ku, device=dev), idx] = mu[idx]
            yG = torch.cat([yG, eta], 0)            # K + ku factors
            # rows over K+ku columns:  -y <= 0  and  x - y <= 0  for unstable
            gy = yG[:, idx].t()                      # [ku, K+ku]
            gx = torch.cat([flat_G[:, idx].t(), torch.zeros((ku, ku), device=dev, dtype=dt)], 1)
            cy = yc[idx]; cx = flat_c[idx]
            rows.add(-gy, cy)                        # -(cy + gy w) <= 0
            rows.add(gx - gy, cy - cx)               # (cx + gx w) - (cy + gy w) <= 0
            K += ku
            env[node.output[0]] = AZ(yc.reshape(x.c.shape), yG.reshape((K,) + tuple(x.c.shape)))
            if log is not None:
                torch.cuda.synchronize()
                log.append(LayerLog(node.name, int(flat_c.numel()), n_sh, ku, time.time() - t0))
        else:
            raise NotImplementedError(op)
    return env[graph.output_name], rows, K


def margin_lower_bounds(out: AZ, rows: RowStore, K: int, t: int, others: List[int], iters: int, lr: float,
                        lp: bool = True) -> torch.Tensor:
    out = out.pad_to(K)
    c = out.c.reshape(-1)
    G = out.G.reshape(K, -1)
    gobj = (G[:, others] - G[:, t:t + 1]).t().contiguous()   # Y_j - Y_t
    cobj = c[others] - c[t]
    if lp:
        A, b = rows.dense(K)
        ubd = lp_upper_bounds(gobj, cobj, A, b, iters, lr)
    else:
        ubd = (cobj + gobj.abs().sum(1)).to(torch.float64)
    return -ubd   # lower bound of Y_t - Y_j
