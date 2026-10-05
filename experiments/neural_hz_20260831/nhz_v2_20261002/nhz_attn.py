"""Neural-HZ v2 attention/Transformer extension (engine n005, float probe).

Extends the projection-aligned engine (nhz_engine.Engine, frozen) with the
operators of the VNN-COMP vit_2023 models: shape arithmetic (constant folded),
Transpose / Concat / ReduceMean on states, bilinear MatMul of two states, and
Softmax.  Diagnostic float version; rigorous rounding is not tracked here.

Tower semantics for the new transformers (THEORY.md Section 8):

* Bilinear z = x y with x = cx + gx^T w, y = cy + gy^T w:
      z = cx cy + cx (gy^T w) + cy (gx^T w) + q,
      q = sum_k gx_k gy_k w_k^2 + sum_{k != l} gx_k gy_l w_k w_l
  q in [m - r, m + r] with m = 1/2 sum_k gx_k gy_k and
  r = |gx|_1 |gy|_1 - 1/2 sum_k |gx_k gy_k|; q is replaced by m + r * eta
  with a fresh factor eta (sigma level).  The LP level adds the McCormick rows
  of the sum (lambda level) when bounds are available.
* Softmax along an axis: with d_j = s_j - s_i, p_i = 1 / (1 + sum_{j!=i} e^{d_j}).
  The value is the tangent plane at the centre d0 plus a remainder bounded with
  an interval Hessian over the box of d (sigma level, fresh factor per entry);
  rows: p_i in [p_lo, p_hi] from the difference bounds and sum_i p_i = 1
  (lambda level; exact for true values).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from onnx import numpy_helper

from nhz_engine import AZ, Engine, PhaseBlock, RowStore, _attrs

ATTN_VERSION = "n005.3"  # .2: attribute-form Slice, factor-group bookkeeping; .3: Sigmoid/Tanh tower


def _bounds(x: AZ):
    r = x.G.abs().sum(0)
    return x.c - r, x.c + r


def bilinear_matmul(x: AZ, y: AZ, K: int):
    """x [..., n, t], y [..., t, m] (same factor space K).  Returns (z, n_new, new_rad)."""
    x = x.pad_to(K); y = y.pad_to(K)
    cz = x.c @ y.c
    Gz = x.c.unsqueeze(0) @ y.G + x.G @ y.c.unsqueeze(0)            # [K, ..., n, m]
    # quadratic part statistics, contracted over t and K
    m_ = 0.5 * torch.einsum("k...nt,k...tm->...nm", x.G, y.G)
    absdiag = torch.einsum("k...nt,k...tm->...nm", x.G.abs(), y.G.abs())
    rx = x.G.abs().sum(0); ry = y.G.abs().sum(0)                     # [..., n, t], [..., t, m]
    rr = rx @ ry                                                     # sum_t |gx|_1 |gy|_1
    rad = (rr - 0.5 * absdiag).clamp(min=0)
    cz = cz + m_
    n_new = int(rad.numel())
    eta = torch.zeros((n_new,) + tuple(cz.shape), device=cz.device, dtype=cz.dtype)
    flat = eta.reshape(n_new, -1)
    flat[torch.arange(n_new, device=cz.device), torch.arange(n_new, device=cz.device)] = rad.reshape(-1)
    return AZ(cz, torch.cat([Gz, eta], 0)), n_new


def softmax_state(s: AZ, axis: int, K: int):
    """Tangent-plane softmax with interval-Hessian remainder.  Returns (p, n_new, (plo, phi))."""
    s = s.pad_to(K)
    axis = axis % s.c.dim()
    # move axis last
    perm = [i for i in range(s.c.dim()) if i != axis] + [axis]
    inv = [perm.index(i) for i in range(s.c.dim())]
    c = s.c.permute(perm)
    G = s.G.permute([0] + [p + 1 for p in perm])
    n = c.shape[-1]
    # differences d_ij = s_j - s_i as affine forms: [..., i, j]
    dc = c.unsqueeze(-2) - c.unsqueeze(-1)
    dG = G.unsqueeze(-2) - G.unsqueeze(-1)
    dr = dG.abs().sum(0)
    dlo, dhi = dc - dr, dc + dr
    # value and gradient at centre
    e0 = torch.exp(dc)                                    # [..., i, j], diag = 1
    S0 = e0.sum(-1)                                       # includes j = i (=1)
    p0 = 1.0 / S0                                         # p_i at centre
    grad = -e0 / (S0 ** 2).unsqueeze(-1)                  # d p_i / d d_ij ; for j = i, d_ii == 0 identically
    eye = torch.eye(n, device=c.device, dtype=c.dtype)
    grad = grad * (1 - eye)
    # linear part: p_i ~ p0_i + sum_j grad_ij (d_ij - dc_ij)
    pc = p0.clone()
    pG = (grad.unsqueeze(0) * dG).sum(-1)                 # [K, ..., i]
    # interval Hessian bound over the box of d: H_jk = 2 e_j e_k / T^3 - delta_jk e_j / T^2, T = 1 + sum e
    elo = torch.exp(dlo) * (1 - eye) + eye; ehi = torch.exp(dhi) * (1 - eye) + eye
    Tlo = elo.sum(-1); Thi = ehi.sum(-1)
    Hoff = 2 * ehi.unsqueeze(-1) * ehi.unsqueeze(-2) / (Tlo ** 3).unsqueeze(-1).unsqueeze(-1)
    Hdiag = torch.maximum(2 * ehi ** 2 / Tlo.unsqueeze(-1) ** 3, ehi / Tlo.unsqueeze(-1) ** 2)
    Hb = Hoff * (1 - torch.eye(n, device=c.device, dtype=c.dtype)) + torch.diag_embed(Hdiag)
    mask = (1 - eye)
    Hb = Hb * mask.unsqueeze(-1) * mask.unsqueeze(-2)
    delta = torch.maximum(dhi - dc, dc - dlo) * mask      # [..., i, j]
    rem = 0.5 * torch.einsum("...ij,...ijk,...ik->...i", delta, Hb, delta)
    # interval bounds of p_i
    plo = 1.0 / Thi; phi = 1.0 / Tlo
    n_new = int(rem.numel())
    eta = torch.zeros((n_new,) + tuple(pc.shape), device=c.device, dtype=c.dtype)
    flat = eta.reshape(n_new, -1)
    flat[torch.arange(n_new, device=c.device), torch.arange(n_new, device=c.device)] = rem.reshape(-1)
    Gp = torch.cat([pG, eta], 0)
    out = AZ(pc.permute(inv), Gp.permute([0] + [i + 1 for i in inv]))
    return out, n_new, (plo.permute(inv), phi.permute(inv))


class AttnEngine(Engine):
    def propagate(self, lb, ub, tighten_iters: int = 0, lr: float = 0.05, log=None):
        dev, dt = self.device, self.dtype
        lb = lb.reshape(self.input_shape).to(dev, dt); ub = ub.reshape(self.input_shape).to(dev, dt)
        c0 = (lb + ub) / 2; r0 = ((ub - lb) / 2).reshape(-1)
        nz = torch.nonzero(r0 > 0).reshape(-1); K = int(nz.numel())
        G0 = torch.zeros((K, r0.numel()), device=dev, dtype=dt)
        G0[torch.arange(K, device=dev), nz] = r0[nz]
        env: Dict[str, object] = {self.input_name: AZ(c0, G0.reshape((K,) + tuple(c0.shape)))}
        self.input_factor_index = nz
        rows = RowStore(dev, dt); phases: List[PhaseBlock] = []
        self.softmax_count = 0; self.bilinear_count = 0
        self.factor_groups = [("input", 0, K)]
        for node in self.nodes:
            op = node.op_type
            ins = [env.get(n, self.consts.get(n)) if n else None for n in node.input]
            a = _attrs(node)
            out = None
            if op == "Constant":
                arr = numpy_helper.to_array(a["value"]).copy()
                out = torch.as_tensor(arr).to(device=dev, dtype=dt if arr.dtype.kind == "f" else torch.int64)
            elif op == "Shape":
                x = ins[0]
                shp = x.c.shape if isinstance(x, AZ) else x.shape
                out = torch.as_tensor(list(shp), device=dev, dtype=torch.int64)
            elif op == "Gather":
                x, idx = ins
                out = torch.index_select(x, int(a.get("axis", 0)), idx.reshape(-1).long()).reshape(
                    tuple(idx.shape) if idx.dim() else ()) if not isinstance(x, AZ) else None
                if out is None:
                    raise NotImplementedError("Gather on state")
            elif op == "Unsqueeze":
                x = ins[0]; axes = a.get("axes") or ins[1].tolist()
                if isinstance(x, AZ):
                    c = x.c; G = x.G
                    for ax in sorted(int(v) for v in axes):
                        c = c.unsqueeze(ax); G = G.unsqueeze(ax + 1)
                    out = AZ(c, G)
                else:
                    for ax in sorted(int(v) for v in axes):
                        x = x.unsqueeze(ax)
                    out = x
            elif op == "Slice":
                x = ins[0]
                if isinstance(x, AZ):
                    raise NotImplementedError("Slice on state")
                if "starts" in a:
                    starts, ends = list(a["starts"]), list(a["ends"])
                    axes = list(a.get("axes", range(len(starts))))
                else:
                    starts, ends = ins[1].tolist(), ins[2].tolist()
                    axes = ins[3].tolist() if len(ins) > 3 and ins[3] is not None else list(range(len(starts)))
                sl = [slice(None)] * x.dim()
                for st, en, ax in zip(starts, ends, axes):
                    sl[ax] = slice(int(st), int(min(en, x.shape[ax])))
                out = x[tuple(sl)]
            elif op == "Concat":
                axis = int(a["axis"])
                if not any(isinstance(v, AZ) for v in ins):
                    out = torch.cat([v.to(ins[0].dtype) for v in ins], axis)
                else:
                    k = max(v.k for v in ins if isinstance(v, AZ))
                    cs, Gs = [], []
                    ref = next(v for v in ins if isinstance(v, AZ))
                    for v in ins:
                        if isinstance(v, AZ):
                            v = v.pad_to(k); cs.append(v.c); Gs.append(v.G)
                        else:
                            v = v.to(dt)
                            cs.append(v); Gs.append(torch.zeros((k,) + tuple(v.shape), device=dev, dtype=dt))
                    ax = axis % ref.c.dim()
                    out = AZ(torch.cat(cs, ax), torch.cat(Gs, ax + 1))
            elif op == "ConstantOfShape":
                shp = [max(1, int(v)) for v in ins[0].tolist()]
                val = numpy_helper.to_array(a["value"]).reshape(-1)[0] if "value" in a else 0.0
                out = torch.full(shp, float(val), device=dev, dtype=dt)
            elif op == "Transpose":
                x = ins[0]; perm = list(a["perm"])
                out = AZ(x.c.permute(perm), x.G.permute([0] + [p + 1 for p in perm])) if isinstance(x, AZ) else x.permute(perm)
            elif op == "ReduceMean":
                x = ins[0]; axes = [int(v) for v in a["axes"]]; kd = bool(a.get("keepdims", 1))
                out = AZ(x.c.mean(axes, keepdim=kd), x.G.mean([v + 1 for v in axes], keepdim=kd))
            elif op == "Softmax":
                x = ins[0]
                out, n_new, (plo, phi) = softmax_state(x, int(a.get("axis", -1)), K)
                self.factor_groups.append(("softmax", K, n_new))
                K += n_new; out = out.pad_to(K)
                fc = out.c.reshape(-1); fG = out.G.reshape(K, -1)
                rows.add(fG.t().contiguous(), (phi.reshape(-1) - fc))        # p <= phi
                rows.add(-fG.t().contiguous(), (fc - plo.reshape(-1)))       # p >= plo
                ax = int(a.get("axis", -1)) % out.c.dim()
                sc = out.c.sum(ax).reshape(-1); sG = out.G.sum(ax + 1).reshape(K, -1)
                rows.add(sG.t().contiguous(), 1.0 - sc); rows.add(-sG.t().contiguous(), sc - 1.0)   # sum p = 1
                self.softmax_count += 1
            elif op == "MatMul" and isinstance(ins[0], AZ) and isinstance(ins[1], AZ):
                out, n_new = bilinear_matmul(ins[0], ins[1], K)
                self.factor_groups.append(("bilinear", K, n_new))
                K += n_new; self.bilinear_count += 1
            elif op in ("Sigmoid", "Tanh"):
                y, n_new, (fG, fc, lows, ups) = smooth_state(op, ins[0], K)
                self.factor_groups.append((op.lower(), K, n_new))
                Kn = K + n_new
                yc = y.c.reshape(-1); yG = y.G.reshape(Kn, -1)
                gx = torch.cat([fG, fG.new_zeros((n_new, fG.shape[1]))], 0)          # [Kn, n]
                for a_, b_ in ups:     # y <= a x + b
                    rows.add((yG - gx * a_.unsqueeze(0)).t().contiguous(), b_ - yc + a_ * fc)
                for a_, b_ in lows:    # y >= a x + b
                    rows.add((gx * a_.unsqueeze(0) - yG).t().contiguous(), yc - a_ * fc - b_)
                K = Kn; out = y
            elif op == "Relu":
                x = ins[0].pad_to(K)
                k_before = K
                out, K = self._relu(node.name, x, K, rows, phases, tighten_iters, lr, log)
                self.factor_groups.append(("relu", k_before, K - k_before))
            else:
                # delegate the remaining (affine) operators to the frozen engine's logic
                out = self._affine(op, node, ins, a, K)
            env[node.output[0]] = out
        res = env[self.output_name]
        if isinstance(res, AZ):
            res = res.pad_to(K)
        return res, rows, K, phases

    def _affine(self, op, node, ins, a, K):
        dev, dt = self.device, self.dtype
        if op in ("Identity", "Dropout"):
            return ins[0]
        if op == "MatMul":
            x, W = ins
            if isinstance(x, AZ):
                return AZ(x.c @ W, x.G @ W)
            return AZ(x @ W.c, x @ W.G)
        if op == "Gemm":
            x, W = ins[0], ins[1]; bb = ins[2] if len(ins) > 2 else None
            if int(a.get("transB", 0)):
                W = W.t()
            al = float(a.get("alpha", 1.0)); be = float(a.get("beta", 1.0))
            return AZ(al * (x.c @ W) + (be * bb if bb is not None else 0), al * (x.G @ W))
        if op in ("Add", "Sub"):
            x, y = ins; sg = 1.0 if op == "Add" else -1.0
            if isinstance(x, AZ) and isinstance(y, AZ):
                k = max(x.k, y.k); xa, ya = x.pad_to(k), y.pad_to(k)
                return AZ(xa.c + sg * ya.c, xa.G + sg * ya.G)
            if isinstance(x, AZ):
                shp = torch.broadcast_shapes(x.c.shape, y.shape)
                return AZ((x.c + sg * y).reshape(shp), x.G.expand((x.k,) + tuple(shp)))
            if isinstance(y, AZ):
                shp = torch.broadcast_shapes(y.c.shape, x.shape)
                return AZ((x + sg * y.c).reshape(shp), (sg * y.G).expand((y.k,) + tuple(shp)))
            return x + sg * y
        if op in ("Mul", "Div"):
            x, y = ins
            if isinstance(x, AZ) and not isinstance(y, AZ):
                s = y if op == "Mul" else 1.0 / y
                return AZ(x.c * s, x.G * (s.unsqueeze(0) if torch.is_tensor(s) and s.dim() else s))
            if isinstance(y, AZ) and op == "Mul" and not isinstance(x, AZ):
                return AZ(y.c * x, y.G * (x.unsqueeze(0) if x.dim() else x))
            if not isinstance(x, AZ) and not isinstance(y, AZ):
                return x * y if op == "Mul" else x / y
            raise NotImplementedError("bilinear Mul")
        if op == "Conv":
            x, W = ins[0], ins[1]; bb = ins[2] if len(ins) > 2 else None
            pads = list(a.get("pads", [0, 0, 0, 0]))
            kw = dict(stride=tuple(a.get("strides", [1, 1])), padding=(pads[0], pads[1]),
                      dilation=tuple(a.get("dilations", [1, 1])), groups=int(a.get("group", 1)))
            cc = F.conv2d(x.c, W, bb, **kw)
            Gx = x.G.reshape((x.k,) + tuple(x.c.shape[1:]))
            parts = [F.conv2d(Gx[s:s + 1024], W, None, **kw) for s in range(0, x.k, 1024)]
            GG = torch.cat(parts, 0) if parts else Gx.new_zeros((0,) + tuple(cc.shape[1:]))
            return AZ(cc, GG.unsqueeze(1))
        if op == "BatchNormalization":
            x = ins[0]; gamma, beta, mean, var = ins[1:5]
            eps = float(a.get("epsilon", 1e-5))
            scale = gamma / torch.sqrt(var + eps); shift = beta - mean * scale
            shp = (1, -1) + (1,) * (x.c.dim() - 2)
            return AZ(x.c * scale.view(shp) + shift.view(shp), x.G * scale.view((1,) + shp))
        if op == "Flatten":
            x = ins[0]; axis = int(a.get("axis", 1))
            shape = (int(np.prod(x.c.shape[:axis])), -1)
            nc = x.c.reshape(shape)
            return AZ(nc, x.G.reshape((x.k,) + tuple(nc.shape)))
        if op == "Reshape":
            x, shp = ins
            shp = [int(s) for s in shp.tolist()]
            if isinstance(x, AZ):
                cs = list(x.c.shape)
                shp = [cs[i] if s == 0 else s for i, s in enumerate(shp)]
                nc = x.c.reshape(shp)
                return AZ(nc, x.G.reshape((x.k,) + tuple(nc.shape)))
            return x.reshape(shp)
        raise NotImplementedError(op)


# ----------------------------------------------------------------------------
# Smooth activations in the tower (n005.3): sigmoid / tanh
# ----------------------------------------------------------------------------

def _f(kind, x):
    return torch.sigmoid(x) if kind == "Sigmoid" else torch.tanh(x)


def _df(kind, x):
    s = torch.sigmoid(x)
    return s * (1 - s) if kind == "Sigmoid" else 1 - torch.tanh(x) ** 2


_D2MAX = {"Sigmoid": 0.0962250448649376, "Tanh": 0.7698003589195010}   # max |f''|


def smooth_lines(kind, l, u, n_tan=5, grid=64):
    """Valid lower and upper lines a*x + b of f on [l, u] (per neuron, vectorised).
    Candidates: chord and tangents at n_tan points; each is shifted by the grid
    maximum of its violation plus a Lipschitz remainder (|phi'| <= max f' + |a|,
    max f' <= 1/4 or 1)."""
    ts = [l + (u - l) * k / (n_tan - 1) for k in range(n_tan)]
    cands = [((_f(kind, u) - _f(kind, l)) / (u - l).clamp(min=1e-12), None)]
    for t in ts:
        cands.append((_df(kind, t), t))
    xs = torch.stack([l + (u - l) * k / grid for k in range(grid + 1)], 0)      # [grid+1, n]
    fx = _f(kind, xs)
    h = (u - l) / grid
    fmax = 0.25 if kind == "Sigmoid" else 1.0
    lows, ups = [], []
    for a, t in cands:
        b0 = (_f(kind, l) - a * l) if t is None else (_f(kind, t) - a * t)
        phi = fx - (a * xs + b0)
        rem = (fmax + a.abs()) * h / 2
        ups.append((a, b0 + phi.max(0).values.clamp(min=0) + rem))       # f <= a x + b_up
        lows.append((a, b0 + phi.min(0).values.clamp(max=0) - rem))      # f >= a x + b_lo
    return lows, ups


def smooth_state(kind, x: AZ, K: int):
    """DeepZ-aligned shadow for sigmoid/tanh + list of LP lines.  Returns (y, n_new, rows_spec)."""
    x = x.pad_to(K)
    shape = x.c.shape
    fc = x.c.reshape(-1); fG = x.G.reshape(K, fc.numel())
    r = fG.abs().sum(0); l = fc - r; u = fc + r
    u = torch.maximum(u, l + 1e-9)
    lam = torch.minimum(_df(kind, l), _df(kind, u))
    lo = _f(kind, l) - lam * l; hi = _f(kind, u) - lam * u
    mu = (lo + hi) / 2; nu = ((hi - lo) / 2).clamp(min=0)
    yc = lam * fc + mu
    yG = fG * lam.unsqueeze(0)
    n = fc.numel()
    eta = torch.zeros((n, n), device=fc.device, dtype=fc.dtype)
    eta[torch.arange(n), torch.arange(n)] = nu
    yG = torch.cat([yG, eta], 0)
    lows, ups = smooth_lines(kind, l, u)
    return AZ(yc.reshape(shape), yG.reshape((K + n,) + tuple(shape))), n, (fG, fc, lows, ups)


