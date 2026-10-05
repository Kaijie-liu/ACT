"""Neural-HZ v2 sound engine (n004): projection-aligned HZ with rigorous rounding.

Semantics.  The network is interpreted in exact real arithmetic with its stored
float weights.  A state carries an affine value map c + G^T w over latent
factors w in [-1,1]^K together with a coordinatewise rounding radius e, with the
invariant

  (INV) for every concrete execution there is w in [-1,1]^K satisfying every
        recorded LP row, such that each true coordinate value lies within
        e of c + G^T w.

Every operator update below preserves INV (THEORY.md Section 5).  LP rows are
relaxed by the radii they involve; terminal bounds add |a|^T e; every dual bound
is re-evaluated in float64 with an explicit summation-error pad.  Multipliers
may come from any (float32) optimiser: weak duality makes every nu >= 0 valid.

The phase list is retained (binaries are not dropped) so the exact MILP can be
emitted; this module only performs LP-level queries.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import numpy_helper

from nhz_engine import _attrs, parse_vnnlib  # noqa: F401  (re-exported)

SOUND_VERSION = "n004.2"  # .2: phases keep x/y forms and radii for rigorous MILP rows


def gamma(n: int, dtype) -> float:
    u = torch.finfo(dtype).eps / 2
    nu = (n + 2) * u
    if nu >= 0.5:
        raise ValueError("gamma undefined")
    return nu / (1 - nu) * (1 + 1e-6)


TINY = 1e-300  # absolute underflow allowance per accumulated term (float64)


@dataclasses.dataclass
class SAZ:
    c: torch.Tensor
    G: torch.Tensor   # [k, *S]
    e: torch.Tensor   # [*S] rounding radius

    @property
    def k(self) -> int:
        return int(self.G.shape[0])

    def pad_to(self, k: int) -> "SAZ":
        if k == self.k:
            return self
        pad = self.G.new_zeros((k - self.k,) + tuple(self.G.shape[1:]))
        return SAZ(self.c, torch.cat([self.G, pad], 0), self.e)

    def radius(self) -> torch.Tensor:
        return self.G.abs().sum(0)


class SRows:
    def __init__(self):
        self.A: List[torch.Tensor] = []
        self.b: List[torch.Tensor] = []
        self._cache = None

    def add(self, A, b):
        if A.shape[0]:
            self.A.append(A); self.b.append(b); self._cache = None

    @property
    def n_rows(self):
        return sum(int(x.numel()) for x in self.b)

    def dense(self, K):
        if self._cache is not None and self._cache[0] == K:
            return self._cache[1], self._cache[2]
        if not self.A:
            dev = None
            A = torch.zeros((0, K), dtype=torch.float64)
            b = torch.zeros((0,), dtype=torch.float64)
        else:
            A = torch.cat([a if a.shape[1] == K else torch.cat([a, a.new_zeros((a.shape[0], K - a.shape[1]))], 1)
                           for a in self.A], 0)
            b = torch.cat(self.b, 0)
        self._cache = (K, A, b)
        return A, b


def optimise_nu(g32: torch.Tensor, A32: torch.Tensor, b32: torch.Tensor, iters: int, lr: float,
                c32: torch.Tensor) -> torch.Tensor:
    """Projected Adam on the weak-duality bound; returns the best (float32) nu."""
    m, R = g32.shape[0], A32.shape[0]
    nu = g32.new_zeros((m, R))
    if R == 0 or iters <= 0:
        return nu
    best = c32 + g32.abs().sum(1); best_nu = nu.clone()
    m1 = torch.zeros_like(nu); m2 = torch.zeros_like(nu)
    for it in range(1, iters + 1):
        resid = g32 - nu @ A32
        if it % 25 == 1 or it == iters:
            val = c32 + nu @ b32 + resid.abs().sum(1)
            better = val < best
            best = torch.where(better, val, best)
            best_nu[better] = nu[better]
        grad = b32.unsqueeze(0) - torch.sign(resid) @ A32.t()
        m1.mul_(0.9).add_(grad, alpha=0.1)
        m2.mul_(0.999).addcmul_(grad, grad, value=0.001)
        nu = (nu - lr * (m1 / (1 - 0.9 ** it)) / ((m2 / (1 - 0.999 ** it)).sqrt() + 1e-8)).clamp_(min=0.0)
    resid = g32 - nu @ A32
    val = c32 + nu @ b32 + resid.abs().sum(1)
    better = val < best
    best_nu[better] = nu[better]
    return best_nu


def sound_eval(g: torch.Tensor, c: torch.Tensor, A: torch.Tensor, b: torch.Tensor, nu32: torch.Tensor) -> torch.Tensor:
    """Rigorous float64 upper bound of max c + g^T w over {A w <= b, |w| <= 1}
    using multipliers nu (made exactly nonnegative in float64)."""
    nu = nu32.double().clamp(min=0.0)
    R = A.shape[0]; K = g.shape[1]
    if R:
        r = g - nu @ A
        val = c + nu @ b + r.abs().sum(1)
        mag = c.abs() + nu @ b.abs() + g.abs().sum(1) + (nu @ A.abs()).sum(1)
    else:
        val = c + g.abs().sum(1)
        mag = c.abs() + g.abs().sum(1)
    pad = gamma(R + K + 4, torch.float64) * mag + TINY * (R + K + 4)
    return val + pad


class SoundEngine:
    def __init__(self, path: str, device: str = "cuda", dtype=torch.float64):
        self.model = onnx.load(path)
        self.device, self.dtype = device, dtype
        self.consts: Dict[str, torch.Tensor] = {}
        for t in self.model.graph.initializer:
            arr = numpy_helper.to_array(t).copy()
            self.consts[t.name] = torch.as_tensor(arr).to(device=device, dtype=dtype if arr.dtype.kind == "f" else torch.int64)
        init_names = set(self.consts)
        self.input_name = [i.name for i in self.model.graph.input if i.name not in init_names][0]
        dims = [d.dim_value for d in [i for i in self.model.graph.input if i.name == self.input_name][0].type.tensor_type.shape.dim]
        self.input_shape = tuple(max(1, d) for d in dims)
        self.output_name = self.model.graph.output[0].name
        self.nodes = list(self.model.graph.node)

    def propagate(self, lb: np.ndarray, ub: np.ndarray, iters: int = 300, lr: float = 0.05, log=None):
        dev, dt = self.device, self.dtype
        lb = torch.as_tensor(lb, dtype=torch.float64).reshape(self.input_shape).to(dev)
        ub = torch.as_tensor(ub, dtype=torch.float64).reshape(self.input_shape).to(dev)
        c0 = (lb + ub) / 2
        r0 = (ub - lb) / 2
        # upward safety: the represented box [c0 - r0 - e0, c0 + r0 + e0] covers [lb, ub]
        e0 = gamma(2, torch.float64) * (lb.abs() + ub.abs()) + TINY
        flat_r = r0.reshape(-1)
        nz = torch.nonzero(flat_r > 0).reshape(-1)
        K = int(nz.numel())
        G0 = torch.zeros((K, flat_r.numel()), device=dev, dtype=torch.float64)
        G0[torch.arange(K, device=dev), nz] = flat_r[nz]
        env: Dict[str, object] = {self.input_name: SAZ(c0.to(dt), G0.reshape((K,) + tuple(c0.shape)).to(dt), e0.to(dt))}
        self.input_factor_index = nz
        rows = SRows()
        phases = []
        remaining: Dict[str, int] = {}
        for node in self.nodes:
            for n in node.input:
                if n:
                    remaining[n] = remaining.get(n, 0) + 1
        for node in self.nodes:
            op = node.op_type
            ins = [env.get(n, self.consts.get(n)) if n else None for n in node.input]
            a = _attrs(node)
            if op == "Constant":
                arr = numpy_helper.to_array(a["value"]).copy()
                out = torch.as_tensor(arr).to(device=dev, dtype=dt if arr.dtype.kind == "f" else torch.int64)
            elif op in ("Identity", "Dropout"):
                out = ins[0]
            elif op == "MatMul":
                x, W = ins
                if not isinstance(x, SAZ) or isinstance(W, SAZ):
                    raise NotImplementedError("MatMul form")
                n = W.shape[0]; Wa = W.abs()
                out = SAZ(x.c @ W, x.G @ W, x.e @ Wa + gamma(n, dt) * ((x.c.abs() + x.radius()) @ Wa) + TINY * n)
            elif op == "Gemm":
                x, W = ins[0], ins[1]; bb = ins[2] if len(ins) > 2 else None
                if int(a.get("transA", 0)):
                    raise NotImplementedError("transA")
                if int(a.get("transB", 0)):
                    W = W.t()
                al = float(a.get("alpha", 1.0)); be = float(a.get("beta", 1.0))
                if al != 1.0 or be != 1.0:
                    raise NotImplementedError("Gemm alpha/beta")
                n = W.shape[0]; Wa = W.abs()
                bias = bb if bb is not None else torch.zeros(W.shape[1], device=dev, dtype=dt)
                out = SAZ(x.c @ W + bias, x.G @ W,
                          x.e @ Wa + gamma(n + 1, dt) * ((x.c.abs() + x.radius()) @ Wa + bias.abs()) + TINY * n)
            elif op in ("Add", "Sub"):
                x, y = ins
                s = 1.0 if op == "Add" else -1.0
                if isinstance(x, SAZ) and isinstance(y, SAZ):
                    k = max(x.k, y.k); xa, ya = x.pad_to(k), y.pad_to(k)
                    out = SAZ(xa.c + s * ya.c, xa.G + s * ya.G,
                              xa.e + ya.e + gamma(1, dt) * (xa.c.abs() + ya.c.abs() + xa.radius() + ya.radius()))
                elif isinstance(x, SAZ):
                    yb = torch.broadcast_to(y, x.c.shape)
                    out = SAZ(x.c + s * yb, x.G, x.e + gamma(1, dt) * (x.c.abs() + yb.abs()))
                elif isinstance(y, SAZ):
                    xb = torch.broadcast_to(x, y.c.shape)
                    out = SAZ(xb + s * y.c, s * y.G, y.e + gamma(1, dt) * (y.c.abs() + xb.abs()))
                else:
                    out = x + s * y
            elif op in ("Mul", "Div"):
                x, y = ins
                if isinstance(x, SAZ) and not isinstance(y, SAZ):
                    sc = y if op == "Mul" else 1.0 / y
                    g = gamma(1 if op == "Mul" else 3, dt)
                    sb = torch.broadcast_to(sc, x.c.shape)
                    out = SAZ(x.c * sb, x.G * sb.unsqueeze(0), x.e * sb.abs() * (1 + g) + g * sb.abs() * (x.c.abs() + x.radius()))
                elif isinstance(y, SAZ) and op == "Mul" and not isinstance(x, SAZ):
                    sb = torch.broadcast_to(x, y.c.shape); g = gamma(1, dt)
                    out = SAZ(y.c * sb, y.G * sb.unsqueeze(0), y.e * sb.abs() * (1 + g) + g * sb.abs() * (y.c.abs() + y.radius()))
                elif not isinstance(x, SAZ) and not isinstance(y, SAZ):
                    out = x * y if op == "Mul" else x / y
                else:
                    raise NotImplementedError("bilinear")
            elif op == "Conv":
                x, W = ins[0], ins[1]; bb = ins[2] if len(ins) > 2 else None
                pads = list(a.get("pads", [0, 0, 0, 0]))
                if pads[0] != pads[2] or pads[1] != pads[3]:
                    raise NotImplementedError("asymmetric padding")
                kw = dict(stride=tuple(a.get("strides", [1, 1])), padding=(pads[0], pads[1]),
                          dilation=tuple(a.get("dilations", [1, 1])), groups=int(a.get("group", 1)))
                n = int(W.shape[1] * W.shape[2] * W.shape[3]); Wa = W.abs()
                cc = F.conv2d(x.c, W, bb, **kw)
                Gx = x.G.reshape((x.k,) + tuple(x.c.shape[1:]))
                parts = [F.conv2d(Gx[s:s + 512], W, None, **kw) for s in range(0, x.k, 512)]
                GG = torch.cat(parts, 0) if parts else Gx.new_zeros((0,) + tuple(cc.shape[1:]))
                mag = F.conv2d(x.c.abs() + x.radius(), Wa, None, **kw)
                if bb is not None:
                    mag = mag + bb.abs().view(1, -1, 1, 1)
                ee = F.conv2d(x.e, Wa, None, **kw) + gamma(n + 1, dt) * mag + TINY * n
                out = SAZ(cc, GG.unsqueeze(1), ee)
            elif op == "BatchNormalization":
                x = ins[0]
                gm, bt, mean, var = ins[1:5]
                eps = float(a.get("epsilon", 1e-5))
                s = gm / torch.sqrt(var + eps)
                t = bt - mean * s
                g4 = gamma(4, dt); g2 = gamma(2, dt)
                ds = g4 * s.abs()                                # |s_true - s|
                dtt = mean.abs() * ds + g2 * (bt.abs() + (mean * s).abs())   # |t_true - t|
                shp = (1, -1) + (1,) * (x.c.dim() - 2)
                sv, tv = s.view(shp), t.view(shp)
                mag = x.c.abs() + x.radius()
                ee = x.e * (sv.abs() + ds.view(shp)) + ds.view(shp) * mag + dtt.view(shp) + g2 * (sv.abs() * mag + tv.abs())
                out = SAZ(x.c * sv + tv, x.G * sv.unsqueeze(0), ee)
            elif op == "Flatten":
                x = ins[0]; axis = int(a.get("axis", 1))
                shape = (int(np.prod(x.c.shape[:axis])), -1)
                nc = x.c.reshape(shape)
                out = SAZ(nc, x.G.reshape((x.k,) + tuple(nc.shape)), x.e.reshape(nc.shape))
            elif op == "Reshape":
                x, shp = ins
                shp = [int(v) for v in shp.tolist()]
                cs = list(x.c.shape)
                shp = [cs[i] if v == 0 else v for i, v in enumerate(shp)]
                nc = x.c.reshape(shp)
                out = SAZ(nc, x.G.reshape((x.k,) + tuple(nc.shape)), x.e.reshape(nc.shape))
            elif op == "Relu":
                x = ins[0].pad_to(K)
                out, K = self._relu(node.name, x, K, rows, phases, iters, lr, log)
            else:
                raise NotImplementedError(op)
            env[node.output[0]] = out
            del ins, out
            for n in node.input:
                if n and n in env and n != self.output_name:
                    remaining[n] -= 1
                    if remaining[n] == 0:
                        del env[n]
        res = env[self.output_name]
        return res.pad_to(K), rows, K, phases

    def _bounds(self, fc, fG, fe, idx, rows: SRows, K: int, iters: int, lr: float):
        """Rigorous [l, u] of the true coordinate values for indices idx."""
        rad = fG[:, idx].abs().sum(0)
        g2 = gamma(K + 2, self.dtype)
        mag = (fc[idx].abs() + rad)
        u = fc[idx] + rad + fe[idx] + g2 * mag + TINY * K
        l = fc[idx] - rad - fe[idx] - g2 * mag - TINY * K
        if iters > 0 and rows.n_rows and idx.numel():
            A, b = rows.dense(K)
            A = A.to(self.device); b = b.to(self.device)
            A32, b32 = A.float(), b.float()
            g = fG[:, idx].t().contiguous().double(); c = fc[idx].double()
            for sign in (1.0, -1.0):
                nu = optimise_nu((sign * g).float(), A32, b32, iters, lr, (sign * c).float())
                bd = sound_eval(sign * g, sign * c, A, b, nu) + fe[idx].double()
                if sign > 0:
                    u = torch.minimum(u, bd.to(u.dtype))
                else:
                    l = torch.maximum(l, (-bd).to(l.dtype))
        return l, u

    def _relu(self, name, x: SAZ, K: int, rows: SRows, phases, iters, lr, log):
        dev, dt = self.device, self.dtype
        shape = x.c.shape
        fc = x.c.reshape(-1); fG = x.G.reshape(K, fc.numel()); fe = x.e.reshape(-1)
        allidx = torch.arange(fc.numel(), device=dev)
        l, u = self._bounds(fc, fG, fe, allidx, SRows(), K, 0, lr)   # shadow bounds (rigorous)
        unst = (l < 0) & (u > 0)
        if iters > 0 and bool(unst.any()) and rows.n_rows:
            idx = torch.nonzero(unst).reshape(-1)
            l2, u2 = self._bounds(fc, fG, fe, idx, rows, K, iters, lr)
            l = l.clone(); u = u.clone()
            l[idx] = torch.maximum(l[idx], l2); u[idx] = torch.minimum(u[idx], u2)
        n_sh = int(unst.sum())
        neg = u <= 0; pos = l >= 0; unst = ~(neg | pos)
        idx = torch.nonzero(unst).reshape(-1); m = int(idx.numel())
        lam = pos.to(dt).clone()
        lu, uu = l[idx], u[idx]
        lam_u = uu / (uu - lu)
        lam_u = lam_u.clamp(min=0.0, max=1.0)
        # M >= max((1-lam)u, -lam l) rounded up; mu = M/2
        Mx = torch.maximum((1 - lam_u) * uu, -lam_u * lu) * (1 + gamma(3, dt)) + TINY
        mu = Mx / 2
        lam[idx] = lam_u
        yc = fc * lam
        yc[idx] = yc[idx] + mu
        yG = fG * lam.unsqueeze(0)
        g2 = gamma(2, dt)
        ye = fe * lam + g2 * (lam * (fc.abs() + fG.abs().sum(0)))
        ye[idx] = ye[idx] + g2 * mu
        ye = torch.where(neg, torch.zeros_like(ye), ye)
        eta = torch.zeros((m, fc.numel()), device=dev, dtype=dt)
        eta[torch.arange(m, device=dev), idx] = mu
        yG = torch.cat([yG, eta], 0)
        gx = fG[:, idx].t().contiguous(); cx = fc[idx]
        gy = yG[:, idx].t().contiguous(); cy = yc[idx]
        ex, ey = fe[idx], ye[idx]
        gx_full = torch.cat([gx, gx.new_zeros((m, m))], 1)
        rows.add((-gy).double(), (cy + ey).double())                       # -y_hat <= e_y (+ c offset)
        rows.add((gx_full - gy).double(), (cy - cx + ex + ey).double())     # x_hat - y_hat <= e_x + e_y
        phases.append(dict(layer=name, idx=idx, eta0=K, l=lu, u=uu, gx=gx, cx=cx, gy=gy, cy=cy, ex=ex, ey=ey,
                           row0=rows.n_rows - 2 * m))
        K = K + m
        if log is not None:
            log.append((name, int(fc.numel()), n_sh, m))
        return SAZ(yc.reshape(shape), yG.reshape((K,) + tuple(shape)), ye.reshape(shape)), K


def sound_terminal(out: SAZ, rows: SRows, K: int, disjuncts, iters: int, lr: float):
    """Rigorous upper bound of the violation b - a^T Y of every unsafe disjunct
    (first atom objective; other atoms as rows).  Negative => disjunct empty."""
    dev = out.c.device
    c = out.c.reshape(-1).double(); G = out.G.reshape(K, c.numel()).double(); e = out.e.reshape(-1).double()
    A, b = rows.dense(K); A = A.to(dev); b = b.to(dev)
    rad = G.abs().sum(0)
    res = []
    n_out = c.numel()
    for atoms in disjuncts:
        a0 = torch.as_tensor(atoms[0][0], device=dev, dtype=torch.float64); b0 = float(atoms[0][1])
        g = -(G @ a0).unsqueeze(0); cc = (b0 - c @ a0).reshape(1)
        coef_pad = gamma(n_out + 2, torch.float64) * (abs(b0) + a0.abs() @ (c.abs() + rad)) + (a0.abs() @ e)
        AA, BB = A, b
        if len(atoms) > 1:
            exA, exb = [], []
            for ak, bk in atoms[1:]:
                at = torch.as_tensor(ak, device=dev, dtype=torch.float64)
                pad_k = gamma(n_out + 2, torch.float64) * (abs(bk) + at.abs() @ (c.abs() + rad)) + (at.abs() @ e)
                exA.append(G @ at); exb.append(bk - c @ at + pad_k)
            AA = torch.cat([A, torch.stack(exA)]); BB = torch.cat([b, torch.stack(exb)])
        nu = optimise_nu(g.float(), AA.float(), BB.float(), iters, lr, cc.float())
        res.append(float(sound_eval(g, cc, AA, BB, nu)[0] + coef_pad))
    return res
