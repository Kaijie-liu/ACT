"""Neural-HZ v2 GPU engine (N003): projection-aligned exact HZ propagation.

Isolated research code; default-off by construction (not imported by ACT).

Domain element (see THEORY.md, Definition 1).  A state is
  (w-space, value map c + G^T w, LP rows A w <= b, MILP rows, phase list)
with continuous latent factors w in [-1,1]^K (input factors first, then one
eta per unstable ReLU) and one binary phase d_j per unstable ReLU.  The value
map is exact for affine operators.  For an unstable ReLU with valid bounds
l < 0 < u the projection-aligned transformer writes

    y = lam x + mu (1 + eta),  lam = u/(u-l),  mu = -lam*l/2,
    LP rows:    -y <= 0,  x - y <= 0
    MILP rows:   y <= u d,  y <= x - l (1 - d),  d in {0, 1}

so that {LP rows, MILP rows, eta box} is exactly y = ReLU(x).  The phase list
keeps (x-form, y-form, l, u) for every binary so the exact MILP can be emitted.

Bounds.  Constraint-free bounds of the value map (the *shadow*) are DeepZ
bounds.  Optional LP tightening uses the baseline's ordinary LP query (the LP
relaxation of the same HZ) solved by batched weak-duality iterations on GPU;
every iterate is a valid bound and nu = 0 equals the shadow bound.

Floating point.  This version tracks NO rounding error; results are diagnostic
until the rigorous version (N004) re-evaluates certificates with explicit
error terms.
"""

from __future__ import annotations

import dataclasses
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import numpy_helper

ENGINE_VERSION = "n003.3"  # .2: consumer-count release; .3: batched terminal_query


# ----------------------------------------------------------------------------
# Affine forms over the global factor space
# ----------------------------------------------------------------------------


@dataclasses.dataclass
class AZ:
    c: torch.Tensor   # value center, shape S (ONNX tensor shape incl. batch 1)
    G: torch.Tensor   # generators, shape [k, *S]; rows = factor ids 0..k-1

    @property
    def k(self) -> int:
        return int(self.G.shape[0])

    def pad_to(self, k: int) -> "AZ":
        if k == self.k:
            return self
        if k < self.k:
            raise ValueError("cannot shrink factor space")
        pad = torch.zeros((k - self.k,) + tuple(self.G.shape[1:]), dtype=self.G.dtype, device=self.G.device)
        return AZ(self.c, torch.cat([self.G, pad], 0))

    def map(self, fc, fg) -> "AZ":
        return AZ(fc(self.c), fg(self.G))


class RowStore:
    """LP relaxation rows A w <= b; columns are global factor ids."""

    def __init__(self, device, dtype):
        self.A_blocks: List[torch.Tensor] = []
        self.b_blocks: List[torch.Tensor] = []
        self.device, self.dtype = device, dtype
        self._cache: Optional[Tuple[int, torch.Tensor, torch.Tensor]] = None

    def add(self, A: torch.Tensor, b: torch.Tensor) -> None:
        if A.shape[0]:
            self.A_blocks.append(A)
            self.b_blocks.append(b)
            self._cache = None

    @property
    def n_rows(self) -> int:
        return sum(int(b.numel()) for b in self.b_blocks)

    def dense(self, K: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._cache is not None and self._cache[0] == K:
            return self._cache[1], self._cache[2]
        if not self.A_blocks:
            A = torch.zeros((0, K), device=self.device, dtype=self.dtype)
            b = torch.zeros((0,), device=self.device, dtype=self.dtype)
        else:
            As = []
            for A in self.A_blocks:
                if A.shape[1] < K:
                    A = torch.cat([A, A.new_zeros((A.shape[0], K - A.shape[1]))], 1)
                As.append(A)
            A = torch.cat(As, 0)
            b = torch.cat(self.b_blocks, 0)
        self._cache = (K, A, b)
        return A, b


@dataclasses.dataclass
class Phase:
    """One binary phase of an unstable ReLU (for exact MILP emission)."""

    layer: str
    index: int            # flat neuron index in its layer
    eta_col: int          # global factor id of its eta
    l: float
    u: float


@dataclasses.dataclass
class PhaseBlock:
    layer: str
    idx: torch.Tensor     # flat indices of unstable neurons
    eta0: int             # first eta column
    gx: torch.Tensor      # [m, K_at_creation] x-forms
    cx: torch.Tensor
    gy: torch.Tensor      # [m, K_at_creation + m] y-forms
    cy: torch.Tensor
    l: torch.Tensor
    u: torch.Tensor


# ----------------------------------------------------------------------------
# Batched LP bound query
# ----------------------------------------------------------------------------


def lp_upper(g: torch.Tensor, c: torch.Tensor, A: torch.Tensor, b: torch.Tensor,
             iters: int, lr: float, chunk: int = 4096, eval_every: int = 25,
             eval_dtype=torch.float64) -> torch.Tensor:
    """Upper bounds of c_i + g_i^T w over {A w <= b, w in [-1,1]^K}."""
    n = g.shape[0]
    out = torch.empty(n, device=g.device, dtype=eval_dtype)
    R = A.shape[0]
    A_e = A.to(eval_dtype); b_e = b.to(eval_dtype)
    for s in range(0, n, chunk):
        gs = g[s:s + chunk]; cs = c[s:s + chunk]
        best = cs.to(eval_dtype) + gs.to(eval_dtype).abs().sum(1)
        if R == 0 or iters <= 0:
            out[s:s + chunk] = best
            continue
        nu = gs.new_zeros((gs.shape[0], R))
        m1 = torch.zeros_like(nu); m2 = torch.zeros_like(nu)
        for it in range(1, iters + 1):
            resid = gs - nu @ A
            grad = b.unsqueeze(0) - torch.sign(resid) @ A.t()
            m1.mul_(0.9).add_(grad, alpha=0.1)
            m2.mul_(0.999).addcmul_(grad, grad, value=0.001)
            nu = (nu - lr * (m1 / (1 - 0.9 ** it)) / ((m2 / (1 - 0.999 ** it)).sqrt() + 1e-8)).clamp_(min=0.0)
            if it % eval_every == 0 or it == iters:
                nu_e = nu.to(eval_dtype)
                val = cs.to(eval_dtype) + nu_e @ b_e + (gs.to(eval_dtype) - nu_e @ A_e).abs().sum(1)
                best = torch.minimum(best, val)
        out[s:s + chunk] = best
    return out


# ----------------------------------------------------------------------------
# ONNX graph and propagation
# ----------------------------------------------------------------------------


def _attrs(node) -> Dict[str, object]:
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


@dataclasses.dataclass
class ReluLog:
    name: str
    n: int
    unstable_shadow: int
    unstable_final: int
    wall_s: float


class Engine:
    def __init__(self, path: str, device: str = "cuda", dtype=torch.float32):
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

    # -- helpers --------------------------------------------------------------
    def _const(self, env, name):
        v = env.get(name, self.consts.get(name))
        return v

    def propagate(self, lb: torch.Tensor, ub: torch.Tensor, tighten_iters: int = 0, lr: float = 0.05,
                  log: Optional[List[ReluLog]] = None):
        dev, dt = self.device, self.dtype
        lb = lb.reshape(self.input_shape).to(dev, dt); ub = ub.reshape(self.input_shape).to(dev, dt)
        c0 = (lb + ub) / 2
        r0 = ((ub - lb) / 2).reshape(-1)
        nz = torch.nonzero(r0 > 0).reshape(-1)
        K = int(nz.numel())
        G0 = torch.zeros((K, r0.numel()), device=dev, dtype=dt)
        G0[torch.arange(K, device=dev), nz] = r0[nz]
        env: Dict[str, object] = {self.input_name: AZ(c0, G0.reshape((K,) + tuple(c0.shape)))}
        self.input_factor_index = nz
        rows = RowStore(dev, dt)
        phases: List[PhaseBlock] = []
        remaining: Dict[str, int] = {}
        for node in self.nodes:
            for n in node.input:
                if n:
                    remaining[n] = remaining.get(n, 0) + 1
        for node in self.nodes:
            op = node.op_type
            ins = [self._const(env, n) if n else None for n in node.input]
            a = _attrs(node)
            out = None
            if op == "Constant":
                v = a["value"]
                arr = numpy_helper.to_array(v).copy()
                out = torch.as_tensor(arr).to(device=dev, dtype=dt if arr.dtype.kind == "f" else torch.int64)
            elif op in ("Identity", "Dropout"):
                out = ins[0]
            elif op == "MatMul":
                x, W = ins
                if isinstance(x, AZ) and not isinstance(W, AZ):
                    out = AZ(x.c @ W, x.G @ W)
                elif isinstance(W, AZ) and not isinstance(x, AZ):
                    out = AZ(x @ W.c, x @ W.G)
                else:
                    raise NotImplementedError("bilinear MatMul")
            elif op == "Gemm":
                x, W = ins[0], ins[1]
                bb = ins[2] if len(ins) > 2 else None
                if int(a.get("transA", 0)):
                    raise NotImplementedError("transA")
                if int(a.get("transB", 0)):
                    W = W.t()
                al = float(a.get("alpha", 1.0)); be = float(a.get("beta", 1.0))
                out = AZ(al * (x.c @ W) + (be * bb if bb is not None else 0), al * (x.G @ W))
            elif op in ("Add", "Sub"):
                x, y = ins
                sgn = 1.0 if op == "Add" else -1.0
                if isinstance(x, AZ) and isinstance(y, AZ):
                    k = max(x.k, y.k); xa, ya = x.pad_to(k), y.pad_to(k)
                    out = AZ(xa.c + sgn * ya.c, xa.G + sgn * ya.G)
                elif isinstance(x, AZ):
                    out = AZ(x.c + sgn * y, x.G.expand((x.k,) + tuple(torch.broadcast_shapes(x.c.shape, y.shape))) if y.dim() > x.c.dim() else x.G)
                elif isinstance(y, AZ):
                    out = AZ(x + sgn * y.c, sgn * y.G)
                else:
                    out = x + sgn * y
            elif op in ("Mul", "Div"):
                x, y = ins
                if isinstance(x, AZ) and not isinstance(y, AZ):
                    s = y if op == "Mul" else 1.0 / y
                    out = AZ(x.c * s, x.G * s.unsqueeze(0) if torch.is_tensor(s) else x.G * s)
                elif isinstance(y, AZ) and op == "Mul" and not isinstance(x, AZ):
                    out = AZ(y.c * x, y.G * x.unsqueeze(0))
                elif not isinstance(x, AZ) and not isinstance(y, AZ):
                    out = x * y if op == "Mul" else x / y
                else:
                    raise NotImplementedError("bilinear Mul/Div")
            elif op == "Conv":
                x, W = ins[0], ins[1]
                bb = ins[2] if len(ins) > 2 else None
                pads = list(a.get("pads", [0, 0, 0, 0]))
                if pads[0] != pads[2] or pads[1] != pads[3]:
                    raise NotImplementedError("asymmetric padding")
                kw = dict(stride=tuple(a.get("strides", [1, 1])), padding=(pads[0], pads[1]),
                          dilation=tuple(a.get("dilations", [1, 1])), groups=int(a.get("group", 1)))
                cc = F.conv2d(x.c, W, bb, **kw)
                Gx = x.G.reshape((x.k,) + tuple(x.c.shape[1:]))
                parts = [F.conv2d(Gx[s:s + 1024], W, None, **kw) for s in range(0, x.k, 1024)]
                GG = torch.cat(parts, 0) if parts else Gx.new_zeros((0,) + tuple(cc.shape[1:]))
                out = AZ(cc, GG.unsqueeze(1))
            elif op == "BatchNormalization":
                x = ins[0]
                gamma, beta, mean, var = ins[1:5]
                eps = float(a.get("epsilon", 1e-5))
                scale = gamma / torch.sqrt(var + eps); shift = beta - mean * scale
                shp = (1, -1) + (1,) * (x.c.dim() - 2)
                out = AZ(x.c * scale.view(shp) + shift.view(shp), x.G * scale.view((1,) + shp))
            elif op == "Flatten":
                x = ins[0]
                axis = int(a.get("axis", 1))
                shape = (int(np.prod(x.c.shape[:axis])), -1)
                out = AZ(x.c.reshape(shape), x.G.reshape((x.k,) + tuple(x.c.reshape(shape).shape))) if isinstance(x, AZ) else x.reshape(shape)
            elif op == "Reshape":
                x, shp = ins
                shp = [int(s) for s in shp.tolist()]
                if isinstance(x, AZ):
                    cshape = list(x.c.shape)
                    shp = [cshape[i] if s == 0 else s for i, s in enumerate(shp)]
                    nc = x.c.reshape(shp)
                    out = AZ(nc, x.G.reshape((x.k,) + tuple(nc.shape)))
                else:
                    out = x.reshape(shp)
            elif op == "Relu":
                x = ins[0].pad_to(K)
                out, K = self._relu(node.name, x, K, rows, phases, tighten_iters, lr, log)
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
        if isinstance(res, AZ):
            res = res.pad_to(K)
        return res, rows, K, phases

    def _relu(self, name, x: AZ, K: int, rows: RowStore, phases: List[PhaseBlock], iters: int, lr: float, log):
        t0 = time.time()
        dev, dt = self.device, self.dtype
        shape = x.c.shape
        fc = x.c.reshape(-1)
        fG = x.G.reshape(K, fc.numel())
        rad = fG.abs().sum(0)
        l = fc - rad; u = fc + rad
        unst = (l < 0) & (u > 0)
        n_sh = int(unst.sum())
        if iters > 0 and n_sh and rows.n_rows:
            idx = torch.nonzero(unst).reshape(-1)
            A, b = rows.dense(K)
            g = fG[:, idx].t().contiguous(); cc = fc[idx]
            ub_lp = lp_upper(g, cc, A, b, iters, lr).to(dt)
            lb_lp = -lp_upper(-g, -cc, A, b, iters, lr).to(dt)
            u = u.clone(); l = l.clone()
            u[idx] = torch.minimum(u[idx], ub_lp)
            l[idx] = torch.maximum(l[idx], lb_lp)
        neg = u <= 0; pos = l >= 0; unst = ~(neg | pos)
        idx = torch.nonzero(unst).reshape(-1)
        m = int(idx.numel())
        lam = pos.to(dt)
        lu, uu = l[idx], u[idx]
        lam_u = uu / (uu - lu)
        mu_u = -lam_u * lu / 2
        lam = lam.clone(); lam[idx] = lam_u
        yc = fc * lam
        yc[idx] = yc[idx] + mu_u
        yG = fG * lam.unsqueeze(0)
        eta = torch.zeros((m, fc.numel()), device=dev, dtype=dt)
        eta[torch.arange(m, device=dev), idx] = mu_u
        yG = torch.cat([yG, eta], 0)
        gx = fG[:, idx].t().contiguous()
        cx = fc[idx]
        gy = yG[:, idx].t().contiguous()
        cy = yc[idx]
        gx_full = torch.cat([gx, gx.new_zeros((m, m))], 1)
        rows.add(-gy, cy)                    # -y <= 0
        rows.add(gx_full - gy, cy - cx)      # x - y <= 0
        phases.append(PhaseBlock(name, idx, K, gx, cx, gy, cy, lu, uu))
        K = K + m
        if log is not None:
            if dev == "cuda":
                torch.cuda.synchronize()
            log.append(ReluLog(name, int(fc.numel()), n_sh, m, time.time() - t0))
        return AZ(yc.reshape(shape), yG.reshape((K,) + tuple(shape))), K


# ----------------------------------------------------------------------------
# VNNLIB (box inputs, disjunction of conjunctions of linear output atoms)
# ----------------------------------------------------------------------------

_NUM = r"[-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?)"


def _tokenize(text: str) -> List[str]:
    text = re.sub(r";[^\n]*", "", text)
    return re.findall(r"\(|\)|[^\s()]+", text)


def _parse(tokens: List[str], i: int = 0):
    if tokens[i] != "(":
        return tokens[i], i + 1
    lst = []; i += 1
    while tokens[i] != ")":
        e, i = _parse(tokens, i)
        lst.append(e)
    return lst, i + 1


@dataclasses.dataclass
class Spec:
    boxes: List[Tuple[np.ndarray, np.ndarray]]     # input disjunction (usually 1)
    disjuncts: List[List[Tuple[np.ndarray, float]]]  # unsafe: OR_d AND_k (a^T Y <= b)


def _linear(expr, n_out: int):
    """Return (a, b0) with expr = a^T Y + b0 for simple atoms."""
    if isinstance(expr, str):
        if expr.startswith("Y_"):
            a = np.zeros(n_out); a[int(expr[2:])] = 1.0
            return a, 0.0
        return np.zeros(n_out), float(expr)
    head = expr[0]
    if head in ("+", "-"):
        parts = [_linear(e, n_out) for e in expr[1:]]
        if head == "-" and len(parts) == 1:
            return -parts[0][0], -parts[0][1]
        a = parts[0][0].copy(); b = parts[0][1]
        for pa, pb in parts[1:]:
            a = a + pa if head == "+" else a - pa
            b = b + pb if head == "+" else b - pb
        return a, b
    if head == "*":
        (a1, b1), (a2, b2) = _linear(expr[1], n_out), _linear(expr[2], n_out)
        if not a1.any():
            return b1 * a2, b1 * b2
        if not a2.any():
            return b2 * a1, b2 * b1
    raise NotImplementedError(str(expr))


def _atom(expr, n_out: int):
    op, lhs, rhs = expr
    al, bl = _linear(lhs, n_out); ar, br = _linear(rhs, n_out)
    if op == "<=":   # lhs <= rhs  ->  (al - ar)^T Y <= br - bl
        return al - ar, br - bl
    if op == ">=":
        return ar - al, bl - br
    raise NotImplementedError(op)


def parse_vnnlib(path: str, n_in: int, n_out: int) -> Spec:
    toks = _tokenize(open(path).read())
    i = 0; asserts = []
    while i < len(toks):
        e, i = _parse(toks, i)
        if isinstance(e, list) and e and e[0] == "assert":
            asserts.append(e[1])
    lb = np.full(n_in, -np.inf); ub = np.full(n_in, np.inf)
    out_forms = []
    in_disj = None

    def is_input(e):
        return isinstance(e, list) and len(e) == 3 and isinstance(e[1], str) and e[1].startswith("X_")

    for e in asserts:
        if is_input(e):
            k = int(e[1][2:]); v = float(e[2])
            if e[0] == "<=":
                ub[k] = min(ub[k], v)
            elif e[0] == ">=":
                lb[k] = max(lb[k], v)
            else:
                raise NotImplementedError
        elif isinstance(e, list) and e[0] == "or" and all(isinstance(d, list) and d[0] == "and" and all(is_input(x) for x in d[1:]) for d in e[1:]):
            in_disj = []
            for d in e[1:]:
                l2 = np.full(n_in, -np.inf); u2 = np.full(n_in, np.inf)
                for x in d[1:]:
                    k = int(x[1][2:]); v = float(x[2])
                    if x[0] == "<=":
                        u2[k] = min(u2[k], v)
                    else:
                        l2[k] = max(l2[k], v)
                in_disj.append((l2, u2))
        else:
            out_forms.append(e)
    boxes = in_disj if in_disj is not None else [(lb, ub)]
    # output: conjunction of top-level asserts, each may be an OR
    disj: List[List[Tuple[np.ndarray, float]]] = [[]]
    for e in out_forms:
        if e[0] == "or":
            alts = []
            for d in e[1:]:
                if d[0] == "and":
                    alts.append([_atom(x, n_out) for x in d[1:]])
                else:
                    alts.append([_atom(d, n_out)])
        elif e[0] == "and":
            alts = [[_atom(x, n_out) for x in e[1:]]]
        else:
            alts = [[_atom(e, n_out)]]
        disj = [p + q for p in disj for q in alts]
    return Spec(boxes, disj)


def certify_disjuncts(out: AZ, rows: RowStore, K: int, spec_disjuncts, iters: int, lr: float):
    """For each unsafe disjunct return an upper bound on max_w a_1^T Y - b_1 over
    the LP relaxation intersected with the other atoms (negative => disjunct empty)."""
    c = out.c.reshape(-1); G = out.G.reshape(K, c.numel())
    A, b = rows.dense(K)
    res = []
    for atoms in spec_disjuncts:
        # atom: a^T Y <= bb  (unsafe).  Disjunct empty iff no w satisfies all atoms.
        # Bound: max over LP of (bb_1 - a_1^T Y) subject to other atoms <0  ->  empty.
        a0, b0 = atoms[0]
        a0t = torch.as_tensor(a0, device=c.device, dtype=c.dtype)
        g = -(G @ a0t).unsqueeze(0)
        cc = (b0 - c @ a0t).reshape(1)
        if len(atoms) > 1:
            extra_A = torch.stack([G @ torch.as_tensor(ak, device=c.device, dtype=c.dtype) for ak, _ in atoms[1:]])
            extra_b = torch.as_tensor([bk for _, bk in atoms[1:]], device=c.device, dtype=c.dtype) - torch.stack(
                [c @ torch.as_tensor(ak, device=c.device, dtype=c.dtype) for ak, _ in atoms[1:]])
            AA = torch.cat([A, extra_A], 0); bb = torch.cat([b, extra_b], 0)
        else:
            AA, bb = A, b
        res.append(float(lp_upper(g, cc, AA, bb, iters, lr)[0]))
    return res


# ----------------------------------------------------------------------------
# Batched terminal query (n003.3): bounds + box-maximiser witness candidates
# ----------------------------------------------------------------------------


def lp_upper_with_argmax(g: torch.Tensor, c: torch.Tensor, A: torch.Tensor, b: torch.Tensor,
                         iters: int, lr: float, eval_every: int = 25):
    """Like lp_upper (single chunk) but also returns, per objective, the box
    maximiser sign(g - A^T nu) at the best evaluated nu (a witness candidate;
    it need not satisfy A w <= b)."""
    R = A.shape[0]
    best = c.double() + g.double().abs().sum(1)
    arg = torch.sign(g)
    if R == 0 or iters <= 0:
        return best, arg
    A64 = A.double(); b64 = b.double()
    nu = g.new_zeros((g.shape[0], R)); m1 = torch.zeros_like(nu); m2 = torch.zeros_like(nu)
    for it in range(1, iters + 1):
        resid = g - nu @ A
        grad = b.unsqueeze(0) - torch.sign(resid) @ A.t()
        m1.mul_(0.9).add_(grad, alpha=0.1)
        m2.mul_(0.999).addcmul_(grad, grad, value=0.001)
        nu = (nu - lr * (m1 / (1 - 0.9 ** it)) / ((m2 / (1 - 0.999 ** it)).sqrt() + 1e-8)).clamp_(min=0.0)
        if it % eval_every == 0 or it == iters:
            nu_e = nu.double()
            r64 = g.double() - nu_e @ A64
            val = c.double() + nu_e @ b64 + r64.abs().sum(1)
            better = val < best
            best = torch.where(better, val, best)
            arg[better] = torch.sign(r64[better]).to(arg.dtype)
    return best, arg


def terminal_query(out: AZ, rows: RowStore, K: int, disjuncts, iters: int, lr: float, chunk: int = 512):
    """Per unsafe disjunct: (upper bound of violation, witness candidate w or None).
    Single-atom disjuncts are batched; multi-atom ones use the extra-row form."""
    c = out.c.reshape(-1); G = out.G.reshape(K, c.numel())
    A, b = rows.dense(K)
    dev, dt = c.device, c.dtype
    bounds = [None] * len(disjuncts); args = [None] * len(disjuncts)
    single = [i for i, d in enumerate(disjuncts) if len(d) == 1]
    for s in range(0, len(single), chunk):
        ids = single[s:s + chunk]
        Am = torch.stack([torch.as_tensor(disjuncts[i][0][0], device=dev, dtype=dt) for i in ids])   # [m, n_out]
        bm = torch.as_tensor([disjuncts[i][0][1] for i in ids], device=dev, dtype=dt)
        g = -(Am @ G.t())                     # [m, K]
        cc = bm - Am @ c
        bd, ag = lp_upper_with_argmax(g.contiguous(), cc, A, b, iters, lr)
        for j, i in enumerate(ids):
            bounds[i] = float(bd[j]); args[i] = ag[j]
    for i, atoms in enumerate(disjuncts):
        if len(atoms) == 1:
            continue
        a0 = torch.as_tensor(atoms[0][0], device=dev, dtype=dt)
        g = -(G @ a0).unsqueeze(0); cc = (atoms[0][1] - c @ a0).reshape(1)
        ex = torch.stack([G @ torch.as_tensor(ak, device=dev, dtype=dt) for ak, _ in atoms[1:]])
        exb = torch.as_tensor([bk for _, bk in atoms[1:]], device=dev, dtype=dt) - torch.stack(
            [c @ torch.as_tensor(ak, device=dev, dtype=dt) for ak, _ in atoms[1:]])
        bd, ag = lp_upper_with_argmax(g, cc, torch.cat([A, ex]), torch.cat([b, exb]), iters, lr)
        bounds[i] = float(bd[0]); args[i] = ag[0]
    return bounds, args
