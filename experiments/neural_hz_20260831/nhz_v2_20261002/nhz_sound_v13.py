"""Neural-HZ rigorous engine n013 = n011.2 + simplex-aware cross term for P @ V.

For MatMul(P, V) with P = Softmax(S, axis = last) (the contraction axis) and V a state:
z_id = sum_j p_ij v_jd.  With p = p_hat + dp, v = v_hat + dv (dp, dv the true deviations
from the value-map centres),
    z = sum p_hat v_hat + sum p_hat dv + sum dp v_hat + sum dp dv .
The first three terms are the DeepZ product's centre and generators.  The cross term
X_id = sum_j dp_ij dv_jd is bounded by the fractional knapsack
    max  sum_j |dp_j| r_jd   s.t.  A_ij <= dp_ij <= B_ij,  sum_j dp_ij = 1 - sum_j p_hat_ij,
with A, B the intersection of the rigorous softmax box [plo, phi] - p_hat and the state
radius, and r = radius of V (incl. its error).  sum_j p_ij = 1 holds exactly for the true
softmax.  Relaxing |dp_j| to u_j + w_j (u, w >= 0) gives an LP solved greedily (take every
capacity, then remove the excess of one sign from the lowest-r tokens).  Per output
coordinate the engine keeps the smaller of this enclosure and the n009 DeepZ product
(which moves 0.5 sum_k gx_k gy_k into the centre).  Both have the same generators; each
coordinate keeps its own fresh factor.  Everything else is n011.2.
"""
import torch

from nhz_sound import SAZ, gamma
from nhz_sound_mp import radius64
from nhz_sound_v9 import F64, _up32, _is_state
from nhz_sound_v11b import SoundEngineV11b

V13_VERSION = "n013.0"


def knapsack_cross(A, B, S, r):
    """A, B: (..., i, j) bounds of dp; S: (..., i) required sum; r: (..., j, d) radii >= 0.
    Returns (..., i, d) upper bound of max sum_j |dp_ij| r_jd over the box with sum = S."""
    Bp = B.clamp(min=0); Am = (-A).clamp(min=0)
    P = Bp.sum(-1); N = Am.sum(-1)                                         # (..., i)
    tot = Bp @ r + Am @ r                                                  # (..., i, d)
    E = P - N - S                                                          # >0: too much positive mass
    order = torch.argsort(r, dim=-2)                                       # ascending r per column d: (..., j, d)
    rs = torch.gather(r, -2, order)                                        # (..., j, d)

    def removal(cap, amount):
        # cap (..., i, j), amount (..., i) >= 0 ; remove 'amount' from lowest-r tokens per (i, d)
        capx = cap.unsqueeze(-1).expand(cap.shape + (r.shape[-1],))          # (..., i, j, d)
        idx = order.unsqueeze(-3).expand(capx.shape)
        cs = torch.gather(capx, -2, idx)                                   # caps in r-order
        before = torch.cumsum(cs, -2) - cs
        take = (amount.unsqueeze(-1).unsqueeze(-1) - before).clamp(min=0)
        take = torch.minimum(take, cs)
        return (take * rs.unsqueeze(-3)).sum(-2), cs.sum(-2)               # (..., i, d)

    rem_pos, cap_pos = removal(Bp, E.clamp(min=0))
    rem_neg, cap_neg = removal(Am, (-E).clamp(min=0))
    out = tot - rem_pos - rem_neg
    # infeasible sum (cannot happen for the true p up to rounding): fall back to the unconstrained bound
    bad = ((E > 0) & (E > P + 1e-12)) | ((E < 0) & (-E > N + 1e-12))
    return torch.where(bad.unsqueeze(-1), tot, out)


class SoundEngineV13(SoundEngineV11b):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._sm_src13 = {}; self.cross_knapsack_coords = 0; self.cross_coords = 0

    def op_Softmax(self, node, ins, a):
        s = ins[0]
        axis = int(a.get("axis", -1)) % s.c.dim()
        if axis == s.c.dim() - 1:
            self._sm_src13[node.output[0]] = s
        return super().op_Softmax(node, ins, a)

    def op_MatMul(self, node, ins, a):
        src = self._sm_src13.get(node.input[0])
        if src is not None and _is_state(ins[0]) and _is_state(ins[1]):
            return self._pv_product(src, ins[0], ins[1])
        return super().op_MatMul(node, ins, a)

    def _softmax_box(self, s):
        K = self.K; s = s.pad_to(K); dev = self.device
        sc = s.c; sG = s.G.double(); se_ = s.e; T = sc.shape[-1]
        dG = sG.unsqueeze(-2) - sG.unsqueeze(-1)
        dr = dG.abs().sum(0) * (1 + gamma(K + 2, F64)); del dG
        de = se_.unsqueeze(-2) + se_.unsqueeze(-1)
        dc = sc.unsqueeze(-2) - sc.unsqueeze(-1)
        eye = torch.eye(T, device=dev, dtype=F64); mask = 1 - eye
        Thi = (torch.exp(dc + dr + de) * mask + eye).sum(-1); Tlo = (torch.exp(dc - dr - de) * mask + eye).sum(-1)
        return (1.0 / Thi) * (1 - 1e-12), (1.0 / Tlo) * (1 + 1e-12)

    def _pv_product(self, s, x, y):
        plo, phi = self._softmax_box(s)
        K = self.K; x = x.pad_to(K); y = y.pad_to(K); dev = self.device
        gx = x.G.double(); gy = y.G.double()
        cz = x.c @ y.c
        Gz = x.c.unsqueeze(0) @ gy + gx @ y.c.unsqueeze(0)
        m_ = 0.5 * torch.einsum("k...nt,k...tm->...nm", gx, gy)
        absd = torch.einsum("k...nt,k...tm->...nm", gx.abs(), gy.abs())
        rx = gx.abs().sum(0); ry = gy.abs().sum(0)
        rad1 = (rx @ ry - 0.5 * absd).clamp(min=0)
        t = x.c.shape[-1]
        ax = x.c.abs() + rx; ay = y.c.abs() + ry
        e1 = ax @ y.e + x.e @ ay + x.e @ y.e
        # option 2: simplex-aware cross term
        rpx = rx * (1 + gamma(K + 2, F64)) + x.e
        A = torch.maximum(plo - x.c, -rpx); B = torch.minimum(phi - x.c, rpx)
        A = torch.minimum(A, B)
        S = 1.0 - x.c.sum(-1)
        rv = ry * (1 + gamma(K + 2, F64)) + y.e
        rad2 = knapsack_cross(A, B, S, rv)
        e2 = x.c.abs() @ y.e + x.e @ y.c.abs()
        mag = ax @ ay
        e_round = gamma(t * K + 4, F64) * 4 * mag
        r1 = rad1 * (1 + gamma(t * K + 4, F64)) + e1
        r2 = rad2 * (1 + gamma(t * K + 8, F64)) + e2
        use2 = r2 < r1
        self.cross_knapsack_coords += int(use2.sum()); self.cross_coords += int(use2.numel())
        cz_out = torch.where(use2, cz, cz + m_)
        rad = torch.where(use2, rad2 * (1 + gamma(t * K + 8, F64)), rad1 * (1 + gamma(t * K + 4, F64))) + e_round
        eout = torch.where(use2, e2, e1) + e_round
        Gz32 = Gz.float(); sstore = (Gz - Gz32.double()).abs().sum(0)
        n_new = int(rad.numel())
        eta = torch.zeros((n_new,) + tuple(cz.shape), device=dev, dtype=torch.float32)
        flat = eta.reshape(n_new, -1); ar = torch.arange(n_new, device=dev)
        flat[ar, ar] = _up32(rad.reshape(-1))
        self.K = K + n_new
        return SAZ(cz_out, torch.cat([Gz32, eta], 0), eout + sstore)
