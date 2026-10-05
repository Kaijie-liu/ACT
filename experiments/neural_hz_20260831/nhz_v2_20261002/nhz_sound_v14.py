"""Neural-HZ rigorous engine n014 = n011.2 + difference-aware score bounds for Softmax.

When the Softmax input is (a constant multiple of) a state-state product S = Q @ K^T, the
score differences d_iab = s_ib - s_ia are bounded directly as  scale * sum_d q_id (k_db - k_da):
the generators of k_b - k_a cancel before the product remainder is taken, instead of adding
the two independent product remainders of s_ib and s_ia.  With x = Q, y = K^T (value maps
c + G w, errors e) and Delta = y_b - y_a:
   |d_true - (d_centre + d_lin^T w)| <= |scale| * sum_d [ (r_q + e_q)(r_Delta + e_Delta) + |c_q| e_Delta + e_q |c_Delta| ]
where d_lin are the difference generators on the factors that are not the product's fresh
factors (the product's centre shift 0.5 sum_k g_q,k g_Delta,k is part of d_centre, and the
bound on the quadratic part around that shift is sum_d r_q r_Delta, see THEORY.md 14).  The
softmax uses min(old radius, this radius) for its bounds (box, Hessian, remainder width);
its value map is unchanged.  Everything else is n011.2.
"""
import torch

from nhz_sound import SAZ, gamma
from nhz_sound_mp import map_G, radius64
from nhz_sound_v9 import F64, _up32, _is_state
from nhz_sound_v11b import SoundEngineV11b

V14_VERSION = "n014.0"


class SoundEngineV14(SoundEngineV11b):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._prod = {}; self.dr_improved = 0; self.dr_total = 0

    def op_MatMul(self, node, ins, a):
        x, W = ins
        if _is_state(x) and _is_state(W):
            K0 = self.K
            out = super().op_MatMul(node, ins, a)
            self._prod[node.output[0]] = dict(x=x.pad_to(K0), y=W.pad_to(K0), K0=K0, K1=self.K, scale=1.0)
            return out
        return super().op_MatMul(node, ins, a)

    def op_Mul(self, node, ins, a):
        out = super().op_Mul(node, ins, a)
        for st, cst in ((0, 1), (1, 0)):
            info = self._prod.get(node.input[st]) if node.input[st] else None
            c = ins[cst]
            if info is not None and torch.is_tensor(c) and c.numel() == 1 and not _is_state(c):
                self._prod[node.output[0]] = dict(info, scale=info["scale"] * float(c.reshape(-1)[0]))
        return out

    def _improve_dr(self, node, dr, dG, K):
        info = self._prod.get(node.input[0])
        if info is None:
            return dr
        x, y, K0, K1, scale = info["x"], info["y"], info["K0"], info["K1"], abs(info["scale"])
        dr_lin = dG[:K0].abs().sum(0) + (dG[K1:].abs().sum(0) if dG.shape[0] > K1 else 0.0)
        dr_lin = dr_lin * (1 + gamma(K + 2, F64))
        gq = x.G.double(); rq = gq.abs().sum(0) * (1 + gamma(K0 + 2, F64)) + x.e                  # (..., T, dk)
        gy = y.G                                                                                   # (K0, ..., dk, T) float32
        T = y.c.shape[-1]
        rD = torch.zeros(tuple(y.c.shape) + (T,), device=self.device, dtype=F64)                   # (..., dk, a, b)
        for s0 in range(0, K0, 64):
            g = gy[s0:s0 + 64].double()
            rD += (g.unsqueeze(-2) - g.unsqueeze(-1)).abs().sum(0)
        rD = rD * (1 + gamma(K0 + 2, F64))
        eD = y.e.unsqueeze(-2) + y.e.unsqueeze(-1)                                                 # (..., dk, a, b)  [y_b - y_a errors]
        cD = y.c.unsqueeze(-2) - y.c.unsqueeze(-1)
        dk = x.c.shape[-1]
        R = torch.einsum("...id,...dab->...iab", rq, rD + eD) + torch.einsum("...id,...dab->...iab", x.c.abs(), eD) \
            + torch.einsum("...id,...dab->...iab", x.e, cD.abs())
        R = scale * R * (1 + gamma(dk * K0 + 8, F64)) + gamma(dk + 4, F64) * scale * torch.einsum("...id,...dab->...iab", x.c.abs() + rq, y.c.abs().unsqueeze(-1) + rD)
        new = dr_lin + R
        self.dr_improved += int((new < dr).sum()); self.dr_total += int(dr.numel())
        return torch.minimum(dr, new)

    def op_Softmax(self, node, ins, a):
        K = self.K; s = ins[0].pad_to(K); dev = self.device; rows = self.rows
        axis = int(a.get("axis", -1)) % s.c.dim()
        perm = [i for i in range(s.c.dim()) if i != axis] + [axis]; inv = [perm.index(i) for i in range(s.c.dim())]
        c = s.c.permute(perm); G = s.G.permute([0] + [p + 1 for p in perm]).double(); e = s.e.permute(perm)
        n = c.shape[-1]; eye = torch.eye(n, device=dev, dtype=F64); mask = 1 - eye
        dc = c.unsqueeze(-2) - c.unsqueeze(-1)
        dG = G.unsqueeze(-2) - G.unsqueeze(-1)
        de = (e.unsqueeze(-2) + e.unsqueeze(-1)) * mask               # radius of the true difference around the form
        dr = dG.abs().sum(0) * (1 + gamma(K + 2, F64))
        if axis == s.c.dim() - 1:
            dr = self._improve_dr(node, dr, dG, K)
        dlo, dhi = dc - dr - de, dc + dr + de
        e0 = torch.exp(dc); S0 = e0.sum(-1); p0 = 1.0 / S0
        grad = -e0 / (S0 ** 2).unsqueeze(-1) * mask
        pG = (grad.unsqueeze(0) * dG).sum(-1)
        elo = torch.exp(dlo) * mask + eye; ehi = torch.exp(dhi) * mask + eye
        Tlo = elo.sum(-1); Thi = ehi.sum(-1)
        Hoff = 2 * ehi.unsqueeze(-1) * ehi.unsqueeze(-2) / (Tlo ** 3).unsqueeze(-1).unsqueeze(-1)
        Hdiag = torch.maximum(2 * ehi ** 2 / Tlo.unsqueeze(-1) ** 3, ehi / Tlo.unsqueeze(-1) ** 2)
        Hb = (Hoff * mask.unsqueeze(0) if False else Hoff) * (1 - torch.eye(n, device=dev, dtype=F64)) + torch.diag_embed(Hdiag)
        Hb = Hb * mask.unsqueeze(-1) * mask.unsqueeze(-2)
        delta = torch.maximum(dhi - dc, dc - dlo) * mask
        rem = 0.5 * torch.einsum("...ij,...ijk,...ik->...i", delta, Hb, delta)
        # the true difference = form + offset with |offset| <= de; that part of the tangent term goes to e_p
        ep = (grad.abs() * de).sum(-1)
        padv = 1e-12 * (p0 + rem + (grad.abs() * (dr + de)).sum(-1)) + gamma(n * K + 8, F64) * (p0 + (grad.abs() * dr).sum(-1))
        rem = rem + padv
        plo = (1.0 / Thi) * (1 - 1e-12); phi = (1.0 / Tlo) * (1 + 1e-12)
        # n011: per-coordinate choice of the smaller sound enclosure (tangent plane + remainder,
        # or the interval [plo, phi] with one fresh factor); each coordinate has its own fresh factor
        r_tan = pG.abs().sum(0) * (1 + gamma(K + 2, F64)) + rem + ep
        r_int = (phi - plo) / 2
        use_int = (rem + ep) >= r_int      # n011.2: switch only where the remainder alone is uninformative
        self.softmax_interval_coords += int(use_int.sum()); self.softmax_coords += int(use_int.numel())
        couple = use_int & ((rem + ep) < r_int)          # tangent relation still informative
        tan_c, tan_G, tan_r = p0.clone(), pG.clone(), (rem + ep).clone()
        p0 = torch.where(use_int, (plo + phi) / 2, p0)
        pG = torch.where(use_int.unsqueeze(0), torch.zeros_like(pG), pG)
        rem = torch.where(use_int, r_int * (1 + gamma(4, F64)) + 1e-300, rem)
        ep = torch.where(use_int, torch.zeros_like(ep), ep)
        n_new = int(rem.numel())
        pG32 = pG.float(); se = (pG - pG32.double()).abs().sum(0)
        eta = torch.zeros((n_new,) + tuple(p0.shape), device=dev, dtype=torch.float32)
        flat = eta.reshape(n_new, -1); ar = torch.arange(n_new, device=dev)
        flat[ar, ar] = _up32(rem.reshape(-1))
        Gp = torch.cat([pG32, eta], 0)
        epp = ep + se
        out = SAZ(p0.permute(inv).contiguous(), Gp.permute([0] + [i + 1 for i in inv]).contiguous(), epp.permute(inv).contiguous())
        Kn = K + n_new; self.K = Kn
        fc = out.c.reshape(-1); fGd = out.G.reshape(Kn, -1).double(); fe = out.e.reshape(-1)
        rows.add(fGd.t().contiguous(), (phi.permute(inv).reshape(-1) - fc + fe))
        rows.add(-fGd.t().contiguous(), (fc - plo.permute(inv).reshape(-1) + fe))
        sc = out.c.sum(axis).reshape(-1); sG = out.G.double().sum(axis + 1).reshape(Kn, -1); se_sum = out.e.sum(axis).reshape(-1)
        pad1 = gamma(n + 2, F64) * (1 + out.c.abs().sum(axis).reshape(-1))
        rows.add(sG.t().contiguous(), 1.0 - sc + se_sum + pad1)
        rows.add(-sG.t().contiguous(), sc - 1.0 + se_sum + pad1)
        # coupling rows for interval coordinates:  |y_i - (tan_c_i + tan_G_i^T w)| <= tan_r_i + e_i (+ pads)
        cpl = couple.permute(inv).reshape(-1)
        if bool(cpl.any()):
            ids = torch.nonzero(cpl).reshape(-1)
            tG = tan_G.permute([0] + [i + 1 for i in inv]).reshape(K, -1)[:, ids]           # (K, m)
            tc = tan_c.permute(inv).reshape(-1)[ids]; tr = tan_r.permute(inv).reshape(-1)[ids]
            yG = fGd[:, ids]                                                                 # (Kn, m), y value map
            D = yG.clone(); D[:K] -= tG
            padc = gamma(K + 4, F64) * (fc[ids].abs() + tc.abs() + yG.abs().sum(0) + tG.abs().sum(0)) + 1e-300
            rhs = tr + fe[ids] + padc
            rows.add(D.t().contiguous(), rhs - (fc[ids] - tc))
            rows.add(-D.t().contiguous(), rhs + (fc[ids] - tc))
            self.softmax_coupled = getattr(self, "softmax_coupled", 0) + int(ids.numel())
        return out

