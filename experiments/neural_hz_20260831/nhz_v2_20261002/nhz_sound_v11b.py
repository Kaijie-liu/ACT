"""Variant n011.2 (selection rule: remainder alone >= interval radius) of engine n011 = n009.2 + per-coordinate softmax enclosure choice.

Softmax (n009): tangent plane in the score differences plus an interval-Hessian remainder.
For wide score ranges the remainder grows like exp(width)^2 and exceeds the trivial range
[plo, phi] of the output by orders of magnitude (LOG N054, ViT pgd_2_3_16).  n011 picks per
output coordinate the enclosure with the smaller radius: either the n009 form or the
interval [plo, phi] written as centre + one fresh factor.  Both are sound for that coordinate
and every coordinate keeps its own fresh factor, so the state stays sound.  The box and
sum-to-one rows are added as before.  Everything else is inherited unchanged.
"""
import torch

from nhz_sound import SAZ, gamma
from nhz_sound_mp import map_G, radius64
from nhz_sound_v9 import SoundEngineV9, F64, _up32

V11_VERSION = "n011.2"  # .1: interval coordinates keep the tangent-plane relation as two coupling rows


class SoundEngineV11b(SoundEngineV9):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.softmax_interval_coords = 0; self.softmax_coords = 0

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

