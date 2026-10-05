"""Neural-HZ rigorous engine n016 = n011.2 + recorded smooth units.

The Sigmoid/Tanh transformer is the n009 one (DeepZ shadow y = lam x + mu + nu eta with a fresh
eta per unit, rigorous shifted chord/tangent rows).  n016 additionally records, per smooth layer,
the x value map (gx, cx, ex), the shadow parameters (lam, mu, nu), the y centre/error and the
bounds (l, u), so terminal plans can write the smooth rows sparsely (explicit x) and add phase
segments (nhz_terminal_v8.py).  Propagation results are identical to n011.2.
"""
import torch

from nhz_sound import SAZ, SRows, gamma
from nhz_sound_mp import map_G, radius64
from nhz_sound_mp2 import rigorous_lines, _sf, _sdf, FPAD
from nhz_sound_v9 import F64, _up32
from nhz_sound_v11b import SoundEngineV11b

V16_VERSION = "n016.0"


class SoundEngineV16(SoundEngineV11b):
    def propagate(self, lb, ub, iters: int = 300, lr: float = 0.05, log=None):
        self.smooth_phases = []
        return super().propagate(lb, ub, iters, lr, log)

    def _smooth(self, kind, x):
        K = self.K; x = x.pad_to(K); dev = self.device; rows = self.rows
        shape = x.c.shape
        fc = x.c.reshape(-1); fG = x.G.reshape(K, fc.numel()); fe = x.e.reshape(-1)
        rad_all = radius64(fG); allidx = torch.arange(fc.numel(), device=dev)
        l, u = self._bounds(fc, fG, fe, allidx, SRows(), K, 0, self.lr, rad_all)
        if self.iters > 0 and rows.n_rows:
            l2, u2 = self._bounds(fc, fG, fe, allidx, rows, K, self.iters, self.lr, rad_all)
            l = torch.maximum(l, l2); u = torch.minimum(u, u2)
        u = torch.maximum(u, l); n = fc.numel()
        lam = torch.minimum(_sdf(kind, l), _sdf(kind, u)) * (1 - 1e-9)
        lo = _sf(kind, l) - lam * l; hi = _sf(kind, u) - lam * u
        padf = FPAD * (1 + _sf(kind, l).abs() + _sf(kind, u).abs() + (lam * l).abs() + (lam * u).abs())
        lo = lo - padf; hi = hi + padf
        mu = (lo + hi) / 2; nu = (hi - lo) / 2 * (1 + gamma(2, F64))
        yc = lam * fc + mu
        yG, se = map_G(fG, lambda g: g * lam.unsqueeze(0))
        ye = lam * fe + gamma(3, F64) * (lam * (fc.abs() + rad_all) + mu.abs()) + se
        eta = torch.zeros((n, n), device=dev, dtype=torch.float32)
        eta[torch.arange(n, device=dev), torch.arange(n, device=dev)] = _up32(nu)
        yG = torch.cat([yG, eta], 0); Kn = K + n
        lows, ups = rigorous_lines(kind, l, u)
        gx = torch.cat([fG, fG.new_zeros((n, n))], 0).double(); gy = yG.double()
        for a_, b_ in ups:
            rows.add((gy - gx * a_.unsqueeze(0)).t().contiguous(), b_ - yc + a_ * fc + ye + a_.abs() * fe)
        for a_, b_ in lows:
            rows.add((gx * a_.unsqueeze(0) - gy).t().contiguous(), yc - a_ * fc - b_ + ye + a_.abs() * fe)
        # n016: record the unit (x value map, shadow parameters, rigorous bounds) for terminal plans
        self.smooth_phases.append(dict(kind=kind, eta0=K, n=n, gx=fG.t().contiguous(), cx=fc.clone(), ex=fe.clone(),
                                       lam=lam.clone(), mu=mu.clone(), nu=_up32(nu).double(), yc=yc.clone(), ye=ye.clone(),
                                       l=l.clone(), u=u.clone()))
        self.K = Kn
        return SAZ(yc.reshape(shape), yG.reshape((Kn,) + tuple(shape)), ye.reshape(shape))

