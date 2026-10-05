"""N008 probe: zonogon pair hulls as extra LP rows (terminal-only effect).

For unstable ReLUs i, j of the same layer, D_ij = {(x_i(w), x_j(w)) : w in box}
is the exact 2-D projection of the latent box (a zonogon).  We over-approximate
it by a polygon P (support function in T directions, intersected with the
valid bounds), and compute H = conv{(x_i, x_j, ReLU x_i, ReLU x_j) : (x_i,x_j) in P}
exactly (lift the vertices of P cut by the axes).  Facets of H that involve an
output coordinate and both neurons are valid rows for the exact HZ; they are
added to the terminal LP of the worst disjunct.  Pairs are chosen structurally
(highest |cosine| of x-forms within the layer), not from LP state.
Diagnostic only (float, no rounding control).
"""

import csv
import os
import sys
import time

import numpy as np
import torch
from scipy.spatial import ConvexHull

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_engine import Engine, lp_upper, parse_vnnlib  # noqa: E402

ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"


def polygon(cx, gx, bounds, T=16):
    """Outer polygon of {(c + G^T w)} for 2-D forms; cx [2], gx [2, K] (numpy)."""
    th = np.linspace(0, 2 * np.pi, T, endpoint=False)
    D = np.stack([np.cos(th), np.sin(th)], 1)
    h = D @ cx + np.abs(D @ gx).sum(1)
    (l1, u1), (l2, u2) = bounds
    D = np.vstack([D, [[1, 0], [-1, 0], [0, 1], [0, -1]]])
    h = np.concatenate([h, [u1, -l1, u2, -l2]])
    # vertices by brute-force pairwise intersection (small T)
    pts = []
    for a in range(len(D)):
        for b in range(a + 1, len(D)):
            M = np.array([D[a], D[b]])
            if abs(np.linalg.det(M)) < 1e-12:
                continue
            p = np.linalg.solve(M, [h[a], h[b]])
            if np.all(D @ p <= h + 1e-9 * (1 + np.abs(h))):
                pts.append(p)
    pts = np.array(pts)
    return D, h, pts


def lifted_points(D, h, pts):
    """Vertices of P cut by the coordinate axes, lifted by ReLU."""
    cand = [p for p in pts]
    # intersections of P's edges with x=0 and y=0: intersect each halfplane line with the axis
    for axis in (0, 1):
        for a in range(len(D)):
            d = D[a]
            other = 1 - axis
            if abs(d[other]) < 1e-12:
                continue
            p = np.zeros(2); p[other] = h[a] / d[other]
            if np.all(D @ p <= h + 1e-9 * (1 + np.abs(h))):
                cand.append(p)
    if np.all(0 <= h + 1e-12):
        cand.append(np.zeros(2))
    P = np.array(cand)
    L = np.concatenate([P, np.maximum(P, 0)], 1)
    return np.unique(np.round(L, 12), axis=0)


def pair_rows(cx, gx, cy, gy, bnds, i, j, K):
    """Facet rows (in w) of the pair hull for neurons i, j of one phase block."""
    D, h, pts = polygon(np.array([cx[i], cx[j]]), np.stack([gx[i], gx[j]]), bnds)
    if len(pts) < 3:
        return []
    L = lifted_points(D, h, pts)
    if len(L) < 6:
        return []
    try:
        hull = ConvexHull(L, qhull_options="QJ")
    except Exception:
        return []
    rows = []
    for eq in hull.equations:   # n . z + off <= 0, z = (x_i, x_j, y_i, y_j)
        n, off = eq[:4], eq[4]
        if abs(n[2]) < 1e-9 and abs(n[3]) < 1e-9:
            continue
        if (abs(n[0]) + abs(n[2])) < 1e-9 or (abs(n[1]) + abs(n[3])) < 1e-9:
            continue          # single-neuron facet (already implied)
        a = n[0] * gx[i] + n[1] * gx[j]
        a = np.concatenate([a, np.zeros(K - a.size)])
        ay = n[2] * gy[i] + n[3] * gy[j]
        a[: ay.size] += ay
        rhs = -(off + n[0] * cx[i] + n[1] * cx[j] + n[2] * cy[i] + n[3] * cy[j])
        rows.append((a, rhs))
    return rows


def solve_ws(g, c, A, b, iters, lr, nu0=None):
    """Projected Adam on the weak-duality bound with optional warm start; returns (best bound, best nu)."""
    R = A.shape[0]
    nu = g.new_zeros((g.shape[0], R)) if nu0 is None else torch.cat([nu0, g.new_zeros((g.shape[0], R - nu0.shape[1]))], 1)
    best = c + nu @ b + (g - nu @ A).abs().sum(1); best_nu = nu.clone()
    m1 = torch.zeros_like(nu); m2 = torch.zeros_like(nu)
    for it in range(1, iters + 1):
        resid = g - nu @ A
        grad = b.unsqueeze(0) - torch.sign(resid) @ A.t()
        m1.mul_(0.9).add_(grad, alpha=0.1); m2.mul_(0.999).addcmul_(grad, grad, value=0.001)
        nu = (nu - lr * (m1 / (1 - 0.9 ** it)) / ((m2 / (1 - 0.999 ** it)).sqrt() + 1e-8)).clamp_(min=0)
        if it % 25 == 0:
            val = c + nu @ b + (g - nu @ A).abs().sum(1)
            bt = val < best
            best = torch.where(bt, val, best); best_nu[bt] = nu[bt]
    return best, best_nu


def main():
    fam, row = sys.argv[1], int(sys.argv[2])
    partners = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    max_layers = int(sys.argv[4]) if len(sys.argv) > 4 else 99
    inst = list(csv.reader(open(f"{ROOT}/{fam}/instances.csv")))
    o, s = inst[row][:2]
    e = Engine(os.path.normpath(f"{ROOT}/{fam}/{o}"), "cuda", torch.float32)
    z = torch.zeros(e.input_shape, device="cuda"); o0, *_ = e.propagate(z, z, 0)
    spec = parse_vnnlib(os.path.normpath(f"{ROOT}/{fam}/{s}"), int(np.prod(e.input_shape)), int(o0.c.numel()))
    lb, ub = spec.boxes[0]
    out, rows, K, phases = e.propagate(torch.as_tensor(lb), torch.as_tensor(ub), 300, 0.05)
    c = out.c.reshape(-1); G = out.G.reshape(K, c.numel())
    A, b = rows.dense(K)
    # worst disjunct under the base LP
    best = None
    objs = []
    for atoms in spec.disjuncts:
        a0, b0 = atoms[0]
        at = torch.as_tensor(a0, device="cuda", dtype=torch.float32)
        objs.append((-(G @ at), b0 - float(c @ at)))
    g = torch.stack([x[0] for x in objs]); cc = torch.as_tensor([x[1] for x in objs], device="cuda")
    base = lp_upper(g, cc, A, b, 1000, 0.05)
    order = torch.argsort(base, descending=True)[:5].tolist()
    print("base worst bounds", [round(float(base[k]), 4) for k in order])
    t0 = time.time(); extra = []
    for p in phases[:max_layers]:
        m = int(p.idx.numel())
        if m < 2:
            continue
        gx = p.gx.double().cpu().numpy(); cx = p.cx.double().cpu().numpy()
        gy = p.gy.double().cpu().numpy(); cy = p.cy.double().cpu().numpy()
        bnds_l = p.l.double().cpu().numpy(); bnds_u = p.u.double().cpu().numpy()
        gn = p.gx / (p.gx.norm(dim=1, keepdim=True) + 1e-30)
        sim = (gn @ gn.t()).abs()
        sim.fill_diagonal_(-1)
        top = torch.topk(sim, min(partners, m - 1), dim=1).indices.cpu().numpy()
        seen = set()
        for i in range(m):
            for j in top[i]:
                key = (min(i, j), max(i, j))
                if key in seen:
                    continue
                seen.add(key)
                extra += pair_rows(cx, gx, cy, gy, [(bnds_l[i], bnds_u[i]), (bnds_l[j], bnds_u[j])], i, j, K)
        print(f"  layer {p.layer}: unstable {m}, pairs {len(seen)}, rows so far {len(extra)}", flush=True)
    if not extra:
        print("no extra rows"); return
    EA = torch.as_tensor(np.stack([r[0] for r in extra]), device="cuda", dtype=torch.float32)
    Eb = torch.as_tensor(np.array([r[1] for r in extra]), device="cuda", dtype=torch.float32)
    A2 = torch.cat([A, EA]); b2 = torch.cat([b, Eb])
    bb, nu_b = solve_ws(g[order], cc[order], A, b, 1500, 0.05)
    print("base (own solver, 1500 it)", [round(float(v), 4) for v in bb])
    new, _ = solve_ws(g[order], cc[order], A2, b2, 1500, 0.02, nu_b)
    print("pair-hull rows", len(extra), "build+solve", round(time.time() - t0, 1), "s")
    print("with pair hulls", [round(float(v), 4) for v in new])


if __name__ == "__main__":
    main()
