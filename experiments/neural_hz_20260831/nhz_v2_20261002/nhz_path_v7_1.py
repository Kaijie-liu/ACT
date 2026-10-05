"""Candidate path v7.1 = path v7.0 + empty-plan fix (LOG N103). Copy; the frozen nhz_path_v7.py of N039 v14 is unchanged.

Candidate path v7.0: domain-only verdicts (LOG N099).

User rule (2026-10-03): gains may not come from any attack or sampling helper; they must come
from Neural-HZ itself.  Path v7 therefore has exactly these verdict sources:

  CERT  every unsafe disjunct is excluded by a rigorous bound of the Neural-HZ element:
        (B) the weak-duality LP over its lambda concretisation, or
        (D/E/F) a MILP over a selective-exactness plan of the element (gamma on the last ReLU
        layer / all ReLU layers / all ReLU layers + smooth phase segments) whose optimum lies
        below the cutoff (HiGHS or SCIP infeasible under the objective bound, or a valid dual
        bound below -margin).
  ADV   an incumbent of one of those MILP plans, i.e. a point of the element's (relaxed or exact)
        concretisation.  Its input part is decoded, rounded to float32 inside the box
        (deterministic numeric rounding of that one point; coordinates within solver tolerance
        of a box bound are put on the bound), and accepted only if ONNX Runtime on the original
        network satisfies the original VNNLIB unsafe condition.

Removed relative to path v6.7: LP box-maximiser witnesses (LP-dual sign corners), the
input-box centre check, MIP starts seeded from the centre, and stage W.  Kept: rigorous
engine bounds, fail-closed handling of non-finite data, terminal v8.3/v7.9 plans (Balas rows,
smooth rows, segment phases, sign rule of Proposition 1', HiGHS seeds + SCIP), the violation
target only on exact plans, the rerun of a stage stopped by a non-real incumbent, stage F.
"""
import time

import numpy as np
import torch

from nhz_terminal_v3 import epigraph_objective, lp_bound_epigraph
from nhz_terminal_v8_4 import plan_milp_v8

PATH_VERSION = "path-v7.1"  # v7.1: terminal v8.4/v7.10 (empty-plan fix, LOG N103); verdict sources unchanged


def inward_f32(x64, lb, ub):
    """float32 point inside [lb, ub] nearest to x64 after one inward step on non-degenerate
    coordinates; None if some coordinate contains no float32 value (then only S1 applies)."""
    lo32 = lb.astype(np.float32); hi32 = ub.astype(np.float32)
    lo32 = np.where(lo32.astype(np.float64) < lb, np.nextafter(lo32, np.float32(np.inf)), lo32)
    hi32 = np.where(hi32.astype(np.float64) > ub, np.nextafter(hi32, np.float32(-np.inf)), hi32)
    wide = hi32 > lo32
    lo32 = np.where(wide, np.nextafter(lo32, np.float32(np.inf)), lo32)
    hi32 = np.where(wide, np.nextafter(hi32, np.float32(-np.inf)), hi32)
    if np.any(lo32 > hi32):
        return None
    return np.minimum(np.maximum(x64.astype(np.float32), lo32), hi32)


def solve_box_v7(e, sess, iname, spec, lb, ub, deadline, iters=300, final_iters=1000, d_share=0.4):
    n_in = int(np.prod(e.input_shape))
    info = {"path": PATH_VERSION}
    t = time.time()
    if hasattr(e, "deadline"):
        e.deadline = deadline; e.t_start = t
    out, rset, K, phases = e.propagate(lb, ub, iters, 0.05)
    torch.cuda.synchronize(); info["propagate_s"] = time.time() - t
    t = time.time()
    A, b = rset.dense(K); A = A.to(out.c.device); b = b.to(out.c.device)
    if not (bool(torch.isfinite(out.c).all()) and bool(torch.isfinite(out.G).all()) and bool(torch.isfinite(out.e).all())
            and bool(torch.isfinite(A).all()) and bool(torch.isfinite(b).all())):
        info["nonfinite"] = True
        return "UNKNOWN", info, None
    objs = [epigraph_objective(out, K, atoms) for atoms in spec.disjuncts]
    ups = []
    for g, cc, pad, extra, T in objs:
        ub_, _ = lp_bound_epigraph(A, b, g, cc, extra, final_iters, 0.05)
        ups.append(ub_ + pad)
    torch.cuda.synchronize(); info["terminal_lp_s"] = time.time() - t
    ups = [float(v) if np.isfinite(v) else float("inf") for v in ups]
    open_ids = sorted([i for i, v in enumerate(ups) if v >= -1e-4], key=lambda i: -ups[i])
    n_layers = sum(1 for p in phases if int(p["idx"].numel()))
    info.update({"K": K, "unstable": sum(int(p["idx"].numel()) for p in phases), "relu_layers_with_phases": n_layers,
                 "lp_worst": max(ups), "lp_open": len(open_ids)})
    if not open_ids:
        return "CERT", info, None
    nzi = e.input_factor_index.cpu().numpy(); center = (lb + ub) / 2; rad = (ub - lb) / 2

    def ort_hit(x):
        y = sess.run(None, {iname: x.reshape(e.input_shape)})[0].reshape(-1).astype(np.float64)
        return next((k for k, atoms in enumerate(spec.disjuncts) if all(float(a @ y) <= bb for a, bb in atoms)), None)

    def check(w):
        """Validate one MILP incumbent of the element on the original network."""
        xi = np.zeros(n_in); xi[nzi] = np.asarray(w)[: nzi.size]
        x64 = np.clip(center + rad * xi, lb, ub)
        snap = np.where(xi >= 1 - 1e-6, ub, np.where(xi <= -1 + 1e-6, lb, x64))
        for xc in ([x64, snap] if np.any(snap != x64) else [x64]):
            x2 = inward_f32(xc, lb, ub)
            hit = ort_hit(x2) if x2 is not None else None
            if hit is not None:
                return hit, (x2, xc), "S2_inward"
            x1 = xc.astype(np.float32)
            hit = ort_hit(x1)
            if hit is not None:
                return hit, (x1, xc), "S1"
        return None, None, None

    milp_log = {}
    info["milp"] = milp_log
    smooth = getattr(e, "smooth_phases", []) or []
    nonsat = None
    if smooth:
        nonsat = np.concatenate([((torch.sigmoid(s_["u"]) - torch.sigmoid(s_["l"])) if s_["kind"] == "Sigmoid"
                                  else (torch.tanh(s_["u"]) - torch.tanh(s_["l"]))).cpu().numpy() > 1e-4 for s_ in smooth])
    stages = [("D", 1, None), ("E", n_layers, None)] if n_layers > 1 else [("D", 1, None)]
    if smooth and nonsat is not None and nonsat.any():
        stages.append(("F", max(n_layers, 1), nonsat))
    has_softmax = bool(getattr(e, "softmax_coords", 0))
    for i in open_ids:
        g, cc, pad, extra, T = objs[i]
        excluded = False
        for stage, gl, seg in stages:
            remaining = deadline - time.time()
            if remaining <= 1:
                return "TIMEOUT", info, None
            tl = remaining * d_share if (stage == "D" and len(stages) > 1) else (
                remaining * 0.5 if (stage == "E" and len(stages) > 2) else remaining)
            exact_plan = (gl >= n_layers) and not smooth and not has_softmax
            t_stage = time.time()
            m = plan_milp_v8(A, b, phases, smooth, K, g, cc, pad, extra, gl, 99, tl, seg_select=seg, K_seg=2,
                             target=(1e-6 if exact_plan else None))
            milp_log[f"{i}:{stage}"] = {k: v for k, v in m.items() if k != "incumbent_w"}
            if m["incumbent_w"] is not None:
                hit, xx, sem = check(m["incumbent_w"])
                if hit is not None:
                    info["witness_semantics"] = sem
                    return "ADV", info, (f"milp_incumbent_{stage}", i, hit, xx)
            if (not m["excluded"]) and "Target for objective reached" in m["status"]:
                left = tl - (time.time() - t_stage)
                if left > 1 and deadline - time.time() > 1:
                    m = plan_milp_v8(A, b, phases, smooth, K, g, cc, pad, extra, gl, 99,
                                     min(left, deadline - time.time() - 0.5), seg_select=seg, K_seg=2, target=None)
                    milp_log[f"{i}:{stage}:rerun"] = {k: v for k, v in m.items() if k != "incumbent_w"}
                    if m["incumbent_w"] is not None:
                        hit, xx, sem = check(m["incumbent_w"])
                        if hit is not None:
                            info["witness_semantics"] = sem
                            return "ADV", info, (f"milp_incumbent_{stage}", i, hit, xx)
            if m["excluded"] and m["upper"] is not None and np.isfinite(m["upper"]):
                excluded = True
                break
        if not excluded:
            return ("TIMEOUT" if time.time() >= deadline - 1 else "UNKNOWN"), info, None
    return "CERT", info, None
