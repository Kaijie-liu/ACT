"""Candidate path v5 = path v4.3 with
  * terminal v7 (disaggregated binaries, monotone-unit integrality; LOG N044, N051), and
  * witness decoding v2: the real decoded point x64 is first rounded to float32 toward the
    box interior (one extra float32 step inward on non-degenerate coordinates) and checked
    as a strict float32-in-box point (S2); if that fails, the baseline check (S1: ORT on the
    nearest float32 of x64) is applied (LOG N052).  The record names the semantics used.
Stages, budgets, margins and seeds unchanged from path v4."""
import time

import numpy as np
import torch

from nhz_terminal_v3 import epigraph_objective, lp_bound_epigraph
from nhz_terminal_v7 import plan_milp_v7

PATH_VERSION = "path-v5.5"  # .5: terminal v7.5 (presolve guard); .4: terminal v7.4 (watchdog); .3: terminal v7.3; .2: witness candidates also snapped to the box bounds within solver tolerance; .1: a stage stopped at the violation target by a spurious incumbent is re-run without the target


def inward_f32(x64, lb, ub):
    lo32 = lb.astype(np.float32); hi32 = ub.astype(np.float32)
    lo32 = np.where(lo32.astype(np.float64) < lb, np.nextafter(lo32, np.float32(np.inf)), lo32)
    hi32 = np.where(hi32.astype(np.float64) > ub, np.nextafter(hi32, np.float32(-np.inf)), hi32)
    wide = hi32 > lo32
    lo32 = np.where(wide, np.nextafter(lo32, np.float32(np.inf)), lo32)
    hi32 = np.where(wide, np.nextafter(hi32, np.float32(-np.inf)), hi32)
    return np.minimum(np.maximum(x64.astype(np.float32), lo32), hi32)


def solve_box_v5(e, sess, iname, spec, lb, ub, deadline, iters=300, final_iters=1000, d_share=0.4):
    n_in = int(np.prod(e.input_shape))
    info = {"path": PATH_VERSION}
    t = time.time()
    if hasattr(e, "deadline"):
        e.deadline = deadline; e.t_start = t
    out, rset, K, phases = e.propagate(lb, ub, iters, 0.05)
    torch.cuda.synchronize(); info["propagate_s"] = time.time() - t
    t = time.time()
    A, b = rset.dense(K); A = A.to(out.c.device); b = b.to(out.c.device)
    objs = [epigraph_objective(out, K, atoms) for atoms in spec.disjuncts]
    ups, cands = [], []
    for g, cc, pad, extra, T in objs:
        ub_, cand = lp_bound_epigraph(A, b, g, cc, extra, final_iters, 0.05)
        ups.append(ub_ + pad); cands.append(cand)
    torch.cuda.synchronize(); info["terminal_lp_s"] = time.time() - t
    open_ids = sorted([i for i, v in enumerate(ups) if v >= -1e-4], key=lambda i: -ups[i])
    n_layers = sum(1 for p in phases if int(p["idx"].numel()))
    info.update({"K": K, "unstable": sum(int(p["idx"].numel()) for p in phases), "relu_layers_with_phases": n_layers,
                 "lp_worst": max(ups), "lp_open": len(open_ids)})
    nzi = e.input_factor_index.cpu().numpy(); center = (lb + ub) / 2; rad = (ub - lb) / 2

    def ort_hit(x):
        y = sess.run(None, {iname: x.reshape(e.input_shape)})[0].reshape(-1).astype(np.float64)
        return next((k for k, atoms in enumerate(spec.disjuncts) if all(float(a @ y) <= bb for a, bb in atoms)), None)

    def check(w):
        xi = np.zeros(n_in); xi[nzi] = np.asarray(w)[: nzi.size]
        x64 = np.clip(center + rad * xi, lb, ub)
        cands = [x64]
        # LP/MILP solutions carry feasibility tolerance (~1e-7 in latent units); coordinates that
        # are within 1e-6 of a box bound are snapped onto it (numeric clean-up, no search)
        snap = x64.copy(); xs = np.asarray(xi)
        snap = np.where(xs >= 1 - 1e-6, ub, np.where(xs <= -1 + 1e-6, lb, snap))
        if np.any(snap != x64):
            cands.append(snap)
        for xc in cands:
            x2 = inward_f32(xc, lb, ub)
            hit = ort_hit(x2)
            if hit is not None:
                return hit, (x2, xc), "S2_inward"
            x1 = xc.astype(np.float32)
            hit = ort_hit(x1)
            if hit is not None:
                return hit, (x1, xc), "S1"
        return None, (x64.astype(np.float32), x64), None

    if not open_ids:
        return "CERT", info, None
    for i in open_ids:
        hit, xx, sem = check(cands[i])
        if hit is not None:
            info["witness_semantics"] = sem
            return "ADV", info, ("lp_box_maximiser", i, hit, xx)
    milp_log = {}
    info["milp"] = milp_log
    stages = [("D", 1), ("E", n_layers)] if n_layers > 1 else [("D", 1)]
    for i in open_ids:
        g, cc, pad, extra, T = objs[i]
        excluded = False
        for stage, gl in stages:
            remaining = deadline - time.time()
            if remaining <= 1:
                return "TIMEOUT", info, None
            tl = remaining * d_share if (stage == "D" and len(stages) > 1) else remaining
            t_stage = time.time()
            m = plan_milp_v7(A, b, phases, K, g, cc, pad, extra, gl, 99, tl)
            milp_log[f"{i}:{stage}"] = {k: v for k, v in m.items() if k != "incumbent_w"}
            if m["incumbent_w"] is not None:
                hit, xx, sem = check(m["incumbent_w"])
                if hit is not None:
                    info["witness_semantics"] = sem
                    return "ADV", info, (f"milp_incumbent_{stage}", i, hit, xx)
            if (not m["excluded"]) and "Target for objective reached" in m["status"]:
                # the early stop fired on a relaxation point that is not a real counterexample:
                # finish the same stage as a pure exclusion query in its remaining time
                left = tl - (time.time() - t_stage)
                if left > 1 and deadline - time.time() > 1:
                    m = plan_milp_v7(A, b, phases, K, g, cc, pad, extra, gl, 99, min(left, deadline - time.time() - 0.5),
                                     target=None)
                    milp_log[f"{i}:{stage}:rerun"] = {k: v for k, v in m.items() if k != "incumbent_w"}
                    if m["incumbent_w"] is not None:
                        hit, xx, sem = check(m["incumbent_w"])
                        if hit is not None:
                            info["witness_semantics"] = sem
                            return "ADV", info, (f"milp_incumbent_{stage}", i, hit, xx)
            if m["excluded"]:
                excluded = True
                break
        if not excluded:
            return ("TIMEOUT" if time.time() >= deadline - 1 else "UNKNOWN"), info, None
    return "CERT", info, None
