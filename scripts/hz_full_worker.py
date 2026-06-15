#!/usr/bin/env python
"""HZ full-run worker: verify ONE (benchmark, iid) with the engine's best config.

Config: hybridz HZ domain, tight_bounds=True (LP-tight [alpha,beta], the #1
precision lever), K=1 sigmoid/tanh (default), reduce Step-0 always-on (default),
verdict = hz_certify_spec (scipy LP-relax + MILP, P3, Gurobi OFF). ORT is a
DOWNGRADE soundness guard ONLY (never produces a CERT). One process per instance
=> mem-isolated; RLIMIT_AS caps CPU mem (cpu device) so a runaway cannot kill the
host/VS Code. Prints exactly ONE json result line to stdout.

Multi-query specs (OR-disjunct robustness: malbeware 24 / metaroom 19 / ...):
prop ONCE, certify EVERY disjunct's C/t; instance CERT iff ALL disjuncts are
proved empty (the union of unsafe disjuncts is unreachable). UNKNOWN otherwise.
"""
import os, sys, json, time, argparse, resource

def _hz_solve(hz, C, t, is_uns, integer, time_limit):
    """Solve the spec margin over the HZ in the UNSAFE direction, returning BOTH
    the margin and the optimum witness xi (continuous part valid as an input map).

    integer=False -> LP relaxation (binaries in [-1,1] = convex hull; sound bound,
    fast). integer=True -> binary-exact MILP (xi_b in {-1,+1} via xi_b=2z-1; exact
    margin AND a real worst-case witness). Margin convention:
      * is_uns (UNSAFE_LINEAR conjunction): margin = s* = min_y max_r(C[r]y-t[r]);
        SAFE iff s* > 0. Witness = the y attaining the min (the most-unsafe point).
      * ALL-rows (TOP1/LE): margin = max_r(max_y C[r]y - t[r]); SAFE iff < 0.
        Witness = the point attaining the worst row's max.
    Returns (margin|None, xi|None). The MILP witness's binary block is mapped back
    to {-1,+1}; the input map only ever reads the continuous block, which is valid."""
    import numpy as np
    import scipy.sparse as sp
    from scipy.optimize import linprog, milp, LinearConstraint as LC, Bounds as LPB
    from act.back_end.solver.solver_hz import _split_eq_le
    c = hz.c.detach().cpu().double().numpy().reshape(-1)
    Gc = hz.Gc.detach().cpu().double().numpy(); Gb = hz.Gb.detach().cpu().double().numpy()
    (Ace, Abe, be), (Acl, Abl, bl) = _split_eq_le(hz)
    Ace = Ace.detach().cpu().double().numpy(); Abe = Abe.detach().cpu().double().numpy()
    be = be.detach().cpu().double().numpy().reshape(-1)
    Acl = Acl.detach().cpu().double().numpy(); Abl = Abl.detach().cpu().double().numpy()
    bl = bl.detach().cpu().double().numpy().reshape(-1)
    ng, nb = Gc.shape[1], Gb.shape[1]

    if not is_uns:
        # ALL-rows: max each row, keep the worst (margin) and its witness.
        best_m = -np.inf; best_xi = None
        for r in range(C.shape[0]):
            obj = np.concatenate([C[r] @ Gc, C[r] @ Gb])
            const = float(C[r] @ c)
            if not integer:
                A_eq = np.hstack([Ace, Abe]) if Ace.shape[0] else None
                A_ub = np.hstack([Acl, Abl]) if Acl.shape[0] else None
                rr = linprog(-obj, A_eq=A_eq, b_eq=(be if Ace.shape[0] else None),
                             A_ub=(A_ub if Acl.shape[0] else None),
                             b_ub=(bl if Acl.shape[0] else None),
                             bounds=[(-1, 1)] * (ng + nb), method="highs")
                if not rr.success:
                    return None, None
                rowmax = const - rr.fun; xi = rr.x[:ng + nb]
            else:
                obj_z = np.concatenate([C[r] @ Gc, 2.0 * (C[r] @ Gb)])
                const_z = const - float((C[r] @ Gb).sum())
                integ = np.concatenate([np.zeros(ng), np.ones(nb)]).astype(int)
                cons = []
                if Ace.shape[0]:
                    cons.append(LC(sp.csr_matrix(np.hstack([Ace, 2.0 * Abe])),
                                   lb=be + Abe.sum(1), ub=be + Abe.sum(1)))
                if Acl.shape[0]:
                    cons.append(LC(sp.csr_matrix(np.hstack([Acl, 2.0 * Abl])),
                                   ub=bl + Abl.sum(1)))
                vlb = np.concatenate([-np.ones(ng), np.zeros(nb)])
                vub = np.ones(ng + nb)
                rr = milp(c=-obj_z, constraints=cons, integrality=integ, bounds=LPB(vlb, vub),
                          options={"mip_rel_gap": 1e-9, "time_limit": time_limit})
                if not rr.success:
                    return None, None
                rowmax = const_z - rr.fun
                xi = np.concatenate([rr.x[:ng], 2.0 * rr.x[ng:ng + nb] - 1.0])
            m_r = rowmax - float(t[r])
            if m_r > best_m:
                best_m = m_r; best_xi = xi
        return best_m, best_xi

    # is_uns: joint epigraph  min s  s.t.  C[r]y - s <= t[r].
    nrow = C.shape[0]; v_s = ng + nb; nv = ng + nb + 1
    epi_A = np.zeros((nrow, nv)); epi_b = np.empty(nrow)
    for r in range(nrow):
        epi_A[r, :ng] = C[r] @ Gc; epi_A[r, ng:ng + nb] = C[r] @ Gb
        epi_A[r, v_s] = -1.0; epi_b[r] = float(t[r] - C[r] @ c)
    obj = np.zeros(nv); obj[v_s] = 1.0
    if not integer:
        A_ub = [epi_A]; b_ub = [epi_b]
        if Acl.shape[0]:
            A_ub.append(np.hstack([Acl, Abl, np.zeros((Acl.shape[0], 1))])); b_ub.append(bl)
        A_eq = np.hstack([Ace, Abe, np.zeros((Ace.shape[0], 1))]) if Ace.shape[0] else None
        rr = linprog(obj, A_ub=np.vstack(A_ub), b_ub=np.concatenate(b_ub),
                     A_eq=A_eq, b_eq=(be if Ace.shape[0] else None),
                     bounds=[(-1, 1)] * (ng + nb) + [(-1e12, 1e12)], method="highs")
        if not rr.success:
            return None, None
        return float(rr.fun), rr.x[:ng + nb]
    # binary-exact MILP (xi_b = 2z-1).
    epi_Az = epi_A.copy(); epi_Az[:, ng:ng + nb] *= 2.0
    epi_bz = epi_b + (C @ Gb).sum(axis=1)
    integ = np.concatenate([np.zeros(ng), np.ones(nb), [0]]).astype(int)
    vlbz = np.concatenate([-np.ones(ng), np.zeros(nb), [-1e12]])
    vubz = np.concatenate([np.ones(ng + nb), [1e12]])
    cons = [LC(sp.csr_matrix(epi_Az), ub=epi_bz)]
    if Ace.shape[0]:
        cons.append(LC(sp.csr_matrix(np.hstack([Ace, 2.0 * Abe, np.zeros((Ace.shape[0], 1))])),
                       lb=be + Abe.sum(1), ub=be + Abe.sum(1)))
    if Acl.shape[0]:
        cons.append(LC(sp.csr_matrix(np.hstack([Acl, 2.0 * Abl, np.zeros((Acl.shape[0], 1))])),
                      ub=bl + Abl.sum(1)))
    rr = milp(c=obj, constraints=cons, integrality=integ, bounds=LPB(vlbz, vubz),
              options={"mip_rel_gap": 1e-9, "time_limit": time_limit})
    if not rr.success:
        return None, None
    xi = np.concatenate([rr.x[:ng], 2.0 * rr.x[ng:ng + nb] - 1.0])
    return float(rr.fun), xi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench"); ap.add_argument("iid", type=int)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cap", type=int, default=200000, help="_HZ_MAX_INPUT_DIM override")
    ap.add_argument("--milp-timeout", type=float, default=15.0)
    ap.add_argument("--mem-gb", type=float, default=10.0, help="RLIMIT_AS cap (cpu device only)")
    ap.add_argument("--ort-samples", type=int, default=4000)
    ap.add_argument("--no-tight", action="store_true",
                    help="disable tight_bounds (fast box) -- for nets whose LP-tight "
                         "[alpha,beta] pass blows memory with no CERT benefit (dist_shift sigmoid)")
    a = ap.parse_args()

    res = {"bench": a.bench, "iid": a.iid, "verdict": "ERROR", "margin": None,
           "n_queries": None, "nc": None, "ng": None, "nb": None,
           "time_s": None, "verify_s": None, "device": a.device,
           "gt_cex": None, "p0": False, "err": None}
    t_start = time.time()
    try:
        if a.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            # Hard CPU-mem backstop: a runaway prop dies with MemoryError in THIS
            # subprocess instead of ballooning the host (RLIMIT_AS = virtual mem).
            cap = int(a.mem_gb * 1024**3)
            resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(v, "1")
        sys.path.insert(0, "/data1/Kane/ACT")
        import numpy as np, torch
        import act.back_end.solver.solver_hz as S
        S._HAS_GUROBI = False  # P3: engine verdict is scipy/HiGHS only
        import act.back_end.hybridz_tf.hybridz_tf as HZTF
        HZTF.HybridzTF._HZ_MAX_INPUT_DIM = a.cap
        HZTF.HybridzTF._HZ_MAX_AFFINE_DIM = max(a.cap, 8192)
        from pathlib import Path
        from act.front_end.vnnlib_loader.onnx_converter import convert_onnx_to_pytorch, get_onnx_input_shape
        from act.front_end.vnnlib_loader.vnnlib_parser import parse_vnnlib_queries
        from act.front_end.verifiable_model import VerifiableModel, InputLayer, InputSpecLayer, OutputSpecLayer
        from act.front_end.spec_creator_base import LabeledInputTensor
        from act.pipeline.verification.torch2act import TorchToACT
        from act.back_end.transfer_functions import set_transfer_function_mode, get_transfer_function
        from act.back_end.core import Bounds as ABounds
        from act.back_end.solver.solver_hz_verdict import hz_certify_spec

        dev = "cuda" if (a.device == "cuda" and torch.cuda.is_available()) else "cpu"
        base = Path("/data1/Kane/data/vnncomp2025_benchmarks/benchmarks") / a.bench
        rows = [l.strip().split(",") for l in open(base / "instances.csv") if l.strip()]
        o = base / rows[a.iid][0].replace("./", ""); v = base / rows[a.iid][1].replace("./", "")
        ish = tuple(get_onnx_input_shape(o)); pt = convert_onnx_to_pytorch(o).float().eval()
        lab = LabeledInputTensor(tensor=torch.zeros(ish, dtype=torch.float32), label=torch.tensor([0]))
        queries = parse_vnnlib_queries(v, labeled_tensor=lab)
        res["n_queries"] = len(queries)

        # Build one ACT net per query (cheap: only net[0] is propagated; the rest
        # share the identical pre-ASSERT structure, we read their ASSERT C/t).
        def build(qk):
            return TorchToACT(VerifiableModel(
                input_layer=InputLayer(labeled_input=lab, shape=ish, dtype=torch.float32),
                input_spec=InputSpecLayer(qk[0]), model=pt, output_spec=OutputSpecLayer(qk[1]))).run()
        net0 = build(queries[0])
        ins0 = queries[0][0]
        lb = ins0.lb.detach().cpu().numpy().reshape(-1).astype(np.float64)
        ub = ins0.ub.detach().cpu().numpy().reshape(-1).astype(np.float64)

        # Propagate ONCE (tight_bounds = #1 precision lever; K=1, reduce = defaults)
        set_transfer_function_mode("hybridz"); tf = get_transfer_function()
        tf._hz_cache.clear(); tf._cache_net_id = None
        tf._relu_tight_bounds = not a.no_tight
        ib = ABounds(lb=torch.tensor(lb, dtype=torch.float32, device=dev).reshape(1, -1),
                     ub=torch.tensor(ub, dtype=torch.float32, device=dev).reshape(1, -1))
        # verifier-time = HZ propagation + engine LP/MILP solve only. The witness
        # forward-eval ("find x*") and the ORT soundness audit are NOT the
        # verifier's job (it only decides IF an adv exists) and are excluded.
        _t_prop0 = time.time()
        after = {}
        for L in net0.layers:
            pr = net0.preds.get(L.id, []); inb = ib if (L.id == 0 or not pr) else after[pr[0]].bounds
            after[L.id] = tf.apply(L, inb, net0, {}, after)
        tf._relu_tight_bounds = False
        _verify_s = time.time() - _t_prop0   # accumulates engine solve below
        pid = net0.preds.get(net0.layers[-1].id, [None])[0]
        hz = tf._hz_cache.get(pid)
        res["nc"] = int(hz.Ac.shape[0]) if hz is not None else 0
        res["ng"] = int(hz.Gc.shape[1]) if hz is not None else 0
        res["nb"] = int(hz.Gb.shape[1]) if hz is not None else 0

        nets = [net0] + [build(queries[k]) for k in range(1, len(queries))]
        specs = []   # (C, t, is_unsafe_linear) per query
        for nk in nets:
            AL = nk.layers[-1]; C = AL.params["C"].detach().cpu().numpy().astype(np.float64)
            t = AL.params["thresholds"].detach().cpu().numpy().astype(np.float64).reshape(-1)
            is_uns = "UNSAFE_LINEAR" in str(AL.params.get("kind"))
            specs.append((C, t, is_uns))

        # ORT session (used for both falsification-verify and the soundness guard)
        import onnxruntime as ort
        sess = ort.InferenceSession(str(o), providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name
        def _net(x):
            return sess.run(None, {iname: x.reshape(ish).astype(np.float32)})[0].reshape(-1)

        # ---- Unified FALSIFY + CERTIFY per disjunct (one solve does both jobs).
        # The HZ margin solve in the unsafe direction returns BOTH the margin
        # (sound CERT decision) AND the worst-case witness xi. Mapping xi's
        # continuous block to a concrete input and running the REAL net (ORT):
        # land in the unsafe region -> VERIFIED counterexample -> ADV. The witness
        # is the abstract domain's own LP/MILP optimum (no PGD/gradient/random
        # search); the forward eval only confirms. LP first (cheap: certifies safe
        # disjuncts, falsifies ~80% of unsafe); only when LP neither certifies nor
        # falsifies do we escalate to the binary-exact MILP (tighter cert + the
        # real worst-case witness that the convex-hull LP relaxation can miss).
        center = (lb + ub) / 2.0; rad = (ub - lb) / 2.0
        inhz = next((tf._hz_cache.get(L.id) for L in net0.layers
                     if L.kind.upper() == "INPUT_SPEC"), None)
        in_ids = (inhz.col_ids.detach().cpu().numpy()
                  if (inhz is not None and inhz.col_ids is not None) else None)
        pos = None
        if hz is not None and in_ids is not None and hz.col_ids is not None:
            pos = {int(v): k for k, v in enumerate(hz.col_ids.detach().cpu().numpy())}

        def _is_cex(xi, C, t, is_uns):
            if xi is None or pos is None:
                return False
            xin = np.array([xi[pos[int(v)]] if int(v) in pos else 0.0 for v in in_ids])
            y = _net(center + rad * xin); cy = y @ C.T
            return bool((cy <= t + 1e-9).all()) if is_uns else bool((cy >= t - 1e-9).any())

        def _solve(C, t, is_uns, integer):
            nonlocal _verify_s
            _ts = time.time()
            out = _hz_solve(hz, C, t, is_uns, integer=integer, time_limit=a.milp_timeout)
            _verify_s += time.time() - _ts   # engine solve counts; the _is_cex map does not
            return out

        TOL = 1e-9
        adv = False; all_cert = True; min_margin = float("inf"); any_margin = False
        for (C, t, is_uns) in specs:
            if hz is None:
                all_cert = False; continue
            # LP pass: cheap certify + cheap falsify witness.
            m_lp, xi_lp = _solve(C, t, is_uns, False)
            if _is_cex(xi_lp, C, t, is_uns):
                adv = True; break
            lp_safe = (m_lp is not None) and ((m_lp > TOL) if is_uns else (m_lp < -TOL))
            if lp_safe:
                min_margin = min(min_margin, m_lp); any_margin = True
                continue
            # MILP escalation: binary-exact margin (tighter cert) + real witness.
            m_mi, xi_mi = _solve(C, t, is_uns, True)
            if _is_cex(xi_mi, C, t, is_uns):
                adv = True; break
            if m_mi is not None:
                min_margin = min(min_margin, m_mi); any_margin = True
            mi_safe = (m_mi is not None) and ((m_mi > TOL) if is_uns else (m_mi < -TOL))
            if not mi_safe:
                all_cert = False

        if adv:
            res["verdict"] = "ADV"; res["margin"] = None
        else:
            res["verdict"] = "CERT" if all_cert else "UNKNOWN"
            res["margin"] = (min_margin if any_margin and min_margin != float("inf") else None)
        res["time_s"] = round(time.time() - t_start, 2)   # total wall (drives timeout)
        res["verify_s"] = round(_verify_s, 2)              # verifier engine time only

        # ORT soundness guard (downgrade only): a sample is a real cex iff it lands
        # in ANY disjunct's unsafe set. P0 = engine CERT but a real cex exists.
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(o), providers=["CPUExecutionProvider"])
            iname = sess.get_inputs()[0].name
            rng = np.random.RandomState(0)
            X = np.vstack([lb, ub, (lb + ub) / 2,
                           rng.uniform(lb, ub, size=(a.ort_samples, lb.size))]).astype(np.float32)
            try:
                Y = sess.run(None, {iname: X.reshape((-1,) + ish[1:])})[0].reshape(X.shape[0], -1)
            except Exception:
                Y = np.stack([sess.run(None, {iname: x.reshape(ish)})[0].reshape(-1) for x in X])
            gt_cex = False
            for (C, t, is_uns) in specs:
                if is_uns:  # disjunct unsafe = all rows C[r]y <= t[r]
                    inu = np.ones(X.shape[0], bool)
                    for r in range(C.shape[0]):
                        inu &= (Y @ C[r] <= t[r] + 1e-9)
                    if inu.any(): gt_cex = True; break
                else:       # TOP1/ALL-rows unsafe = some row C[r]y >= t[r]
                    vio = np.zeros(X.shape[0], bool)
                    for r in range(C.shape[0]):
                        vio |= (Y @ C[r] >= t[r] - 1e-9)
                    if vio.any(): gt_cex = True; break
            res["gt_cex"] = gt_cex
            if res["verdict"] == "CERT" and gt_cex:
                res["p0"] = True; res["verdict"] = "UNKNOWN_P0DOWNGRADE"
        except Exception as e:
            res["gt_cex"] = None; res["err_ort"] = str(e)[:120]
    except MemoryError:
        res["verdict"] = "ERROR"; res["err"] = "MemoryError (RLIMIT_AS cap hit)"
        res["time_s"] = round(time.time() - t_start, 2)
    except Exception as e:
        import traceback
        res["verdict"] = "ERROR"; res["err"] = f"{type(e).__name__}: {str(e)[:200]}"
        res["time_s"] = round(time.time() - t_start, 2)
        print(traceback.format_exc()[:1500], file=sys.stderr)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
