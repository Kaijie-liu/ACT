"""HyZor adapter as Tier-2 ACT Solver.

Walks ACT cons IR (verified correct in Step 0) and dispatches to:
  - HyZor ops (dense/conv2d/relu eq_lagr_v8/add SGM/...) via `import HyZor`
  - ACT ops (sigmoid/tanh K-piece tangent parallelogram) via
    `import act.back_end.hybridz_tf.tf_mlp` -- preserves ACT's innovation

NO HyZor algorithm code is duplicated. HyZor upgrades on its own ->
ACT updates automatically.
"""
from __future__ import annotations
import os
import sys
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import torch

from act.back_end.solver.solver_base import Solver, SolverCaps, SolveStatus
from act.back_end.core import Bounds, Con, ConSet, Fact, Layer, Net


# ─── Make HyZor importable ─────────────────────────────────────────────
# HyZor uses flat-layout imports (e.g. `from HybridZonotope import ...`),
# so both the parent dir and the package dir need to be on sys.path.
_HYZOR_ROOT = os.environ.get(
    "HYZOR_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__),
                                  "..", "..", "..", "..", "HyZor")),
)
_HYZOR_PARENT = os.path.dirname(_HYZOR_ROOT)
# Order matters: parent first so `import HyZor` finds the package
# (HyZor/__init__.py), not the module file (HyZor/HyZor.py).
# HyZor's flat imports (e.g. `from HybridZonotope import ...`) work
# because _HYZOR_ROOT is also on sys.path.
if _HYZOR_ROOT not in sys.path:
    sys.path.insert(0, _HYZOR_ROOT)
if _HYZOR_PARENT not in sys.path:
    sys.path.insert(0, _HYZOR_PARENT)  # inserted later → searched first


class HyZorSolver(Solver):
    """Tier-2 verification solver -- walks ACT cons IR, dispatches to
    HyZor (eq_lagr_v8 family) and ACT (sigmoid/tanh K-piece) operators."""

    # v9.2: benchmark-tuned profile presets (ports HyZor's 4 canonical overrides)
    # Each profile is an overlay dict applied on top of constructor defaults.
    BENCHMARK_PROFILES = {
        # safenlp_2024: B14 profile. Heavy proof techniques + eq_lagr_v8 last 3.
        # NOTE: dual_cert_n=1 (LP single cert is enough; UNC too loose).
        # The B14 power comes from extra_env enabling HyZor-internal LP techniques.
        "safenlp_b14": {
            "relu_method": "eq_lagr_v8",
            "girard_cap": 6000,
            "mace_enabled": True,
            "sgm_enabled": True,
            "dual_cert_n": 1,
            "large_cls_proof_mode": "off",  # safenlp is small-out, large_cls not relevant
            "extra_env": {
                "HYZOR_L2_EXACTLP_TAIL": "1",
                "HYZOR_L2_EXACTLP_RELU_CAP": "128",
                "HYZOR_L2_EXACTLP_OUT_CAP": "1024",
                "HYZOR_MARGIN_LAGR": "1",
                "HYZOR_MARGIN_LAGR_ITERS": "260",
            },
        },
        # nn4sys: A0_triangle profile. Force triangle (no eq_lagr_v8, faster).
        "nn4sys_triangle": {
            "relu_method": "triangle",
            "girard_cap": 6000,
            "mace_enabled": False,
            "sgm_enabled": False,
            "dual_cert_n": 1,
            "large_cls_proof_mode": "off",
            "extra_env": {},
        },
        # metaroom_2023: PDNT N=1 bypass override.
        "metaroom_n1": {
            "relu_method": "eq_lagr_v8",
            "girard_cap": 6000,
            "mace_enabled": True,
            "sgm_enabled": True,
            "dual_cert_n": 1,                # ★ key knob for metaroom
            "large_cls_proof_mode": "auto",
            "extra_env": {},
        },
        # dist_shift_2023: B9 profile (lighter than B14).
        "dist_shift_b9": {
            "relu_method": "eq_lagr_v8",
            "girard_cap": 6000,
            "mace_enabled": True,
            "sgm_enabled": True,
            "dual_cert_n": 1,
            "large_cls_proof_mode": "off",
            "extra_env": {
                "HYZOR_L2_EXACTLP_TAIL": "1",
                "HYZOR_L2_EXACTLP_RELU_CAP": "96",
                "HYZOR_L2_EXACTLP_OUT_CAP": "512",
                "HYZOR_MARGIN_LAGR": "1",
                "HYZOR_MARGIN_LAGR_ITERS": "220",
            },
        },
        "default": {
            "relu_method": "eq_lagr_v8",
            "girard_cap": 6000,
            "mace_enabled": True,
            "sgm_enabled": True,
            "dual_cert_n": 1,
            "large_cls_proof_mode": "auto",
            "extra_env": {},
        },
    }

    def __init__(
        self, *,
        # Profile shortcut (v9.2): applies a benchmark-tuned overlay
        benchmark_profile: Optional[str] = None,
        # Individual knobs (override profile if specified)
        relu_method: str = "eq_lagr_v8",
        girard_cap: int = 6000,
        mace_enabled: bool = True,
        sgm_enabled: bool = True,
        strict_replay: bool = True,
        sigmoid_K: int = 2,
        tanh_K: int = 2,
        # large_cls_proof_mode (HyZor scheduling for cifar100/tinyimagenet)
        large_cls_proof_mode: str = "auto",   # "on" | "off" | "auto"
        large_cls_eq_layers: int = 3,         # last N relus use eq_lagr_v8
        large_cls_conv_threshold: int = 4,    # min conv count to trigger
        large_cls_out_dim_threshold: int = 100,  # min output dim
        # PDNT multi-cert (v9.2): N independent sound certs required.
        # N=1 (default) = trust LP single cert (sound; matches metaroom override).
        # N≥2 = belt-and-suspenders mode: require additional cert(s) to confirm.
        # NOTE: only UNC cert ("U") implemented in v9.2. Full E/Z/F pool deferred.
        # Since UNC is much looser than LP, requiring N≥2 UNC-confirmation will
        # lose precision (LP-verified + UNC-can't-confirm → UNKNOWN). Use N=1
        # unless you specifically need the extra paranoia.
        dual_cert_n: int = 1,
        dual_cert_pool: str = "U",
        dual_cert_margin: float = 1e-8,
        timeout_s: float = 300.0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        # Apply benchmark profile if specified
        if benchmark_profile is not None:
            if benchmark_profile not in self.BENCHMARK_PROFILES:
                raise ValueError(
                    f"Unknown benchmark_profile {benchmark_profile!r}. "
                    f"Valid: {list(self.BENCHMARK_PROFILES.keys())}"
                )
            profile = self.BENCHMARK_PROFILES[benchmark_profile]
            relu_method = profile.get("relu_method", relu_method)
            girard_cap = profile.get("girard_cap", girard_cap)
            mace_enabled = profile.get("mace_enabled", mace_enabled)
            sgm_enabled = profile.get("sgm_enabled", sgm_enabled)
            large_cls_proof_mode = profile.get("large_cls_proof_mode",
                                                large_cls_proof_mode)
            dual_cert_n = profile.get("dual_cert_n", dual_cert_n)
            # Set HyZor-internal env vars that affect proof aggressiveness
            for k, v in profile.get("extra_env", {}).items():
                os.environ.setdefault(k, v)

        self.cfg = dict(
            benchmark_profile=benchmark_profile,
            relu_method=relu_method, girard_cap=girard_cap,
            mace_enabled=mace_enabled, sgm_enabled=sgm_enabled,
            strict_replay=strict_replay,
            sigmoid_K=sigmoid_K, tanh_K=tanh_K,
            large_cls_proof_mode=large_cls_proof_mode,
            large_cls_eq_layers=large_cls_eq_layers,
            large_cls_conv_threshold=large_cls_conv_threshold,
            large_cls_out_dim_threshold=large_cls_out_dim_threshold,
            dual_cert_n=dual_cert_n,
            dual_cert_pool=dual_cert_pool,
            dual_cert_margin=dual_cert_margin,
            timeout_s=timeout_s, device=device, dtype=dtype,
        )
        self._reset_state()

    def _reset_state(self):
        self._status: str = SolveStatus.UNKNOWN
        self._witness: Optional[np.ndarray] = None
        self._has_solution: bool = False
        self._var_count: int = 0
        self._stats: Dict[str, Any] = {}

    # ----- Solver interface stubs (no-op; HyZor uses consume_cons) -----
    def capabilities(self) -> SolverCaps:
        return SolverCaps(supports_gpu=True, supports_csp=True, supports_hz=True)

    def begin(self, name: str = "verify", device: Optional[str] = None):
        self._reset_state()
        if device is not None:
            self.cfg["device"] = device
        # Aggressive GPU memory cleanup to avoid fragmentation across
        # sequential verify_once calls (esp. on large nets like cifar100/
        # tinyimagenet — fragmented allocator causes spurious OOM).
        try:
            import torch as _t
            if _t.cuda.is_available() and self.cfg.get("device", "cpu").startswith("cuda"):
                _t.cuda.empty_cache()
                _t.cuda.synchronize()
        except Exception:
            pass

    @property
    def n(self) -> int:
        return self._var_count

    def add_vars(self, n: int) -> None:
        self._var_count += n

    def add_binary_vars(self, n: int) -> List[int]:
        ids = list(range(self._var_count, self._var_count + n))
        self._var_count += n
        return ids

    def set_bounds(self, idxs, lb, ub): pass
    def add_lin_eq(self, vids, coeffs, rhs): pass
    def add_lin_le(self, vids, coeffs, rhs): pass
    def add_lin_ge(self, vids, coeffs, rhs): pass
    def add_sum_eq(self, vids, rhs): pass
    def add_ge_zero(self, vids): pass
    def add_sos2(self, var_ids, weights=None): pass
    def set_objective_linear(self, vids, coeffs, const=0.0, sense="min"): pass
    def optimize(self, timelimit: Optional[float] = None) -> None: pass

    # ----- Real entry: cons walker -----
    def consume_cons(
        self, globalC: ConSet, before: Dict[int, Fact], after: Dict[int, Fact],
        *, net: Net, input_ids: List[int], output_ids: List[int],
        assert_layer: Layer,
    ) -> str:
        # Lazy imports
        try:
            from HyZor import (
                hz_from_bounds, hz_dense, hz_conv2d, hz_add_const, hz_scale,
                hz_bn, hz_minkowski_sum, hz_sgm_add, shares_generator,
                hz_concat, hz_intersect_polytope,
                hz_apply_relu_v8, hz_apply_leaky_relu_v8,
                check_unsafe_for_act, lp_witness_to_input,
                strict_replay_for_act,
            )
        except ImportError as e:
            raise RuntimeError(
                f"HyZorSolver: cannot import HyZor from {_HYZOR_ROOT}. "
                f"Set $HYZOR_ROOT or `pip install -e {_HYZOR_ROOT}`. "
                f"Underlying: {e}"
            )

        # ACT operators (sigmoid/tanh K-piece -- ACT innovation)
        from act.back_end.hybridz_tf.tf_mlp import (
            hz_apply_sigmoid as act_hz_apply_sigmoid,
            hz_apply_tanh as act_hz_apply_tanh,
        )

        # ─── Phase 1: cons walker ───
        cons_by_layer: Dict[int, List[Con]] = {}
        global_polys: List[Con] = []
        for con in globalC:
            tag = con.meta.get("tag", "")
            if tag == "in:linpoly":
                global_polys.append(con); continue
            if tag.startswith("box:"): continue
            if ":" in tag:
                try:
                    lid = int(tag.split(":")[-1])
                    cons_by_layer.setdefault(lid, []).append(con)
                except ValueError: pass

        # Build initial input HZ
        input_box = self._extract_input_box(globalC, input_ids, before)
        device = torch.device(self.cfg["device"])
        dtype = self.cfg["dtype"]
        input_hz = hz_from_bounds(input_box, dtype=dtype, device=device)
        for poly_con in global_polys:
            input_hz = hz_intersect_polytope(
                input_hz, poly_con.meta["A"], poly_con.meta["b"])

        var_to_hz: Dict[Tuple[int, ...], Any] = {tuple(input_ids): input_hz}

        # ── Pre-scan: detect large_cls_proof_mode + count relus/convs ──
        relu_layer_ids: List[int] = []
        conv_count = 0
        for L in net.layers:
            ku = L.kind.upper()
            if ku == "RELU": relu_layer_ids.append(L.id)
            elif ku in ("CONV2D", "CONV1D", "CONV3D"): conv_count += 1
        total_relu = len(relu_layer_ids)
        out_dim = len(output_ids)

        lc_mode = self.cfg["large_cls_proof_mode"]
        if lc_mode == "auto":
            large_cls_active = (
                self.cfg["relu_method"] == "eq_lagr_v8"
                and conv_count >= self.cfg["large_cls_conv_threshold"]
                and out_dim >= self.cfg["large_cls_out_dim_threshold"]
            )
        else:
            large_cls_active = (lc_mode == "on")

        # ReLU index bookkeeping: which relu (1..total) is the next one we hit
        relu_idx_map: Dict[int, int] = {
            lid: i + 1 for i, lid in enumerate(relu_layer_ids)
        }
        eq_last = self.cfg["large_cls_eq_layers"]

        if large_cls_active:
            print(f"  [hyzor] large_cls_proof_mode ACTIVE: "
                  f"conv={conv_count} out_dim={out_dim} relus={total_relu} "
                  f"(triangle for relu 1..{total_relu - eq_last}, "
                  f"eq_lagr_v8 for last {eq_last})", flush=True)
            self._stats["large_cls_active"] = True
            self._stats["total_relu"] = total_relu
            self._stats["eq_last"] = eq_last
        self._lc_active = large_cls_active
        self._relu_idx_map = relu_idx_map
        self._total_relu = total_relu

        op_counts: Dict[str, int] = {}
        for L in net.layers:
            if L.kind in ("INPUT", "INPUT_SPEC", "ASSERT"):
                continue

            in_var_tuple = tuple(L.in_vars)
            hz_in = var_to_hz.get(in_var_tuple)
            multi_in_hzs = (
                self._collect_multi_input_hzs(L, var_to_hz, net)
                if hz_in is None else None
            )

            cons_list = cons_by_layer.get(L.id, [])
            op_con = next(
                (c for c in cons_list
                 if not c.meta.get("tag", "").startswith("box:")),
                None
            )

            # Per-layer logging (so OOM-kill leaves a trail)
            tag_for_log = (op_con.meta["tag"] if op_con else f"box-fallback({L.kind})")
            in_dim = hz_in.dim if hz_in is not None else "n/a"
            in_ng = hz_in.ng if hz_in is not None else "n/a"
            print(f"  [hyzor L{L.id}] {tag_for_log}  "
                  f"in: dim={in_dim} ng={in_ng}", flush=True)
            try:
                hz_out = self._dispatch(
                    L, op_con, hz_in, multi_in_hzs, before, after,
                    # HyZor ops
                    hz_dense=hz_dense, hz_conv2d=hz_conv2d,
                    hz_add_const=hz_add_const, hz_scale=hz_scale, hz_bn=hz_bn,
                    hz_sgm_add=hz_sgm_add, hz_minkowski_sum=hz_minkowski_sum,
                    shares_generator=shares_generator, hz_concat=hz_concat,
                    hz_apply_relu_v8=hz_apply_relu_v8,
                    hz_apply_leaky_relu_v8=hz_apply_leaky_relu_v8,
                    hz_from_bounds=hz_from_bounds,
                    # ACT ops (sigmoid/tanh K-piece)
                    act_hz_apply_sigmoid=act_hz_apply_sigmoid,
                    act_hz_apply_tanh=act_hz_apply_tanh,
                )

                # Girard reduction: cap ng to keep memory bounded
                hz_out = self._maybe_reduce(hz_out)

                print(f"  [hyzor L{L.id}] done  "
                      f"out: dim={hz_out.dim} ng={hz_out.ng} nb={hz_out.nb} nc={hz_out.nc}",
                      flush=True)
            except Exception as e:
                # Sound fallback on any per-layer failure
                self._stats[f"error@{L.id}"] = f"{type(e).__name__}: {e}"
                hz_out = self._box_fallback(L, after, hz_from_bounds)
                print(f"  [hyzor L{L.id}] FALLBACK ({type(e).__name__})", flush=True)

            var_to_hz[tuple(L.out_vars)] = hz_out
            op_kind = (op_con.meta["tag"].split(":")[0]
                       if op_con else f"box-fallback({L.kind})")
            op_counts[op_kind] = op_counts.get(op_kind, 0) + 1

        out_hz = var_to_hz.get(tuple(output_ids))
        if out_hz is None:
            self._status = SolveStatus.UNKNOWN
            self._stats["error"] = "no output HZ"
            return self._status
        self._stats["op_counts"] = op_counts

        # ─── Phase 2: LP feasibility ───
        try:
            feas, xi_star = check_unsafe_for_act(
                out_hz, assert_layer,
                output_ids=output_ids,
                timeout_s=self.cfg["timeout_s"]
            )
        except Exception as e:
            self._status = SolveStatus.UNKNOWN
            self._stats["feasibility_error"] = f"{type(e).__name__}: {e}"
            return self._status

        if feas == "infeasible":
            # v9.2: PDNT multi-cert. dual_cert_n=1 trusts LP single cert.
            # dual_cert_n>=2 requires additional independent cert(s) to
            # confirm before declaring "verified".
            n_certs_required = self.cfg["dual_cert_n"]
            if n_certs_required <= 1:
                self._status = SolveStatus.UNSAT
                self._stats["dual_cert_n"] = 1
                return self._status
            # Run additional cert: UNC bound on out_hz vs assert_layer
            extra_certs_pass = self._extra_certs_verified(
                out_hz, assert_layer, output_ids)
            n_passing = 1 + sum(extra_certs_pass.values())  # +1 for LP
            self._stats["dual_cert_n_required"] = n_certs_required
            self._stats["dual_cert_passes"] = n_passing
            self._stats["dual_cert_detail"] = extra_certs_pass
            if n_passing >= n_certs_required:
                self._status = SolveStatus.UNSAT
                return self._status
            # Multi-cert disagreement → conservative downgrade to UNKNOWN
            self._status = SolveStatus.UNKNOWN
            self._stats["dual_cert_rejected"] = True
            return self._status
        if feas == "timeout":
            self._status = SolveStatus.UNKNOWN
            self._stats["timeout"] = True
            return self._status

        # ─── Phase 3: witness back to input space ───
        try:
            x_star = lp_witness_to_input(xi_star, input_hz)
        except Exception as e:
            self._status = SolveStatus.UNKNOWN
            self._stats["witness_error"] = f"{type(e).__name__}: {e}"
            return self._status

        # ─── Phase 4: strict replay ───
        if self.cfg["strict_replay"]:
            try:
                ok = strict_replay_for_act(
                    net=net, x_star=x_star, assert_layer=assert_layer
                )
            except Exception as e:
                ok = False
                self._stats["replay_error"] = f"{type(e).__name__}: {e}"
            if not ok:
                self._status = SolveStatus.UNKNOWN
                self._stats["phantom_rejected"] = True
                return self._status

        self._status = SolveStatus.SAT
        self._witness = np.asarray(x_star, dtype=np.float64).ravel()
        self._has_solution = True
        # Final cleanup: release intermediate HZ tensors held in var_to_hz
        try:
            del var_to_hz, out_hz, input_hz
            import torch as _t
            if _t.cuda.is_available() and self.cfg.get("device","cpu").startswith("cuda"):
                _t.cuda.empty_cache()
        except Exception:
            pass
        return self._status

    # ----- Per-layer dispatch (cons tag -> HyZor or ACT op) -----
    def _dispatch(self, L, op_con, hz_in, multi_in_hzs, before, after, **ops):
        if op_con is None:
            return self._box_fallback(L, after, ops["hz_from_bounds"])
        tag = op_con.meta["tag"]; op = tag.split(":")[0]; meta = op_con.meta

        # ── HyZor ops ──
        if op == "dense":
            return ops["hz_dense"](hz_in, meta["W"], meta.get("b"))
        if op == "conv2d":
            cp = meta.get("conv_params", {})
            return ops["hz_conv2d"](
                hz_in, meta["weight"], meta.get("b"),
                input_shape=meta["input_shape"],
                stride=cp.get("stride", 1), padding=cp.get("padding", 0),
                dilation=cp.get("dilation", 1), groups=cp.get("groups", 1)
            )
        if op == "bias":  return ops["hz_add_const"](hz_in, meta["c"])
        if op == "scale": return ops["hz_scale"](hz_in, meta["a"])
        if op == "bn":    return ops["hz_bn"](hz_in, meta["A"], meta["c"])
        if op == "add":
            if multi_in_hzs is None or len(multi_in_hzs) < 2:
                return self._box_fallback(L, after, ops["hz_from_bounds"])
            hz_x, hz_y = multi_in_hzs[0], multi_in_hzs[1]
            return (ops["hz_sgm_add"](hz_x, hz_y)
                    if (self.cfg["sgm_enabled"] and
                        ops["shares_generator"](hz_x, hz_y))
                    else ops["hz_minkowski_sum"](hz_x, hz_y))
        if op == "sub":
            # z = x - y. Negate hz_y, then minkowski sum.
            if multi_in_hzs is None or len(multi_in_hzs) < 2:
                return self._box_fallback(L, after, ops["hz_from_bounds"])
            hz_x, hz_y = multi_in_hzs[0], multi_in_hzs[1]
            from HybridZonotope import HybridZonotope as _HZ
            hz_y_neg = _HZ(
                Gc=-hz_y.Gc, Gb=-hz_y.Gb, c=-hz_y.c,
                Ac=hz_y.Ac.clone(), Ab=hz_y.Ab.clone(), b=hz_y.b.clone(),
                device=hz_y.device, dtype=hz_y.dtype,
            )
            return ops["hz_minkowski_sum"](hz_x, hz_y_neg)
        if op == "concat":
            if multi_in_hzs is None: multi_in_hzs = [hz_in]
            return ops["hz_concat"](multi_in_hzs)
        if op == "relu":
            # large_cls_proof_mode: triangle for early relus, eq_lagr_v8 for last N
            method = self.cfg["relu_method"]
            if getattr(self, "_lc_active", False):
                ridx = self._relu_idx_map.get(L.id, 0)
                eq_last = self.cfg["large_cls_eq_layers"]
                if ridx <= self._total_relu - eq_last:
                    method = "triangle"
            return ops["hz_apply_relu_v8"](
                hz_in,
                method=method,
                mace=self.cfg["mace_enabled"],
                girard_cap=self.cfg["girard_cap"]
            )
        if op == "lrelu":
            return ops["hz_apply_leaky_relu_v8"](hz_in, alpha=meta["alpha"])

        # ── ACT ops (sigmoid/tanh K-piece -- ACT innovation) ──
        # Dim guard: ACT's hz_apply_piecewise has a Python-level loop over
        # wide neurons; for dim > sigmoid_dim_cap it becomes prohibitively
        # slow. Fall back to box (sound) on large dims.
        if op == "sigmoid":
            cap = int(os.environ.get("HYZOR_SIGMOID_DIM_CAP", "256"))
            if int(hz_in.dim) > cap:
                return self._box_fallback(L, after, ops["hz_from_bounds"])
            hzono_in = self._hyzor_to_hzono(hz_in)
            hzono_out = ops["act_hz_apply_sigmoid"](
                hzono_in, K=self.cfg["sigmoid_K"]
            )
            return self._hzono_to_hyzor(hzono_out)
        if op == "tanh":
            cap = int(os.environ.get("HYZOR_TANH_DIM_CAP", "256"))
            if int(hz_in.dim) > cap:
                return self._box_fallback(L, after, ops["hz_from_bounds"])
            hzono_in = self._hyzor_to_hzono(hz_in)
            hzono_out = ops["act_hz_apply_tanh"](
                hzono_in, K=self.cfg["tanh_K"]
            )
            return self._hzono_to_hyzor(hzono_out)

        # ── Shape ops ──
        if op in ("flatten", "reshape", "transpose", "squeeze",
                  "unsqueeze", "tile", "expand"):
            return hz_in

        # ── Fallback ──
        return self._box_fallback(L, after, ops["hz_from_bounds"])

    # ----- Helpers -----
    def _extract_input_box(self, globalC, input_ids, before):
        for con in globalC:
            tag = con.meta.get("tag", "")
            if tag.startswith("box:") and set(con.var_ids) == set(input_ids):
                return Bounds(lb=con.meta["lb"], ub=con.meta["ub"])
        for lid, fact in before.items():
            return fact.bounds
        raise RuntimeError("HyZorSolver: cannot find input box")

    def _collect_multi_input_hzs(self, L, var_to_hz, net):
        out = []
        for pid in net.preds.get(L.id, []):
            pred_layer = net.by_id[pid]
            tup = tuple(pred_layer.out_vars)
            if tup in var_to_hz:
                out.append(var_to_hz[tup])
        return out

    def _box_fallback(self, L, after, hz_from_bounds):
        b = after[L.id].bounds
        return hz_from_bounds(
            Bounds(b.lb, b.ub),
            dtype=self.cfg["dtype"],
            device=torch.device(self.cfg["device"])
        )

    def _maybe_reduce(self, hz):
        """Apply Girard generator reduction if ng exceeds budget.

        Bounds memory by capping continuous generators at girard_cap.
        Sound (over-approximation)."""
        cap = self.cfg["girard_cap"]
        if int(hz.ng) <= cap:
            return hz
        try:
            return hz.reduce_constraints(ng_budget=cap)
        except Exception:
            return hz

    def _extra_certs_verified(self, out_hz, assert_layer, output_ids) -> dict:
        """v9.2 PDNT extra certs (besides the LP cert that just said infeasible).

        Returns dict of {cert_name: bool}. Each True means that independent
        cert also proves the unsafe predicate is false (i.e. spec verified).

        Pool currently implemented:
          - U: UNC bound (cheapest, sound). Just compute Girard interval
               bounds on out_hz and check spec is unsafe-infeasible.

        Future certs (E/Z/F) deferred to a follow-up minor; this 2-cert
        (LP + UNC) is the minimum useful multi-cert for the "directly verify"
        comparison setting.
        """
        pool = set(self.cfg["dual_cert_pool"].upper())
        results = {}
        if "U" in pool:
            results["U"] = self._cert_U_unc_verified(out_hz, assert_layer)
        return results

    def _cert_U_unc_verified(self, out_hz, assert_layer) -> bool:
        """UNC cert: use Girard interval bounds to prove unsafe-infeasible."""
        try:
            lb_t, ub_t = out_hz._bounds_unconstrained()
            import numpy as np
            lb = lb_t.detach().cpu().numpy().reshape(-1)
            ub = ub_t.detach().cpu().numpy().reshape(-1)
            kind = assert_layer.params.get("kind")
            kstr = str(kind).split(".")[-1] if hasattr(kind, "__class__") else str(kind)
            eps = self.cfg["dual_cert_margin"]

            def _unwrap_int(x):
                if hasattr(x, "item"): return int(x.item())
                if hasattr(x, "__len__"): return int(x[0])
                return int(x)

            def _to_np(x):
                import torch as _t
                if _t.is_tensor(x): return x.detach().cpu().numpy()
                return np.asarray(x)

            if kstr == "TOP1_ROBUST":
                t = _unwrap_int(assert_layer.params["y_true"])
                # Spec: y[t] > y[j] for all j ≠ t
                # Spec verified iff lb[t] > max_{j≠t} ub[j]   (with eps)
                others_ub = np.delete(ub, t)
                return float(lb[t]) > float(others_ub.max()) + eps
            if kstr == "MARGIN_ROBUST":
                t = _unwrap_int(assert_layer.params["y_true"])
                m = float(_to_np(assert_layer.params["margin"]).reshape(-1)[0])
                others_ub = np.delete(ub, t)
                return float(lb[t]) > float(others_ub.max()) + m + eps
            if kstr == "LINEAR_LE":
                # Spec: c·y ≤ d.   Verified iff max c·y ≤ d.
                coef = _to_np(assert_layer.params["c"]).reshape(-1)
                d = float(_to_np(assert_layer.params["d"]).reshape(-1)[0])
                pos = np.clip(coef, 0, None); neg = np.clip(coef, None, 0)
                cy_max = float(pos @ ub + neg @ lb)
                return cy_max <= d - eps
            if kstr == "UNSAFE_LINEAR":
                # Spec violated (unsafe) iff ALL rows of C·y ≤ d.
                # Verified (NOT unsafe) iff ANY row C[i]·y > d[i] always.
                C = _to_np(assert_layer.params["c"])
                d_vec = _to_np(assert_layer.params["d"]).reshape(-1)
                if C.ndim == 1: C = C.reshape(1, -1)
                for i in range(C.shape[0]):
                    coef = C[i]
                    pos = np.clip(coef, 0, None); neg = np.clip(coef, None, 0)
                    cy_min = float(pos @ lb + neg @ ub)
                    if cy_min > float(d_vec[i]) + eps:
                        return True
                return False
            if kstr == "RANGE":
                # Spec: lb_t ≤ y ≤ ub_t. Verified iff actual bounds
                # are strictly inside spec bounds.
                lb_spec = assert_layer.params.get("lb")
                ub_spec = assert_layer.params.get("ub")
                if lb_spec is not None:
                    lb_v = _to_np(lb_spec).reshape(-1)
                    if (lb < lb_v + eps).any(): return False
                if ub_spec is not None:
                    ub_v = _to_np(ub_spec).reshape(-1)
                    if (ub > ub_v - eps).any(): return False
                return True
            return False
        except Exception:
            return False

    # ----- HZ data structure conversion (HyZor <-> ACT) -----
    def _hyzor_to_hzono(self, hyzor_hz):
        from act.back_end.solver.solver_hz import HZono
        return HZono(
            c=hyzor_hz.c, Gc=hyzor_hz.Gc, Gb=hyzor_hz.Gb,
            Ac=hyzor_hz.Ac, Ab=hyzor_hz.Ab, b=hyzor_hz.b,
        )

    def _hzono_to_hyzor(self, hzono):
        from HybridZonotope import HybridZonotope
        return HybridZonotope(
            Gc=hzono.Gc, Gb=hzono.Gb, c=hzono.c,
            Ac=hzono.Ac, Ab=hzono.Ab, b=hzono.b,
        )

    # ----- Result accessors -----
    def status(self) -> str:
        return self._status

    def has_solution(self) -> bool:
        return self._has_solution

    def get_values(self, vids: List[int]) -> np.ndarray:
        if self._witness is None:
            return np.zeros(len(vids), dtype=np.float64)
        return self._witness[: len(vids)]

    def get_counterexample(self, input_ids: List[int]) -> np.ndarray:
        return self.get_values(input_ids)

    def stats(self) -> dict:
        return dict(self._stats)
