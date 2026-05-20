"""HyZor adapter as Tier-2 ACT Solver.

Walks ACT cons IR (verified correct in Step 0) and dispatches to:
  - HyZor ops (dense/conv2d/relu eq_lagr_v8/add SGM/...) via `import HyZor`
  - ACT ops (sigmoid/tanh K-piece tangent parallelogram) via
    `import act.back_end.hybridz_tf.tf_mlp` -- preserves ACT's innovation

NO HyZor algorithm code is duplicated. HyZor upgrades on its own ->
ACT updates automatically.

================================================================
MIGRATION STATUS (2026-05-19)
================================================================
HyZor algorithms now have a parallel home under
``act/back_end/hybridz_tf/algorithms/`` (and
``act/back_end/hybridz_tf/representations.py``). Future HZ solvers
should compose those modules instead of importing the HyZor repo.

  ✓ P1 -- Plumbing: cli/config/hybridz_tf SLICE/vnnlib_loader/
       torch2act/utils/validate_verifier — landed in this merge.

  ✓ P2 -- HZ algorithms in ACT (parallel implementations,
       feature-light but verified by self-tests):
       * ``algorithms/sgm.py``         -- shares_generator + hz_sgm_add
       * ``algorithms/eq_elim.py``     -- project_eq_elim
       * ``algorithms/binary_probe.py`` -- RIIM + LP singleton probing
       * ``algorithms/lp_verify.py``   -- check_unsafe_for_act + witness
       * ``algorithms/cascade.py``     -- ReLU encoding scheduler
       * ``representations.py``        -- BoxHZ / LazyChainHZ / SparseGcZ
       * ``solver_hz.HZono``           -- extended with eq_mask + ineq

  ✓ P5 -- ``solver_simple_hz.py`` (50-line template) demonstrates
       that the ACT algorithms compose into a working HZ solver
       without any HyZor-repo dependency.

  ◌ P4 -- This file (solver_hyzor.py) still imports HyZor's
       full implementations (binary_probe_v8 + project_eq_elim +
       hz_apply_relu_v8 cascade) via ``import HyZor``. Reason: the
       ACT-side ports are deliberately simplified (no pairwise RIIM
       v2, no GPU PEE QR, no LinearPrefilter, no profiling). Until
       feature parity lands, this solver continues to use HyZor's
       feature-rich versions so the verified cifar100 154/200 and
       tinyimagenet 175/175 results remain reproducible bit-for-bit.

       The MIGRATION-READY state is: every algorithm has both a
       full implementation in HyZor and a simpler port in ACT; we
       choose feature-richness today and migrate progressively.

  ◌ P6 -- ``verify_once_legacy_batch1`` shim below remains required
       until the HyZor solver is replaced by an analyze() +
       hybridz_tf-native dispatch flow.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
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
        onnx_path: Optional[str] = None,
        vnnlib_path: Optional[str] = None,
        # Tier 1A: LP-aggressive tail pass. After cons-walker returns UNKNOWN,
        # retry with `exact_lp` ReLU encoding (more LP cons per ReLU = tighter).
        # Gating mirrors HyZor's HYZOR_L2_EXACTLP_TAIL: small/medium nets only.
        l2_exactlp_tail: bool = False,        # off by default (opt-in)
        l2_exactlp_relu_cap: int = 128,       # only retry if ≤ this many ReLUs
        l2_exactlp_out_cap: int = 512,        # only retry if output dim ≤ this
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
            # Tier 1A: profiles that set HYZOR_L2_EXACTLP_TAIL in extra_env
            # also enable the cons-walker's LP-aggressive retry path. Caps
            # come from profile env (RELU_CAP / OUT_CAP) when set.
            _eenv = profile.get("extra_env", {})
            if _eenv.get("HYZOR_L2_EXACTLP_TAIL") == "1":
                l2_exactlp_tail = True
                l2_exactlp_relu_cap = int(float(_eenv.get("HYZOR_L2_EXACTLP_RELU_CAP", l2_exactlp_relu_cap)))
                l2_exactlp_out_cap = int(float(_eenv.get("HYZOR_L2_EXACTLP_OUT_CAP", l2_exactlp_out_cap)))
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
            onnx_path=onnx_path,
            vnnlib_path=vnnlib_path,
            l2_exactlp_tail=l2_exactlp_tail,
            l2_exactlp_relu_cap=l2_exactlp_relu_cap,
            l2_exactlp_out_cap=l2_exactlp_out_cap,
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

    # ─── LEGACY_SHIM_TO_REMOVE_AT_P3 ───────────────────────────────────
    # HyZor's verification path walks the ACT cons IR via consume_cons()
    # (a custom analyze-walking pipeline) rather than consuming a
    # BatchLPProblem. Until eq_lagr_v8 / project_eq_elim / Phase-1-3
    # representations land in act/back_end/hybridz_tf/ and the cascade
    # controller moves to hybridz_tf/algorithms/, the new
    # setup_and_solve_batch + verify_once entry points cannot drive
    # HyZor end-to-end. solve_batch therefore mirrors HZSolver's design
    # (raise with redirect message); callers must use
    # ``verify_once_legacy_batch1`` (defined below) for HyZor-mode
    # verification of a single (model, vnnlib) instance.
    def solve_batch(self, problem, timelimit: Optional[float] = None):  # noqa: D401
        """HyZorSolver does not accept BatchLPProblem inputs.

        HyZor walks the ACT cons IR directly via ``consume_cons``; it
        does not consume a pre-built LP. Callers verifying a single
        instance through HyZor should use
        ``verify_once_legacy_batch1(net, solver=..., timelimit=...)``
        from this module. Batch-native HyZor integration via
        ``hybridz_tf`` is a follow-up (see comment block above).
        """
        raise NotImplementedError(
            "HyZorSolver does not consume BatchLPProblem; use "
            "act.back_end.solver.solver_hyzor.verify_once_legacy_batch1"
            "(net, solver=..., timelimit=...) for single-instance HyZor "
            "verification until hybridz_tf integration lands."
        )

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
        # Phase 5.2 (2026-05-20): hyzor_compat re-exports ACT-native
        # versions where parity-tested at the ALGORITHM level (61 tests,
        # 0.0e+00 element-wise error). However v108 A/B regression
        # revealed ACT-default is more memory-hungry than HyZor on
        # cifar/tiny: HyZor's hz_dense/hz_conv2d auto-dispatch to
        # Phase 1-3 representations (BoxHZ / LazyChainHZ / SparseGcZ)
        # when memory tight, ACT lacks that routing. 37/37 OOMs on
        # tinyimagenet under vLLM contention. Default ROLLED BACK to
        # LEGACY until Phase 1-3 routing is ported (Phase 6 follow-up).
        # ``HYZOR_USE_ACT=1`` opts in to the ACT-default path for tests
        # / small models where memory pressure is absent.
        _use_act = os.environ.get("HYZOR_USE_ACT", "0") == "1"
        if not _use_act:
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
        else:
            from act.back_end.hybridz_tf.hyzor_compat import (
                hz_from_bounds, hz_dense, hz_conv2d, hz_add_const, hz_scale,
                hz_bn, hz_minkowski_sum, hz_sgm_add, shares_generator,
                hz_concat, hz_intersect_polytope,
                hz_apply_relu_v8, hz_apply_leaky_relu_v8,
                check_unsafe_for_act, lp_witness_to_input,
                strict_replay_for_act,
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

        # ─── Optional dispatch: SpecAwareLP for small-dense networks ───
        # Forward + LP only, no split / BaB / backward / Gurobi. Sound.
        # GlobalTriangleLP baseline: 61/186 on acasxu (paper §6.7).
        # +unsafe-spec-conditioned bound refinement: 74/186 (+13, 0 regr).
        # See paper §6.7 + project_specaware_refinement_20260516 memory.
        # Gated by HYZOR_USE_GLOBAL_LP env: "auto" (default) | "on" | "off".
        # auto: dispatch iff is_small_dense(onnx_path) (no conv, in_dim≤32, relus≤500).
        # Detector lives in GlobalTriangleLP module (same heuristic).
        _gtlp_mode = os.environ.get("HYZOR_USE_GLOBAL_LP", "auto").lower()
        _onnx_p = self.cfg.get("onnx_path")
        _vnn_p = self.cfg.get("vnnlib_path")
        if _gtlp_mode != "off" and _onnx_p and _vnn_p and conv_count == 0:
            try:
                # Phase 6.4: GlobalTriangleLP / WitnessExtract are byte-
                # identical copies relocated under
                # ``act.back_end.hybridz_tf.algorithms``. Phase 5.2 rollback:
                # default LEGACY for safety; HYZOR_USE_ACT=1 opts in.
                if os.environ.get("HYZOR_USE_ACT", "0") == "1":
                    from act.back_end.hybridz_tf.algorithms.global_triangle_lp import is_small_dense as _is_sd
                    from act.back_end.hybridz_tf.algorithms.witness_extract import verify_with_falsification as _we_verify
                else:
                    import sys as _sys_gtlp
                    _hz_root = os.environ.get("HYZOR_ROOT", _HYZOR_ROOT)
                    if _hz_root not in _sys_gtlp.path:
                        _sys_gtlp.path.insert(0, _hz_root)
                    from GlobalTriangleLP import is_small_dense as _is_sd
                    from WitnessExtract import verify_with_falsification as _we_verify
                _dispatch = (_gtlp_mode == "on") or (
                    _gtlp_mode == "auto" and _is_sd(_onnx_p)
                )
                if _dispatch:
                    print(f"  [hyzor] WitnessExtract(SA+falsify) dispatch: "
                          f"mode={_gtlp_mode} onnx={Path(_onnx_p).name} "
                          f"vnnlib={Path(_vnn_p).name}", flush=True)
                    _verdict, _x_wit, _y_wit, _elapsed = _we_verify(
                        _onnx_p, _vnn_p,
                        time_limit_per_lp=15.0,
                        max_refinement_passes=3,
                        return_witness=True,
                    )
                    self._stats["global_lp_dispatch"] = True
                    self._stats["global_lp_verdict"] = _verdict
                    self._stats["global_lp_elapsed_s"] = _elapsed
                    if _verdict == "verified":
                        self._status = SolveStatus.UNSAT  # safe
                        return self._status
                    elif _verdict == "falsified":
                        # Stash witness so verify_once promotes SAT → FALSIFIED
                        import numpy as _np_wit
                        self._witness = _np_wit.asarray(_x_wit, dtype=_np_wit.float64).ravel()
                        self._has_solution = True
                        self._status = SolveStatus.SAT
                        return self._status
                    else:
                        # WitnessExtract demotes all non-verified-non-falsified
                        # (unknown OR solver fail) to 'unknown'. No fallback to
                        # HZ cascade for small-dense — HZ K-sweep best is 5/186
                        # vs WE 74V+15A=89/186.
                        self._status = SolveStatus.UNKNOWN
                        return self._status
            except Exception as _e_gtlp:
                print(f"  [hyzor] SpecAwareLP dispatch error "
                      f"({type(_e_gtlp).__name__}: {_e_gtlp}); falling back",
                      flush=True)
                self._stats["global_lp_dispatch"] = False
                self._stats["global_lp_error"] = f"{type(_e_gtlp).__name__}: {_e_gtlp}"

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
                # MaxPool: ACT cons_exporter doesn't generate constraints
                # for max-pool (it's not a linear op), so op_con is None
                # and the generic _dispatch fall-through would land in
                # _box_fallback (discarding all HZ correlation). Instead,
                # detect MAXPOOL2D explicitly here and route to the HZ
                # max_pool_node_evaluate via the hz_maxpool2d facade,
                # which preserves stable-winner rows exactly and falls
                # back to interval only on unstable blocks.
                if op_con is None and L.kind == "MAXPOOL2D" and hz_in is not None:
                    try:
                        # Phase 5.2 rollback: default LEGACY for memory
                        # safety; HYZOR_USE_ACT=1 opts in.
                        if os.environ.get("HYZOR_USE_ACT", "0") == "1":
                            from act.back_end.hybridz_tf.hyzor_compat import hz_maxpool2d
                        else:
                            from HyZor import hz_maxpool2d
                        params = L.params
                        in_shape = params.get("input_shape")
                        if in_shape is None:
                            # fall back to box if shape missing
                            hz_out = self._box_fallback(L, after, hz_from_bounds)
                        else:
                            hz_out = hz_maxpool2d(
                                hz_in,
                                kernel_size=params["kernel_size"],
                                stride=params.get("stride"),
                                padding=params.get("padding", 0),
                                input_shape=in_shape,
                            )
                    except Exception as e:
                        # Sound fallback on any failure (shape mismatch etc.)
                        self._stats[f"maxpool_fallback@{L.id}"] = f"{type(e).__name__}: {e}"
                        hz_out = self._box_fallback(L, after, hz_from_bounds)
                else:
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
                import os as _osdbg, traceback as _tb
                if _osdbg.environ.get("HYZOR_DEBUG_FALLBACK", "0") == "1":
                    print(f"  [hyzor L{L.id}] FALLBACK ({type(e).__name__}): {e}",
                          flush=True)
                    _tb.print_exc()
                else:
                    print(f"  [hyzor L{L.id}] FALLBACK ({type(e).__name__})",
                          flush=True)

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

        # Stash for Tier 1A LP-aggressive retry: if Phase 2/4 ends UNKNOWN,
        # we re-walk with relu_method='exact_lp' (tighter ReLU encoding).
        self._first_pass_method = self.cfg["relu_method"]

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

        # Tier 1A: LP-aggressive tail. Try retry with exact_lp on UNKNOWN-likely
        # outcomes (timeout or feasible-but-replay-rejected). Gating: small/med
        # nets only, not large_cls, not already exact_lp.
        def _should_try_lp_tail() -> bool:
            return (self.cfg.get("l2_exactlp_tail", False)
                    and not large_cls_active
                    and self.cfg["relu_method"] != "exact_lp"
                    and total_relu <= self.cfg.get("l2_exactlp_relu_cap", 128)
                    and out_dim <= self.cfg.get("l2_exactlp_out_cap", 512))

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
            # Tier 1A: retry with exact_lp before giving up
            if _should_try_lp_tail() and not getattr(self, "_lp_tail_active", False):
                self._stats["l2_exactlp_tail_attempted"] = True
                return self._retry_with_exact_lp(
                    globalC, before, after,
                    net=net, input_ids=input_ids,
                    output_ids=output_ids, assert_layer=assert_layer,
                    reason="phase2_timeout")
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
                # F1: prefer ORT replay over torch fallback (avoids 1e-6 fp drift
                # that caused phantom_rejected on acasxu).
                if self.cfg.get("onnx_path") and not getattr(net, "onnx_path", None):
                    try: net.onnx_path = self.cfg["onnx_path"]
                    except Exception: pass
                ok = strict_replay_for_act(
                    net=net, x_star=x_star, assert_layer=assert_layer
                )
            except Exception as e:
                ok = False
                self._stats["replay_error"] = f"{type(e).__name__}: {e}"
            if not ok:
                # Tier 1A: retry with exact_lp before giving up
                if _should_try_lp_tail() and not getattr(self, "_lp_tail_active", False):
                    self._stats["l2_exactlp_tail_attempted"] = True
                    return self._retry_with_exact_lp(
                        globalC, before, after,
                        net=net, input_ids=input_ids,
                        output_ids=output_ids, assert_layer=assert_layer,
                        reason="phase4_replay_rejected")
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

        # PR3 memory guard: predict output dim; if dim × ng exceeds budget,
        # fall back to box (interval) to avoid OOM. Used for ImageNet-scale
        # models like VGG16 (n=3.2M after first conv, ng=150K input pixels).
        # Budget defaults to 4 GB (configurable via env).
        import os as _os
        guard_gb = float(_os.environ.get("HYZOR_DISPATCH_GUARD_GB", "4.0"))
        guard_bytes = int(guard_gb * 1024 ** 3)
        def _would_oom(out_dim, in_ng):
            est = out_dim * in_ng * 8 * 3  # ×3 for intermediate copies
            return est > guard_bytes
        # Estimate output dim for the layer
        try:
            if op == "dense":
                out_dim = int(meta["W"].shape[0])
            elif op == "conv2d":
                out_shape = meta.get("output_shape")
                out_dim = int(out_shape[1] * out_shape[2] * out_shape[3]) if out_shape else 0
            else:
                out_dim = 0
            in_ng = int(hz_in.ng) if hz_in is not None else 0
            if out_dim > 0 and in_ng > 0 and _would_oom(out_dim, in_ng):
                self._stats[f"oom_guard@{L.id}"] = f"{op}: out_dim={out_dim}, ng={in_ng}, est_GB={out_dim*in_ng*8*3/(1024**3):.1f}"
                return self._box_fallback(L, after, ops["hz_from_bounds"])
        except Exception:
            pass  # if estimation fails, just attempt the op normally

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
        # SLICE actually subsets dims; box-fallback is sound (looser but correct)
        if op == "slice":
            return self._box_fallback(L, after, ops["hz_from_bounds"])

        # ── Fallback ──
        return self._box_fallback(L, after, ops["hz_from_bounds"])

    # ----- Helpers -----
    def _retry_with_exact_lp(
        self, globalC, before, after, *,
        net, input_ids, output_ids, assert_layer, reason: str
    ):
        """Tier 1A: re-run consume_cons with relu_method='exact_lp'.

        exact_lp encoding has more LP constraints per ReLU than eq_lagr_v8
        (tighter relaxation). For UNKNOWN cases on small/medium nets, this
        often promotes to verified. Only called once per query (guarded by
        self._lp_tail_active flag) — sound by construction.
        """
        prev_method = self.cfg["relu_method"]
        self._lp_tail_active = True
        self.cfg["relu_method"] = "exact_lp"
        try:
            print(f"  [hyzor] LP-aggressive retry (reason={reason}, method=exact_lp)",
                  flush=True)
            result = self.consume_cons(
                globalC, before, after,
                net=net, input_ids=input_ids,
                output_ids=output_ids, assert_layer=assert_layer,
            )
            self._stats["l2_exactlp_tail_outcome"] = result
            return result
        finally:
            self.cfg["relu_method"] = prev_method
            self._lp_tail_active = False

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


def _slice_facts_lane(facts: Dict[int, Fact], lane: int) -> Dict[int, Fact]:
    """Return a new facts dict where each Fact's batched bounds are
    sliced to the requested batch lane. Cons (if any) are passed through
    unchanged — legacy consume_cons only reads bounds from facts."""
    out: Dict[int, Fact] = {}
    for lid, f in facts.items():
        lb, ub = f.bounds.lb, f.bounds.ub
        if lb.dim() >= 2:
            lb = lb[lane]
            ub = ub[lane]
        out[lid] = Fact(bounds=Bounds(lb=lb, ub=ub), cons=f.cons)
    return out


def _slice_assert_layer_lane(assert_layer: Layer, lane: int) -> Layer:
    """Build a de-batched copy of ASSERT layer for single-instance consumers.

    PR #66's ASSERT params carry leading B axis on per-kind fields
    (``y_true: [B]``, ``c: [B, ...]``, ``d: [B, ...]``, ``margin: [B]``,
    ``lb/ub: [B, n_out]``, ``thresholds: [B, M]``, ``C: [B*M, n_out]``).
    HyZor's ``check_unsafe_for_act`` / ``_cert_U_unc_verified`` predate
    this and expect the unbatched per-kind layout. This helper slices
    lane ``lane`` so those readers see the same shapes they always did.
    """
    new_params: Dict[str, Any] = {}
    B_hint: Optional[int] = None
    for k, v in assert_layer.params.items():
        if k in ("kind", "M"):
            new_params[k] = v
            continue
        if hasattr(v, "dim") and hasattr(v, "shape"):
            if v.dim() >= 1 and v.shape[0] > 0:
                # Heuristic: leading dim is B for per-kind fields. For the
                # pre-encoded C of shape [B*M, n_out] we keep as-is — readers
                # of "C" key are PR-#66 callers, not the legacy path.
                if k == "C":
                    new_params[k] = v
                else:
                    new_params[k] = v[lane]
                    if B_hint is None:
                        B_hint = int(v.shape[0])
            else:
                new_params[k] = v
        else:
            new_params[k] = v
    # Construct a shallow Layer-like wrapper. ASSERT consumers only read
    # `.params`; building a fresh Layer keeps the Layer dataclass invariants
    # of the host module untouched.
    from dataclasses import replace as _dc_replace
    try:
        return _dc_replace(assert_layer, params=new_params)
    except Exception:
        # Fallback if Layer isn't a dataclass on this main: use a thin proxy
        class _LayerProxy:  # noqa: D401
            pass
        proxy = _LayerProxy()
        for attr in ("id", "kind", "in_vars", "out_vars"):
            if hasattr(assert_layer, attr):
                setattr(proxy, attr, getattr(assert_layer, attr))
        proxy.params = new_params
        return proxy  # type: ignore[return-value]


def _slice_globalC_lane(globalC: ConSet, lane: int) -> ConSet:
    """Slice batched ``box:`` meta-stored lb/ub to a single batch lane.
    Non-box cons are passed through unchanged."""
    out = ConSet()
    for sig, con in list(globalC.S.items()):
        meta = dict(con.meta) if con.meta else {}
        lb_meta = meta.get("lb")
        ub_meta = meta.get("ub")
        sliced = False
        if lb_meta is not None and hasattr(lb_meta, "dim") and lb_meta.dim() >= 2:
            meta["lb"] = lb_meta[lane].reshape(-1)
            sliced = True
        if ub_meta is not None and hasattr(ub_meta, "dim") and ub_meta.dim() >= 2:
            meta["ub"] = ub_meta[lane].reshape(-1)
            sliced = True
        if sliced:
            out.replace(Con(kind=con.kind, var_ids=con.var_ids, meta=meta))
        else:
            out.S[sig] = con
    return out


# ─── LEGACY_SHIM_TO_REMOVE_AT_P3 ───────────────────────────────────────
# Single-instance verifier entry that mirrors the pre-PR-#66
# ``verify_once(net, solver=..., timelimit=...)`` semantics on top of
# HyZor's ``consume_cons`` cons-IR walker. Driver scripts that pre-date
# the batch-native verifier (v100/v101/v102 and similar) call this
# helper instead of ``act.back_end.verifier.verify_once`` (which no
# longer accepts a ``solver=`` argument).
#
# Inputs:
#   - ``net``: ACT Net whose first layer is INPUT, last is ASSERT.
#     INPUT_SPEC may be batched [B, *shape]; this helper takes lane
#     ``batch_lane`` only (default 0; raises if B>1 and lane unset).
#   - ``solver``: HyZorSolver instance.
#   - ``timelimit``: optional wall-clock budget (seconds).
#
# Returns: ``(status: str, ce_input: Optional[np.ndarray], stats: dict)``
# matching the pre-PR-#66 return type.
#
# REMOVAL PLAN: once HyZor's HZ propagation lives in hybridz_tf and the
# cascade controller in hybridz_tf/algorithms, the new
# ``setup_and_solve_batch`` will dispatch to HyZor natively and this
# helper can be deleted.
def verify_once_legacy_batch1(
    net,
    *,
    solver: "HyZorSolver",
    timelimit: Optional[float] = None,
    batch_lane: int = 0,
) -> Tuple[str, Optional[np.ndarray], Dict[str, Any]]:
    """Pre-PR-#66 verify_once API on top of HyZorSolver.consume_cons.

    See module-level comment ``LEGACY_SHIM_TO_REMOVE_AT_P3``.
    """
    from act.back_end.analyze import analyze
    from act.back_end.transfer_functions import set_transfer_function_mode
    from act.back_end.verifier import (
        find_entry_layer_id, get_input_ids, get_output_ids,
        gather_input_spec_layers, get_assert_layer,
        seed_from_input_specs, add_all_input_specs, validate_constraints,
    )

    # consume_cons reads `.bounds` from before/after for _box_fallback
    # paths, so we need TIGHT bounds. An earlier attempt forced
    # interval-only TF for speed; it caused a regression on
    # cifar100_resnet_large (20 V down from baseline 52) because
    # looser box fallbacks turned verified instances into "unknown".
    # Default mode honored (hybridz). Set HYZOR_TF_MODE=interval to
    # force the fast-but-lossy path when speed matters more than recall.
    _tf_mode = os.environ.get("HYZOR_TF_MODE", "").strip().lower()
    if _tf_mode in ("interval", "hybridz"):
        set_transfer_function_mode(_tf_mode)

    entry_id = find_entry_layer_id(net)
    input_ids = get_input_ids(net)
    output_ids = get_output_ids(net)
    spec_layers = gather_input_spec_layers(net)
    assert_layer = get_assert_layer(net)

    # Run analyze with the batched seed (new TFs require [B, *shape]).
    seed_bounds = seed_from_input_specs(spec_layers)
    if seed_bounds.lb.dim() < 2:
        # Legacy 1-D seed: synthesize a B=1 batch so new TFs accept it.
        seed_bounds = Bounds(
            lb=seed_bounds.lb.unsqueeze(0), ub=seed_bounds.ub.unsqueeze(0)
        )
    B = int(seed_bounds.lb.shape[0])
    if batch_lane >= B:
        raise IndexError(
            f"verify_once_legacy_batch1: batch_lane={batch_lane} out of "
            f"range [0, {B})"
        )

    entry_fact = Fact(bounds=seed_bounds, cons=ConSet())
    add_all_input_specs(entry_fact.cons, input_ids, spec_layers)
    before, after, globalC = analyze(net, entry_id, entry_fact)
    validate_constraints(globalC, after, net)

    # HyZor's consume_cons predates batch-native analyze and expects
    # single-lane (1-D) bounds in before/after Facts. Slice lane
    # ``batch_lane`` so consume_cons sees the same input shape it always
    # has. globalC may also carry per-batch box rows; rewrite them to
    # the single-lane view.
    before_b1, after_b1 = _slice_facts_lane(before, batch_lane), _slice_facts_lane(after, batch_lane)
    globalC_b1 = _slice_globalC_lane(globalC, batch_lane)
    assert_layer_b1 = _slice_assert_layer_lane(assert_layer, batch_lane)

    if timelimit is not None and hasattr(solver, "cfg"):
        solver.cfg["timeout_s"] = float(timelimit)

    st = solver.consume_cons(
        globalC_b1, before_b1, after_b1,
        net=net, input_ids=input_ids, output_ids=output_ids,
        assert_layer=assert_layer_b1,
    )
    ce_input = None
    if st == SolveStatus.SAT and solver.has_solution():
        ce_input = solver.get_values(input_ids)

    stats: Dict[str, Any] = {
        "status": st, "ncons": len(globalC), "solver": "hyzor",
    }
    try:
        stats.update(solver.stats() if hasattr(solver, "stats") else {})
    except Exception:
        pass
    return st, ce_input, stats
