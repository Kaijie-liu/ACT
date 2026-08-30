# ===- act/back_end/config.py - Backend Configuration ---------------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------====#

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Final, List, Optional, Union

import yaml

_DEFAULT_YAML = Path(__file__).parent / "config.yaml"

_VALID_SOLVERS = {"auto", "gurobi", "torchlp", "dual", "hybridz"}
_VALID_DEVICES = {"cpu", "cuda", "gpu"}
_VALID_DTYPES = {"float32", "float64"}
_VALID_REGISTRY_MODES = {"intersection", "union"}
_VALID_COVERAGE_MODES = {"basic", "full"}
VALID_SOLVER_TIERS: Final[tuple[str, ...]] = ("lp", "dual", "dual_alpha", "dual_alpha_eta")
VALID_BERT_METHODS: Final[tuple[str, ...]] = (
    "planar",
    "rule",
    "alpha",
    "ibp",
    "discrete",
)


def normalize_query_dual_feedback_targets(value: Any) -> tuple[int, ...]:
    """Normalize query-dual layer ids from YAML or a comma-separated CLI.

    The first occurrence wins when an id is repeated.  Rejecting booleans and
    non-integral YAML values keeps the serialized experiment identity
    unambiguous (``True`` must never silently become layer ``1``).
    """

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            items: list[Any] = []
        else:
            tokens = [token.strip() for token in stripped.split(",")]
            if any(not token for token in tokens):
                raise ValueError(
                    "query_dual_feedback_targets must be a comma-separated "
                    "list of nonnegative integers"
                )
            try:
                items = [int(token, 10) for token in tokens]
            except ValueError as exc:
                raise ValueError(
                    "query_dual_feedback_targets must be a comma-separated "
                    "list of nonnegative integers"
                ) from exc
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        raise ValueError(
            "query_dual_feedback_targets must be a YAML list/tuple or a "
            "comma-separated CLI string"
        )

    normalized: list[int] = []
    seen: set[int] = set()
    for item in items:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(
                "query_dual_feedback_targets entries must be integers"
            )
        if item < 0:
            raise ValueError(
                "query_dual_feedback_targets entries must be nonnegative"
            )
        if item not in seen:
            seen.add(item)
            normalized.append(item)
    return tuple(normalized)


@dataclass(frozen=True)
class BertMethodSelection:
    """Resolved attention-relaxation BERT verification method."""

    method: str
    internal_method: str
    baf: bool
    alpha_mode: str
    solver_tier: str
    use_bab: bool = True


_BERT_METHOD_SELECTIONS: Final[dict[str, BertMethodSelection]] = {
    "planar": BertMethodSelection("planar", "planar", True, "fixed", "dual"),
    "rule": BertMethodSelection("rule", "rule", True, "rule", "dual"),
    "alpha": BertMethodSelection("alpha", "alpha", True, "optimized", "dual_alpha"),
    "ibp": BertMethodSelection("ibp", "ibp", False, "none", "dual"),
    "discrete": BertMethodSelection("discrete", "discrete", False, "none", "dual"),
}

BERT_METHOD_TIERS: Final[dict[str, str]] = {
    key: value.solver_tier for key, value in _BERT_METHOD_SELECTIONS.items()
}


def normalize_bert_method(method: str) -> str:
    """Normalize a public BERT method name."""
    key = method.strip().lower().replace("-", "_")
    if key not in _BERT_METHOD_SELECTIONS:
        valid = ", ".join(name.replace("_", "-") for name in VALID_BERT_METHODS)
        raise ValueError(f"Invalid bert method {method!r}; expected one of: {valid}")
    return key


def select_bert_method(method: str) -> BertMethodSelection:
    """Resolve a user-facing SST/Yelp method into ACT back-end settings."""
    return _BERT_METHOD_SELECTIONS[normalize_bert_method(method)]


# ---------------------------------------------------------------------------
# BaBConfig — Branch-and-Bound algorithm parameters
# ---------------------------------------------------------------------------


@dataclass
class BaBConfig:
    """Configuration for Branch-and-Bound verification algorithm.

    Construction::

        BaBConfig()                     # programmatic defaults
        BaBConfig.from_yaml()           # load from act/back_end/config.yaml
        BaBConfig.from_yaml(path, **kw) # custom YAML + overrides
    """

    max_depth: int = 20
    max_nodes: int = 2000
    frontier_cap: int = 0
    input_split_fanout: int = 2

    branching_method: str = "random"
    bounding_method: str = "random"
    bounding_order: str = "depth_lb"
    bounding_depth_weight: float = 0.5
    bounding_bound_weight: float = 0.5
    sa_cooling_rate: float = 0.99

    # Dual-tier solver knobs — support solver_tier="dual_alpha_eta" with
    # Iterative slope + Lagrange-multiplier optimization for the dual backward pass.
    solver_tier: str = "lp"
    f"""Solver tier for BaB bound computation. Valid: {VALID_SOLVER_TIERS}."""

    dual_n_iters: int = 50
    """Number of Adam iterations for α/η optimization (only used in ``dual_alpha`` / ``dual_alpha_eta`` tiers)."""

    lr_alpha: float = 0.1
    """Adam learning rate for α (slope) variables."""

    lr_beta: float = 0.1
    """Adam learning rate for η (split-constraint KKT multipliers). 0.1 default; tune per network."""

    lr_decay: float = 0.98
    """Multiplicative learning-rate decay applied each Adam iteration."""

    incremental_start_enabled: bool = True
    """Reuse α/η tensors from the parent subproblem as the initial point for child optimization."""

    per_class_alpha: bool = True
    """Allocate separate α tensors per output class (tighter bounds) rather than sharing one α."""

    provenance_enabled: bool = False
    """Track logical BaB node ids and parent ids in TopKBounding."""

    eta_only_children: bool = False
    """Freeze alpha in child subproblems (depth > 0): children inherit the
    parent's optimized alpha and refine only the split multipliers (eta).
    Cuts the per-node Adam graph and, combined with reuse_root_bounds,
    removes the per-iteration forward pass entirely."""

    presplit_levels: int = 0
    """Pre-split the root's top-k scored unstable neurons into all 2^k sign
    combinations before the main loop (LEAPS-style leap: descendants are
    materialized directly, intermediate tree levels are never bounded). The
    combinations exactly partition the root region, so soundness is
    unaffected. Requires a dual tier with neuron branching state."""

    intermediate_refine: str = "none"
    """Backward refinement of intermediate pre-activation bounds at the root:
    'none' (off), 'auto' (refine activation layers whose mean width exceeds
    intermediate_refine_ratio x the median - targets wide fan-in
    concretization loss), 'all' (every unstable activation layer)."""

    intermediate_refine_ratio: float = 10.0
    """Width-blowup threshold multiplier for intermediate_refine='auto'."""

    reuse_root_bounds: bool = False
    """Reuse the root box's forward bounds for every descendant (dual tiers).

    Sound by monotonicity: a child box is contained in the root box, so the
    root's per-layer bounds remain valid over-approximations. Children only
    override the INPUT/INPUT_SPEC bounds with their own sub-box; intermediate
    ReLU relaxations stay at root tightness, with branching gains recovered by
    the input-term concretization and the eta split multipliers. Eliminates
    the per-node forward pass (the dominant time and memory cost)."""

    per_subproblem_refine: str = "none"
    """Per-subproblem sparse backward refinement of intermediate bounds in the
    BaB loop (requires reuse_root_bounds): 'none' (off), 'tail' (last two
    unstable activation layers), 'all' (every unstable activation layer), or
    'split_successors' (the first reachable unstable ReLU descendants in
    network topological order, capped by ``per_subproblem_refine_layer_cap``).
    For each child batch, the split-hardened bounds are
    re-tightened by a K-lane backward pass over the unstable-neuron union only
    (stable phases are exact, so refining them gains nothing), so splits
    propagate relationally downstream instead of only through interval
    refresh."""

    per_subproblem_refine_iters: int = 0
    """Adam iterations for per-subproblem refine rows (0 = single fixed-slope
    backward, cheapest)."""

    per_subproblem_refine_rows_cap: int = 64
    """Max refined neurons per layer per batch (top-cap by interval width);
    bounds the K x 2*cap backward cost."""

    per_subproblem_refine_layer_cap: int = 2
    """Maximum downstream ReLU layers selected by ``split_successors``."""

    auto_batch_safety: float = 0.55
    """Fraction of GPU memory the auto batch sizer (max_batch_size='auto') may
    target; lowered on a shared GPU. The sizer also never exceeds 90% of the
    currently-reclaimable memory (free + this process's reserved cache)."""

    auto_batch_cap: int = 2048
    """Hard upper bound on the auto-sized batch (also the CPU fallback)."""

    auto_batch_floor: int = 8
    """Lower bound on the auto-sized batch."""

    multi_split_levels: int = 1
    """Simultaneous neuron splits per branching step (gain branching only).
    Each lane splits its top-k scored neurons jointly into all 2^k sign
    combinations. Joint splits are super-additive: the bound gain of
    constraining k neurons together exceeds the sum of the k individual
    split gains, because the split multipliers are optimized jointly
    against all constraints.     1 = single-split behavior."""

    joint_gain_groups: int = 1
    """Number of alternative joint ReLU groups measured before a gain split.

    ``1`` preserves the score-only top-k baseline.  Values above one add
    layer-diverse groups from the same finite BaBSR pool and select the group
    whose complete ``2^k`` child partition has the best measured worst lower
    bound.  The measurement is heuristic only; every selected group is still
    expanded into all phase combinations."""

    property_branch_focus: str = "sum"
    """How gain branching aggregates dual sensitivities across property rows.

    ``"sum"`` preserves the baseline sum of absolute sensitivities.
    ``"worst"`` uses only the currently smallest-slack (least certified)
    property row in each BaB lane to propose split neurons.  Every property
    row remains present in the bound solve and verdict."""

    property_separable_bab: bool = False
    """Prove every unresolved conjunct in its own complete BaB tree.

    This is valid only for ALL-rows output semantics.  Each tree starts from
    the full input region and carries one immutable original ASSERT-row id;
    the overall property is certified only after every tree is exhausted.
    The default keeps the ordinary shared multi-row tree."""

    branch_requires_unstable_successor: bool = False
    """Restrict split proposals to ReLUs with an unstable ReLU descendant.

    This is a heuristic-only long-horizon filter: terminal activation splits
    cannot propagate phase information into another relaxation.  If the
    filter would remove every candidate, branching fails safe to the original
    unfiltered set."""

    frontier_contraction_target: float = 0.0
    """Optional survivor-aware cap on joint split depth.

    ``0`` disables the policy.  A value in ``(0, 1]`` chooses the largest
    split depth ``k`` whose observed wave survivor rate ``r`` satisfies
    ``2**k * r <= target``.  Depth one is the fail-safe minimum when no
    available depth can contract the frontier.  The root keeps the ordinary
    batch-fill depth because it has no child-survival observation yet."""

    llm_probe_enabled: bool = False
    llm_probe_backend: str = "mock"
    llm_probe_model: str = ""
    llm_probe_base_url: str = ""
    llm_probe_api_key_env: str = ""
    llm_probe_temperature: float = 0.0
    llm_probe_timeout: float = 30.0
    llm_probe_max_candidates: int = 8
    llm_probe_max_candidates_total: int = 1024
    llm_probe_neuron_topk: int = 512
    llm_probe_cadence: int = 1
    llm_probe_history: int = 8
    llm_probe_max_failures: int = 3
    llm_probe_decisions: str = "split,frontier,refine"
    """Comma-separated decision types the LLM may steer: 'split' (joint neuron
    split depth), 'frontier' (wave width), 'refine' (per-subproblem refinement),
    'neuron' (joint neuron-group selection), 'input_split' (which input
    dimension to bisect and its fanout, input-domain-splitting BaB only)."""
    llm_probe_log: bool = False

    verbose: bool = False

    method: Optional[str] = None
    baf: bool = True
    alpha_mode: str = "fixed"
    p: float = 2.0
    perturbed_words: int = 1
    eps: float = 1e-5
    max_eps: float = 0.01
    num_verify_iters: int = 5
    k: int = 1
    alpha_opt_steps: int = 1000

    def __post_init__(self) -> None:
        if self.solver_tier not in VALID_SOLVER_TIERS:
            raise ValueError(
                f"Invalid solver_tier {self.solver_tier!r}; expected {VALID_SOLVER_TIERS}"
            )
        if self.method is not None:
            selection = select_bert_method(self.method)
            self.method = selection.method
            self.baf = selection.baf
            self.alpha_mode = selection.alpha_mode
            if self.solver_tier == "lp":
                self.solver_tier = selection.solver_tier
        if self.perturbed_words not in (1, 2):
            raise ValueError("perturbed_words must be 1 or 2")
        if self.num_verify_iters < 0:
            raise ValueError("num_verify_iters must be non-negative")
        if self.max_eps < 0 or self.eps < 0:
            raise ValueError("eps and max_eps must be non-negative")
        if self.joint_gain_groups < 1:
            raise ValueError("joint_gain_groups must be positive")
        if self.property_branch_focus not in {"sum", "worst"}:
            raise ValueError(
                "property_branch_focus must be 'sum' or 'worst'"
            )
        if not 0.0 <= self.frontier_contraction_target <= 1.0:
            raise ValueError(
                "frontier_contraction_target must be in [0, 1]"
            )
        if self.per_subproblem_refine_layer_cap < 1:
            raise ValueError(
                "per_subproblem_refine_layer_cap must be positive"
            )

    @classmethod
    def from_yaml(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        **overrides,
    ) -> BaBConfig:
        """Load BaB settings from YAML with optional keyword overrides.

        Reads from ``backend.bab`` in the unified backend config, falling
        back to a top-level ``bab`` key for standalone BaB YAML files.
        """
        path = Path(config_path) if config_path else _DEFAULT_YAML

        if not path.exists():
            raise FileNotFoundError(
                f"Backend config not found: {path}\nExpected: act/back_end/config.yaml"
            )

        with open(path) as f:
            yaml_data = yaml.safe_load(f) or {}

        # Support both nested (backend.bab) and flat (bab) YAML layouts.
        backend_section = yaml_data.get("backend", {})
        yaml_config: dict[str, Any] = backend_section.get("bab", yaml_data.get("bab", {}))

        valid_keys = {fld.name for fld in fields(cls)}
        merged = {k: v for k, v in yaml_config.items() if k in valid_keys}
        merged.update({k: v for k, v in overrides.items() if k in valid_keys})

        return cls(**merged)

    def to_yaml(self, path: Union[str, Path]) -> Path:
        """Write BaB settings to a standalone YAML file (top-level ``bab`` key)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(
                {"bab": asdict(self)}, f, default_flow_style=False, sort_keys=False
            )

        return path


# ---------------------------------------------------------------------------
# GenerationConfig — network generation (net_factory) parameters
# ---------------------------------------------------------------------------

_DEFAULT_GEN_CONFIG = str(
    Path(__file__).parent / "examples" / "config_gen_act_net.yaml"
)


@dataclass
class GenerationConfig:
    """Configuration for network generation via ``NetFactory``.

    Controls the simple knobs (how many, where, seed, TF filtering).
    The architecture sampling DSL lives in a separate file referenced
    by ``gen_config_path``.
    """

    gen_config_path: str = _DEFAULT_GEN_CONFIG
    output_dir: str = "act/back_end/examples/nets"
    num_instances: int = 15
    base_seed: int = 42
    name_prefix: str = "cfg_seed"
    tf_targets: Optional[List[str]] = None
    registry_mode: str = "intersection"
    coverage_mode: str = "basic"
    coverage_max_attempts: int = 1000
    coverage_report: bool = True
    write_manifest: bool = True

    def __post_init__(self) -> None:
        if self.registry_mode not in _VALID_REGISTRY_MODES:
            raise ValueError(
                f"Invalid registry_mode {self.registry_mode!r}; "
                f"expected one of {_VALID_REGISTRY_MODES}"
            )
        if self.coverage_mode not in _VALID_COVERAGE_MODES:
            raise ValueError(
                f"Invalid coverage_mode {self.coverage_mode!r}; "
                f"expected one of {_VALID_COVERAGE_MODES}"
            )

# ---------------------------------------------------------------------------
# BackendConfig — unified back-end configuration
# ---------------------------------------------------------------------------


@dataclass
class HybridZConfig:
    """Configuration for the strict pure-HybridZ verifier path.

    These are generic formulation and resource knobs for the HybridZ backend.
    Benchmark-specific profiles and frozen result tables intentionally live
    outside the ACT package.
    """

    timeout: Optional[float] = None
    engine: str = "dense_hz_objbound"
    sigmoid_k: Optional[int] = None
    tanh_k: Optional[int] = None
    scurve_domain_cuts: Optional[bool] = None
    scurve_graph_cuts: Optional[bool] = None
    compressed_relu: Optional[bool] = None
    relu_valid_cuts: Optional[bool] = None
    cell_budget: Optional[int] = None
    operator_exact_budget: int = 0
    operator_phase_projection_time_limit: float = 0.0
    operator_phase_clique_time_limit: float = 0.0
    operator_materialize_add: bool = True
    preactivation_lp_budget: int = 0
    preactivation_lp_time_limit: float = 0.0
    property_correlation_budget: int = 0
    property_correlation_time_limit: float = 0.0
    residual_phase_screen: bool = False
    residual_bound_screen: bool = False
    property_residual_budget: int = 0
    property_residual_time_limit: float = 0.0
    property_residual_max_adjoint_cells: int = 30_000_000
    property_residual_pool_per_rival: int = 8
    property_tail_upper: bool = False
    property_micro_rlt_product_cap: int = 0
    property_micro_rlt_packet_mode: str = "both"
    property_micro_rlt_parent_prefilter_seconds: float = 0.0
    property_micro_rlt_parent_only_diagnostic: bool = False
    property_tail_add_source_planes: bool = False
    property_tail_alpha_steps: int = 0
    property_tail_alpha_time_limit: float = 0.0
    property_tail_alpha_learning_rate: float = 0.08
    property_tail_alpha_max_cells: int = 50_000_000
    property_tail_alpha_device: str = "auto"
    property_tail_mixture_grid_bits: int = 0
    property_tail_pairhull_budget: int = 0
    property_tail_pairhull_time_limit: float = 0.0
    property_tail_suffix_blocks: int = 0
    property_tail_suffix_alpha_steps: int = 0
    property_tail_suffix_alpha_time_limit: float = 0.0
    property_tail_suffix_alpha_device: str = "auto"
    query_dual_feedback_targets: tuple[int, ...] = ()
    query_dual_feedback_steps: int = 0
    query_dual_feedback_time_limit: float = 0.0
    query_dual_feedback_block_size: int = 1024
    query_dual_feedback_device: str = "cuda"
    gpu_dual_steps: int = 0
    gpu_dual_time_limit: float = 0.0
    gpu_dual_row_topk: int = 0
    gpu_dual_learning_rate: float = 0.08
    lp_prefilter_fraction: float = 0.20
    lp_prefilter_max_seconds: float = 8.0

    def __post_init__(self) -> None:
        valid_engines = {
            "dense_hz_objbound",
            "sparse_hz_objbound",
            "operator_hz_objbound",
        }
        if self.engine not in valid_engines:
            raise ValueError(
                f"Invalid HybridZ engine {self.engine!r}; "
                f"expected one of {sorted(valid_engines)}"
            )
        if (
            isinstance(self.operator_exact_budget, bool)
            or not isinstance(self.operator_exact_budget, Integral)
        ):
            raise ValueError(
                "operator_exact_budget must be an integer"
            )
        self.operator_exact_budget = int(self.operator_exact_budget)
        if self.operator_exact_budget < -1:
            raise ValueError(
                "operator_exact_budget must be -1, 0, or a positive integer"
            )
        if (
            isinstance(self.operator_phase_projection_time_limit, bool)
            or not isinstance(
                self.operator_phase_projection_time_limit, Real
            )
        ):
            raise ValueError(
                "operator_phase_projection_time_limit must be numeric"
            )
        phase_projection_seconds = float(
            self.operator_phase_projection_time_limit
        )
        if (
            not math.isfinite(phase_projection_seconds)
            or not 0.0 <= phase_projection_seconds <= 30.0
        ):
            raise ValueError(
                "operator_phase_projection_time_limit must be finite and "
                "lie in [0, 30]"
            )
        self.operator_phase_projection_time_limit = (
            phase_projection_seconds
        )
        if phase_projection_seconds > 0.0 and (
            self.engine != "operator_hz_objbound"
            or self.operator_exact_budget != -1
        ):
            raise ValueError(
                "operator phase projection requires operator_hz_objbound "
                "and operator_exact_budget=-1"
            )
        if (
            isinstance(self.operator_phase_clique_time_limit, bool)
            or not isinstance(
                self.operator_phase_clique_time_limit, Real
            )
        ):
            raise ValueError(
                "operator_phase_clique_time_limit must be numeric"
            )
        phase_clique_seconds = float(
            self.operator_phase_clique_time_limit
        )
        if (
            not math.isfinite(phase_clique_seconds)
            or not 0.0 <= phase_clique_seconds <= 40.0
        ):
            raise ValueError(
                "operator_phase_clique_time_limit must be finite and lie "
                "in [0, 40]"
            )
        if phase_clique_seconds == 0.0:
            phase_clique_seconds = 0.0
        self.operator_phase_clique_time_limit = phase_clique_seconds
        if int(self.preactivation_lp_budget) < 0:
            raise ValueError("preactivation_lp_budget must be nonnegative")
        preactivation_seconds = float(self.preactivation_lp_time_limit)
        if (
            not math.isfinite(preactivation_seconds)
            or preactivation_seconds < 0.0
        ):
            raise ValueError(
                "preactivation_lp_time_limit must be finite and nonnegative"
            )
        if int(self.property_correlation_budget) < 0:
            raise ValueError(
                "property_correlation_budget must be nonnegative"
            )
        correlation_seconds = float(self.property_correlation_time_limit)
        if (
            not math.isfinite(correlation_seconds)
            or correlation_seconds < 0.0
        ):
            raise ValueError(
                "property_correlation_time_limit must be finite and "
                "nonnegative"
            )
        if (int(self.property_correlation_budget) > 0) != (
            correlation_seconds > 0.0
        ):
            raise ValueError(
                "property correlation budget and time limit must be "
                "enabled together"
            )
        if (
            int(self.property_correlation_budget) > 0
            and not bool(self.operator_materialize_add)
        ):
            raise ValueError(
                "property correlation shadows require "
                "operator_materialize_add=true"
            )
        if not isinstance(self.residual_phase_screen, bool):
            raise ValueError("residual_phase_screen must be a boolean")
        if (
            self.residual_phase_screen
            and not bool(self.operator_materialize_add)
        ):
            raise ValueError(
                "residual_phase_screen requires "
                "operator_materialize_add=true"
            )
        if not isinstance(self.residual_bound_screen, bool):
            raise ValueError("residual_bound_screen must be a boolean")
        if (
            self.residual_bound_screen
            and not bool(self.operator_materialize_add)
        ):
            raise ValueError(
                "residual_bound_screen requires "
                "operator_materialize_add=true"
            )
        if self.residual_phase_screen and self.residual_bound_screen:
            raise ValueError(
                "residual phase-only and bound screens are mutually "
                "exclusive modes"
            )
        if (
            isinstance(self.property_residual_budget, bool)
            or not isinstance(self.property_residual_budget, Integral)
        ):
            raise ValueError(
                "property_residual_budget must be an integer"
            )
        self.property_residual_budget = int(
            self.property_residual_budget
        )
        if self.property_residual_budget < 0:
            raise ValueError("property_residual_budget must be nonnegative")
        if (
            isinstance(self.property_residual_time_limit, bool)
            or not isinstance(self.property_residual_time_limit, Real)
        ):
            raise ValueError(
                "property_residual_time_limit must be numeric"
            )
        residual_seconds = float(self.property_residual_time_limit)
        if not math.isfinite(residual_seconds) or residual_seconds < 0.0:
            raise ValueError(
                "property_residual_time_limit must be finite and nonnegative"
            )
        self.property_residual_time_limit = residual_seconds
        if int(self.property_residual_max_adjoint_cells) <= 0:
            raise ValueError(
                "property_residual_max_adjoint_cells must be positive"
            )
        if int(self.property_residual_pool_per_rival) <= 0:
            raise ValueError(
                "property_residual_pool_per_rival must be positive"
            )
        phase_split_mode = bool(
            self.property_tail_upper
            and int(self.operator_exact_budget) > 0
            and int(self.property_residual_budget) > 0
        )
        phase_clique_enabled = phase_clique_seconds > 0.0
        micro_rlt_cap = self.property_micro_rlt_product_cap
        if isinstance(micro_rlt_cap, bool) or not isinstance(
            micro_rlt_cap, int
        ):
            raise ValueError(
                "property_micro_rlt_product_cap must be an integer"
            )
        if not 0 <= micro_rlt_cap <= 4096:
            raise ValueError(
                "property_micro_rlt_product_cap must lie in [0, 4096]"
            )
        micro_rlt_packet_mode = self.property_micro_rlt_packet_mode
        if (
            not isinstance(micro_rlt_packet_mode, str)
            or micro_rlt_packet_mode
            not in {"both", "first", "second"}
        ):
            raise ValueError(
                "property_micro_rlt_packet_mode must be one of "
                "both|first|second"
            )
        if micro_rlt_cap <= 0 and micro_rlt_packet_mode != "both":
            raise ValueError(
                "property_micro_rlt_packet_mode first/second requires "
                "property micro-RLT to be enabled"
            )
        raw_micro_rlt_seconds = (
            self.property_micro_rlt_parent_prefilter_seconds
        )
        if isinstance(raw_micro_rlt_seconds, bool) or not isinstance(
            raw_micro_rlt_seconds, (int, float)
        ):
            raise ValueError(
                "property_micro_rlt_parent_prefilter_seconds must be numeric"
            )
        micro_rlt_seconds = float(raw_micro_rlt_seconds)
        if (
            not math.isfinite(micro_rlt_seconds)
            or not 0.0 <= micro_rlt_seconds <= 10.0
        ):
            raise ValueError(
                "property_micro_rlt_parent_prefilter_seconds must be finite "
                "and lie in [0, 10]"
            )
        self.property_micro_rlt_parent_prefilter_seconds = (
            micro_rlt_seconds
        )
        if not isinstance(
            self.property_micro_rlt_parent_only_diagnostic, bool
        ):
            raise ValueError(
                "property_micro_rlt_parent_only_diagnostic must be a boolean"
            )
        if (micro_rlt_cap > 0) != (micro_rlt_seconds > 0.0):
            raise ValueError(
                "property micro-RLT product cap and parent prefilter time "
                "must be enabled together"
            )
        if (
            self.property_micro_rlt_parent_only_diagnostic
            and micro_rlt_cap <= 0
        ):
            raise ValueError(
                "property_micro_rlt_parent_only_diagnostic requires "
                "property micro-RLT to be enabled"
            )
        if micro_rlt_cap > 0:
            if self.engine != "operator_hz_objbound":
                raise ValueError(
                    "property micro-RLT requires "
                    "engine=operator_hz_objbound"
                )
            if self.property_tail_upper is not True:
                raise ValueError(
                    "property micro-RLT requires property_tail_upper=true"
                )
            if (
                not phase_split_mode
                or int(self.operator_exact_budget) != 2
                or int(self.property_residual_budget) != 2
            ):
                raise ValueError(
                    "property micro-RLT requires the depth-2 property-tail "
                    "phase split with operator_exact_budget="
                    "property_residual_budget=2"
                )
        if phase_split_mode:
            if not 1 <= int(self.operator_exact_budget) <= 2:
                raise ValueError(
                    "property-tail exact phase cover supports depth 1 or 2"
                )
            if int(self.property_residual_budget) != int(
                self.operator_exact_budget
            ):
                raise ValueError(
                    "property-tail exact phase cover requires "
                    "property_residual_budget=operator_exact_budget; the "
                    "residual selector is used only to choose split ReLUs"
                )
            if residual_seconds <= 0.0:
                raise ValueError(
                    "property-tail exact phase cover requires "
                    "property_residual_time_limit>0"
                )
        elif bool(self.property_tail_upper) and int(
            self.property_residual_budget
        ) > 0:
            raise ValueError(
                "property_tail_upper and property_residual_budget are "
                "mutually exclusive candidates"
            )
        if int(self.property_correlation_budget) > 0 and (
            int(self.property_residual_budget) > 0
            or bool(self.property_tail_upper)
        ):
            raise ValueError(
                "property correlation, residual normal form, and property "
                "tail are isolated candidate families"
            )
        if phase_clique_enabled:
            if self.engine != "operator_hz_objbound":
                raise ValueError(
                    "operator phase cliques require "
                    "engine=operator_hz_objbound"
                )
            if self.operator_materialize_add is not True:
                raise ValueError(
                    "operator phase cliques require "
                    "operator_materialize_add=true"
                )
            if int(self.operator_exact_budget) != 4:
                raise ValueError(
                    "operator phase cliques require "
                    "operator_exact_budget=4"
                )
            if int(self.property_residual_budget) != 4:
                raise ValueError(
                    "operator phase cliques require "
                    "property_residual_budget=4"
                )
            if residual_seconds <= 0.0:
                raise ValueError(
                    "operator phase cliques require "
                    "property_residual_time_limit>0"
                )
            if bool(self.property_tail_upper):
                raise ValueError(
                    "operator phase cliques require "
                    "property_tail_upper=false"
                )
            if int(self.property_correlation_budget) != 0:
                raise ValueError(
                    "operator phase cliques require "
                    "property_correlation_budget=0"
                )
            if self.residual_phase_screen or self.residual_bound_screen:
                raise ValueError(
                    "operator phase cliques require residual screens off"
                )
            if int(self.preactivation_lp_budget) != 0:
                raise ValueError(
                    "operator phase cliques require "
                    "preactivation_lp_budget=0"
                )
            if preactivation_seconds != 0.0:
                raise ValueError(
                    "operator phase cliques require "
                    "preactivation_lp_time_limit=0"
                )
            if micro_rlt_cap != 0:
                raise ValueError(
                    "operator phase cliques require "
                    "property_micro_rlt_product_cap=0"
                )
        if not isinstance(self.property_tail_add_source_planes, bool):
            raise ValueError(
                "property_tail_add_source_planes must be a boolean"
            )
        if (
            self.property_tail_add_source_planes
            and self.property_tail_upper is not True
        ):
            raise ValueError(
                "property_tail_add_source_planes requires "
                "property_tail_upper=true"
            )
        if (
            self.property_tail_add_source_planes
            and self.operator_materialize_add is not True
        ):
            raise ValueError(
                "property_tail_add_source_planes requires "
                "operator_materialize_add=true"
            )
        if int(self.property_tail_alpha_steps) < 0:
            raise ValueError(
                "property_tail_alpha_steps must be nonnegative"
            )
        property_tail_alpha_seconds = float(
            self.property_tail_alpha_time_limit
        )
        if (
            not math.isfinite(property_tail_alpha_seconds)
            or property_tail_alpha_seconds < 0.0
        ):
            raise ValueError(
                "property_tail_alpha_time_limit must be finite and "
                "nonnegative"
            )
        if (
            int(self.property_tail_alpha_steps) > 0
        ) != (property_tail_alpha_seconds > 0.0):
            raise ValueError(
                "property-tail alpha steps and time limit must be enabled "
                "together"
            )
        property_tail_alpha_lr = float(
            self.property_tail_alpha_learning_rate
        )
        if (
            not math.isfinite(property_tail_alpha_lr)
            or property_tail_alpha_lr <= 0.0
        ):
            raise ValueError(
                "property_tail_alpha_learning_rate must be finite and "
                "positive"
            )
        if int(self.property_tail_alpha_max_cells) <= 0:
            raise ValueError(
                "property_tail_alpha_max_cells must be positive"
            )
        property_tail_alpha_device = str(
            self.property_tail_alpha_device
        ).lower()
        if property_tail_alpha_device not in {"auto", "cpu", "cuda"}:
            raise ValueError(
                "property_tail_alpha_device must be auto, cpu, or cuda"
            )
        self.property_tail_alpha_device = property_tail_alpha_device
        if (
            int(self.property_tail_alpha_steps) > 0
            and not bool(self.property_tail_upper)
        ):
            raise ValueError(
                "property-tail alpha candidates require "
                "property_tail_upper"
            )
        if (
            int(self.property_tail_alpha_steps) > 0
            and int(self.operator_exact_budget) != 0
        ):
            raise ValueError(
                "property-tail alpha candidates currently require "
                "operator_exact_budget=0"
            )
        mixture_grid_bits = self.property_tail_mixture_grid_bits
        if isinstance(mixture_grid_bits, bool) or not isinstance(
            mixture_grid_bits, int
        ):
            raise ValueError(
                "property_tail_mixture_grid_bits must be an integer"
            )
        if not 0 <= mixture_grid_bits <= 24:
            raise ValueError(
                "property_tail_mixture_grid_bits must lie in [0, 24]"
            )
        if (
            mixture_grid_bits > 0
            and self.property_tail_upper is not True
        ):
            raise ValueError(
                "property_tail_mixture_grid_bits>0 requires "
                "property_tail_upper=true"
            )
        if (
            mixture_grid_bits > 0
            and int(self.property_tail_alpha_steps) <= 0
        ):
            raise ValueError(
                "property_tail_mixture_grid_bits>0 requires "
                "property_tail_alpha_steps>0"
            )
        if (
            mixture_grid_bits > 0
            and property_tail_alpha_seconds <= 0.0
        ):
            raise ValueError(
                "property_tail_mixture_grid_bits>0 requires "
                "property_tail_alpha_time_limit>0"
            )
        if (
            mixture_grid_bits > 0
            and int(self.operator_exact_budget) != 0
        ):
            raise ValueError(
                "property_tail_mixture_grid_bits>0 requires "
                "operator_exact_budget=0"
            )
        pairhull_budget = self.property_tail_pairhull_budget
        if isinstance(pairhull_budget, bool) or not isinstance(
            pairhull_budget, int
        ):
            raise ValueError(
                "property_tail_pairhull_budget must be an integer"
            )
        if not 0 <= pairhull_budget <= 8:
            raise ValueError(
                "property_tail_pairhull_budget must lie in [0, 8]"
            )
        raw_pairhull_seconds = self.property_tail_pairhull_time_limit
        if isinstance(raw_pairhull_seconds, bool) or not isinstance(
            raw_pairhull_seconds, (int, float)
        ):
            raise ValueError(
                "property_tail_pairhull_time_limit must be numeric"
            )
        pairhull_seconds = float(raw_pairhull_seconds)
        if (
            not math.isfinite(pairhull_seconds)
            or not 0.0 <= pairhull_seconds <= 1.5
        ):
            raise ValueError(
                "property_tail_pairhull_time_limit must be finite and "
                "lie in [0, 1.5]"
            )
        self.property_tail_pairhull_time_limit = pairhull_seconds
        if (pairhull_budget > 0) != (pairhull_seconds > 0.0):
            raise ValueError(
                "property-tail PairHull budget and time limit must be "
                "enabled together"
            )
        if pairhull_budget > 0 and self.property_tail_upper is not True:
            raise ValueError(
                "property_tail_pairhull_budget>0 requires "
                "property_tail_upper=true"
            )
        if (
            pairhull_budget > 0
            and int(self.operator_exact_budget) != 0
        ):
            raise ValueError(
                "property_tail_pairhull_budget>0 requires "
                "operator_exact_budget=0"
            )
        suffix_blocks = self.property_tail_suffix_blocks
        if isinstance(suffix_blocks, bool) or not isinstance(
            suffix_blocks, int
        ):
            raise ValueError(
                "property_tail_suffix_blocks must be an integer"
            )
        if not 0 <= suffix_blocks <= 8:
            raise ValueError(
                "property_tail_suffix_blocks must lie in [0, 8]"
            )
        if suffix_blocks > 0 and self.property_tail_upper is not True:
            raise ValueError(
                "property_tail_suffix_blocks>0 requires "
                "property_tail_upper=true"
            )
        if suffix_blocks > 0 and self.operator_materialize_add is not True:
            raise ValueError(
                "property_tail_suffix_blocks>0 requires "
                "operator_materialize_add=true"
            )
        if phase_split_mode and not 1 <= suffix_blocks <= 7:
            raise ValueError(
                "property-tail exact phase cover requires a shared-suffix "
                "constraint prefix with property_tail_suffix_blocks in [1, 7]"
            )
        suffix_alpha_steps = self.property_tail_suffix_alpha_steps
        if isinstance(suffix_alpha_steps, bool) or not isinstance(
            suffix_alpha_steps, int
        ):
            raise ValueError(
                "property_tail_suffix_alpha_steps must be an integer"
            )
        if not 0 <= suffix_alpha_steps <= 64:
            raise ValueError(
                "property_tail_suffix_alpha_steps must lie in [0, 64]"
            )
        suffix_alpha_seconds = float(
            self.property_tail_suffix_alpha_time_limit
        )
        if (
            not math.isfinite(suffix_alpha_seconds)
            or not 0.0 <= suffix_alpha_seconds <= 20.0
        ):
            raise ValueError(
                "property_tail_suffix_alpha_time_limit must be finite and "
                "lie in [0, 20]"
            )
        if (suffix_alpha_steps > 0) != (suffix_alpha_seconds > 0.0):
            raise ValueError(
                "property-tail suffix alpha steps and time limit must be "
                "enabled together"
            )
        if suffix_alpha_steps > 0 and suffix_blocks <= 0:
            raise ValueError(
                "property_tail_suffix_alpha_steps>0 requires "
                "property_tail_suffix_blocks>0"
            )
        suffix_alpha_device = str(
            self.property_tail_suffix_alpha_device
        ).lower()
        if suffix_alpha_device not in {"auto", "cpu", "cuda"}:
            raise ValueError(
                "property_tail_suffix_alpha_device must be auto, cpu, or cuda"
            )
        self.property_tail_suffix_alpha_device = suffix_alpha_device
        self.query_dual_feedback_targets = (
            normalize_query_dual_feedback_targets(
                self.query_dual_feedback_targets
            )
        )
        query_steps = self.query_dual_feedback_steps
        if isinstance(query_steps, bool) or not isinstance(query_steps, int):
            raise ValueError(
                "query_dual_feedback_steps must be an integer"
            )
        if not 0 <= query_steps <= 64:
            raise ValueError(
                "query_dual_feedback_steps must lie in [0, 64]"
            )
        raw_query_seconds = self.query_dual_feedback_time_limit
        if isinstance(raw_query_seconds, bool) or not isinstance(
            raw_query_seconds, (int, float)
        ):
            raise ValueError(
                "query_dual_feedback_time_limit must be numeric"
            )
        query_seconds = float(raw_query_seconds)
        if (
            not math.isfinite(query_seconds)
            or not 0.0 <= query_seconds <= 20.0
        ):
            raise ValueError(
                "query_dual_feedback_time_limit must be finite and lie in "
                "[0, 20]"
            )
        self.query_dual_feedback_time_limit = query_seconds
        query_block_size = self.query_dual_feedback_block_size
        if (
            isinstance(query_block_size, bool)
            or not isinstance(query_block_size, int)
            or not 1 <= query_block_size <= 4096
        ):
            raise ValueError(
                "query_dual_feedback_block_size must be an integer in "
                "[1, 4096]"
            )
        query_device = str(self.query_dual_feedback_device).lower()
        if query_device not in {"cpu", "cuda"}:
            raise ValueError(
                "query_dual_feedback_device must be cpu or cuda"
            )
        self.query_dual_feedback_device = query_device
        if query_steps == 0:
            if query_seconds != 0.0:
                raise ValueError(
                    "disabled query-dual feedback requires "
                    "query_dual_feedback_time_limit=0"
                )
        else:
            property_only_bound_replay = bool(
                self.residual_bound_screen
                and not self.query_dual_feedback_targets
            )
            if (
                not self.query_dual_feedback_targets
                and not property_only_bound_replay
            ):
                raise ValueError(
                    "enabled query-dual feedback requires target ReLUs or "
                    "residual_bound_screen=true for property-only replay"
                )
            if query_seconds <= 0.0:
                raise ValueError(
                    "enabled query-dual feedback requires "
                    "query_dual_feedback_time_limit>0"
                )
            if (
                not property_only_bound_replay
                and self.property_tail_upper is not True
            ):
                raise ValueError(
                    "enabled query-dual feedback requires "
                    "property_tail_upper=true"
                )
            if (
                property_only_bound_replay
                and self.property_tail_upper is not False
            ):
                raise ValueError(
                    "property-only residual-bound query replay requires "
                    "property_tail_upper=false"
                )
            if self.engine != "operator_hz_objbound":
                raise ValueError(
                    "enabled query-dual feedback requires "
                    "engine=operator_hz_objbound"
                )
            if int(self.operator_exact_budget) != 0:
                raise ValueError(
                    "enabled query-dual feedback requires "
                    "operator_exact_budget=0"
                )
        if phase_clique_enabled and query_steps != 0:
            raise ValueError(
                "operator phase cliques require "
                "query_dual_feedback_steps=0"
            )
        if int(self.gpu_dual_steps) < 0:
            raise ValueError("gpu_dual_steps must be nonnegative")
        gpu_dual_seconds = float(self.gpu_dual_time_limit)
        if not math.isfinite(gpu_dual_seconds) or gpu_dual_seconds < 0.0:
            raise ValueError(
                "gpu_dual_time_limit must be finite and nonnegative"
            )
        if int(self.gpu_dual_row_topk) < 0:
            raise ValueError("gpu_dual_row_topk must be nonnegative")
        if phase_clique_enabled and (
            int(self.gpu_dual_steps) != 0
            or gpu_dual_seconds != 0.0
            or int(self.gpu_dual_row_topk) != 0
        ):
            raise ValueError(
                "operator phase cliques require GPU dual candidates off"
            )
        if phase_projection_seconds > 0.0 and any(
            (
                phase_clique_enabled,
                int(self.preactivation_lp_budget) != 0,
                float(self.preactivation_lp_time_limit) != 0.0,
                int(self.property_correlation_budget) != 0,
                float(self.property_correlation_time_limit) != 0.0,
                bool(self.residual_phase_screen),
                bool(self.residual_bound_screen),
                int(self.property_residual_budget) != 0,
                float(self.property_residual_time_limit) != 0.0,
                bool(self.property_tail_upper),
                int(self.property_micro_rlt_product_cap) != 0,
                query_steps != 0,
                query_seconds != 0.0,
                int(self.gpu_dual_steps) != 0,
                gpu_dual_seconds != 0.0,
                int(self.gpu_dual_row_topk) != 0,
            )
        ):
            raise ValueError(
                "operator phase projection is a single-path mode and "
                "cannot be combined with optional phase/property/dual "
                "enhancements"
            )
        gpu_dual_lr = float(self.gpu_dual_learning_rate)
        if not math.isfinite(gpu_dual_lr) or gpu_dual_lr <= 0.0:
            raise ValueError(
                "gpu_dual_learning_rate must be finite and positive"
            )
        fraction = float(self.lp_prefilter_fraction)
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("lp_prefilter_fraction must lie in [0, 1]")
        lp_seconds = float(self.lp_prefilter_max_seconds)
        if not math.isfinite(lp_seconds) or lp_seconds < 0.0:
            raise ValueError(
                "lp_prefilter_max_seconds must be finite and nonnegative"
            )

    def verdict_timeout(self, fallback_timeout: Optional[float] = None) -> float:
        """Resolve the HybridZ verdict wall time in seconds."""

        if self.timeout is not None:
            return float(self.timeout)
        if fallback_timeout is not None:
            return float(fallback_timeout)
        return 30.0


@dataclass
class BackendConfig:
    """Unified configuration for the ACT back-end.

    Covers runtime selectors (solver / device / dtype), verification timeout,
    and nested BaB settings.  The canonical source is ``act/back_end/config.yaml``;
    CLI flags and environment variables override it at load time.

    Construction::

        BackendConfig()                     # programmatic defaults
        BackendConfig.from_yaml()           # load from default YAML
        BackendConfig.from_yaml(path, **kw) # custom YAML + overrides
    """

    solver: str = "auto"
    device: str = "cpu"
    dtype: str = "float64"
    verbose: bool = False
    timeout: float = 300.0

    bab_enabled: bool = False
    bab: BaBConfig = field(default_factory=BaBConfig)

    # -- batched-API knobs (C11) --------------------------------------------
    lp_enabled: bool = True
    """Enable the LP-batched tier (tier 2) in the 3-tier cascade.

    Set to False to skip verify_lp_batched and fall through directly to BaB.
    Must be False when solver='gurobi' (Gurobi solve_batch is N=1 only;
    see commit af797ff / C6).
    """

    bab_max_batch_size: int = 8
    """Maximum K for BaB sub-problem batching (tier 3).

    BaB dispatches up to K sub-problems per solve_batch call.  Set to 1 to
    disable batching inside BaB (equivalent to the legacy sequential loop).
    Must be 1 when solver='gurobi' (same N=1 restriction as lp_enabled).
    """

    generation: GenerationConfig = field(default_factory=GenerationConfig)
    hybridz: HybridZConfig = field(default_factory=HybridZConfig)

    method: Optional[str] = None
    p: float = 2.0
    perturbed_words: int = 1
    eps: float = 1e-5
    max_eps: float = 0.01
    num_verify_iters: int = 5
    k: int = 1
    alpha_opt_steps: int = 1000

    # -- validation ---------------------------------------------------------

    def __post_init__(self) -> None:
        if self.solver not in _VALID_SOLVERS:
            raise ValueError(
                f"Invalid solver {self.solver!r}; expected one of {_VALID_SOLVERS}"
            )
        if self.device not in _VALID_DEVICES:
            raise ValueError(
                f"Invalid device {self.device!r}; expected one of {_VALID_DEVICES}"
            )
        if self.dtype not in _VALID_DTYPES:
            raise ValueError(
                f"Invalid dtype {self.dtype!r}; expected one of {_VALID_DTYPES}"
            )
        if self.method is not None:
            selection = select_bert_method(self.method)
            self.method = selection.method
            self.bab.method = selection.method
            self.bab.baf = selection.baf
            self.bab.alpha_mode = selection.alpha_mode
            self.bab.solver_tier = selection.solver_tier
            self.bab.p = float(self.p)
            self.bab.perturbed_words = int(self.perturbed_words)
            self.bab.eps = float(self.eps)
            self.bab.max_eps = float(self.max_eps)
            self.bab.num_verify_iters = int(self.num_verify_iters)
            self.bab.k = int(self.k)
            self.bab.alpha_opt_steps = int(self.alpha_opt_steps)
        # Gurobi solve_batch is restricted to N=1 (commit af797ff / C6).
        # Fail loud at config-load time rather than at the first batched call.
        if self.solver == "gurobi":
            if self.lp_enabled:
                raise ValueError(
                    "BackendConfig: solver='gurobi' is incompatible with "
                    "lp_enabled=True.  GurobiSolver.solve_batch raises for N>1 "
                    "(Gurobi does not expose a truly parallel multi-LP API for "
                    "varying constraint matrices; see commit af797ff).  "
                    "Either set lp_enabled=False or switch to solver='torchlp'."
                )
            if self.bab_max_batch_size > 1:
                raise ValueError(
                    f"BackendConfig: solver='gurobi' is incompatible with "
                    f"bab_max_batch_size={self.bab_max_batch_size} > 1.  "
                    f"GurobiSolver.solve_batch raises for N>1.  "
                    f"Either set bab_max_batch_size=1 or switch to solver='torchlp'."
                )

    # -- YAML I/O -----------------------------------------------------------

    @classmethod
    def from_yaml(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        **overrides,
    ) -> BackendConfig:
        """Load config from YAML with optional keyword overrides.

        YAML layout::

            backend:
              solver: "torchlp"
              ...
              bab:
                enabled: true
                ...
              generation:
                num_instances: 15
                ...

        Override naming:
          - ``bab_<field>`` → ``BaBConfig.<field>``
          - ``gen_<field>`` → ``GenerationConfig.<field>``
          - ``hybridz_<field>`` → ``HybridZConfig.<field>``
          - ``bab_enabled`` → top-level ``bab_enabled``
        """
        path = Path(config_path) if config_path else _DEFAULT_YAML
        if not path.exists():
            raise FileNotFoundError(f"Backend config not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        backend_raw: dict[str, Any] = raw.get("backend", {})
        bab_raw: dict[str, Any] = backend_raw.pop("bab", {})
        gen_raw: dict[str, Any] = backend_raw.pop("generation", {})
        hz_raw: dict[str, Any] = backend_raw.pop("hybridz", {})

        # Extract "enabled" from bab section → top-level bab_enabled
        bab_enabled = bab_raw.pop("enabled", None)

        # Route prefixed overrides to the right sub-config
        bab_fields = {fld.name for fld in fields(BaBConfig)}
        gen_fields = {fld.name for fld in fields(GenerationConfig)}
        hz_fields = {fld.name for fld in fields(HybridZConfig)}
        bab_overrides: dict[str, Any] = {}
        gen_overrides: dict[str, Any] = {}
        hz_overrides: dict[str, Any] = {}
        top_overrides: dict[str, Any] = {}
        for k, v in overrides.items():
            if k.startswith("bab_") and k[4:] in bab_fields:
                bab_overrides[k[4:]] = v
            elif k.startswith("gen_") and k[4:] in gen_fields:
                gen_overrides[k[4:]] = v
            elif k.startswith("hybridz_") and k[8:] in hz_fields:
                hz_overrides[k[8:]] = v
            else:
                top_overrides[k] = v

        # Build BaBConfig
        bab_merged = {k: v for k, v in bab_raw.items() if k in bab_fields}
        bab_merged.update(bab_overrides)
        bab_config = BaBConfig(**bab_merged)

        # Build GenerationConfig
        gen_merged = {k: v for k, v in gen_raw.items() if k in gen_fields}
        gen_merged.update(gen_overrides)
        gen_config = GenerationConfig(**gen_merged)
        
        hz_merged = {k: v for k, v in hz_raw.items() if k in hz_fields}
        hz_merged.update(hz_overrides)
        hz_config = HybridZConfig(**hz_merged)

        # Build top-level config
        top_fields = {fld.name for fld in fields(cls)} - {"bab", "generation", "hybridz"}
        top_merged: dict[str, Any] = {}
        for k, v in backend_raw.items():
            if k in top_fields:
                top_merged[k] = v

        if bab_enabled is not None:
            top_merged["bab_enabled"] = bab_enabled

        top_merged.update({k: v for k, v in top_overrides.items() if k in top_fields})

        return cls(bab=bab_config, generation=gen_config, hybridz=hz_config, **top_merged)

    def to_yaml(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        d = asdict(self)
        bab_d = d.pop("bab")
        gen_d = d.pop("generation")
        hz_d = d.pop("hybridz")
        # ``yaml.safe_load`` cannot read PyYAML's ``!!python/tuple`` tag.
        # Serialize the immutable in-memory target tuple as portable YAML.
        hz_d["query_dual_feedback_targets"] = list(
            hz_d["query_dual_feedback_targets"]
        )
        bab_enabled = d.pop("bab_enabled")
        bab_d["enabled"] = bab_enabled

        with open(path, "w") as f:
            yaml.dump(
                {"backend": {**d, "bab": bab_d, "generation": gen_d, "hybridz": hz_d}},
                f,
                default_flow_style=False,
                sort_keys=False,
            )
        return path


if __name__ == "__main__":
    import sys

    passed = 0
    failed = 0

    def _check(label: str, fn) -> None:  # pragma: no cover
        global passed, failed
        try:
            fn()
            print(f"  PASS  {label}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {label}: {exc}")
            failed += 1

    print("BackendConfig.__post_init__ rejection tests")

    def _t1():  # pragma: no cover
        try:
            BackendConfig(solver="gurobi", lp_enabled=True)
            raise AssertionError("expected ValueError not raised")
        except ValueError as e:
            assert "lp_enabled" in str(e), f"wrong message: {e}"

    def _t2():  # pragma: no cover
        try:
            BackendConfig(solver="gurobi", lp_enabled=False, bab_max_batch_size=2)
            raise AssertionError("expected ValueError not raised")
        except ValueError as e:
            assert "bab_max_batch_size" in str(e), f"wrong message: {e}"

    def _t3():  # pragma: no cover
        cfg = BackendConfig(solver="gurobi", lp_enabled=False, bab_max_batch_size=1)
        assert cfg.solver == "gurobi"
        assert not cfg.lp_enabled
        assert cfg.bab_max_batch_size == 1

    def _t4():  # pragma: no cover
        cfg = BackendConfig()
        assert cfg.lp_enabled is True
        assert cfg.bab_max_batch_size == 8

    _check("gurobi + lp_enabled=True raises ValueError", _t1)
    _check("gurobi + bab_max_batch_size=2 raises ValueError", _t2)
    _check("gurobi + lp_enabled=False + bab_max_batch_size=1 succeeds", _t3)
    _check("default config has lp_enabled=True, bab_max_batch_size=8", _t4)

    print(f"\n{passed}/{passed + failed} passed")
    sys.exit(0 if failed == 0 else 1)


def build_vnncomp_bab_config(
    config_label: str,
    *,
    llm_backend: str = "mock",
    llm_decisions: str = "split,frontier,refine,input_split",
    llm_timeout: float = 30.0,
    llm_model: str = "",
    llm_cadence: int = 1,
    llm_neuron_topk: int = 0,
    llm_log: bool = False,
    multi_split_levels: int = 4,
    max_depth: int = 1_000_000,
    max_nodes: int = 1_000_000_000,
    solver_tier: str = "dual_alpha_eta",
    dual_n_iters: int = 100,
) -> BaBConfig:
    """BaBConfig for real VNNLIB instances (the VNN-COMP runner profile):
    ``fsb``/``babsr`` keep single-neuron splits, ``gain``/``gain+llm`` use joint-split
    depth, and only ``gain+llm`` enables the LLM probe."""
    branching_method = config_label if config_label in ("fsb", "babsr") else "gain"
    common: dict[str, Any] = dict(
        solver_tier=solver_tier,
        branching_method=branching_method,
        bounding_method="topk",
        bounding_order="depth_lb",
        frontier_cap=25000,
        max_depth=max_depth,
        max_nodes=max_nodes,
        dual_n_iters=dual_n_iters,
        lr_alpha=0.25,
        lr_beta=0.1,
        lr_decay=0.98,
        incremental_start_enabled=True,
        per_class_alpha=True,
        reuse_root_bounds=True,
        intermediate_refine="all",
        presplit_levels=0,
        eta_only_children=False,
        multi_split_levels=1 if branching_method != "gain" else max(1, int(multi_split_levels)),
    )
    if config_label != "gain+llm":
        return BaBConfig(**common)
    cfg = BaBConfig(
        llm_probe_enabled=True,
        llm_probe_backend=llm_backend,
        llm_probe_decisions=llm_decisions,
        llm_probe_timeout=llm_timeout,
        llm_probe_cadence=llm_cadence,
        llm_probe_neuron_topk=llm_neuron_topk,
        llm_probe_log=llm_log,
        **common,
    )
    if llm_model:
        cfg.llm_probe_model = llm_model
    return cfg
