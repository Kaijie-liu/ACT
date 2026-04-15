#!/usr/bin/env python3
"""
Validation Core Module

Shared infrastructure for experiments (RQ1-RQ6).
Implements the correct validation flow:
  load network -> analyze -> mutate bounds -> detect via BBL/CBR -> localize

All RQ experiment scripts import from this module for consistency.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from act.back_end.analyze import analyze
from act.back_end.core import Bounds, Fact, ConSet
from act.back_end.transfer_functions import set_transfer_function_mode
from act.back_end.validation import (
    set_all_seeds,
    derive_seed,
    MutationType,
    MutationConfig,
    TFMutator,
)
from act.back_end.verifier import (
    find_entry_layer_id,
    gather_input_spec_layers,
    seed_from_input_specs,
)
from act.pipeline.verification.model_factory import ModelFactory
from act.pipeline.verification.per_neuron_bounds import (
    PerNeuronCheckConfig,
    compare_bounds_per_neuron,
    collect_concrete_activations,
)
from act.util.options import PerformanceOptions

logger = logging.getLogger(__name__)

# Disable debug TF logging for experiments (avoid I/O overhead)
PerformanceOptions.disable_all()


# ============================================================================
# Generated ModelFactory (adapts experiments to current ACT serializer)
# ============================================================================
#
# Background: the networks bundled under act/back_end/examples/nets/ are saved
# with format_version "1.0", but ACT's serializer currently requires "2.0".
# To avoid modifying any ACT code, experiments generate fresh networks via
# NetFactory (which writes 2.0 JSONs) and expose them through ModelFactory
# pointed at that temporary directory.
#
# The factory is cached per (seed, num_networks, tf_targets) so repeated calls
# within a run reuse the same generated set.

_generated_factory_cache: Dict[Tuple[int, int, Tuple[str, ...]], "ModelFactory"] = {}

# UCU_Aiware's experiments were run against a pre-generated set with this
# base_seed (recorded in cuc/back_end/examples/nets/_meta/manifest.json).
# We reuse the same seed so the ACT run covers a comparable network
# population, modulo backend differences in NetFactory itself.
UCU_NETFACTORY_BASE_SEED = 1015796661
UCU_NETFACTORY_NUM_INSTANCES = 100


def _write_override_gen_config(dest_path: "Path") -> "Path":
    """Write a NetFactory config that restores the operator mix UCU had.

    ACT's bundled ``config_gen_act_net.yaml`` disables several toggles
    with comments like "act2torch issues" and "cons_exportor doesn't
    support". We tested each toggle empirically (see investigation in
    commit history): the BBL/CBR path, which is what these experiments
    use, works fine for all of them. The disabled comments apply only
    to the MILP constraint-export path (``cons_exportor``), which the
    experiments do not invoke.

    Re-enabled here:
      - ``variant.residual`` for both mlp and cnn2d  -> ADD op coverage
      - ``use_unsqueeze_squeeze`` (mlp)              -> SQUEEZE, UNSQUEEZE
      - ``use_transpose`` (cnn2d)                    -> TRANSPOSE
      - ``use_bias_layer``, ``use_scale_layer``      -> BIAS, SCALE

    Left alone:
      - ``use_batchnorm`` -> BN is not registered in ACT's layer_schema
        REGISTRY, so generation would fail.
      - RESHAPE -> no toggle exposes it from the generator.
    """
    import copy
    from pathlib import Path as _Path
    import yaml as _yaml

    default_path = (
        _Path(__file__).resolve().parent.parent
        / "act" / "back_end" / "examples" / "config_gen_act_net.yaml"
    )
    with open(default_path, "r") as f:
        cfg = _yaml.safe_load(f) or {}
    cfg = copy.deepcopy(cfg)

    mlp = cfg.setdefault("families", {}).setdefault("mlp", {})
    cnn = cfg.setdefault("families", {}).setdefault("cnn2d", {})

    # Re-enable residual variants (matches UCU_Aiware's original weights).
    if "weighted" in mlp.get("variant", {}):
        mlp["variant"]["weighted"] = {
            "plain": 0.33, "block": 0.33, "residual": 0.34,
        }
    if "weighted" in cnn.get("variant", {}):
        cnn["variant"]["weighted"] = {
            "plain": 0.33, "residual": 0.34, "stage": 0.33,
        }

    # Re-enable layer toggles that expose extra operators. BBL/CBR
    # handles them fine; only MILP export (cons_exportor) doesn't.
    mlp["use_unsqueeze_squeeze"] = {"probability": 0.3}
    mlp["use_bias_layer"] = {"probability": 0.3}
    mlp["use_scale_layer"] = {"probability": 0.3}
    cnn["use_transpose"] = {"probability": 0.3}
    cnn["use_scale_layer"] = {"probability": 0.3}

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "w") as f:
        _yaml.safe_dump(cfg, f, sort_keys=False)
    return dest_path


def build_generated_factory(
    seed: int = UCU_NETFACTORY_BASE_SEED,
    num_networks: int = UCU_NETFACTORY_NUM_INSTANCES,
    tf_targets: Optional[List[str]] = None,
) -> "ModelFactory":
    """Build a ModelFactory backed by networks generated on-the-fly.

    Uses NetFactory to materialize ``num_networks`` networks into a temp
    directory, then returns a ModelFactory pointed at that directory.
    Results are cached per (seed, num_networks, tf_targets).

    Defaults mirror the UCU_Aiware paper's generation config (base_seed
    1015796661, 100 instances, residual variants enabled) so ACT results
    can be compared against the published baseline on a comparable
    network population.
    """
    import tempfile
    from pathlib import Path as _Path
    from act.back_end.net_factory import NetFactory

    if tf_targets is None:
        tf_targets = ["interval", "hybridz", "dual"]

    cache_key = (int(seed), int(num_networks), tuple(tf_targets))
    cached = _generated_factory_cache.get(cache_key)
    if cached is not None:
        return cached

    gen_dir = _Path(tempfile.mkdtemp(
        prefix=f"act_exp_seed{seed}_n{num_networks}_"
    ))
    gen_config = _write_override_gen_config(gen_dir / "_gen_config.yaml")

    nf = NetFactory(
        gen_config_path=str(gen_config),
        base_seed=int(seed),
        num_instances=int(num_networks),
        output_dir=str(gen_dir),
        tf_targets=list(tf_targets),
    )
    nf.generate()

    factory = ModelFactory(nets_dir=str(gen_dir))
    if not factory.list_networks():
        raise RuntimeError(
            f"NetFactory generated 0 loadable networks in {gen_dir}"
        )
    _generated_factory_cache[cache_key] = factory
    return factory

# Layer kinds that are hookable (have concrete activations via forward hooks)
HOOKABLE_KINDS = frozenset({
    "DENSE", "CONV1D", "CONV2D", "CONV3D",
    "RELU", "SIGMOID", "TANH", "SILU", "LRELU",
    "FLATTEN",
    "MAXPOOL1D", "MAXPOOL2D", "MAXPOOL3D",
    "AVGPOOL1D", "AVGPOOL2D", "AVGPOOL3D",
    "ADAPTIVEAVGPOOL1D", "ADAPTIVEAVGPOOL2D", "ADAPTIVEAVGPOOL3D",
})

# ============================================================================
# Result Data Classes
# ============================================================================

@dataclass
class FullDetectionResult:
    """Full result of a detection experiment."""
    network_name: str
    network_seed: int
    domain: str
    mutation: str
    target_layer_id: int
    # BBL
    bca_detected: bool
    bca_violations_total: int
    bca_top_violation_layers: List[int]
    bca_worst_gap: float
    bca_time_ms: float
    # CBR
    scc_detected: bool
    scc_num_samples: int
    scc_time_ms: float
    # Localization
    localized_top1: bool
    localized_top5: bool
    # Bounds info
    bound_width_avg: float
    # Error
    error: Optional[str] = None


# ============================================================================
# Core Functions
# ============================================================================

def load_network_pair(
    factory: ModelFactory,
    network_name: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Tuple[Any, nn.Module]:
    """Load ACT Net and corresponding PyTorch model."""
    act_net = factory.get_act_net(network_name)
    model = factory.create_model(network_name, load_weights=True)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return act_net, model


def get_clean_bounds(
    act_net,
    domain: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Tuple[Dict[int, Bounds], Fact]:
    """Run abstract analysis and return clean bounds per layer.

    Returns:
        (bounds_by_layer, entry_fact)
    """
    set_transfer_function_mode(domain)
    entry_id = find_entry_layer_id(act_net)

    spec_layers = gather_input_spec_layers(act_net)
    seed_bounds = seed_from_input_specs(spec_layers)
    seed_bounds = Bounds(
        lb=seed_bounds.lb.to(device=device, dtype=dtype),
        ub=seed_bounds.ub.to(device=device, dtype=dtype),
    )
    entry_fact = Fact(bounds=seed_bounds, cons=ConSet())

    _before, after, _globalC = analyze(act_net, entry_id, entry_fact)

    bounds_by_layer: Dict[int, Bounds] = {}
    for layer in act_net.layers:
        lid = layer.id
        if lid not in after:
            continue
        b = after[lid].bounds
        if torch.isfinite(b.lb).all() and torch.isfinite(b.ub).all():
            bounds_by_layer[lid] = Bounds(lb=b.lb.clone(), ub=b.ub.clone())

    return bounds_by_layer, entry_fact


def select_target_layer(
    act_net,
    bounds_by_layer: Dict[int, Bounds],
    target_index: int = 0,
) -> int:
    """Select a hookable computation layer for mutation.

    Returns:
        layer_id of the target layer

    Raises:
        ValueError: if no hookable layers with bounds exist
    """
    candidate_ids = [
        layer.id for layer in act_net.layers
        if layer.kind in HOOKABLE_KINDS and layer.id in bounds_by_layer
    ]
    if not candidate_ids:
        raise ValueError("No hookable layers with finite bounds found")
    return candidate_ids[target_index % len(candidate_ids)]


def run_bbl_detection(
    act_net,
    model: nn.Module,
    mutated_bounds: Dict[int, Bounds],
    input_tensor: torch.Tensor,
    config: PerNeuronCheckConfig = PerNeuronCheckConfig(atol=1e-6, rtol=0.0, topk=10),
) -> Dict[str, Any]:
    """BBL detection: compare concrete activations against mutated bounds.

    Returns dict with status, violations_total, violations_topk, layerwise_stats.
    """
    concrete_by_layer, errors, warnings, alignment_meta = collect_concrete_activations(
        act_net, model, input_tensor,
    )

    if errors:
        return {
            "status": "ERROR",
            "errors": errors,
            "violations_total": 0,
            "violations_topk": [],
            "layerwise_stats": [],
            "worst_gap": 0.0,
        }

    # Filter to layers present in BOTH mutated_bounds and concrete
    common_ids = set(mutated_bounds.keys()) & set(concrete_by_layer.keys())
    if not common_ids:
        return {
            "status": "ERROR",
            "errors": ["No overlapping layers between bounds and concrete activations"],
            "violations_total": 0,
            "violations_topk": [],
            "layerwise_stats": [],
            "worst_gap": 0.0,
        }

    bounds_for_compare = {lid: mutated_bounds[lid] for lid in common_ids}
    concrete_for_compare = {lid: concrete_by_layer[lid] for lid in common_ids}

    result = compare_bounds_per_neuron(
        bounds_by_layer=bounds_for_compare,
        concrete_by_layer=concrete_for_compare,
        layer_by_id=getattr(act_net, "by_id", {}),
        atol=config.atol,
        rtol=config.rtol,
        topk=config.topk,
        nan_policy=config.nan_policy,
    )

    # Add worst_gap
    worst_gap = 0.0
    for s in result.get("layerwise_stats", []):
        worst_gap = max(worst_gap, float(s.get("max_gap", 0.0)))
    result["worst_gap"] = worst_gap

    return result


def run_cbr_detection(
    model: nn.Module,
    input_bounds: Bounds,
    output_bounds: Bounds,
    num_samples: int = 20,
    seed: int = 42,
    input_shape: Optional[tuple] = None,
) -> Dict[str, Any]:
    """CBR detection: sample inputs and check if outputs violate mutated output bounds.

    Returns dict with discovered (bool), num_violations, num_samples.
    """
    set_all_seeds(seed)

    # Flatten bounds so the random sample broadcasts correctly regardless
    # of the bound tensor rank (MLPs are 2D, CNNs are 4D, etc.).
    lb_flat = input_bounds.lb.reshape(-1)
    ub_flat = input_bounds.ub.reshape(-1)
    dim = lb_flat.numel()
    num_violations = 0

    with torch.no_grad():
        for _ in range(num_samples):
            t = torch.rand(dim, device=lb_flat.device, dtype=lb_flat.dtype)
            x_flat = lb_flat + t * (ub_flat - lb_flat)

            if input_shape is not None:
                x = x_flat.reshape(input_shape)
            else:
                x = x_flat.unsqueeze(0)

            result = model(x)

            if isinstance(result, dict):
                output = result.get("output", result.get("logits", None))
                if output is None:
                    for v in result.values():
                        if isinstance(v, torch.Tensor):
                            output = v
                            break
            elif isinstance(result, torch.Tensor):
                output = result
            else:
                continue

            if output is None:
                continue

            output_flat = output.flatten()
            out_lb = output_bounds.lb
            out_ub = output_bounds.ub

            if output_flat.numel() != out_lb.numel():
                continue

            lb_viol = (output_flat < out_lb - 1e-6).any()
            ub_viol = (output_flat > out_ub + 1e-6).any()

            if lb_viol or ub_viol:
                num_violations += 1

    return {
        "discovered": num_violations > 0,
        "num_violations": num_violations,
        "num_samples": num_samples,
    }


def _get_input_shape(act_net) -> Optional[tuple]:
    """Extract input shape from network's INPUT layer.

    ACT stores shape in ``layer.params["shape"]``; UCU_Aiware used
    ``layer.meta["shape"]``. Prefer params (ACT's current convention)
    and fall back to meta for cross-version compatibility.
    """
    for layer in act_net.layers:
        if layer.kind == "INPUT":
            params = getattr(layer, "params", {}) or {}
            meta = getattr(layer, "meta", {}) or {}
            shape = params.get("shape") or meta.get("shape")
            if shape is not None:
                return tuple(shape)
    return None


def _get_last_computation_layer_id(act_net) -> Optional[int]:
    """Get the last computation layer ID (before ASSERT)."""
    comp_layers = [
        L for L in act_net.layers
        if L.kind not in ("INPUT", "INPUT_SPEC", "ASSERT")
    ]
    if comp_layers:
        return comp_layers[-1].id
    return None


def run_full_detection(
    factory: ModelFactory,
    network_name: str,
    domain: str,
    mutation_type: MutationType,
    net_seed: int,
    target_layer_index: int = 0,
    mutation_factor: float = 0.1,
    num_cbr_samples: int = 20,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> FullDetectionResult:
    """Complete detection pipeline: analyze -> mutate -> BBL + CBR -> localize.

    This is the main entry point for RQ1, RQ3, and RQ5 experiments.
    """
    try:
        # Step 1: Load
        act_net, model = load_network_pair(factory, network_name, device, dtype)

        # Step 2: Clean bounds
        bounds_by_layer, entry_fact = get_clean_bounds(act_net, domain, device, dtype)

        if not bounds_by_layer:
            return _error_result(network_name, net_seed, domain, mutation_type,
                                 "No finite bounds from analysis")

        # Step 3: Target layer
        target_layer_id = select_target_layer(act_net, bounds_by_layer, target_layer_index)

        # Step 4: Mutate
        mutation_config = MutationConfig(
            mutation_type=mutation_type,
            target_layer_id=target_layer_id,
            factor=mutation_factor,
            seed=derive_seed(net_seed, "mutation"),
        )
        mutated_bounds = TFMutator.mutate_layer_bounds(bounds_by_layer, mutation_config)

        # Step 5: BBL
        input_tensor = factory.generate_test_input(network_name, "random")
        input_tensor = input_tensor.to(device=device, dtype=dtype)

        bca_start = time.perf_counter()
        bbl_result = run_bbl_detection(act_net, model, mutated_bounds, input_tensor)
        bca_time_ms = (time.perf_counter() - bca_start) * 1000

        bca_detected = bbl_result.get("status") == "FAIL"
        bca_violations = bbl_result.get("violations_total", 0)
        bca_worst_gap = bbl_result.get("worst_gap", 0.0)

        # Top violation layers (unique, sorted by gap)
        top_violation_layers = []
        seen = set()
        for v in bbl_result.get("violations_topk", []):
            lid = v.get("layer_id")
            if lid is not None and lid not in seen:
                top_violation_layers.append(lid)
                seen.add(lid)

        # Step 6: CBR
        spec_layers = gather_input_spec_layers(act_net)
        input_bounds = seed_from_input_specs(spec_layers)
        input_bounds = Bounds(
            lb=input_bounds.lb.to(device=device, dtype=dtype),
            ub=input_bounds.ub.to(device=device, dtype=dtype),
        )

        output_layer_id = _get_last_computation_layer_id(act_net)
        output_mutated_bounds = mutated_bounds.get(
            output_layer_id, bounds_by_layer.get(output_layer_id)
        ) if output_layer_id is not None else None

        scc_start = time.perf_counter()
        if output_mutated_bounds is not None:
            input_shape = _get_input_shape(act_net)
            cbr_result = run_cbr_detection(
                model=model,
                input_bounds=input_bounds,
                output_bounds=output_mutated_bounds,
                num_samples=num_cbr_samples,
                seed=derive_seed(net_seed, "cbr"),
                input_shape=input_shape,
            )
            scc_detected = cbr_result["discovered"]
        else:
            scc_detected = False
            cbr_result = {"num_samples": 0}
        scc_time_ms = (time.perf_counter() - scc_start) * 1000

        # Step 7: Localization
        localized_top1 = target_layer_id in top_violation_layers[:1]
        localized_top5 = target_layer_id in top_violation_layers[:5]

        # Bound width (clean bounds, reflects domain precision)
        widths = []
        for b in bounds_by_layer.values():
            w = (b.ub - b.lb).mean().item()
            if not (w != w):  # NaN check
                widths.append(w)
        bound_width_avg = sum(widths) / len(widths) if widths else 0.0

        return FullDetectionResult(
            network_name=network_name,
            network_seed=net_seed,
            domain=domain,
            mutation=mutation_type.value,
            target_layer_id=target_layer_id,
            bca_detected=bca_detected,
            bca_violations_total=bca_violations,
            bca_top_violation_layers=top_violation_layers,
            bca_worst_gap=bca_worst_gap,
            bca_time_ms=bca_time_ms,
            scc_detected=scc_detected,
            scc_num_samples=cbr_result.get("num_samples", 0),
            scc_time_ms=scc_time_ms,
            localized_top1=localized_top1,
            localized_top5=localized_top5,
            bound_width_avg=bound_width_avg,
        )

    except Exception as e:
        logger.warning("Detection failed for %s: %s", network_name, e)
        return _error_result(network_name, net_seed, domain, mutation_type, str(e))


def _error_result(name, seed, domain, mutation_type, error_msg):
    return FullDetectionResult(
        network_name=name,
        network_seed=seed,
        domain=domain,
        mutation=mutation_type.value if isinstance(mutation_type, MutationType) else str(mutation_type),
        target_layer_id=-1,
        bca_detected=False,
        bca_violations_total=0,
        bca_top_violation_layers=[],
        bca_worst_gap=0.0,
        bca_time_ms=0.0,
        scc_detected=False,
        scc_num_samples=0,
        scc_time_ms=0.0,
        localized_top1=False,
        localized_top5=False,
        bound_width_avg=0.0,
        error=error_msg,
    )


# ============================================================================
# Architecture Filtering (RQ3)
# ============================================================================

def get_networks_for_architecture(
    factory: ModelFactory,
    architecture: str,
) -> List[str]:
    """Get pre-built network names matching the given architecture type."""
    arch_keywords = {
        "sequential_mlp": ["mlp"],
        "sequential_cnn": ["cnn2d", "cnn"],
        "residual": ["residual", "resnet"],
    }
    keywords = arch_keywords.get(architecture, [architecture])
    matching = []

    for name in factory.list_networks():
        name_lower = name.lower()
        if any(kw in name_lower for kw in keywords):
            matching.append(name)

    return matching


# ============================================================================
# Overhead Measurement (RQ6)
# ============================================================================

def run_overhead_measurement(
    factory: ModelFactory,
    network_name: str,
    net_seed: int,
    scc_budget: int = 20,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Dict[str, Any]:
    """Measure overhead of analysis, CBR, and BBL."""
    act_net, model = load_network_pair(factory, network_name, device, dtype)
    n_params = sum(p.numel() for p in model.parameters())

    # Measure analysis time
    t0 = time.perf_counter()
    bounds_by_layer, entry_fact = get_clean_bounds(act_net, "interval", device, dtype)
    analyze_time_ms = (time.perf_counter() - t0) * 1000

    # Measure CBR time
    spec_layers = gather_input_spec_layers(act_net)
    input_bounds = seed_from_input_specs(spec_layers)
    input_bounds = Bounds(
        lb=input_bounds.lb.to(device=device, dtype=dtype),
        ub=input_bounds.ub.to(device=device, dtype=dtype),
    )

    output_layer_id = _get_last_computation_layer_id(act_net)
    output_bounds = bounds_by_layer.get(output_layer_id) if output_layer_id else None

    input_shape = _get_input_shape(act_net)

    scc_start = time.perf_counter()
    if output_bounds:
        run_cbr_detection(model, input_bounds, output_bounds,
                          num_samples=scc_budget, seed=net_seed,
                          input_shape=input_shape)
    scc_time_ms = (time.perf_counter() - scc_start) * 1000

    # Measure BBL time
    input_tensor = factory.generate_test_input(network_name, "random")
    input_tensor = input_tensor.to(device=device, dtype=dtype)

    bca_start = time.perf_counter()
    run_bbl_detection(act_net, model, bounds_by_layer, input_tensor)
    bca_time_ms = (time.perf_counter() - bca_start) * 1000

    return {
        "network_name": network_name,
        "n_params": n_params,
        "analyze_time_ms": analyze_time_ms,
        "scc_time_ms": scc_time_ms,
        "bca_time_ms": bca_time_ms,
        "overhead_ratio": (scc_time_ms + bca_time_ms) / analyze_time_ms if analyze_time_ms > 0 else 0.0,
    }
