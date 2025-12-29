# act/pipeline/verification/validation/level2_bounds.py
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Tuple

from .level1_inputs import InputSamplingPlan, sample_concrete_inputs
from .results import hash_tensor


@dataclass
class BoundsViolation:
    tf_mode: str
    layer_id: int
    layer_kind: str
    sample_idx: int
    num_violations: int
    total_neurons: int
    concrete_min: float
    concrete_max: float
    abstract_lb_min: float
    abstract_ub_max: float
    lb_min: float
    ub_max: float
    first_bad_index: int | None = None
    first_bad_value: float | None = None
    first_bad_lb: float | None = None
    first_bad_ub: float | None = None
    topk: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Level2CaseResult:
    case_name: str
    status: str  # PASS | VIOLATION_FOUND | ERROR
    metadata: Dict[str, Any] = field(default_factory=dict)
    violations: List[BoundsViolation] = field(default_factory=list)


def _normalize_tf_mode(mode: str) -> str:
    return str(mode).strip().lower()


def _collect_concrete_activations_strict(model, act_net, x) -> Tuple[Dict[int, Any], Dict[str, Any]]:
    import torch  # lazy import

    hook_classes = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ReLU, torch.nn.Flatten)
    act_kind_map = {
        "DENSE": "Linear",
        "LINEAR": "Linear",
        "FC": "Linear",
        "CONV": "Conv2d",
        "CONV2D": "Conv2d",
        "RELU": "ReLU",
        "FLATTEN": "Flatten",
    }

    # Ordered ACT layers eligible for alignment
    act_layers = [L for L in getattr(act_net, "layers", []) if act_kind_map.get(getattr(L, "kind", "").upper())]
    activations: Dict[int, Any] = {}
    collected = []
    hooks = []

    def make_hook():
        def hook(module, _inp, out):
            tensor_out = out
            if isinstance(out, (list, tuple)) and out:
                tensor_out = out[0]
            if isinstance(tensor_out, torch.Tensor):
                collected.append(tensor_out.detach())
            else:
                raise RuntimeError(f"Unsupported activation type from module {type(module)}: {type(out)}")

        return hook

    try:
        for module in model.modules():
            if isinstance(module, hook_classes):
                hooks.append(module.register_forward_hook(make_hook()))

        model.eval()
        with torch.no_grad():
            _ = model(x)

        if len(collected) != len(act_layers):
            raise RuntimeError(
                f"AlignmentMismatch: hooked {len(collected)} outputs but found {len(act_layers)} ACT layers"
            )

        for idx, layer in enumerate(act_layers):
            activations[layer.id] = collected[idx]

        align_meta = {
            "num_hooked": len(collected),
            "num_act_layers": len(act_layers),
            "act_layer_ids": [L.id for L in act_layers],
        }
        return activations, align_meta
    finally:
        for h in hooks:
            h.remove()


def _plan_num_samples(plan) -> int:
    for attr in ("k_center", "k_random", "k_boundary"):
        if not hasattr(plan, attr):
            return None
    return int(plan.k_center + plan.k_random + plan.k_boundary)


def _plan_repr(plan) -> Dict[str, Any]:
    if plan is None:
        return {}
    if all(hasattr(plan, a) for a in ("k_center", "k_random", "k_boundary")):
        return {"k_center": plan.k_center, "k_random": plan.k_random, "k_boundary": plan.k_boundary}
    return {"repr": str(plan)}


def _topk_violations(a_flat, lb, ub, violation_mask, k: int = 5):
    import torch  # lazy import

    bad_idx = torch.nonzero(violation_mask, as_tuple=False).flatten()
    if bad_idx.numel() == 0:
        return []
    gap_low = lb - a_flat
    gap_high = a_flat - ub
    gap = torch.maximum(gap_low, gap_high)
    order = torch.argsort(gap[bad_idx], descending=True)
    top_idx = bad_idx[order][:k]
    top_list = []
    for i in top_idx:
        top_list.append(
            {
                "idx": int(i.item()),
                "value": float(a_flat[i].item()),
                "lb": float(lb[i].item()),
                "ub": float(ub[i].item()),
                "gap": float(gap[i].item()),
            }
        )
    return top_list


def _compute_interval_after_bounds(act_net, device, dtype):
    import torch  # lazy import

    from act.back_end.analyze import analyze, initialize_tf_mode
    from act.back_end.core import Bounds, ConSet, Fact
    from act.back_end.verifier import (
        find_entry_layer_id,
        gather_input_spec_layers,
        seed_from_input_specs,
    )

    spec_layers = gather_input_spec_layers(act_net)
    seed_bounds = seed_from_input_specs(spec_layers)
    seed_lb = seed_bounds.lb.to(device=device, dtype=dtype)
    seed_ub = seed_bounds.ub.to(device=device, dtype=dtype)
    entry_fact = Fact(bounds=Bounds(seed_lb, seed_ub), cons=ConSet())
    entry_id = find_entry_layer_id(act_net)

    initialize_tf_mode("interval")
    _before, after, _ = analyze(act_net, entry_id, entry_fact)
    bounds_map = {}
    for lid, fact in after.items():
        bounds_map[lid] = (fact.bounds.lb, fact.bounds.ub)
    return bounds_map


def run_level2_bounds_check(
    case,
    seed: int,
    plan: InputSamplingPlan,
    device: str,
    dtype,
    tf_modes: List[str] | Tuple[str, ...] = ("interval",),
    atol: float = 1e-9,
) -> Dict[str, Any]:
    import torch  # lazy import

    per_mode: List[Dict[str, Any]] = []
    overall_status = "PASS"
    requested_modes = list(tf_modes) if isinstance(tf_modes, (list, tuple)) else [tf_modes]

    for mode in requested_modes:
        tf_mode = _normalize_tf_mode(mode)
        case_meta = getattr(case, "metadata", {}) or {}
        metadata: Dict[str, Any] = {
            "tf_mode": tf_mode,
            "seed": seed,
            "supported_modes": ["interval"],
            "confignet_source": case_meta.get("source", "<unknown>"),
            "atol": float(atol),
            "device": str(device),
            "dtype": str(dtype),
        }
        violations: List[BoundsViolation] = []
        status = "PASS"

        if tf_mode != "interval":
            metadata["error_type"] = "UnsupportedTFMode"
            metadata["requested_mode"] = mode
            per_mode.append(asdict(Level2CaseResult(case_name=getattr(case, "name", "<unknown>"), status="ERROR", metadata=metadata)))
            overall_status = "ERROR"
            continue

        try:
            inputs = sample_concrete_inputs(
                case.act_net, seed=seed, plan=plan, device=device, dtype=dtype, require_satisfy=True
            )
            metadata["num_inputs"] = len(inputs)
            metadata["input_hashes"] = []
            metadata["input_ranges"] = []

            model = getattr(case, "torch_model", None)
            if model is None:
                raise ValueError("Case missing torch_model.")
            model = model.to(device=device, dtype=dtype)

            n_samples = _plan_num_samples(plan)
            metadata["num_samples"] = n_samples if n_samples is not None else metadata["num_inputs"]
            metadata["plan"] = _plan_repr(plan)

            bounds_map = _compute_interval_after_bounds(case.act_net, device=device, dtype=dtype)

            for sample_idx, x in enumerate(inputs):
                x_hash = hash_tensor(x)
                metadata["input_hashes"].append(x_hash)
                metadata["input_ranges"].append({"min": float(x.min().item()), "max": float(x.max().item())})

                activations, align_meta = _collect_concrete_activations_strict(model, case.act_net, x)
                metadata.setdefault("alignment", align_meta)

                for lid, act_tensor in activations.items():
                    if lid not in bounds_map:
                        continue
                    lb, ub = bounds_map[lid]
                    a_flat = act_tensor.flatten()
                    if a_flat.numel() != lb.numel() or a_flat.numel() != ub.numel():
                        raise RuntimeError(f"ShapeMismatch at layer {lid}: act {a_flat.shape} vs bounds {lb.shape}")

                    violation_mask = (a_flat < lb - atol) | (a_flat > ub + atol)
                    if torch.any(violation_mask):
                        bad_idx = torch.nonzero(violation_mask, as_tuple=False).flatten()
                        idx = int(bad_idx[0].item())
                        topk = _topk_violations(a_flat, lb, ub, violation_mask, k=5)
                        violation = BoundsViolation(
                            tf_mode=tf_mode,
                            layer_id=lid,
                            layer_kind=getattr(case.act_net.by_id.get(lid), "kind", "<unknown>") if hasattr(case, "act_net") else "<unknown>",
                            sample_idx=sample_idx,
                            num_violations=int(violation_mask.sum().item()),
                            total_neurons=int(a_flat.numel()),
                            concrete_min=float(a_flat.min().item()),
                            concrete_max=float(a_flat.max().item()),
                            abstract_lb_min=float(lb.min().item()),
                            abstract_ub_max=float(ub.max().item()),
                            lb_min=float(lb.min().item()),
                            ub_max=float(ub.max().item()),
                            first_bad_index=idx,
                            first_bad_value=float(a_flat[idx].item()),
                            first_bad_lb=float(lb[idx].item()),
                            first_bad_ub=float(ub[idx].item()),
                            topk=topk,
                        )
                        violations.append(violation)
                        status = "VIOLATION_FOUND"

        except Exception as exc:  # pylint: disable=broad-except
            metadata["error_type"] = type(exc).__name__
            metadata["error"] = str(exc)
            status = "ERROR"

        case_result = Level2CaseResult(
            case_name=getattr(case, "name", "<unknown>"),
            status=status,
            metadata=metadata,
            violations=violations,
        )
        per_mode.append(asdict(case_result))
        if status == "ERROR":
            overall_status = "ERROR"
        elif status == "VIOLATION_FOUND" and overall_status == "PASS":
            overall_status = "VIOLATION_FOUND"

    return {
        "case_name": getattr(case, "name", "<unknown>"),
        "overall_status": overall_status,
        "per_mode": per_mode,
    }
