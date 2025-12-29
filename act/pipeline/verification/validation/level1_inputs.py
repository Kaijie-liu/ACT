# act/pipeline/verification/validation/level1_inputs.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class InputSamplingPlan:
    k_center: int = 1
    k_random: int = 1
    k_boundary: int = 1
    boundary_delta: float = 1e-3


def _normalize_spec_kind(kind: Optional[str]) -> Optional[str]:
    if kind is None:
        return None
    name = str(kind).upper().replace("-", "_")
    if name in {"LINF_BALL", "LINFBALL", "L_INF", "LINF"}:
        return "LINF_BALL"
    return name


def _extract_input_layer(act_net) -> Tuple[Dict[str, Any], Tuple[int, ...]]:
    input_layer = next((L for L in getattr(act_net, "layers", []) if getattr(L, "kind", None) == "INPUT"), None)
    if input_layer is None:
        raise ValueError("ACT net missing INPUT layer.")
    meta = getattr(input_layer, "meta", None) or {}
    shape = meta.get("shape")
    if not shape:
        raise ValueError("INPUT layer missing 'shape' metadata.")
    return meta, tuple(shape)


def _extract_spec_layer(act_net):
    spec_layers = [L for L in getattr(act_net, "layers", []) if getattr(L, "kind", None) == "INPUT_SPEC"]
    if len(spec_layers) > 1:
        raise ValueError(f"Expected at most one INPUT_SPEC; found {len(spec_layers)}.")
    return spec_layers[0] if spec_layers else None


def _value_range_fallback(input_meta: Dict[str, Any]):
    vr = input_meta.get("value_range")
    if isinstance(vr, (list, tuple)) and len(vr) >= 2:
        return vr[0], vr[1]
    return 0.0, 1.0


def _spec_info(input_meta: Dict[str, Any], shape: Tuple[int, ...], spec_layer, device: str, dtype):
    import torch  # lazy import

    if spec_layer is None:
        lb_val, ub_val = _value_range_fallback(input_meta)
        return {
            "kind": None,
            "lb": torch.full(shape, lb_val, device=device, dtype=dtype),
            "ub": torch.full(shape, ub_val, device=device, dtype=dtype),
        }

    spec_meta = getattr(spec_layer, "meta", None) or {}
    spec_params = getattr(spec_layer, "params", None) or {}
    spec_kind = _normalize_spec_kind(spec_meta.get("kind"))
    if spec_kind not in {"LINF_BALL", "BOX"}:
        raise NotImplementedError(f"Unsupported INPUT_SPEC kind: {spec_kind}")

    if spec_kind == "LINF_BALL":
        eps = spec_meta.get("eps")
        if eps is None:
            raise ValueError("LINF_BALL requires 'eps' in meta.")
        eps_tensor = eps if hasattr(eps, "to") else torch.tensor(eps, device=device, dtype=dtype)
        eps_tensor = eps_tensor.to(device=device, dtype=dtype)

        center = spec_params.get("center")
        if center is None:
            center_val = spec_meta.get("center_val", 0.0)
            center = torch.full(shape, center_val, device=device, dtype=dtype)
        else:
            center = center.to(device=device, dtype=dtype)

        lb = spec_params.get("lb")
        ub = spec_params.get("ub")
        lb = lb.to(device=device, dtype=dtype) if lb is not None else center - eps_tensor
        ub = ub.to(device=device, dtype=dtype) if ub is not None else center + eps_tensor

        return {
            "kind": "LINF_BALL",
            "lb": lb,
            "ub": ub,
            "center": center,
            "eps": eps_tensor,
            "spec_kind": spec_kind,
        }

    # BOX
    lb = spec_params.get("lb")
    ub = spec_params.get("ub")
    if lb is None or ub is None:
        lb_val = spec_meta.get("lb_val")
        ub_val = spec_meta.get("ub_val")
        if lb_val is None or ub_val is None:
            lb_val, ub_val = _value_range_fallback(input_meta)
        lb = torch.full(shape, lb_val, device=device, dtype=dtype)
        ub = torch.full(shape, ub_val, device=device, dtype=dtype)
    else:
        lb = lb.to(device=device, dtype=dtype)
        ub = ub.to(device=device, dtype=dtype)

    return {"kind": "BOX", "lb": lb, "ub": ub, "spec_kind": spec_kind}


def assert_satisfies_input_spec(x, spec_meta: Dict[str, Any], atol: Optional[float] = None) -> None:
    """Assert tensor satisfies BOX or LINF_BALL spec."""
    import torch  # lazy import

    kind = _normalize_spec_kind((spec_meta or {}).get("kind"))
    if kind is None:
        kind = "BOX"
    if atol is None:
        atol = 1e-9 if x.dtype == torch.float64 else 1e-6
    if kind == "BOX":
        lb = spec_meta.get("lb")
        ub = spec_meta.get("ub")
        if lb is None or ub is None:
            lb_val = spec_meta.get("lb_val", 0.0)
            ub_val = spec_meta.get("ub_val", 1.0)
            lb = torch.full_like(x, lb_val)
            ub = torch.full_like(x, ub_val)
        else:
            lb = lb.to(device=x.device, dtype=x.dtype)
            ub = ub.to(device=x.device, dtype=x.dtype)
        if torch.any(x < lb - atol) or torch.any(x > ub + atol):
            raise AssertionError("Input violates BOX bounds.")
        return

    if kind == "LINF_BALL":
        eps = spec_meta.get("eps")
        if eps is None:
            raise ValueError("LINF_BALL requires 'eps'.")
        eps_tensor = eps if hasattr(eps, "to") else torch.tensor(eps, device=x.device, dtype=x.dtype)
        eps_tensor = eps_tensor.to(device=x.device, dtype=x.dtype)
        center = spec_meta.get("center")
        if center is None:
            center_val = spec_meta.get("center_val", 0.0)
            center = torch.full_like(x, center_val)
        else:
            center = center.to(device=x.device, dtype=x.dtype)
        if torch.any((x - center).abs() > eps_tensor + atol):
            raise AssertionError("Input violates LINF_BALL constraints.")
        return

    raise NotImplementedError(f"Unsupported INPUT_SPEC kind: {kind}")


def _make_generator(device: str, seed: int):
    import torch

    try:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        return gen
    except Exception:
        gen = torch.Generator()
        gen.manual_seed(seed)
        return gen


def sample_concrete_inputs(act_net, seed: int, plan: InputSamplingPlan, device: str, dtype, require_satisfy: bool = True) -> List[Any]:
    """Sample concrete inputs deterministically based on act_net metadata."""
    import torch  # lazy import

    input_meta, shape = _extract_input_layer(act_net)
    spec_layer = _extract_spec_layer(act_net)
    info = _spec_info(input_meta, shape, spec_layer, device=device, dtype=dtype)
    kind = _normalize_spec_kind(info.get("kind"))

    gen = _make_generator(device, seed)
    delta = torch.tensor(plan.boundary_delta, device=device, dtype=dtype)

    samples: List[Any] = []

    def _append(t):
        if require_satisfy:
            assert_satisfies_input_spec(t, info)
        samples.append(t)

    if kind == "LINF_BALL":
        center = info["center"]
        eps = info["eps"]
        rand_kwargs = {"device": device, "dtype": dtype, "generator": gen}
        for _ in range(plan.k_center):
            _append(center.clone())
        for _ in range(plan.k_random):
            noise = (torch.rand(shape, **rand_kwargs) - 0.5) * 2 * eps
            _append(center + noise)
        for _ in range(plan.k_boundary):
            _append(center + eps - delta)
            _append(center - eps + delta)
    else:  # BOX or None treated as BOX fallback
        lb = info["lb"]
        ub = info["ub"]
        span = ub - lb
        center = lb + 0.5 * span
        rand_kwargs = {"device": device, "dtype": dtype, "generator": gen}
        for _ in range(plan.k_center):
            _append(center.clone())
        for _ in range(plan.k_random):
            rnd = torch.rand(shape, **rand_kwargs)
            _append(lb + rnd * span)
        for _ in range(plan.k_boundary):
            _append(ub - delta)
            _append(lb + delta)

    return samples
