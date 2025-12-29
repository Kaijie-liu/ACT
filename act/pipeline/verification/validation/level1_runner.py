# act/pipeline/verification/validation/level1_runner.py
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from .level1_inputs import InputSamplingPlan, sample_concrete_inputs, _normalize_spec_kind
from .level1_properties import check_property_concrete
from .results import CaseResult, Counterexample, hash_tensor


def _spec_kind(act_net) -> str:
    spec_layers = [L for L in getattr(act_net, "layers", []) if getattr(L, "kind", None) == "INPUT_SPEC"]
    if not spec_layers:
        return None
    if len(spec_layers) > 1:
        return "MULTIPLE"
    return _normalize_spec_kind((getattr(spec_layers[0], "meta", None) or {}).get("kind"))


def _select_logits(output):
    import torch  # lazy import

    if isinstance(output, dict):
        if "output" in output:
            return output["output"]
        if "logits" in output:
            return output["logits"]
        raise ValueError("Model dict output missing 'output'/'logits'.")
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)) and output:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
        raise ValueError("First element of model output must be Tensor.")
    raise ValueError(f"Unsupported model output type: {type(output)}")


def run_level1_counterexample_check(case, seed: int, plan: InputSamplingPlan, device: str, dtype) -> CaseResult:
    """Run Level 1 counterexample check for a ConfigNet case."""
    import torch  # lazy import

    metadata: Dict[str, Any] = {
        "seed": seed,
        "source": getattr(case, "metadata", {}).get("source"),
        "arch": getattr(case, "metadata", {}).get("arch"),
        "spec_kind": _spec_kind(getattr(case, "act_net", None)),
    }
    counterexamples = []
    status = "CERTIFIED"

    try:
        inputs = sample_concrete_inputs(
            case.act_net, seed=seed, plan=plan, device=device, dtype=dtype, require_satisfy=True
        )
        model = getattr(case, "torch_model", None)
        if model is None:
            raise ValueError("Case missing torch_model.")
        model = model.to(device=device, dtype=dtype)
        model.eval()
        assert_layer = getattr(case, "assert_layer", None)
        assert_meta = deepcopy(getattr(assert_layer, "meta", None) or {})
        if not assert_meta:
            raise ValueError("Missing ASSERT metadata.")

        for idx, sample in enumerate(inputs):
            with torch.no_grad():
                out = model(sample)
            logits = _select_logits(out)
            prop = check_property_concrete(logits, assert_meta)
            if not prop.satisfied:
                violation_idx = prop.details.get("first_violation_index", 0)
                if logits.dim() > 1 and violation_idx < logits.shape[0]:
                    pred_tensor = torch.argmax(logits[violation_idx])
                else:
                    pred_tensor = torch.argmax(logits)
                ce = Counterexample(
                    input_index=idx,
                    input_hash=hash_tensor(sample),
                    min_val=float(sample.min().item()),
                    max_val=float(sample.max().item()),
                    pred=int(pred_tensor.item()),
                    true_label=int(assert_meta.get("y_true", -1)),
                    kind=assert_meta.get("kind"),
                    details=prop.details,
                )
                counterexamples.append(ce)
                status = "COUNTEREXAMPLE_FOUND"
                break

        metadata["num_inputs"] = len(inputs)

    except Exception as exc:  # pylint: disable=broad-except
        metadata["error"] = str(exc)
        metadata["error_type"] = type(exc).__name__
        status = "ERROR"

    return CaseResult(
        case_name=getattr(case, "name", "<unknown>"),
        status=status,
        counterexamples=counterexamples,
        metadata=metadata,
    )
