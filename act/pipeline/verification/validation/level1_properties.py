# act/pipeline/verification/validation/level1_properties.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class PropertyCheckResult:
    satisfied: bool
    kind: str
    details: Dict[str, Any]


def _normalize_assert_kind(kind: str) -> str:
    return str(kind).upper().replace("-", "_")


def check_property_concrete(logits, assert_meta: Dict[str, Any], atol: float = 1e-9) -> PropertyCheckResult:
    import torch  # lazy import

    if assert_meta is None or "kind" not in assert_meta:
        raise ValueError("ASSERT metadata with 'kind' required.")
    kind = _normalize_assert_kind(assert_meta["kind"])

    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"logits must be torch.Tensor, got {type(logits)}")
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    if logits.dim() != 2:
        raise ValueError(f"Expected logits shape [B, C], got {tuple(logits.shape)}")

    y_true = assert_meta.get("y_true")
    if y_true is None:
        raise ValueError("ASSERT metadata missing y_true.")
    num_classes = logits.shape[1]
    if y_true < 0 or y_true >= num_classes:
        raise ValueError(f"y_true {y_true} out of range for {num_classes} classes.")

    if kind == "TOP1_ROBUST":
        preds = torch.argmax(logits, dim=1)
        violations = (preds != y_true).nonzero(as_tuple=False)
        if violations.numel() == 0:
            return PropertyCheckResult(True, kind, {"pred": int(preds[0].item())})
        idx = int(violations[0].item())
        details = {"first_violation_index": idx, "pred": int(preds[idx].item()), "true_label": int(y_true)}
        return PropertyCheckResult(False, kind, details)

    if kind == "MARGIN_ROBUST":
        margin = float(assert_meta.get("margin", 0.0))
        true_logits = logits[:, y_true]
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[:, y_true] = False
        other_logits = logits.masked_select(mask).view(logits.shape[0], -1)
        best_other, _ = torch.max(other_logits, dim=1)
        ok = true_logits >= (best_other + margin - atol)
        violations = (~ok).nonzero(as_tuple=False)
        if violations.numel() == 0:
            return PropertyCheckResult(True, kind, {"margin": margin})
        idx = int(violations[0].item())
        details = {
            "first_violation_index": idx,
            "pred": int(torch.argmax(logits[idx]).item()),
            "true_label": int(y_true),
            "true_logit": float(true_logits[idx].item()),
            "best_other_logit": float(best_other[idx].item()),
            "margin": margin,
        }
        return PropertyCheckResult(False, kind, details)

    raise NotImplementedError(f"Unsupported ASSERT kind: {kind}")
