"""Numerical bridge for softmax-target KL underflow in the AdvMoE trainer."""

from __future__ import annotations

from collections import Counter
from contextlib import AbstractContextManager
import inspect
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


class SoftmaxUnderflowGradientBridge(AbstractContextManager):
    """Replace only undefined KL target gradients at exact softmax zeros.

    A finite logit can underflow to an exact zero after float32 softmax.  The
    derivative of ``xlogy(target, target)`` with respect to ``target`` is then
    non-finite, although its composite derivative through softmax has a finite
    limiting value.  This bridge replaces a non-finite incoming gradient only
    where the corresponding softmax probability is exactly zero.  Any
    non-finite gradient at a positive probability fails closed.
    """

    def __init__(
        self,
        *,
        allowed_callsites: Iterable[tuple[str | Path, int]] | None = None,
    ) -> None:
        self._original = None
        self.allowed_callsites = (
            None
            if allowed_callsites is None
            else frozenset(
                (str(Path(path).resolve()), int(line))
                for path, line in allowed_callsites
            )
        )
        self.softmax_calls = 0
        self.eligible_softmax_calls = 0
        self.skipped_softmax_calls = 0
        self.gradient_hook_calls = 0
        self.replaced_elements = 0
        self.observed_callsites: Counter[str] = Counter()
        self._resolved_caller_paths: dict[str, str] = {}

    def __enter__(self) -> "SoftmaxUnderflowGradientBridge":
        if self._original is not None:
            raise RuntimeError("softmax bridge is already active")
        self._original = F.softmax
        original = self._original

        def bridged_softmax(
            input: torch.Tensor,
            dim: int | None = None,
            _stacklevel: int = 3,
            dtype: torch.dtype | None = None,
        ) -> torch.Tensor:
            output = original(
                input, dim=dim, _stacklevel=_stacklevel, dtype=dtype
            )
            self.softmax_calls += 1
            frame = inspect.currentframe()
            caller = None if frame is None else frame.f_back
            try:
                raw_path = None if caller is None else caller.f_code.co_filename
                resolved_path = None
                if raw_path is not None:
                    resolved_path = self._resolved_caller_paths.get(raw_path)
                    if resolved_path is None:
                        resolved_path = str(Path(raw_path).resolve())
                        self._resolved_caller_paths[raw_path] = resolved_path
                callsite = (
                    "<unknown>",
                    -1,
                ) if caller is None else (
                    resolved_path,
                    int(caller.f_lineno),
                )
            finally:
                del caller
                del frame
            callsite_label = f"{callsite[0]}:{callsite[1]}"
            self.observed_callsites[callsite_label] += 1
            eligible = (
                self.allowed_callsites is None
                or callsite in self.allowed_callsites
            )
            if not eligible:
                self.skipped_softmax_calls += 1
                return output
            self.eligible_softmax_calls += 1
            if not output.requires_grad:
                return output
            frozen_output = output.detach()

            def bridge_gradient(gradient: torch.Tensor) -> torch.Tensor:
                self.gradient_hook_calls += 1
                nonfinite = ~torch.isfinite(gradient)
                if not bool(nonfinite.any().item()):
                    return gradient
                allowed = nonfinite & (frozen_output == 0)
                forbidden = nonfinite & ~allowed
                if bool(forbidden.any().item()):
                    raise RuntimeError(
                        "non-finite softmax gradient occurred at a positive probability"
                    )
                self.replaced_elements += int(allowed.sum().item())
                return torch.where(allowed, torch.zeros_like(gradient), gradient)

            output.register_hook(bridge_gradient)
            return output

        F.softmax = bridged_softmax
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._original is None:
            raise RuntimeError("softmax bridge is not active")
        F.softmax = self._original
        self._original = None

    def summary(self) -> dict[str, Any]:
        return {
            "softmax_calls": self.softmax_calls,
            "eligible_softmax_calls": self.eligible_softmax_calls,
            "skipped_softmax_calls": self.skipped_softmax_calls,
            "gradient_hook_calls": self.gradient_hook_calls,
            "replaced_elements": self.replaced_elements,
            "scope": (
                "ALL_CALLSITES_LEGACY"
                if self.allowed_callsites is None
                else "EXACT_CALLSITE_ALLOWLIST"
            ),
            "allowed_callsites": (
                None
                if self.allowed_callsites is None
                else [
                    {"path": path, "line": line}
                    for path, line in sorted(self.allowed_callsites)
                ]
            ),
            "observed_callsites": dict(sorted(self.observed_callsites.items())),
        }
