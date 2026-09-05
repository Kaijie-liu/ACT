"""Numerical bridge for softmax-target KL underflow in the AdvMoE trainer."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

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

    def __init__(self) -> None:
        self._original = None
        self.softmax_calls = 0
        self.gradient_hook_calls = 0
        self.replaced_elements = 0

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

    def summary(self) -> dict[str, int]:
        return {
            "softmax_calls": self.softmax_calls,
            "gradient_hook_calls": self.gradient_hook_calls,
            "replaced_elements": self.replaced_elements,
        }
