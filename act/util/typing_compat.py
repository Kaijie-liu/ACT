"""Central Python-version compatibility helpers for ACT entry points."""

from __future__ import annotations

import typing

try:
    from typing import override
except ImportError:  # Python 3.11 compatibility environments.
    from typing_extensions import override


def install_typing_override() -> None:
    """Expose :func:`override` on ``typing`` for legacy transitive imports."""

    if not hasattr(typing, "override"):
        typing.override = override  # type: ignore[attr-defined]


# Blackwell and CROWN runners import this module before the rest of ACT.  Keep
# the compatibility action centralized so individual entry points cannot drift.
install_typing_override()


__all__ = ["install_typing_override", "override"]
