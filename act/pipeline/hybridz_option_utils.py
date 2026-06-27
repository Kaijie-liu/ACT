# ===- act/pipeline/hybridz_option_utils.py - HybridZ CLI helpers -------===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#
"""Small package-local helpers shared by HybridZ pipeline entry points."""

from __future__ import annotations

from typing import Dict, Iterable


def parse_key_value_options(items: Iterable[str], flag_name: str) -> Dict[str, object]:
    """Parse repeated ``--<flag_name> name=value`` style arguments."""

    out: Dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"invalid --{flag_name} {item!r}; expected name=value")
        key, raw = item.split("=", 1)
        low = raw.lower()
        if low in {"true", "false"}:
            val: object = low == "true"
        else:
            try:
                val = int(raw)
            except ValueError:
                try:
                    val = float(raw)
                except ValueError:
                    val = raw
        out[key] = val
    return out
