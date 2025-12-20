from __future__ import annotations

from typing import Any, Dict, Optional


def extract_effective_spec(act_net: Any) -> Dict[str, Optional[Any]]:
    """Return a small dict with effective eps/assert_kind/y_true from an ACT Net."""
    from act.back_end.verifier import gather_input_spec_layers, get_assert_layer

    input_specs = gather_input_spec_layers(act_net)
    assert_layer = get_assert_layer(act_net)
    return {
        "eps": input_specs[0].meta.get("eps") if input_specs else None,
        "assert_kind": assert_layer.meta.get("kind") if assert_layer else None,
        "y_true": assert_layer.meta.get("y_true") if assert_layer else None,
    }
