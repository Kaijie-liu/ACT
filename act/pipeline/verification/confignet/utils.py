from __future__ import annotations

from typing import Any, Dict, Optional


def extract_effective_spec(act_net: Any) -> Dict[str, Optional[Any]]:
    """
    Return a dict with effective eps/assert_kind/y_true.
    Raises if multiple INPUT_SPEC or ASSERT layers are present to avoid ambiguity.
    """
    from act.back_end.verifier import gather_input_spec_layers, get_assert_layer

    input_specs = gather_input_spec_layers(act_net)
    assert_layers = [L for L in getattr(act_net, "layers", []) if getattr(L, "kind", None) == "ASSERT"]
    if len(input_specs) != 1:
        raise ValueError(f"Expected exactly one INPUT_SPEC, found {len(input_specs)}.")
    if len(assert_layers) != 1:
        raise ValueError(f"Expected exactly one ASSERT, found {len(assert_layers)}.")

    assert_layer = assert_layers[0]
    return {
        "eps": input_specs[0].meta.get("eps"),
        "assert_kind": assert_layer.meta.get("kind"),
        "y_true": assert_layer.meta.get("y_true"),
    }
