from __future__ import annotations

import hashlib

import pytest

from convert_flat_vnnlib_v2 import convert_text


def _flat() -> str:
    return """; demo
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const X_2 Real)
(declare-const X_3 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (<= X_0 X_3))
(assert (or (>= Y_0 Y_1) (<= Y_1 Y_0)))
"""


def test_lossless_token_rewrite_and_metadata():
    original = _flat()
    converted, record = convert_text(
        original, input_shape=(1, 2, 2), output_shape=(1, 2)
    )
    assert converted.startswith("(vnnlib-version 2.0)\n")
    assert "(declare-input X Real [1,2,2])" in converted
    assert "(declare-output Y Real [1,2])" in converted
    assert "X[0,0,0]" in converted
    assert "X[0,1,1]" in converted
    assert "Y[0,0]" in converted
    assert "Y[0,1]" in converted
    assert "declare-const" not in converted
    assert record["source_sha256"] == hashlib.sha256(
        original.encode("utf-8")
    ).hexdigest()
    assert record["round_trip"] == "EXACT_TOKEN_IDENTITY"


def test_rewrite_is_deterministic():
    kwargs = {"input_shape": (1, 2, 2), "output_shape": (1, 2)}
    assert convert_text(_flat(), **kwargs) == convert_text(_flat(), **kwargs)


@pytest.mark.parametrize(
    "text,error",
    [
        ("(vnnlib-version 2.0)\n", "already declares"),
        (
            "(declare-const X_0 Real)\n(declare-const Y_0 Real)\n",
            "input declarations",
        ),
        (
            "\n".join(
                [f"(declare-const X_{i} Real)" for i in range(4)]
                + ["(declare-const Y_1 Real)"]
            ),
            "output declarations",
        ),
    ],
)
def test_invalid_source_fails_closed(text, error):
    with pytest.raises(ValueError, match=error):
        convert_text(text, input_shape=(1, 2, 2), output_shape=(1, 1))


@pytest.mark.parametrize("shape", [(), (1, 0), (-1, 2)])
def test_invalid_shape_fails_closed(shape):
    with pytest.raises(ValueError, match="dimensions must be positive"):
        convert_text(_flat(), input_shape=shape, output_shape=(1, 2))
