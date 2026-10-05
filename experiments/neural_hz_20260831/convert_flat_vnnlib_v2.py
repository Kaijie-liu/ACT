#!/usr/bin/env python3
"""Losslessly rewrite a flat VNNLIB 1.0 property as tensor-indexed 2.0."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

from freeze_family_manifest import EXPERIMENT_ROOT, exclusive_atomic_write


_DECLARATION = re.compile(
    r"^\s*\(declare-const\s+([XY])_(\d+)\s+Real\s*\)\s*$",
    re.MULTILINE,
)
_FLAT_TOKEN = re.compile(r"\b([XY])_(\d+)\b")


def _shape(text: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in text.split(","))
    if not result or any(value <= 0 for value in result):
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return result


def _size(shape: tuple[int, ...]) -> int:
    result = 1
    for value in shape:
        result *= value
    return result


def _unravel(index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    if index < 0 or index >= _size(shape):
        raise ValueError(f"flat index {index} is outside shape {shape}")
    coordinates = []
    for width in reversed(shape):
        index, coordinate = divmod(index, width)
        coordinates.append(coordinate)
    return tuple(reversed(coordinates))


def _tensor_token(symbol: str, index: int, shape: tuple[int, ...]) -> str:
    coordinates = ",".join(str(value) for value in _unravel(index, shape))
    return f"{symbol}[{coordinates}]"


def convert_text(
    original: str,
    *,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> tuple[str, dict[str, object]]:
    """Return a token-level lossless VNNLIB 1.0-to-2.0 rewrite."""

    if not input_shape or any(int(value) <= 0 for value in input_shape):
        raise ValueError("input_shape dimensions must be positive")
    if not output_shape or any(int(value) <= 0 for value in output_shape):
        raise ValueError("output_shape dimensions must be positive")
    if "(vnnlib-version" in original or "(declare-network" in original:
        raise ValueError("source already declares VNNLIB 2.0")
    declarations = _DECLARATION.findall(original)
    x_declared = {int(index) for symbol, index in declarations if symbol == "X"}
    y_declared = {int(index) for symbol, index in declarations if symbol == "Y"}
    if x_declared != set(range(_size(input_shape))):
        raise ValueError("input declarations are not one complete flat tensor")
    if y_declared != set(range(_size(output_shape))):
        raise ValueError("output declarations are not one complete flat tensor")

    body = _DECLARATION.sub("", original)

    def replace(match: re.Match[str]) -> str:
        symbol = match.group(1)
        index = int(match.group(2))
        shape = input_shape if symbol == "X" else output_shape
        return _tensor_token(symbol, index, shape)

    converted_body = _FLAT_TOKEN.sub(replace, body)
    digest = hashlib.sha256(original.encode()).hexdigest()
    header = (
        "(vnnlib-version 2.0)\n"
        "(declare-network N\n"
        f"  (declare-input X Real [{','.join(map(str, input_shape))}])\n"
        f"  (declare-output Y Real [{','.join(map(str, output_shape))}])\n"
        ")\n"
        f"; lossless flat-to-tensor rewrite; source_sha256={digest}\n"
    )
    converted = header + converted_body

    # Prove the non-declaration body round-trips exactly at token level.
    round_trip = converted_body
    for symbol, shape in (("X", input_shape), ("Y", output_shape)):
        pattern = re.compile(rf"\b{symbol}\[([0-9,]+)\]")

        def flatten(match: re.Match[str]) -> str:
            coordinates = tuple(int(value) for value in match.group(1).split(","))
            if len(coordinates) != len(shape):
                raise ValueError("converted tensor rank mismatch")
            index = 0
            for coordinate, width in zip(coordinates, shape):
                if coordinate < 0 or coordinate >= width:
                    raise ValueError("converted tensor coordinate out of range")
                index = index * width + coordinate
            return f"{symbol}_{index}"

        round_trip = pattern.sub(flatten, round_trip)
    if round_trip != body:
        raise AssertionError("flat/tensor token conversion did not round-trip")

    return converted, {
        "source_sha256": digest,
        "input_declarations": len(x_declared),
        "output_declarations": len(y_declared),
        "input_shape": [int(value) for value in input_shape],
        "output_shape": [int(value) for value in output_shape],
        "round_trip": "EXACT_TOKEN_IDENTITY",
    }


def _isolated_destination(path: Path) -> Path:
    destination = path.resolve(strict=False)
    try:
        destination.relative_to(EXPERIMENT_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"destination must remain below {EXPERIMENT_ROOT}: {destination}"
        ) from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--input-shape", type=_shape, required=True)
    parser.add_argument("--output-shape", type=_shape, required=True)
    args = parser.parse_args()

    source = args.source.resolve(strict=True)
    destination = _isolated_destination(args.destination)
    original = source.read_text(encoding="utf-8")
    converted, record = convert_text(
        original,
        input_shape=args.input_shape,
        output_shape=args.output_shape,
    )

    exclusive_atomic_write(destination, converted.encode("utf-8"))
    print(
        f"converted {source.name}: X={record['input_declarations']}, "
        f"Y={record['output_declarations']}, "
        f"sha256={record['source_sha256']}, destination={destination}"
    )


if __name__ == "__main__":
    main()
