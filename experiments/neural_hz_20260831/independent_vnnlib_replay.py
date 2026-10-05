"""Small independent concrete evaluator for VNNLIB 1.0 assertions.

This module intentionally imports neither ACT nor the historical SATSidecar.
It evaluates a concrete `(x, y)` against the original SMT-LIB assertions at
literal zero tolerance.  It is not a symbolic verifier or a VNNLIB converter.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


_TOKEN = re.compile(r"\(|\)|[^\s()]+")
_VARIABLE = re.compile(r"^([XY])_(\d+)$")


@dataclass(frozen=True)
class BoolResult:
    holds: bool
    robustness: float


@dataclass(frozen=True)
class ConcreteSpecResult:
    input_holds: bool
    unsafe_holds: bool
    all_assertions_hold: bool
    input_min_slack: float
    unsafe_min_slack: float
    input_assertion_count: int
    unsafe_assertion_count: int


def _strip_comments(text: str) -> str:
    return "\n".join(line.split(";", 1)[0] for line in text.splitlines())


def parse_sexpressions(text: str):
    tokens = _TOKEN.findall(_strip_comments(text))
    cursor = 0

    def parse_one():
        nonlocal cursor
        if cursor >= len(tokens):
            raise ValueError("unexpected end of VNNLIB")
        token = tokens[cursor]
        cursor += 1
        if token == "(":
            values = []
            while True:
                if cursor >= len(tokens):
                    raise ValueError("unclosed VNNLIB list")
                if tokens[cursor] == ")":
                    cursor += 1
                    return values
                values.append(parse_one())
        if token == ")":
            raise ValueError("unexpected closing parenthesis")
        return token

    forms = []
    while cursor < len(tokens):
        forms.append(parse_one())
    return forms


def _symbols(expression) -> set[str]:
    if isinstance(expression, str):
        return {expression} if _VARIABLE.fullmatch(expression) else set()
    result: set[str] = set()
    for child in expression:
        result.update(_symbols(child))
    return result


def _finite(value: float, *, context: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite numeric result in {context}")
    return result


def _number(token: str) -> float:
    try:
        return _finite(float(token), context=f"literal {token!r}")
    except ValueError as exc:
        raise ValueError(f"unsupported numeric atom: {token!r}") from exc


def _numeric(expression, environment: Mapping[str, float]) -> float:
    if isinstance(expression, str):
        if expression in environment:
            return _finite(environment[expression], context=expression)
        return _number(expression)
    if not expression:
        raise ValueError("empty numeric expression")
    operator = expression[0]
    arguments = expression[1:]
    if operator == "+":
        return _finite(sum(_numeric(arg, environment) for arg in arguments), context="+")
    if operator == "-":
        if not arguments:
            raise ValueError("'-' requires at least one argument")
        first = _numeric(arguments[0], environment)
        result = -first if len(arguments) == 1 else first
        for argument in arguments[1:]:
            result -= _numeric(argument, environment)
        return _finite(result, context="-")
    if operator == "*":
        result = 1.0
        for argument in arguments:
            result *= _numeric(argument, environment)
            result = _finite(result, context="*")
        return result
    if operator == "/":
        if not arguments:
            raise ValueError("'/' requires at least one argument")
        first = _numeric(arguments[0], environment)
        result = 1.0 / first if len(arguments) == 1 else first
        for argument in arguments[1:]:
            result /= _numeric(argument, environment)
        return _finite(result, context="/")
    if operator in ("to_real", "to_int") and len(arguments) == 1:
        value = _numeric(arguments[0], environment)
        return value if operator == "to_real" else float(math.floor(value))
    if operator == "ite" and len(arguments) == 3:
        condition = _boolean(arguments[0], environment)
        return _numeric(arguments[1] if condition.holds else arguments[2], environment)
    raise ValueError(f"unsupported numeric operator: {operator!r}")


def _and(results: Sequence[BoolResult]) -> BoolResult:
    if not results:
        return BoolResult(True, math.inf)
    return BoolResult(
        all(result.holds for result in results),
        min(result.robustness for result in results),
    )


def _or(results: Sequence[BoolResult]) -> BoolResult:
    if not results:
        return BoolResult(False, -math.inf)
    return BoolResult(
        any(result.holds for result in results),
        max(result.robustness for result in results),
    )


def _comparison(operator: str, left: float, right: float) -> BoolResult:
    difference = _finite(left - right, context=operator)
    if operator == "<=":
        return BoolResult(difference <= 0.0, -difference)
    if operator == "<":
        return BoolResult(difference < 0.0, -difference)
    if operator == ">=":
        return BoolResult(difference >= 0.0, difference)
    if operator == ">":
        return BoolResult(difference > 0.0, difference)
    if operator == "=":
        return BoolResult(difference == 0.0, -abs(difference))
    if operator == "distinct":
        return BoolResult(difference != 0.0, abs(difference))
    raise ValueError(f"unsupported comparison: {operator!r}")


def _boolean(expression, environment: Mapping[str, float]) -> BoolResult:
    if isinstance(expression, str):
        if expression == "true":
            return BoolResult(True, math.inf)
        if expression == "false":
            return BoolResult(False, -math.inf)
        raise ValueError(f"unsupported Boolean atom: {expression!r}")
    if not expression:
        raise ValueError("empty Boolean expression")
    operator = expression[0]
    arguments = expression[1:]
    if operator == "and":
        return _and([_boolean(arg, environment) for arg in arguments])
    if operator == "or":
        return _or([_boolean(arg, environment) for arg in arguments])
    if operator == "not" and len(arguments) == 1:
        child = _boolean(arguments[0], environment)
        return BoolResult(not child.holds, -child.robustness)
    if operator == "=>" and len(arguments) == 2:
        premise = _boolean(arguments[0], environment)
        conclusion = _boolean(arguments[1], environment)
        return _or(
            [BoolResult(not premise.holds, -premise.robustness), conclusion]
        )
    if operator == "ite" and len(arguments) == 3:
        condition = _boolean(arguments[0], environment)
        return _boolean(arguments[1] if condition.holds else arguments[2], environment)
    if operator in ("<=", "<", ">=", ">", "=", "distinct"):
        if len(arguments) < 2:
            raise ValueError(f"{operator!r} requires at least two arguments")
        values = [_numeric(argument, environment) for argument in arguments]
        return _and(
            [
                _comparison(operator, left, right)
                for left, right in zip(values[:-1], values[1:], strict=True)
            ]
        )
    raise ValueError(f"unsupported Boolean operator: {operator!r}")


def evaluate_vnnlib(text: str, x, y) -> ConcreteSpecResult:
    """Evaluate all original VNNLIB assertions on concrete finite arrays."""

    x_values = np.asarray(x, dtype=np.float64).reshape(-1)
    y_values = np.asarray(y, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(y_values)):
        raise ValueError("concrete input/output contains a non-finite value")
    environment = {
        **{f"X_{index}": float(value) for index, value in enumerate(x_values)},
        **{f"Y_{index}": float(value) for index, value in enumerate(y_values)},
    }

    declared_x: set[int] = set()
    declared_y: set[int] = set()
    input_results: list[BoolResult] = []
    unsafe_results: list[BoolResult] = []
    for form in parse_sexpressions(text):
        if not isinstance(form, list) or not form:
            raise ValueError(f"invalid top-level VNNLIB form: {form!r}")
        operator = form[0]
        if operator in ("set-logic", "set-info", "set-option"):
            continue
        if operator == "declare-const":
            if len(form) != 3 or form[2] != "Real":
                raise ValueError(f"unsupported declaration: {form!r}")
            match = _VARIABLE.fullmatch(form[1])
            if match is None:
                raise ValueError(f"unsupported declared symbol: {form[1]!r}")
            destination = declared_x if match.group(1) == "X" else declared_y
            index = int(match.group(2))
            if index in destination:
                raise ValueError(f"duplicate declaration: {form[1]}")
            destination.add(index)
            continue
        if operator != "assert" or len(form) != 2:
            raise ValueError(f"unsupported top-level form: {operator!r}")
        symbols = _symbols(form[1])
        prefixes = {symbol[0] for symbol in symbols}
        if prefixes == {"X"}:
            input_results.append(_boolean(form[1], environment))
        elif prefixes == {"Y"}:
            unsafe_results.append(_boolean(form[1], environment))
        else:
            raise ValueError(
                f"assertion must be purely X or purely Y, got prefixes={prefixes}"
            )

    if declared_x != set(range(x_values.size)):
        raise ValueError("X declarations do not match concrete input width")
    if declared_y != set(range(y_values.size)):
        raise ValueError("Y declarations do not match concrete output width")
    if not input_results:
        raise ValueError("VNNLIB has no input assertion")
    if not unsafe_results:
        raise ValueError("VNNLIB has no UNSAFE output assertion")
    input_result = _and(input_results)
    unsafe_result = _and(unsafe_results)
    return ConcreteSpecResult(
        input_holds=input_result.holds,
        unsafe_holds=unsafe_result.holds,
        all_assertions_hold=input_result.holds and unsafe_result.holds,
        input_min_slack=float(input_result.robustness),
        unsafe_min_slack=float(unsafe_result.robustness),
        input_assertion_count=len(input_results),
        unsafe_assertion_count=len(unsafe_results),
    )
