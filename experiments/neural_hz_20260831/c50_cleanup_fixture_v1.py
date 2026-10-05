"""Tiny authentic source for loaded-bytecode rejection tests only."""


def test_scalar(value):
    assert value == 1
    return value


def helper(value):
    assert value > 0
    return value
