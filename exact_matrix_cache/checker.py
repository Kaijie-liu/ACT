"""Opt-in exact parser reuse, with immutable original mathematical functions.

Function-local dependency binding uses the original code objects and fresh
globals dictionaries. No original module/global/sys.modules entry is patched.
The only substituted numerical operations are CSR entries/row parsing; source,
property, rectangle, dual and completeness checks execute for every obligation.
"""
from types import FunctionType

from act.back_end.solver import lp_certificate as lp
from act.back_end.solver import sparse_lp_certificate as sparse
from act.back_end.solver import check_hz_lp_export as hz
from act.back_end.solver import check_rational_mccormick as weighted
from moe_evidence import checker as request_checker
from moe_evidence.schema import validate_request
from exact_matrix_cache.cache import MatrixCache


def _bind(function, **overrides):
    if function.__closure__:
        raise ValueError('only reviewed closure-free checker functions may be bound')
    if not set(overrides) <= set(function.__globals__):
        raise ValueError('unrecognized checker dependency')
    local_globals = {**function.__globals__, **overrides}
    bound = FunctionType(function.__code__, local_globals, function.__name__, function.__defaults__)
    bound.__kwdefaults__ = function.__kwdefaults__
    return bound


def check_manifest(manifest, request, load, *, enabled=True, tick=lambda: None):
    scope = validate_request(request)
    cache = MatrixCache(scope, enabled=enabled, tick=tick)
    evaluate = _bind(sparse.evaluate, rows=cache.rows)
    sparse_check = _bind(sparse.check, evaluate=evaluate)
    def check_lp(program, certificate):
        return sparse_check(program, certificate) if 'matrix_format' in program else lp.check(program, certificate)
    export_check = _bind(hz.check_export, _entries=cache.entries, check=check_lp)
    weighted_check = _bind(weighted.check_construction, _entries=cache.entries, check=check_lp)
    aggregate = _bind(request_checker.check_manifest, check_export=export_check,
                      check_construction=weighted_check, check=check_lp)
    try:
        result = aggregate(manifest, request, load, tick=tick)
    finally:
        cache.clear()  # no parsed object or fact survives a request, even on error
    return {'result': result, 'cache': cache.stats(), 'scope': scope,
            'acceptance_changed': False, 'source_validation_cached': False,
            'property_or_dual_result_cached': False}
