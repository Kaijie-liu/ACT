"""Reuse parsing in four pair-check functions, retaining their exact bytecode.

Private function globals are cloned; no module-level monkey patch, shared
cache, source producer or change to guard/route/property predicates. Router
and network checking remain untouched. No proof verdict or bound is cached.
"""
import time
from types import FunctionType

from scoped_source.graph import clock
from source_enclosure.format import identity
from parsed_source_reuse.cache import SourceParser


def clone(fn, replacements):
    if fn.__closure__ is not None: raise ValueError('unsupported checker closure')
    if any(k not in fn.__globals__ for k in replacements): raise ValueError('checker dependency drift')
    result=FunctionType(fn.__code__,dict(fn.__globals__,**replacements),fn.__name__,fn.__defaults__)
    result.__kwdefaults__=None if fn.__kwdefaults__ is None else dict(fn.__kwdefaults__)
    return result


def private_checker(parser, scope):
    from residual_proof import check as original
    unpack=lambda state:parser.unpack(state,scope=scope)
    replacements={name:clone(getattr(original,name),{'unpack':unpack})
                  for name in ('check_join','check_guards','check_projection','check_outputs')}
    return clone(original.check,replacements)


def check(doc,bundle,*,invocation,expected_source_sha256,deadline,enabled,stats_sink=None,limits=None,cache_invocation=None):
    started=time.monotonic();tick=clock(deadline)
    if type(enabled) is not bool:raise ValueError('explicit opt-in boolean')
    # Do not trust a caller's source digest to select a cache namespace.
    if identity(doc)!=expected_source_sha256:raise ValueError('current source/request binding')
    scope={'invocation':invocation if cache_invocation is None else cache_invocation,'source_sha256':expected_source_sha256}
    parser=SourceParser(scope,enabled=enabled,tick=tick,limits=limits)
    state='ERROR'
    try:
        result=private_checker(parser,scope)(doc,bundle,invocation=invocation,
                    expected_source_sha256=expected_source_sha256,deadline=deadline)
        tick();state='COMPLETED';return result
    finally:
        retained=parser.stats();parser.close()
        stats={'schema':'SOURCE_PARSE_REUSE_CHECK_R1','status':state,'parser':retained,
               'after_close':parser.stats(),'seconds_before_stats':time.monotonic()-started,
               'complete_output_positive_proof':False,'native_solver_queries':0,
               'scope':'same source-construction checking, NOT output lower-bound certificates'}
        if stats_sink is not None:stats_sink(stats)
