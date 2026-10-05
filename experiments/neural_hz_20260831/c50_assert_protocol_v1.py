"""Default-off ordinary-Python assertion execution audit, never python -O.

Source/function bytecode and assertion canaries remain checked. This changes
failure-report formatting work only, not any HZ code, test or assertion source.
"""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import sys
import types
import numpy as np
from _pytest.assertion.rewrite import rewrite_asserts


def guard():
    if not __debug__ or sys.flags.optimize!=0:
        raise ValueError('normal debug assertions are mandatory; Python optimization is forbidden')
    try:exec(compile("assert False, 'required_failure_canary'",'<c50-canary>','exec',dont_inherit=True,optimize=0),{})
    except AssertionError as exc:
        if str(exc)!='required_failure_canary':raise ValueError('assertion exception semantics changed')
    else:raise ValueError('mandatory failed assertion did not raise')


def compiled(source,filename,*,rewritten=False):
    guard();tree=ast.parse(source,filename=filename)
    if rewritten:rewrite_asserts(tree,source,filename)
    return compile(tree,filename,'exec',dont_inherit=True,optimize=0)


def code_key(code):return code.co_qualname,code.co_firstlineno


def code_tree(code):
    out=[code]
    for value in code.co_consts:
        if type(value) is types.CodeType:out.extend(code_tree(value))
    return out


def code_image(code):
    def value(v):
        if type(v) is types.CodeType:return item(v)
        if type(v) in (tuple,frozenset):
            vals=[value(x) for x in v]
            return [type(v).__name__,sorted(vals,key=repr) if type(v) is frozenset else vals]
        if type(v) is bytes:return ['bytes',v.hex()]
        return [type(v).__name__,repr(v)]
    def item(c):
        return ['code',c.co_name,c.co_qualname,c.co_filename,c.co_firstlineno,c.co_argcount,
            c.co_posonlyargcount,c.co_kwonlyargcount,c.co_nlocals,c.co_stacksize,c.co_flags,
            c.co_code.hex(),c.co_linetable.hex(),c.co_exceptiontable.hex(),
            c.co_names,c.co_varnames,c.co_freevars,c.co_cellvars,[value(v) for v in c.co_consts]]
    return hashlib.sha256(json.dumps(item(code),sort_keys=True).encode()).hexdigest()


def unwrap_function(value):
    if isinstance(value,(staticmethod,classmethod)):value=value.__func__
    if isinstance(value,types.MethodType):value=value.__func__
    if hasattr(value,'__wrapped__'):value=inspect.unwrap(value)
    return value if type(value) is types.FunctionType else None


def check_module(module,path,expected_sha,*,extra_functions=(),enabled=False):
    if not enabled:return None
    guard();path=Path(path).resolve();source=path.read_bytes()
    if hashlib.sha256(source).hexdigest()!=expected_sha:raise ValueError('test source differs from independent frozen hash')
    if Path(module.__file__).resolve()!=path:raise ValueError('loaded test module does not belong to its frozen source')
    tree=ast.parse(source,filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in (
                'pytest_assertion_pass','pytest_assertrepr_compare'):
            raise ValueError('assertion reporting hook requires a separate semantic proof')
    # Compile the SAME complete source with the filename used by its loader.
    # No rewriter/cache/global success token can substitute for this comparison.
    functions=[]
    def visit(namespace):
        for obj in namespace.values():
            fn=unwrap_function(obj)
            if fn is not None and fn.__module__==module.__name__:functions.append(fn)
            elif isinstance(obj,type) and obj.__module__==module.__name__:visit(vars(obj))
    visit(vars(module));functions.extend(extra_functions)
    functions=list({id(f):f for f in functions}.values())
    expected_by_filename={};seen=set();images=[]
    for fn in functions:
        fn=unwrap_function(fn)
        if fn is None or Path(fn.__code__.co_filename).resolve()!=path:
            raise ValueError('unregistered wrapped test or helper code')
        filename=fn.__code__.co_filename
        if filename not in expected_by_filename:
            reference=compiled(source,filename)
            expected_by_filename[filename]={code_key(c):c for c in code_tree(reference)}
        key=code_key(fn.__code__);expected=expected_by_filename[filename].get(key)
        if expected is None or code_image(fn.__code__)!=code_image(expected):
            raise ValueError('actual loaded test/helper bytecode differs from source with assertions enabled')
        seen.add(key);images.append(code_image(fn.__code__))
    if not functions:raise ValueError('no actual test/helper functions checked')
    plain=compiled(source,str(path));rewritten=compiled(source,str(path),rewritten=True)
    return dict(file=str(path),source_sha256=expected_sha,loaded_functions_checked=len(functions),
        source_assert_statements=sum(isinstance(n,ast.Assert) for n in ast.walk(tree)),
        plain_code_bytes=sum(len(c.co_code) for c in code_tree(plain)),
        rewritten_code_bytes=sum(len(c.co_code) for c in code_tree(rewritten)),
        loaded_code_sha256=hashlib.sha256('\n'.join(sorted(images)).encode()).hexdigest(),
        all_loaded_functions_equal_debug_source_code=True,formal_gain=0)


CANARY=b'''
def evaluate(mode, trace):
    def v(label, value):
        trace.append(label)
        return value
    if mode == 0:
        assert v('condition', True)
    elif mode == 1:
        assert v('condition', False), v('message', 'explicit failure')
    elif mode == 2:
        assert v('first', False) and v('second', True)
    elif mode == 3:
        assert v('first', True) or v('second', False)
    elif mode == 4:
        assert v('a', 1) < v('b', 2) <= v('c', 3)
    elif mode == 5:
        assert v('a', 3) < v('b', 2) < v('c', 1)
    elif mode == 6:
        assert all(v(str(i), i >= 0) for i in range(4))
    elif mode == 7:
        assert np.array_equal(v('array', np.array([1., 2.])), np.array([1., 2.]))
    elif mode == 8:
        assert not v('condition', False)
    elif mode == 9:
        assert v('array', np.array([1., 2.]))
    elif mode == 10:
        assert v('before', 1) / v('zero', 0)
    elif mode == 11:
        assert np.bool_(v('condition', False))
    elif mode == 12:
        assert v('condition', True), v('message', 'must not execute')
    elif mode == 13:
        assert (a := v('walrus', 2)) == 2 and a > 0
    elif mode == 14:
        assert np.bool_(v('condition', True))
    elif mode == 15:
        assert False, 'forced failure'
    return 'returned'
'''


def canaries(*,enabled=False):
    if not enabled:return None
    guard();versions=[]
    expected={1:'AssertionError',2:'AssertionError',5:'AssertionError',9:'ValueError',10:'ZeroDivisionError',11:'AssertionError',15:'AssertionError'}
    for rewritten in (False,True):
        namespace={'np':np};exec(compiled(CANARY,'<c50-assertion-semantics>',rewritten=rewritten),namespace)
        rows=[]
        for mode in range(16):
            trace=[]
            try:result=namespace['evaluate'](mode,trace);kind='returned'
            except Exception as exc:kind=type(exc).__name__
            if kind!=expected.get(mode,'returned'):raise ValueError('forced-failure/exception canary was not enforced')
            rows.append(dict(mode=mode,outcome=kind,evaluation_trace=trace))
        versions.append(rows)
    if versions[0]!=versions[1]:raise ValueError('normal source assertion side effects/short circuits differ from reference')
    return dict(cases=16,all_outcomes_and_evaluation_traces_equal=True,normal_python_assertions_enforced=True,
        expected_failed_assertions=5,expected_other_exceptions=2,rows=versions[0],
        failure_formatting_text_is_not_identical=True,arbitrary_custom_truthiness_equivalence_not_claimed=True,
        runtime_or_HZ_payment_proved=False,formal_gain=0)
