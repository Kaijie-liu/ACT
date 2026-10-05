"""Keep pytest assertion semantics; shorten proved-defined private cleanup.

Only compiler-generated function-local scratch targets are eligible. A
possibly unassigned short-circuit temporary keeps the original None store.
No condition, message, comparison, exception branch or test is removed.
"""
import ast
import hashlib
import re
import numpy as np
from _pytest.assertion.rewrite import rewrite_asserts
from experiments.neural_hz_20260831.c50_assert_protocol_v1 import guard,compiled,code_tree,CANARY


def private(name):return name.startswith(('@py_assert','@py_format'))


def targets(node):
    return {n.id for n in ast.walk(node) if isinstance(n,ast.Name) and private(n.id)}


class Cleanup(ast.NodeTransformer):
    def __init__(self):
        self.changed_assignments=0;self.deleted_targets=0;self.retained_none_targets=0

    def visit_FunctionDef(self,node):
        node.body,_=self.block(node.body,set());return node

    visit_AsyncFunctionDef=visit_FunctionDef

    def block(self,body,bound):
        out=[];bound=set(bound)
        for node in body:
            clean=(isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant) and node.value.value is None
                and node.targets and all(isinstance(t,ast.Name) and private(t.id) for t in node.targets))
            if clean and any(t.id in bound for t in node.targets):
                # Preserve left-to-right release order. No speculative DEL
                # on a short-circuited or otherwise unassigned private slot.
                groups=[]
                for t in node.targets:
                    deleting=t.id in bound
                    if not groups or groups[-1][0]!=deleting:groups.append((deleting,[]))
                    groups[-1][1].append(ast.copy_location(ast.Name(t.id,ast.Del() if deleting else ast.Store()),t))
                    if deleting:bound.discard(t.id);self.deleted_targets+=1
                    else:bound.add(t.id);self.retained_none_targets+=1
                for deleting,names in groups:
                    replacement=ast.Delete(names) if deleting else ast.Assign(names,ast.copy_location(ast.Constant(None),node.value))
                    out.append(ast.copy_location(replacement,node))
                self.changed_assignments+=1
                continue
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                node=self.visit(node)
            elif isinstance(node,ast.If):
                node.body,a=self.block(node.body,bound);node.orelse,b=self.block(node.orelse,bound)
                bound=a&b
            elif isinstance(node,(ast.For,ast.AsyncFor,ast.While)):
                # Loop bodies can execute repeatedly: never carry an entry
                # binding through a prior iteration's deletion by assumption.
                node.body,_=self.block(node.body,set());node.orelse,_=self.block(node.orelse,set())
                bound-={x.id for n in ast.walk(node) if isinstance(n,ast.Delete) for x in n.targets if isinstance(x,ast.Name)}
            elif isinstance(node,(ast.Try,ast.TryStar)):
                node.body,_=self.block(node.body,set());node.orelse,_=self.block(node.orelse,set())
                node.finalbody,_=self.block(node.finalbody,set())
                for handler in node.handlers:handler.body,_=self.block(handler.body,set())
                bound-={x.id for n in ast.walk(node) if isinstance(n,ast.Delete) for x in n.targets if isinstance(x,ast.Name)}
            elif isinstance(node,(ast.With,ast.AsyncWith)):
                node.body,_=self.block(node.body,set())
                bound-={x.id for n in ast.walk(node) if isinstance(n,ast.Delete) for x in n.targets if isinstance(x,ast.Name)}
            elif isinstance(node,ast.Assign):
                for t in node.targets:bound|=targets(t)
            elif isinstance(node,ast.AnnAssign) and node.value is not None:bound|=targets(node.target)
            elif isinstance(node,ast.AugAssign):bound|=targets(node.target)
            elif isinstance(node,ast.Delete):
                for t in node.targets:bound-=targets(t)
            out.append(node)
        return out,bound


def compile_cleanup(source,filename,*,config=None,enabled=False):
    if not enabled:return None
    guard();tree=ast.parse(source,filename=filename)
    source_asserts=sum(isinstance(n,ast.Assert) for n in ast.walk(tree))
    rewrite_asserts(tree,source,filename,config)
    reference=compile(tree,filename,'exec',dont_inherit=True,optimize=0)
    worker=Cleanup();tree=worker.visit(tree)
    candidate=compile(tree,filename,'exec',dont_inherit=True,optimize=0)
    return candidate,dict(source_assert_statements=source_asserts,
        changed_cleanup_assignments=worker.changed_assignments,
        proved_defined_private_deletes=worker.deleted_targets,
        possibly_unassigned_targets_still_cleared=worker.retained_none_targets,
        original_rewritten_code_bytes=sum(len(c.co_code) for c in code_tree(reference)),
        cleanup_code_bytes=sum(len(c.co_code) for c in code_tree(candidate)),
        source_sha256=hashlib.sha256(source).hexdigest(),
        original_assertion_rewriting_retained=True,condition_or_failure_branch_removed=False,formal_gain=0)


OWNER_CANARY=b'''
class Box:
    def __init__(self, trace, name): self.trace, self.name = trace, name
    def __eq__(self, other): self.trace.append('equal'); return True
    def __del__(self): self.trace.append('release:' + self.name)
def evaluate(mode, trace):
    for i in range(4):
        assert (i % 2 == 0) or (i % 2 == 1 and i >= 0)
    assert Box(trace, 'left') == Box(trace, 'right')
    trace.append('after')
    return 'returned'
'''


def exact_canaries(*,enabled=False):
    if not enabled:return None
    guard();summaries=[]
    for source,modes in ((CANARY,range(16)),(OWNER_CANARY,range(1))):
        reference=compiled(source,'<c50-exact-rewritten-canaries>',rewritten=True)
        candidate,stats=compile_cleanup(source,'<c50-exact-rewritten-canaries>',enabled=True)
        runs=[]
        for code in (reference,candidate):
            ns={'np':np};exec(code,ns);rows=[]
            for mode in modes:
                trace=[]
                try:value=ns['evaluate'](mode,trace);outcome='returned';message=str(value)
                except Exception as exc:outcome=type(exc).__name__;message=re.sub(r'0x[0-9a-f]+','0xADDR',str(exc))
                rows.append(dict(mode=mode,outcome=outcome,message=message,trace=trace))
            runs.append(rows)
        if runs[0]!=runs[1]:raise ValueError('cleanup changed assertion outcome, message, effects or owner retirement')
        summaries.append(dict(compile=stats,identical_rows=runs[0]))
    return dict(cases=17,all_outcomes_messages_and_side_effects_equal=True,
        original_chained_comparison_and_walrus_evaluations_retained=True,
        short_circuit_loop_and_temporary_owner_release_preserved=True,
        compiler_only_payment_not_HZ_gain=True,corpus=summaries,formal_gain=0)
