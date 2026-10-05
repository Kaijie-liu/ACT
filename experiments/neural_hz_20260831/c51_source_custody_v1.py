"""Default-off, source-first compile custody for the complete test population.

Only authenticated source may mint an immutable expected-code receipt, before
module execution. Collection checks actual code and definition/collection
identity without repeating compilation. No cached test outcomes are accepted.
"""
import ast
import hashlib
from pathlib import Path
import sys
import types
from types import MappingProxyType
from typing import NamedTuple
from _pytest.fixtures import FixtureFunctionDefinition
from experiments.neural_hz_20260831.c50_assert_protocol_v1 import guard,code_image,code_key,code_tree,compiled
from experiments.neural_hz_20260831.c50_cleanup_compile_v1 import compile_cleanup


def function(value):
    if type(value) in (staticmethod,classmethod):value=value.__func__
    if type(value) is FixtureFunctionDefinition:
        actual=value._get_wrapped_function()
        if value.__wrapped__ is not actual:raise ValueError('fixture wrapper does not bind its executing function')
        value=actual
    if type(value) is types.MethodType:value=value.__func__
    if type(value) is not types.FunctionType:return None
    if hasattr(value,'__wrapped__'):raise ValueError('unproved custom function wrapper')
    return value


class Receipt(NamedTuple):
    source_sha256:str
    policy:str
    code_images:tuple
    stats:tuple


class Custody:
    def __init__(self,selected,source_manifest,root):
        guard();self.root=Path(root).resolve()
        self.selected=frozenset(Path(p).resolve() for p in selected)
        manifest={Path(p).resolve():sha for p,sha in source_manifest.items()}
        specs={};edges={};pending=list(sorted(self.selected))
        while pending:
            path=pending.pop(0)
            if path in specs:continue
            if path not in manifest:raise ValueError('definition source is absent from the frozen manifest')
            source=path.read_bytes()
            if hashlib.sha256(source).hexdigest()!=manifest[path]:raise ValueError('frozen definition source changed')
            tree=ast.parse(source,filename=str(path));imports=[]
            for node in ast.walk(tree):
                if isinstance(node,ast.Call) and ((isinstance(node.func,ast.Name) and node.func.id=='locals')
                        or (isinstance(node.func,ast.Attribute) and node.func.attr in ('currentframe','_getframe'))):
                    raise ValueError('private frame inspection is outside the cleanup proof')
                if isinstance(node,ast.Attribute) and node.attr in ('f_locals','tb_frame'):
                    raise ValueError('reflected scratch state is outside the cleanup proof')
                if isinstance(node,ast.ImportFrom) and any(a.name=='*' for a in node.names):
                    if node.level or not node.module:raise ValueError('unregistered relative star import')
                    dependency=(self.root/(node.module.replace('.','/')+'.py')).resolve()
                    if dependency not in manifest:raise ValueError('star import definition is not frozen')
                    imports.append((node.module,dependency));pending.append(dependency)
            specs[path]=(manifest[path],'cleanup' if path in self.selected else 'rewrite')
            edges[path]=tuple(imports)
        self.specs=MappingProxyType(specs);self.edges=MappingProxyType(edges)
        self._receipts={};self._images={};self._audited={};self._compiled={};self.compile_count=0

    def prepare(self,path,config=None):
        """Mint only from source before exec; never from a loaded function."""
        guard();path=Path(path).resolve()
        if path not in self.specs:raise ValueError('unregistered compilation source')
        sha,policy=self.specs[path];source=path.read_bytes()
        if hashlib.sha256(source).hexdigest()!=sha:raise ValueError('source changed before authenticated compilation')
        if config is not None and (config.getoption('assertmode')!='rewrite' or config.getini('enable_assertion_pass_hook')):
            raise ValueError('ordinary assertion evaluation is required')
        # Python can import one unchanged file under its top-level and package
        # names. Both module bodies still execute; only immutable compilation
        # is reused, after source/policy checks, never a module or test result.
        if path in self._compiled:
            self._receipt(path);return self._compiled[path]
        if policy=='cleanup':code,stats=compile_cleanup(source,str(path),config=config,enabled=True)
        else:
            # Same original rewriter on unselected imported definitions; no
            # cleanup-policy expansion to a dependency just to pass an audit.
            from _pytest.assertion.rewrite import rewrite_asserts
            tree=ast.parse(source,filename=str(path));count=sum(isinstance(n,ast.Assert) for n in ast.walk(tree))
            rewrite_asserts(tree,source,str(path),config)
            code=compile(tree,str(path),'exec',dont_inherit=True,optimize=0)
            stats=dict(source_assert_statements=count,original_assertion_rewriting_retained=True)
        images=tuple((code_key(c),code_image(c)) for c in code_tree(code))
        self._receipts[path]=Receipt(sha,policy,images,tuple(sorted(stats.items())))
        # Several comprehensions/lambdas can share a line and qualname.
        # The full recursive code image is part of the key, not discarded.
        self._images[path]=frozenset(images);self.compile_count+=1
        self._compiled[path]=code
        return code

    def _receipt(self,path):
        if path not in self._receipts:raise ValueError('no pre-execution source receipt; old cache is not authority')
        receipt=self._receipts[path]
        if (receipt.source_sha256,receipt.policy)!=self.specs[path]:raise ValueError('source or compiler policy receipt changed')
        if frozenset(receipt.code_images)!=self._images[path]:raise ValueError('immutable code image custody changed')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=receipt.source_sha256:raise ValueError('definition source changed after loading')
        return receipt

    def audit_module(self,module,path):
        guard();path=Path(path).resolve();receipt=self._receipt(path)
        if Path(module.__file__).resolve()!=path:raise ValueError('actual definition module has the wrong file')
        functions=[];classes=set()
        def visit(namespace):
            for value in namespace.values():
                # Imported library functions belong to their frozen library,
                # not to this test source's compiler-policy transaction.
                if getattr(value,'__module__',None)!=module.__name__:continue
                fn=function(value)
                if fn is not None and fn.__module__==module.__name__:functions.append(fn)
                elif isinstance(value,type) and value.__module__==module.__name__ and id(value) not in classes:
                    classes.add(id(value));visit(vars(value))
        visit(vars(module));functions=list({id(fn):fn for fn in functions}.values())
        if not functions:raise ValueError('no actual definition code was checked')
        images=[]
        for fn in functions:
            if fn.__globals__ is not vars(module) or Path(fn.__code__.co_filename).resolve()!=path:
                raise ValueError('function globals or definition path changed')
            image=code_image(fn.__code__)
            if (code_key(fn.__code__),image) not in self._images[path]:
                raise ValueError('loaded code differs from authenticated source compilation')
            images.append(image)
        # Code objects and their constants are immutable. Retaining the
        # checked object lets repeated parametrized items verify its identity
        # without hashing identical code again; each callable's namespace and
        # code binding are still checked. No test result is cached here.
        self._audited[path,id(module)]=(module,{id(fn):(fn,fn.__code__) for fn in functions})
        return dict(file=str(path),source_sha256=receipt.source_sha256,policy=receipt.policy,
            functions=len(functions),compile=dict(receipt.stats),
            actual_loaded_code_image=hashlib.sha256('\n'.join(sorted(images)).encode()).hexdigest(),
            all_loaded_test_and_helper_code_checked=True,collection_recompilations=0)

    def definition(self,collection_module,collection_path,obj,name,modules=None):
        """Bind the selected callable, not merely its claimed name/filename."""
        collection_path=Path(collection_path).resolve();modules=sys.modules if modules is None else modules
        if collection_path not in self.selected:raise ValueError('unregistered collecting source')
        if Path(collection_module.__file__).resolve()!=collection_path:raise ValueError('wrong collecting module')
        fn=function(obj)
        if fn is None:raise ValueError('unknown selected callable')
        if fn.__name__!=name:raise ValueError('selected name was rebound to a different source function')
        if function(getattr(collection_module,name,None)) is not fn:raise ValueError('selected item is not the collected callable')
        path=Path(fn.__code__.co_filename).resolve()
        if path==collection_path:module=collection_module
        else:
            admitted=dict((p,n) for n,p in self.edges[collection_path])
            if path not in admitted or admitted[path]!=fn.__module__ or name.startswith('_'):
                raise ValueError('selected definition is not a declared frozen import')
            module=modules.get(admitted[path])
            if module is None or Path(module.__file__).resolve()!=path:raise ValueError('missing or wrong imported definition module')
            if function(getattr(module,name,None)) is not fn:raise ValueError('imported callable was substituted')
        if fn.__globals__ is not vars(module) or fn.__module__!=module.__name__:
            raise ValueError('selected callable does not execute in its source namespace')
        for origin,namespace in ((collection_path,collection_module),(path,module)):
            if (origin,id(namespace)) not in self._audited:self.audit_module(namespace,origin)
        checked=self._audited[path,id(module)][1].get(id(fn))
        if checked is None or checked[0] is not fn or checked[1] is not fn.__code__:
            raise ValueError('selected test code binding changed after its complete source audit')
        return path,module


def custody(selected,source_manifest,root,*,enabled=False):
    if not enabled:return None
    return Custody(selected,source_manifest,root)
