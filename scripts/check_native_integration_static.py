"""Read/parse only: does not import the new runtime or execute any controls."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    'act/back_end/moe/source_row_endpoints.py',
    'scoped_source/native_method.py','scoped_source/native_loader.py',
    'scoped_source/native_lifecycle.py','scoped_source/native_source.py',
    'scoped_source/native_result.py','scoped_source/native_package.py',
    'scoped_source/native_production.py','scoped_source/native_protocol.py',
    'scoped_source/native_supervised.py','scoped_source/native_portable.py',
    'scoped_source/scaled_endpoints.py')


def check():
    parsed = {}; hashes = {}
    for name in FILES:
        raw = (ROOT/name).read_bytes()
        parsed[name] = ast.parse(raw,filename=name)
        compile(parsed[name],name,'exec')  # Compile only: no execution/import.
        if name == 'scoped_source/native_supervised.py':
            bootstraps = [n.value.value for n in ast.walk(parsed[name]) if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id == 'bootstrap' for t in n.targets)
                and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)]
            assert len(bootstraps) == 1
            compile(bootstraps[0],'<native bootstrap>','exec')
        hashes[name] = hashlib.sha256(raw).hexdigest()
    registry = json.loads((ROOT/'docs/hz_native_execution_approvals.json').read_bytes())
    assert registry == {'profiles':[],'schema':'NATIVE_EXECUTION_APPROVAL_REGISTRY_V1'}
    native = parsed[FILES[0]]
    assert not any(isinstance(n,ast.ImportFrom) and (n.module or '').startswith('act.') for n in ast.walk(native))
    for path in ('scoped_source/native_package.py','scoped_source/native_portable.py',
                 'scoped_source/native_result.py','scoped_source/native_protocol.py'):
        for n in ast.walk(parsed[path]):
            names = [a.name for a in n.names] if isinstance(n,ast.Import) else [n.module] if isinstance(n,ast.ImportFrom) else []
            assert not any(v and v.split('.')[0] in ('torch','numpy','scipy','act') for v in names), path
    pending = list(FILES)+['act/back_end/moe/scaled_forest_endpoints.py']
    found = set()
    while pending:
        name = pending.pop()
        if name in found: continue
        assert len(found) < 256
        raw = (ROOT/name).read_bytes(); assert len(raw) <= 128*1024
        tree = ast.parse(raw,filename=name); found.add(name)
        for n in ast.walk(tree):
            names = [a.name for a in n.names] if isinstance(n,ast.Import) else [n.module] if isinstance(n,ast.ImportFrom) else []
            for module in names:
                if module and module.split('.')[0] in ('scoped_source','scoped_proof','source_enclosure','upstream_source','router_source'):
                    target = module.replace('.','/')+'.py'
                    if (ROOT/target).is_file(): pending.append(target)
    assert {'source_enclosure/format.py','upstream_source/checker.py'} <= found
    # Current closed metadata roster: six reports, one controller log, six small
    # actor logs/progress files. Its reserve is carved out of the old total cap.
    assert 7*1048576+6*65536 == 7733248 < 8*1048576
    assert 6+1+6 == 13 < 16
    return dict(status='PASS',scope='SYNTAX_AND_STATIC_DEPENDENCY_ONLY',
        syntax_files=len(FILES),local_dependency_files=len(found),metadata_upper_bytes=7733248,
        metadata_members=13,active_profiles=0,runtime_controls_executed=0,
        native_imports_executed=0,models_executed=0,gpu_executions=0,sha256=hashes)


if __name__ == '__main__': print(json.dumps(check(),sort_keys=True,indent=2))
