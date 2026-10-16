"""Load only two frozen native files, without importing ACT package initializers.

This is a production-only loader, invoked after the supervisor's hard limits.
It changes no existing module or global backend function. Private module objects
are retained solely to give the owned dual handle one stable concrete type.
"""
import importlib.util
from pathlib import Path

from scoped_source.hz_request_store import require, digest, sha

REFERENCE = 'act/back_end/moe/scaled_forest_endpoints.py'
BLOCKS = 'act/back_end/moe/source_row_endpoints.py'
FILES = (REFERENCE, BLOCKS)
_reference = _blocks = _binding = None


def reference_operator():
    require(_reference is not None, 'reference loaded before optional block module')
    return _reference.Operator


def load(root, expected, work):
    global _reference, _blocks, _binding
    root = Path(root)
    require(root.is_absolute() and root.resolve() == root and type(expected) is dict,
            'canonical frozen native implementation')
    binding = (str(root), tuple((name, sha(expected[name])) for name in FILES))
    if _binding is not None:
        require(_binding == binding and _blocks is not None, 'one native implementation per actor')
        work.tick()
        return _blocks
    # Read and check the actual bytes to execute, not hash then reopen the file.
    sources = []
    for name in FILES:
        work.tick(); path = root/name
        require(all(not p.is_symlink() for p in (path, *path.parents))
                and path.is_file() and path.stat().st_nlink == 1
                and path.stat().st_size <= 64*2**10, 'bounded unaliased native code')
        with path.open('rb') as stream: raw = stream.read(64*2**10+1)
        require(len(raw) <= 64*2**10 and digest(raw) == expected[name], 'native code hash')
        sources.append((path, raw))
    require(_reference is None and _blocks is None, 'failed loader cannot be retried')
    _binding = binding
    for i, (path, raw) in enumerate(sources):
        spec = importlib.util.spec_from_file_location(f'_act_moe_native_v1_{i}', path)
        require(spec is not None, 'native standalone module spec')
        module = importlib.util.module_from_spec(spec)
        exec(compile(raw, str(path), 'exec'), module.__dict__)
        work.tick()
        if i == 0: _reference = module
        else: _blocks = module
    return _blocks
