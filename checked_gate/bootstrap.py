"""Load only pure ACT proof modules, avoiding eager model initializers."""
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]


def setup(checker=False):
    if checker:
        if not sys.flags.no_site:
            raise ValueError('checker requires python -S')
        def guard(event, args):
            if event == 'import' and args[0].split('.')[0] in {
                    'torch', 'numpy', 'scipy', 'highspy', 'gurobipy', 'decimal'}:
                raise ImportError('forbidden independent-checker dependency')
            if event.startswith(('subprocess.', 'socket.')) or event in {'os.system', 'os.fork', 'os.exec'}:
                raise PermissionError('external execution forbidden in checker')
        sys.addaudithook(guard)
    for name in ('act', 'act.back_end', 'act.back_end.solver', 'act.pipeline', 'act.pipeline.moe'):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(ROOT.joinpath(*name.split('.')))]
            sys.modules[name] = module
