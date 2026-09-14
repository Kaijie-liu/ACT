"""Standard-library-only CLI for the supplied-HZ request proof checker.

Run with ``python -S scripts/check_moe_request_lp.py PACKAGE``. Bypass ACT's
eager package initializers (which import Torch), not the checker itself. Only
the pure checker modules are imported. This launcher never loads a model or
executes a proof producer; it is intentionally an isolated-process entry.
"""
from pathlib import Path
import runpy
import sys
import types


def main():
    root = Path(__file__).resolve().parents[1]
    packages = ('act', 'act.back_end', 'act.back_end.solver',
                'act.pipeline', 'act.pipeline.moe')
    for name in packages:
        if name in sys.modules:
            raise RuntimeError('checker launcher requires a fresh process')
        package = types.ModuleType(name)
        package.__path__ = [str(root.joinpath(*name.split('.')))]
        package.__package__ = name
        sys.modules[name] = package
    runpy.run_module('act.pipeline.moe.check_request_lp', run_name='__main__')


if __name__ == '__main__':
    main()
