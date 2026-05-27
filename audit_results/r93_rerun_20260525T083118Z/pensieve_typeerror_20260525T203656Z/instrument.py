"""Find the exact site that raises 'only integer tensors of a single element can be converted to an index'."""
import os, sys, traceback
sys.path.insert(0, '/data1/Kane/ACT')
sys.argv = ['act.pipeline', '--verify', 'vnnlib',
            '--category', 'nn4sys', '--instance-ids', '1',
            '--max-instances', '1', '--timeout', '15',
            '--device', 'cpu', '--dtype', 'float64', '--solvers', 'hybridz']
os.environ.setdefault('ACT_VNNLIB_ROOT', '/data1/Kane/data/vnncomp2025_benchmarks/benchmarks')
try:
    from act.pipeline.__main__ import main
    main()
except SystemExit:
    pass
except Exception as e:
    print(f"\nTOP-LEVEL: {type(e).__name__}: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
