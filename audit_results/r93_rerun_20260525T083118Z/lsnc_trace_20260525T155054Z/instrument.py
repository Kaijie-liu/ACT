import os, sys, torch, traceback
sys.path.insert(0, '/data1/Kane/ACT')
_orig_view = torch.Tensor.view
def _traced_view(self, *args):
    try:
        return _orig_view(self, *args)
    except RuntimeError as e:
        if "is invalid for input of size" in str(e):
            print(f"\n>>> VIEW FAIL: tensor numel={self.numel()} shape={tuple(self.shape)} -> args={args}", file=sys.stderr)
            traceback.print_stack(file=sys.stderr)
        raise
torch.Tensor.view = _traced_view
sys.argv = ['act.pipeline', '--verify', 'vnnlib', '--category', 'lsnc_relu', '--max-instances', '1', '--timeout', '25', '--device', 'cpu', '--dtype', 'float64', '--solvers', 'hybridz']
os.environ.setdefault('ACT_VNNLIB_ROOT', '/data1/Kane/data/vnncomp2025_benchmarks/benchmarks')
from act.pipeline.__main__ import main
main()
