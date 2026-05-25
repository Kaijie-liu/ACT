"""Monkey-patch F.conv2d to log shapes before each call so we find the
exact site that produces input (21x8) vs kernel (6x20) for collins."""
import os, sys, traceback
sys.path.insert(0, '/data1/Kane/ACT')
import torch
import torch.nn.functional as F

_orig = F.conv2d
def _traced(input, weight, *args, **kwargs):
    in_shape = tuple(input.shape)
    w_shape = tuple(weight.shape)
    # Only print suspicious calls: kernel spatial > input spatial
    if len(in_shape) == 4 and len(w_shape) == 4:
        in_h, in_w = in_shape[2], in_shape[3]
        k_h, k_w = w_shape[2], w_shape[3]
        if k_h > in_h or k_w > in_w:
            print(f"\n>>> SUSPECT CONV2D: input={in_shape} weight={w_shape}", file=sys.stderr)
            traceback.print_stack(file=sys.stderr)
    return _orig(input, weight, *args, **kwargs)
F.conv2d = _traced

# Now run the CLI
sys.argv = ['act.pipeline', '--verify', 'vnnlib',
            '--category', 'collins_rul_cnn_2022',
            '--max-instances', '1', '--timeout', '30',
            '--device', 'cpu', '--dtype', 'float64', '--solvers', 'hybridz']
os.environ.setdefault('ACT_VNNLIB_ROOT', '/data1/Kane/data/vnncomp2025_benchmarks/benchmarks')
from act.pipeline.__main__ import main
main()
