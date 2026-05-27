import os, sys
sys.path.insert(0, '/data1/Kane/ACT')
import torch
_orig_tf_gather = None
def _traced_tf_gather(L, Bin):
    print(f"\n>>> tf_gather called id={L.id}", file=sys.stderr)
    print(f"  L.params input_shape={L.params.get('input_shape')} axis={L.params.get('axis')} indices_shape={tuple(L.params['indices'].shape) if hasattr(L.params['indices'], 'shape') else None}", file=sys.stderr)
    print(f"  Bin.lb.shape={tuple(Bin.lb.shape)} numel={Bin.lb.numel()}", file=sys.stderr)
    print(f"  L.in_vars[:8]={L.in_vars[:8]} (total {len(L.in_vars)})", file=sys.stderr)
    print(f"  L.out_vars={L.out_vars}", file=sys.stderr)
    return _orig_tf_gather(L, Bin)

from act.back_end.interval_tf import tf_mlp as _tfm
_orig_tf_gather = _tfm.tf_gather
_tfm.tf_gather = _traced_tf_gather

from act.back_end.interval_tf import interval_tf as _itf
# Re-patch the lambda registry
old_registry = _itf.IntervalTF._LAYER_REGISTRY
from act.back_end.layer_schema import LayerKind
old_registry[LayerKind.GATHER.value] = lambda L, bounds, tf: _traced_tf_gather(L, bounds)

sys.argv = ['act.pipeline', '--verify', 'vnnlib', '--category', 'lsnc_relu', '--max-instances', '1', '--timeout', '25', '--device', 'cpu', '--dtype', 'float64', '--solvers', 'hybridz']
os.environ.setdefault('ACT_VNNLIB_ROOT', '/data1/Kane/data/vnncomp2025_benchmarks/benchmarks')
from act.pipeline.__main__ import main
try:
    main()
except Exception as e:
    print(f"\nFINAL: {type(e).__name__}: {e}", file=sys.stderr)
