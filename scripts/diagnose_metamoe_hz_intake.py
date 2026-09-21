"""Old-smoke ROUTER propagation only: record exact HZ drop location, no solver."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recent_moe_deployment import sha256, supervise
from robust_experts_workflow_control import write


def run(config, root):
    started = time.monotonic()
    from metamoe_paired_execution_r2 import validate
    cfg = json.loads(config.read_text())
    validate(cfg)
    import numpy as np
    import torch
    from metamoe_paired_model import load_full
    from metamoe_functional_intake import adapted_class_separated
    from act.util.device_manager import initialize_device
    from act.back_end.moe.class_separated_top1 import classification_rows
    from act.back_end.moe.factory import build_act_moe_program
    from act.front_end.specs import OutKind, OutputSpec
    from act.back_end.hybridz_tf import HybridzTF
    from act.back_end.moe.route_a import _analyze_router
    from act.back_end.transfer_functions import set_transfer_function, set_solver_mode
    initialize_device('cpu', 'float64')
    torch.set_num_threads(2)
    request = next(r for r in cfg['requests'] if r['id'] == 'mnist_0')
    with np.load(request['tensor_file'], allow_pickle=False) as f:
        x, lo, hi = [torch.from_numpy(f[k].copy()) for k in ['center','lower','upper']]
    model = load_full(cfg['repo'], cfg['checkpoint'], cfg['files'][cfg['checkpoint']])
    adapter, _ = adapted_class_separated(model, x)
    q = classification_rows(adapter.total_classes, request['label'])
    program = build_act_moe_program(adapter.reduced_components(x), center=x, lower=lo, upper=hi,
        output_spec=OutputSpec(kind=OutKind.LINEAR_LE, c=-q,
            d=torch.full((len(q),), -cfg['margin'], dtype=torch.float64)))
    events = []
    class Recorded(HybridzTF):
        def apply(self, layer, *args, **kwargs):
            out = super().apply(layer, *args, **kwargs)
            hz = self.get_sparse_hz(layer.id)
            events.append({'id': layer.id, 'kind': layer.kind, 'outputs': len(layer.out_vars),
                'drop': self._sparse_drop_reasons.get(layer.id),
                'sparse': None if hz is None else {'n_cont': hz.n_cont, 'n_bin': hz.n_bin,
                    'n_out': hz.n_out}, 'dense_retained': self.get_hz(layer.id) is not None})
            return out
    tf = Recorded()
    set_transfer_function(tf)
    set_solver_mode('hybridz')
    failure = None
    try:
        _analyze_router(program.router, tf)
    except RuntimeError as e:
        failure = str(e)
    write(root/'diagnostic.json', {'config_sha256': sha256(config), 'events': events,
        'output_error': failure, 'sparse_affine_cell_cap': tf._SPARSE_MAX_AFFINE_CELLS,
        'seconds': time.monotonic()-started, 'new_solver_queries': 0,
        'scope': 'old MNIST0 representation diagnostic; no bound/feasibility solve'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--worker', action='store_true')
    a = p.parse_args()
    if a.worker:
        run(a.config, a.output)
    else:
        cfg = json.loads(a.config.read_text())
        receipt = supervise([cfg['python']['act'], str(Path(__file__).resolve()), '--config',
            str(a.config.resolve()), '--output', str(a.output), '--worker'],
            str(Path(__file__).resolve().parents[1]), a.output, 90, 'NO_SOLVER_INTAKE_DIAGNOSTIC')
        print(receipt['status'])
