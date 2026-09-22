"""Old MNIST0 no-solver CSR/router/guarded-expert admission, bounded at 90s."""
import argparse
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metamoe_csr_execution import supervise, resource_config
from metamoe_paired_execution_r2 import validate
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write


def run(config, root):
    start = time.monotonic()
    cfg = json.loads(config.read_text())
    resource_config(cfg)
    validate(cfg)
    import numpy as np
    import scipy.sparse as sp
    import torch
    from metamoe_paired_model import load_full
    from metamoe_functional_intake import adapted_class_separated
    from act.util.device_manager import initialize_device
    from act.back_end.moe.class_separated_top1 import classification_rows
    from act.back_end.moe.factory import build_act_moe_program
    from act.front_end.specs import OutKind, OutputSpec
    from act.back_end.hybridz_tf import HybridzTF
    from act.back_end.hybridz_tf.sparse_budget import SparseResourceLimit, storage
    from act.back_end.moe.route_a import _analyze_router
    from act.back_end.moe.hz_routing import guarded_input_domain
    from act.back_end.transfer_functions import set_transfer_function, set_solver_mode
    from act.config.config import HybridZConfig
    initialize_device('cpu', 'float64')
    torch.set_num_threads(2)
    request = next(r for r in cfg['requests'] if r['id'] == 'mnist_0')
    with np.load(request['tensor_file'], allow_pickle=False) as f:
        x, lo, hi = [torch.from_numpy(f[k].copy()) for k in ['center', 'lower', 'upper']]
    model = load_full(cfg['repo'], cfg['checkpoint'], cfg['files'][cfg['checkpoint']])
    adapter, _ = adapted_class_separated(model, x)
    q = classification_rows(adapter.total_classes, request['label'])
    program = build_act_moe_program(adapter.reduced_components(x), center=x, lower=lo, upper=hi,
        output_spec=OutputSpec(kind=OutKind.LINEAR_LE, c=-q,
            d=torch.full((len(q),), -cfg['margin'], dtype=torch.float64)))
    events, phase, completed = [], 'router', []
    class Recorded(HybridzTF):
        def apply(self, layer, *args, **kwargs):
            began = time.monotonic()
            try:
                return super().apply(layer, *args, **kwargs)
            finally:
                hz = self.get_sparse_hz(layer.id)
                event = {'phase': phase, 'id': layer.id, 'kind': layer.kind,
                         'seconds': time.monotonic()-began, 'outputs': len(layer.out_vars),
                         'drop': self._sparse_drop_reasons.get(layer.id),
                         'dense_retained': self.get_hz(layer.id) is not None,
                         'sparse': None if hz is None else storage(hz),
                         'last_admission': self.sparse_resource_events()[-1:]}
                events.append(event)
                # Immutable progress survives later process-group termination.
                write(root/f'layer_{len(events):03d}.json', event)
    tf = Recorded(HybridZConfig(**cfg['hybridz']))
    set_transfer_function(tf)
    set_solver_mode('hybridz')
    failure, state = None, 'COMPLETE'
    with (patch('act.back_end.solver.solver_hz.milp', side_effect=AssertionError('no solver in intake')) as milp,
          patch('act.back_end.solver.solver_hz.linprog', side_effect=AssertionError('no solver in intake')) as lp,
          patch.object(sp.csr_matrix, 'toarray', side_effect=AssertionError('no CSR densification'))):
        try:
            router_hz, input_hz = _analyze_router(program.router, tf)
            completed.append('router')
            for i, expert in enumerate(program.experts):
                phase = f'guarded_expert_{i}'
                guard = guarded_input_domain(input_hz, router_hz, i, top_k=1)
                tf.set_entry_hz(guard.hz)
                _analyze_router(expert, tf)
                completed.append(phase)
        except SparseResourceLimit as exc:
            state, failure = 'RESOURCE_REFUSED', exc.event
        except Exception as exc:
            state, failure = 'ERROR', repr(exc)
    write(root/'diagnostic.json', {'config_sha256': sha256(config), 'status': state,
        'completed_phases': completed, 'failure_phase': phase if failure else None,
        'failure': failure, 'events': events, 'admissions': tf.sparse_resource_events(),
        'seconds': time.monotonic()-start, 'solver_calls': milp.call_count+lp.call_count,
        'scope': 'old MNIST0 representation and both guarded experts; no feasibility or property query'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        run(args.config, args.output)
    else:
        cfg = json.loads(args.config.read_text())
        receipt = supervise([cfg['python']['act'], str(Path(__file__).resolve()), '--config',
            str(args.config.resolve()), '--output', str(args.output.resolve()), '--worker'],
            str(Path(__file__).resolve().parents[1]), args.output, 90, cfg['group_rss_limit_bytes'])
        print(receipt['status'])
