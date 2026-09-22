"""R4 OLD-input representation-only diagnostic; no paired/cohort entry point."""
import argparse
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recent_moe_deployment import sha256, git_identity
from recent_moe_env_inventory import inventory
from robust_experts_workflow_control import write
from metamoe_csr_execution import supervise


def contract(cfg):
    if (cfg['protocol'] != 'metamoe_csr_spatial_r4_old_input' or cfg['seconds'] != 90 or
            cfg['group_rss_limit_bytes'] != 8*2**30 or
            cfg['hybridz'] != {'sparse_resource_policy': 'csr_spatial_v2', 'sparse_representation_bytes': 2**31} or
            cfg['epsilon'] != 2/255 or cfg['margin'] != 1e-7 or
            len(cfg['requests']) != 1 or cfg['requests'][0]['id'] != 'mnist_0' or
            cfg['requests'][0]['index'] != 0 or cfg['requests'][0]['dataset'] != 'MNIST'):
        raise ValueError('R4 old-input/resource contract changed')


def validate(cfg):
    contract(cfg)
    parent_path = Path('configs/recent_moe/metamoe_csr_smoke_r3.json')
    if sha256(parent_path) != cfg['parent_config_sha256']:
        raise ValueError('R3 parent binding')
    parent = json.loads(parent_path.read_text())
    if cfg['requests'] != [next(r for r in parent['requests'] if r['id'] == 'mnist_0')]:
        raise ValueError('old physical tensor/request changed')
    for field in ('python', 'environment', 'repositories', 'checkpoint', 'repo', 'backend_repo'):
        if cfg[field] != parent[field]:
            raise ValueError('inherited execution identity changed: '+field)
    for name, digest in cfg['files'].items():
        if sha256(name) != digest:
            raise ValueError('frozen source/input drift: '+name)
    for repo, digest in cfg['repositories'].items():
        state = git_identity(repo)
        if state['head'] != digest or state['status']:
            raise ValueError('author source changed')
    if inventory(list(cfg['python'].values())) != cfg['environment']:
        raise ValueError('environment changed')


def worker(config):
    started = time.monotonic()
    cfg = json.loads(config.read_text())
    validate(cfg)
    root = Path(cfg['output_root'])
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
    request = cfg['requests'][0]
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
            first = len(self.sparse_resource_events())
            try:
                return super().apply(layer, *args, **kwargs)
            finally:
                hz = self.get_sparse_hz(layer.id)
                event = {'phase': phase, 'id': layer.id, 'kind': layer.kind,
                    'seconds': time.monotonic()-began, 'outputs': len(layer.out_vars),
                    'drop': self._sparse_drop_reasons.get(layer.id),
                    'dense_retained': self.get_hz(layer.id) is not None,
                    'sparse': None if hz is None else storage(hz),
                    'admission_start': first, 'admission_end': len(self.sparse_resource_events())}
                events.append(event)
                write(root/f'layer_{len(events):03d}.json', event)
    tf = Recorded(HybridZConfig(**cfg['hybridz']))
    set_transfer_function(tf)
    set_solver_mode('hybridz')
    state, failure = 'COMPLETE', None
    with (patch('act.back_end.solver.solver_hz.milp', side_effect=AssertionError('no solver in intake')) as milp,
          patch('act.back_end.solver.solver_hz.linprog', side_effect=AssertionError('no solver in intake')) as lp,
          patch.object(sp.csr_matrix, 'toarray', side_effect=AssertionError('no densification'))):
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
        'seconds': time.monotonic()-started, 'solver_calls': milp.call_count+lp.call_count,
        'scope': 'old MNIST0 full router and two guarded experts; no property or feasibility query'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--worker', action='store_true')
    args = p.parse_args()
    if args.worker:
        worker(args.config)
    else:
        cfg = json.loads(args.config.read_text())
        contract(cfg)
        receipt = supervise([cfg['python']['act'], str(Path(__file__).resolve()), '--config',
            str(args.config.resolve()), '--worker'], str(Path(__file__).resolve().parents[1]),
            Path(cfg['output_root']), cfg['seconds'], cfg['group_rss_limit_bytes'])
        print(receipt['status'])
