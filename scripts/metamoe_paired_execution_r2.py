"""R2 functional-ReLU intake and symmetric original-center replay; same300s."""
import argparse
from decimal import Decimal
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recent_moe_deployment import sha256, supervise, git_identity
from recent_moe_env_inventory import inventory
from robust_experts_workflow_control import write


def spec_text(lower, upper, outputs):
    rows = [f'(declare-const X_{i} Real)' for i in range(lower.size)]
    rows += [f'(declare-const Y_{i} Real)' for i in range(outputs)]
    for i, (lo, hi) in enumerate(zip(lower.reshape(-1), upper.reshape(-1))):
        rows += [f'(assert (>= X_{i} {Decimal.from_float(float(lo)):f}))',
                 f'(assert (<= X_{i} {Decimal.from_float(float(hi)):f}))']
    rows += ['(assert (or ' + ' '.join(f'(and (>= Y_0 Y_{i}))' for i in range(1, outputs)) + '))']
    return '\n'.join(rows)+'\n'


def validate(cfg):
    for file, h in cfg['files'].items():
        if sha256(file) != h:
            raise ValueError('frozen identity: '+file)
    for repo, digest in cfg['repositories'].items():
        state = git_identity(repo)
        if state['head'] != digest or state['status']:
            raise ValueError('author/backend source drift: '+repo)
    if inventory(list(cfg['python'].values())) != cfg['environment']:
        raise ValueError('execution environment changed')
    if cfg['seconds'] != 300 or cfg['epsilon'] != 2/255 or cfg['margin'] != 1e-7:
        raise ValueError('request protocol changed')


def worker(cfg, path, request_id, arm):
    start = time.monotonic()
    validate(cfg)
    import numpy as np
    import torch
    from metamoe_paired_model import load_full, InvariantObligations
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(100)
    request = next(r for r in cfg['requests'] if r['id'] == request_id)
    root = Path(cfg['output_root']) / f'{request_id}_{arm}'
    with np.load(request['tensor_file'], allow_pickle=False) as arrays:
        tensor = {k: torch.from_numpy(arrays[k].copy()) for k in arrays.files}
    x, lo, hi = (tensor[k] for k in ['center', 'lower', 'upper'])
    model = load_full(cfg['repo'], cfg['checkpoint'], cfg['files'][cfg['checkpoint']])
    with torch.no_grad():
        clean, scores = model(x)
    if not torch.isfinite(clean).all() or not torch.isfinite(scores).all():
        raise ValueError('undefined clean output')
    label, route = request['label'], int(scores.argmax(1))
    common = {'request_id': request_id, 'arm': arm, 'config_sha256': sha256(path),
              'tensor_file_sha256': sha256(request['tensor_file']), 'label': label,
              'clean_prediction': int(clean.argmax(1)), 'clean_route': route,
              'numerical_object': 'same stored coefficients lifted to float64 real abstraction; not source-float proof'}
    write(root / 'prepared.json', common)
    from metamoe_functional_intake import center_witness
    early = center_witness(model, x, lo, hi, label, cfg['margin'])
    if early:
        early.update(common, worker_seconds=time.monotonic()-start)
        write(root / 'result.json', early)
        return
    if arm == 'act':
        from act.util.device_manager import initialize_device
        from act.back_end.moe.class_separated_top1 import ClassSeparatedTop1, classification_rows, verify_class_separated_box
        initialize_device('cpu', 'float64')
        from metamoe_functional_intake import adapted_class_separated
        adapted, mapping = adapted_class_separated(model, x)
        write(root / 'functional_rewrite.json', {'mapping': mapping,
            'rule': 'functional non-inplace ReLU to module ReLU; no BN folding',
            'finite_probe_is_not_domain_equivalence_proof': True})
        rows = classification_rows(model.total_classes, label)
        out = verify_class_separated_box(adapted, center=x, lower=lo, upper=hi,
            rows=rows, thresholds=torch.full((len(rows),), cfg['margin'], dtype=torch.float64),
            total_seconds=max(.001, cfg['seconds']-(time.monotonic()-start)))
    else:
        # The sufficient route-invariance path includes nonzero division and
        # zero-filled OTHER-domain outputs, not just a local expert label.
        sign = 1 if float(scores[0, route]) > 0 else -1
        joint = InvariantObligations(model.meta_gating_net, model.experts[route],
                    model.num_classes_list, route, label, sign, cfg['margin'])
        with torch.no_grad():
            probe = joint(x)
        write(root / 'obligation_identity.json', {'route': route, 'sign': sign,
            'output_rows': model.total_classes-1, 'router_rows': model.num_experts,
            'probe': probe.tolist(), 'checkpoint_sha256': cfg['files'][cfg['checkpoint']]})
        spec = root / 'request.vnnlib'
        spec.write_text(spec_text(lo.numpy(), hi.numpy(), probe.shape[1]))
        csv = root / 'instances.csv'
        csv.write_text(str(spec)+'\n')
        sys.path.insert(0, str(Path(cfg['repo']) / 'src/Formal_Neural_Network_Verification/alpha-beta-crown'))
        sys.path.insert(0, str(Path(cfg['repo']) / 'src/Vision_Transformer_Pytorch'))
        from verify_expert_abcrown import create_verification_config
        import yaml
        dest = root / 'backend.yaml'
        create_verification_config(cfg['checkpoint'], str(root / 'unused.onnx'), request['dataset'],
            cfg['epsilon'], 1, max(.001, cfg['seconds']-(time.monotonic()-start)), str(dest),
            None, None, vnnlib_dir=str(root), csv_file=str(csv))
        backend = yaml.safe_load(dest.read_text())
        backend['model'].pop('onnx_path')
        loader = str(Path(__file__).with_name('metamoe_paired_model.py').resolve())
        backend['model']['name'] = "Customized(%r, 'load_obligations', %r, %r, %r, %r)" % (
            loader, str(path.resolve()), request_id, route, sign)
        backend['general'].update(device='cpu', seed=100, double_fp=True,
                                  results_file=str(root / 'backend_results.pkl'))
        dest.write_text(yaml.safe_dump(backend, sort_keys=False))
        began = time.monotonic()
        with (root / 'backend.stdout').open('x') as stdout, (root / 'backend.stderr').open('x') as stderr:
            code = subprocess.run([sys.executable, str(Path(cfg['backend_repo']) / 'complete_verifier/abcrown.py'),
                '--config', str(dest)], cwd=root, stdout=stdout, stderr=stderr).returncode
        from metamoe_component_control import parse_backend_result
        out = parse_backend_result((root / 'backend.stdout').read_text(), code, time.monotonic()-began,
                                   cfg['seconds']-(began-start))
        if out['status'] == 'BACKEND_UNSAFE_UNREPLAYED':
            out.update(status='UNKNOWN', reason='component counterexample is not a full-model witness')
        if out['status'] == 'BACKEND_POSITIVE':
            out['evidence_grade'] = 'AUTHOR_BACKEND_NUMERICAL_SUFFICIENT_FILTER'
    out.update(common, worker_seconds=time.monotonic()-start)
    write(root / 'result.json', out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--worker')
    p.add_argument('--arm', choices=['act', 'author'])
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    if a.worker:
        worker(cfg, a.config, a.worker, a.arm)
    else:
        validate(cfg)
        root = Path(cfg['output_root'])
        root.mkdir(parents=True, exist_ok=False)
        rows, blocked = [], False
        for rank, request in enumerate(cfg['requests']):
            for arm in (['act', 'author'] if rank % 2 == 0 else ['author', 'act']):
                if blocked:
                    rows.append({'id': request['id'], 'arm': arm, 'status': 'NOT_STARTED_AFTER_ERROR'})
                    continue
                folder = root / f"{request['id']}_{arm}"
                receipt = supervise([cfg['python'][arm], str(Path(__file__).resolve()), '--config', str(a.config.resolve()),
                    '--worker', request['id'], '--arm', arm], str(Path(__file__).resolve().parents[1]), folder,
                    cfg['seconds'], 'MATCHED_FULL_MODEL_REQUEST')
                result = json.loads((folder / 'result.json').read_text()) if (folder / 'result.json').exists() else None
                status = result['status'] if receipt['status'] == 'COMPLETED' and result else (
                    'ERROR' if receipt['status'] == 'COMPLETED' else receipt['status'])
                row = {'id': request['id'], 'arm': arm, 'status': status,
                    'seconds': receipt['execution_including_preflight_seconds'],
                    'receipt_sha256': sha256(folder / 'receipt.json'),
                    'result_sha256': sha256(folder / 'result.json') if result else None}
                write(folder / 'terminal.json', row)
                rows.append(row)
                blocked = status in ['ERROR', 'SOURCE_CHANGED']
        write(root / 'summary.json', {'config_sha256': sha256(a.config), 'rows': rows})

