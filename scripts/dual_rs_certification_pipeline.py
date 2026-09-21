"""Final-epoch-only two-stage native Smooth pilot, with count audit in budget.

Author certify/_sample_noise implementations remain unchanged. The subclass only
records completed count batches. This is a probabilistic L2 result about smoothed
functions under the author's numerical path, never deterministic HZ/formal SAFE.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from dual_rs_certification_evidence import combine, from_counts, recording_smooth, summarize
from recent_moe_deployment import sha256, supervise
from dual_rs_training_control import validate_config, write_json


def validate(cfg):
    fixed = {'test_indices': [0, 1], 'n0': 100, 'n': 10000, 'alpha': .0005,
             'selector_sigma': 1., 'sigma_candidates': [.25, .5, 1.],
             'batch_size': 16, 'total_seconds': 7200, 'epoch': 90}
    if any(cfg[k] != v for k, v in fixed.items()):
        raise ValueError('frozen certification recipe differs')
    if cfg['schema'] != 'dual_rs_final_epoch_certification_execution_v1':
        raise ValueError('recipe-only document cannot execute')
    for path, h in cfg['files'].items():
        if sha256(path) != h:
            raise ValueError('certification identity: ' + path)
    parent = json.loads(Path(cfg['training_parent']).read_text())
    validate_config(parent)
    train = json.loads(Path(cfg['training_config']).read_text())
    root = Path(train['output_root'])
    terminal = json.loads((root / 'outer_terminal.json').read_text())
    audit = json.loads((root / 'audit.json').read_text())
    info = json.loads((root / 'training/epoch090.json').read_text())
    if (terminal['status'] != 'TRAINING_LANDED' or audit['audit'] != 'PASS'
            or audit['epochs'] != 90 or audit['config_sha256'] != sha256(cfg['training_config'])
            or info['file_sha256'] != sha256(cfg['selector_checkpoint'])
            or Path(cfg['selector_checkpoint']) != root / 'training/epoch090.pt'):
        raise ValueError('final-epoch audit gate')
    return parent, train, info


def worker(cfg, config):
    start = time.monotonic()
    parent, train, info = validate(cfg)
    import numpy as np
    import torch
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 24 * 1024**3:
        raise ValueError('24GiB GPU gate failed; no fallback/retry')
    torch.cuda.set_per_process_memory_fraction(.25)
    sys.path.insert(0, str(Path(parent['author_repo']) / 'code'))
    from architectures import get_architecture
    from DRM_sigma_est import DiffusionModel
    from DRM_classifier import DiffusionRobustModel
    from certify_sigma_est import CertifyModel
    from core import Smooth
    from dual_rs_epoch_pipeline import check_epoch_state
    state = torch.load(cfg['selector_checkpoint'], map_location='cpu', weights_only=True)
    check_epoch_state(state, info['binding'], 90, 90 * 2 * math.ceil(train['train_count']/256))
    selector = get_architecture('cifar_resnet110', 'cifar10', 3, False)
    selector.load_state_dict(state['model'], strict=True)
    selector_model = CertifyModel(DiffusionModel(parent['diffusion']), selector).eval()
    classifier_model = DiffusionRobustModel(parent['diffusion'], cfg['vit']).eval()
    # Checks before argmax; no fallback/clamp/dtype/forward replacement.
    def finite(_model, _inputs, out):
        if not torch.isfinite(out).all():
            raise ValueError('nonfinite Monte Carlo logits')
    selector_model.register_forward_hook(finite)
    classifier_model.register_forward_hook(finite)
    dataset = CIFAR10(parent['data_root'], train=False, download=False, transform=ToTensor())
    root = Path(cfg['output_root'])
    write_json(root / 'prepared.json', {'config_sha256': sha256(config),
        'setup_seconds': time.monotonic() - start,
        'selector_checkpoint_sha256': sha256(cfg['selector_checkpoint']),
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0)})

    def run_stage(x, model, diffusion, sigma, classes, index, stage, seed):
        t = 0
        while diffusion.sqrt_one_minus_alphas_cumprod[t] / diffusion.sqrt_alphas_cumprod[t] < 2*sigma:
            t += 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        began = time.monotonic()
        prefix = f'input{index:05d}_{stage}'
        write_json(root / (prefix + '_started.json'), {'index': index, 'stage': stage,
            'seed': seed, 'sigma': sigma, 'timestep': t, 'monotonic': began})
        def publish(number, counts):
            write_json(root / (prefix + f'_counts{number}.json'), {
                'config_sha256': sha256(config), 'index': index, 'stage': stage,
                'count_number': number, 'counts': counts,
                'cumulative_stage_seconds': time.monotonic() - began})
        RecordedSmooth = recording_smooth(Smooth, cfg['n0'], cfg['n'], classes, cfg['batch_size'], publish)
        smoothed = RecordedSmooth(model, classes, sigma, t)
        with torch.no_grad(), torch.cuda.amp.autocast():
            prediction, radius = smoothed.certify(x, cfg['n0'], cfg['n'], cfg['alpha'], cfg['batch_size'])
        samples = smoothed.samples
        if len(samples) != 2:
            raise ValueError('incomplete native certification')
        checked = from_counts(*samples, cfg['n0'], cfg['n'], classes, sigma, cfg['alpha'])
        if prediction != checked['prediction'] or not math.isclose(radius, checked['radius_l2'], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('native/count result mismatch')
        result = {**checked, 'index': index, 'stage': stage, 'classes': classes,
                  'sigma': sigma, 'seed': seed, 'timestep': t, 'selection_counts': samples[0],
                  'estimation_counts': samples[1], 'stage_seconds': time.monotonic() - began}
        write_json(root / (prefix + '.json'), result)
        return result

    for index in cfg['test_indices']:
        began = time.monotonic()
        x, label = dataset[index]
        input_hash = hashlib.sha256(x.numpy().tobytes()).hexdigest()
        x = x.cuda()
        # Independent predetermined stage streams; never reuse selector samples
        # to select/estimate the chosen classifier. Counts audit is not RNG proof.
        s = run_stage(x, selector_model, selector_model.denoiser.diffusion,
                      cfg['selector_sigma'], 3, index, 'selector', cfg['seeds'][str(index)][0])
        c = None
        if s['prediction'] != -1:
            sigma = cfg['sigma_candidates'][s['prediction']]
            c = run_stage(x, classifier_model, classifier_model.diffusion,
                          sigma, 10, index, 'classifier', cfg['seeds'][str(index)][1])
        result = {'config_sha256': sha256(config), 'index': index, 'label': label,
                  'input_float32_sha256': input_hash, 'selector': s, 'classifier': c,
                  'combined': combine(s, c, cfg['sigma_candidates'], label),
                  'input_seconds': time.monotonic() - began}
        write_json(root / f'input{index:05d}.json', result)
    write_json(root / 'worker_complete.json', {'status': 'COMPLETED',
        'worker_seconds': time.monotonic() - start, 'indices': cfg['test_indices'],
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--phase', choices=['worker', 'manager'])
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    root = Path(cfg['output_root'])
    if a.phase == 'worker':
        worker(cfg, a.config)
        return
    if a.phase == 'manager':
        for phase in ['worker', 'audit']:
            command = ([sys.executable, str(Path(__file__).resolve()), '--config', str(a.config.resolve()), '--phase', phase]
                if phase == 'worker' else [sys.executable, str(Path(__file__).with_name('audit_dual_rs_certification.py')),
                    '--config', str(a.config.resolve()), '--output', str(root / 'audit.json')])
            began = time.monotonic()
            with (root / (phase + '.stdout')).open('x') as out, (root / (phase + '.stderr')).open('x') as err:
                code = subprocess.run(command, stdout=out, stderr=err).returncode
            write_json(root / (phase + '_finished.json'), {'returncode': code, 'seconds': time.monotonic()-began})
            if code:
                raise SystemExit(code)
        write_json(root / 'inner_terminal.json', {'status': 'CERTIFICATION_PILOT_AUDITED'})
        return
    receipt = supervise([cfg['python'], str(Path(__file__).resolve()), '--config', str(a.config.resolve()), '--phase', 'manager'],
        str(Path(__file__).resolve().parents[1]), root, cfg['total_seconds'],
        'TWO_STAGE_NATIVE_RS_PILOT', cfg['author_repo'], cpu_only=False)
    path = root / 'inner_terminal.json'
    inner = json.loads(path.read_text()) if path.exists() else None
    value = summarize(receipt, inner)
    write_json(root / 'outer_terminal.json', value)
    print(json.dumps(value, indent=2))
    raise SystemExit(0 if value['accepted'] else 1)


if __name__ == '__main__':
    main()
