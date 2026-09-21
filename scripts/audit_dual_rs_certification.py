"""Independent completed-count/identity audit; not formal floating-point proof."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from dual_rs_certification_evidence import from_counts
from recent_moe_deployment import sha256


def audit(config):
    start = time.monotonic()
    cfg = json.loads(config.read_text())
    root = Path(cfg['output_root'])
    for path, h in cfg['files'].items():
        if sha256(path) != h:
            raise ValueError('source/weight/config identity changed')
    complete = json.loads((root / 'worker_complete.json').read_text())
    if complete['status'] != 'COMPLETED' or complete['indices'] != [0, 1]:
        raise ValueError('incomplete fixed cohort')
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor
    parent = json.loads(Path(cfg['training_parent']).read_text())
    dataset = CIFAR10(parent['data_root'], train=False, download=False, transform=ToTensor())
    rows, files = [], {}
    for index in cfg['test_indices']:
        path = root / f'input{index:05d}.json'
        row = json.loads(path.read_text())
        x, label = dataset[index]
        if (row['index'] != index or row['label'] != label or row['config_sha256'] != sha256(config)
                or row['input_float32_sha256'] != hashlib.sha256(x.numpy().tobytes()).hexdigest()):
            raise ValueError('input identity/order/label')
        s = row['selector']
        c = row['classifier']
        stages = [('selector', s, 3, 1., 0)]
        if s['prediction'] == -1:
            if c is not None:
                raise ValueError('unregistered classifier after abstention')
            prediction, radius, sigma = -1, 0., None
        else:
            if not 0 <= s['prediction'] < 3 or c is None:
                raise ValueError('missing second stage')
            sigma = cfg['sigma_candidates'][s['prediction']]
            stages.append(('classifier', c, 10, sigma, 1))
            prediction = c['prediction']
            radius = 0. if prediction == -1 else min(s['radius_l2'], c['radius_l2'])
        for name, stage, classes, noise, seed_pos in stages:
            stage_path = root / f'input{index:05d}_{name}.json'
            if (json.loads(stage_path.read_text()) != stage or stage['stage'] != name
                    or stage['index'] != index or stage['classes'] != classes or stage['sigma'] != noise
                    or stage['seed'] != cfg['seeds'][str(index)][seed_pos]):
                raise ValueError('stage/source binding')
            checked = from_counts(stage['selection_counts'], stage['estimation_counts'],
                cfg['n0'], cfg['n'], classes, noise, cfg['alpha'])
            for key, value in checked.items():
                if stage[key] != value:
                    raise ValueError('count-derived result changed')
            for number, counts in [(1, stage['selection_counts']), (2, stage['estimation_counts'])]:
                count_path = root / f'input{index:05d}_{name}_counts{number}.json'
                stored = json.loads(count_path.read_text())
                if (stored['counts'] != counts or stored['count_number'] != number or
                        stored['config_sha256'] != sha256(config) or stored['index'] != index or stored['stage'] != name):
                    raise ValueError('count trace inconsistent')
                files[str(count_path)] = sha256(count_path)
            files[str(stage_path)] = sha256(stage_path)
        expected = {'selected_sigma': sigma, 'prediction': prediction, 'radius_l2': radius,
                    'correct': prediction == label, 'correct_radius_l2': radius if prediction == label else 0.,
                    'grade': 'PROBABILISTIC_RS_NATIVE_NUMERICAL', 'deterministic_formal_SAFE': False}
        if row['combined'] != expected or not math.isfinite(row['input_seconds']) or row['input_seconds'] < 0:
            raise ValueError('aggregation or cost mismatch')
        rows.append({'index': index, 'label': label, **expected, 'input_seconds': row['input_seconds']})
        files[str(path)] = sha256(path)
    return {'status': 'COUNT_IDENTITY_AUDIT_PASS', 'config_sha256': sha256(config), 'rows': rows,
        'files': files, 'audit_seconds': time.monotonic() - start,
        'confidence': 'per-input <=0.001 total stage failure probability; for BOTH pilot inputs <=0.002 by union bound',
        'trust': ['native models/noise implementation', 'numerical probability tails', 'Monte Carlo randomness'],
        'not_checked': 'full-domain numerical soundness or deterministic routed-model equivalence',
        'scope': 'fresh two-input deployment pilot, not paper-scale certified accuracy'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    value = audit(a.config)
    with a.output.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
    print('Count/identity audit PASS')
