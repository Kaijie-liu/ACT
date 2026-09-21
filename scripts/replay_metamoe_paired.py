"""Saved-only original full-model witness replay, independently of ACT/HZ."""
import argparse
import json
from pathlib import Path
import time
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write


def replay(path):
    start = time.monotonic()
    cfg = json.loads(path.read_text())
    for file, digest in cfg['files'].items():
        if sha256(file) != digest:
            raise ValueError('changed frozen input/source')
    import numpy as np
    import torch
    from metamoe_paired_model import load_full
    torch.set_num_threads(2)
    model = load_full(cfg['repo'], cfg['checkpoint'], cfg['files'][cfg['checkpoint']])
    root = Path(cfg['output_root'])
    summary = json.loads((root/'summary.json').read_text())
    rows = []
    for row in summary['rows']:
        if row['status'] != 'UNSAFE_REPLAYED':
            continue
        request = next(r for r in cfg['requests'] if r['id'] == row['id'])
        result_path = root/f"{row['id']}_{row['arm']}"/'result.json'
        if sha256(result_path) != row['result_sha256']:
            raise ValueError('result binding')
        result = json.loads(result_path.read_text())
        point = torch.tensor(result['witness'], dtype=torch.float64)
        with np.load(request['tensor_file'], allow_pickle=False) as data:
            lo, hi = torch.from_numpy(data['lower']), torch.from_numpy(data['upper'])
        if point.numel() != lo.numel():
            raise ValueError('witness dimensions')
        point = point.reshape_as(lo)
        if not torch.isfinite(point).all() or (point < lo).any() or (point > hi).any():
            raise ValueError('witness outside box')
        with torch.no_grad():
            output, scores = model(point)
        if not torch.isfinite(output).all() or not torch.isfinite(scores).all():
            raise ValueError('undefined original model')
        label = request['label']
        margins = [float(output[0,label]-output[0,j]) for j in range(model.total_classes) if j != label]
        if min(margins) >= cfg['margin']:
            raise ValueError('not a full original-model property violation')
        rows.append({'id': row['id'], 'arm': row['arm'], 'result_sha256': sha256(result_path),
            'label': label, 'prediction': int(output.argmax(1)), 'minimum_margin': min(margins)})
    return {'audit': 'INDEPENDENT_ORIGINAL_MODEL_REPLAY_PASS', 'rows': rows,
        'config_sha256': sha256(path), 'summary_sha256': sha256(root/'summary.json'),
        'separate_audit_seconds': time.monotonic()-start,
        'scope': 'same explicit float64 original full model; no positive bound reproof'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    write(a.output, replay(a.config))
