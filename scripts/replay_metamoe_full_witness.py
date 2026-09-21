"""Independent complete-source replay for the frozen original intake witness."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recent_moe_deployment import sha256
from metamoe_full_intake import load_original, validate

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    validate(cfg)
    import torch
    torch.set_num_threads(2)
    start = time.monotonic()
    original, _ = load_original(cfg)
    original.double().eval()
    records = []
    for request in cfg['requests']:
        folder = Path(cfg['output_root']) / request['id']
        result = json.loads((folder / 'result.json').read_text())
        if result['status'] != 'UNSAFE_REPLAYED':
            continue
        materialized = torch.load(folder / 'request.pt', map_location='cpu', weights_only=True)
        for key, value in materialized.items():
            if hashlib.sha256(value.numpy().tobytes()).hexdigest() != result['materialized_tensor_hashes'][key]:
                raise ValueError('materialized request identity')
        x = torch.tensor(result['witness'], dtype=torch.float64).reshape_as(materialized['center'])
        if not torch.isfinite(x).all() or (x < materialized['lower']).any() or (x > materialized['upper']).any():
            raise ValueError('witness outside domain')
        with torch.no_grad():
            output, scores = original(x)
        if not torch.isfinite(output).all() or not torch.isfinite(scores).all():
            raise ValueError('nonfinite source output')
        label = result['global_label']
        competitors = [i for i in range(original.total_classes) if i != label]
        margins = output[0, label] - output[0, competitors]
        if not (margins < 0).any():
            raise ValueError('no actual global classification reversal')
        records.append({'request': request, 'label': label, 'prediction': int(output.argmax(1)),
            'minimum_class_margin': float(margins.min()), 'witness_in_box': True,
            'checkpoint_sha256': sha256(cfg['checkpoint']), 'result_sha256': sha256(folder / 'result.json')})
    if not records:
        raise ValueError('no replayed witness')
    with a.output.open('x') as f:
        json.dump({'audit': 'PASS', 'independent_original_model_replays': records,
            'replay_seconds': time.monotonic() - start,
            'scope': 'explicit CPU float64 original-module snapshot, not arbitrary IEEE execution proof'}, f, indent=2)
        f.write('\n')
    print('Original full-model witness replay PASS')
