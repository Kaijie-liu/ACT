"""Read-only tensor/state audit; no author imports, data, training or attacks.

This checks an already generated native checkpoint against its frozen control
result. It does NOT establish exact optimizer continuation or attack validity.
"""
import argparse
import json
from pathlib import Path
import torch

from recent_moe_deployment import sha256


def check_state(checkpoint, prediction, result):
    if result.get('status') != 'NATIVE_WORKFLOW_CONTROL_PASS':
        raise ValueError('control did not complete')
    if checkpoint.get('epoch') != 0 or checkpoint.get('global_step') != 1:
        raise ValueError('unexpected completed training position')
    native = checkpoint['state_dict']
    if not native or native.keys() != prediction.keys():
        raise ValueError('state coverage mismatch')
    for name, value in native.items():
        if (not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
                or not torch.equal(value, prediction[name])):
            raise ValueError('invalid or inconsistent tensor: ' + name)
    optimizers = checkpoint.get('optimizer_states', [])
    schedules = checkpoint.get('lr_schedulers', [])
    if len(optimizers) != 1 or len(schedules) != 1:
        raise ValueError('optimizer or scheduler missing')
    optimizer = optimizers[0]
    parameters = [i for group in optimizer['param_groups'] for i in group['params']]
    if len(set(parameters)) != len(parameters):
        raise ValueError('duplicated optimizer parameter index')
    momenta = optimizer['state']
    if not momenta or not set(momenta).issubset(parameters):
        raise ValueError('invalid optimizer-state index')
    for value in momenta.values():
        buffer = value.get('momentum_buffer')
        if not isinstance(buffer, torch.Tensor) or not torch.isfinite(buffer).all():
            raise ValueError('invalid SGD momentum buffer')
    if schedules[0].get('last_epoch') != 1:
        raise ValueError('scheduler position mismatch')
    return {'audit': 'SAVED_TENSORS_AND_STATE_PASS', 'accepted': True,
        'prediction_tensors': len(native), 'optimizer_parameter_indices': len(parameters),
        'finite_momentum_buffers': len(momenta), 'epoch': 0, 'global_step': 1,
        'scheduler_last_epoch': 1, 'checkpoint_prediction_equal': True,
        'full_training_resume_proven': False, 'formal_SAFE': False,
        'scope': 'saved finite tensors, SGD state indices, scheduler position and prediction-state equality only'}


def collect(config_path):
    cfg = json.loads(config_path.read_text())
    for path, expected in cfg['files'].items():
        if sha256(path) != expected:
            raise ValueError('frozen file changed')
    root = Path(cfg['output_root'])
    result_path = root/'result.json'
    result = json.loads(result_path.read_text())
    receipt = json.loads((root/'receipt.json').read_text())
    terminal = json.loads((root/'terminal.json').read_text())
    if (receipt['status'] != 'COMPLETED' or not receipt['source_unchanged']
            or not terminal['accepted'] or result['config_sha256'] != sha256(config_path)):
        raise ValueError('execution identity or terminal mismatch')
    prediction_path = root/'trained_state.pt'
    checkpoint_path = root/'checkpoints/last.ckpt'
    if sha256(prediction_path) != result['state_sha256']:
        raise ValueError('prediction-state file changed')
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
    prediction = torch.load(prediction_path, weights_only=True, map_location='cpu')
    differences = []
    if checkpoint['state_dict'].keys() == prediction.keys():
        for name, before in checkpoint['state_dict'].items():
            after = prediction[name]
            if not torch.equal(before, after):
                differences.append({'name': name,
                    'max_abs_difference': float((before-after).abs().max())})
    if differences:
        value = {'audit': 'SAVED_CHECKPOINT_PREDICTION_MISMATCH', 'accepted': False,
            'differences': differences, 'checkpoint_prediction_equal': False,
            'full_training_resume_proven': False, 'formal_SAFE': False,
            'scope': 'training-end and post-evaluation stored states differ; not frozen-model evaluation evidence'}
    else:
        value = check_state(checkpoint, prediction, result)
    value['files'] = {str(p): sha256(p) for p in [config_path, result_path,
        checkpoint_path, prediction_path, root/'terminal.json', root/'receipt.json']}
    value['native_checkpoint_hash_first_bound_by_this_audit'] = True
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    value = collect(args.config)
    if args.check:
        if json.loads(args.output.read_text()) != value:
            raise ValueError('saved audit differs')
    else:
        with args.output.open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(value['audit'])
    if not value['accepted']:
        raise SystemExit(1)
