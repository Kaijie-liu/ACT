"""Archive audited final training without changing or selecting checkpoints."""
import argparse
import json
from pathlib import Path

from recent_moe_deployment import sha256


def collect(config, reaudit):
    cfg = json.loads(config.read_text())
    root = Path(cfg['output_root'])
    for name, digest in cfg['execution_files'].items():
        if sha256(name) != digest:
            raise ValueError('training source identity changed')
    old = json.loads((root / 'audit.json').read_text())
    new = json.loads(reaudit.read_text())
    without_time = lambda value: {k: v for k, v in value.items() if k != 'audit_seconds'}
    if without_time(old) != without_time(new):
        raise ValueError('independent saved-state audit differs')
    if new['audit'] != 'PASS' or new['epochs'] != 90 or new['config_sha256'] != sha256(config):
        raise ValueError('final audit binding')
    outer = json.loads((root / 'outer_terminal.json').read_text())
    inner = json.loads((root / 'training/terminal.json').read_text())
    if outer['status'] != 'TRAINING_LANDED' or inner['status'] != 'COMPLETED':
        raise ValueError('not landed')
    records = new['epoch_records']['training']
    if len(records) != 90:
        raise ValueError('epoch coverage')
    for epoch, info in enumerate(records, 1):
        path = root / 'training' / f'epoch{epoch:03d}.pt'
        if sha256(path) != info['file_sha256']:
            raise ValueError('saved checkpoint changed')
    return {
        'schema': 'dual_rs_final_training_archive_v1', 'status': 'TRAINING_LANDED',
        'config': str(config), 'config_sha256': sha256(config),
        'outer_terminal': outer, 'worker_terminal': inner,
        'final_epoch': records[-1], 'epochs_checked': len(records),
        'independent_saved_state_audit': {'path': str(reaudit), 'sha256': sha256(reaudit),
                                          'audit_seconds': new['audit_seconds'], 'result': 'PASS'},
        'epoch_metadata': {str(p): sha256(p) for p in sorted((root / 'training').glob('epoch*.json'))},
        'terminal_evidence': {str(p): sha256(p) for p in [root / 'audit.json',
                              root / 'outer_terminal.json', root / 'training/terminal.json']},
        'selection': 'final epoch90, no accuracy/certification selection',
        'metric_scope': 'test_author_metrics are sigma-selector loss/accuracy, NOT CIFAR class accuracy',
        'implementation_scope': 'named log-domain consistency compatibility variant; original R1 failure retained',
        'guarantee': 'training identity/accounting only; no certification claim',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--reaudit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    value = collect(args.config, args.reaudit)
    if args.check:
        if value != json.loads(args.output.read_text()):
            raise ValueError('archive changed')
    else:
        with args.output.open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print('Final90 landing archive PASS')
