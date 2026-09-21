"""Audit saved deployment/control records, never infer paper performance."""
import argparse
import json
from pathlib import Path
from recent_moe_deployment import sha256


def collect(executed=False):
    base = Path('/data1/Kane/MOE/baseline_runs')
    expected = {'robust_experts_workflow_deps_20260921_r1': 'ERROR',
        'robust_experts_workflow_deps_20260921_r2': 'COMPLETED',
        'robust_experts_workflow_imports_20260921_r1': 'COMPLETED',
        'robust_experts_config_controls_20260921_r1': 'COMPLETED',
        'robust_experts_cifar100_download_20260921_r1': 'TIMEOUT',
        'robust_experts_cifar100_download_20260921_r2': 'ERROR',
        'robust_experts_cifar100_download_20260921_r3': 'COMPLETED'}
    records = []
    for name, status in expected.items():
        root = base / name
        receipt = json.loads((root/'receipt.json').read_text())
        if receipt['status'] != status or not receipt['source_unchanged']:
            raise ValueError('unexpected deployment terminal')
        for stream in ['stdout', 'stderr']:
            if sha256(root / (stream+'.txt')) != receipt[stream+'_sha256']:
                raise ValueError('saved deployment log changed')
        records.append({'name': name, 'status': status, 'receipt_sha256': sha256(root/'receipt.json'),
            'execution_seconds': receipt['execution_including_preflight_seconds'],
            'with_postflight_seconds': receipt['total_with_postflight_seconds']})
    path = Path('configs/recent_moe/robust_experts_workflow_r1.json')
    config = json.loads(path.read_text())
    for file, h in config['files'].items():
        if sha256(file) != h:
            raise ValueError('frozen workflow identity changed')
    value = {'audit': 'SAVED_DEPLOYMENT_RECORDS_PASS', 'records': records,
        'config_sha256': sha256(path), 'execution_started_at_archive': executed,
        'compatibility': config['execution_compatibility'], 'frozen_math': config['frozen_math'],
        'data': json.loads((Path(config['data_root'])/'manifest.json').read_text()),
        'scope': 'isolated source/dependency/data controls; not paper accuracy or certification', 'formal_SAFE': False,
        'setup_failure_attribution': 'R1 dependency pin was our nonexistent ClearML2.0.4 choice, NOT author-code failure; R2 uses published1.18.0'}
    if executed:
        root = Path(config['output_root'])
        receipt = json.loads((root/'receipt.json').read_text())
        terminal = json.loads((root/'terminal.json').read_text())
        result = json.loads((root/'result.json').read_text()) if (root/'result.json').exists() else None
        if not receipt['source_unchanged']:
            raise ValueError('source changed during workflow')
        for stream in ['stdout', 'stderr']:
            if sha256(root/(stream+'.txt')) != receipt[stream+'_sha256']:
                raise ValueError('workflow log changed')
        if terminal['accepted']:
            if (receipt['status'] != 'COMPLETED' or result['status'] != 'NATIVE_WORKFLOW_CONTROL_PASS'
                    or result['config_sha256'] != sha256(path) or result['global_steps'] != 1
                    or result['changed_parameter_tensors'] < 1 or len(result['test_calls']) != 3
                    or not result['prediction_state_reload_exact'] or sha256(root/'trained_state.pt') != result['state_sha256']):
                raise ValueError('incomplete native workflow acceptance')
            for i, record in enumerate(result['test_calls']):
                if record['call'] != i or json.loads((root/f'test_call{i}.json').read_text()) != record:
                    raise ValueError('test-stage accounting mismatch')
        elif terminal['status'] == 'NATIVE_WORKFLOW_CONTROL_PASS':
            raise ValueError('false successful terminal')
        value.update(terminal=terminal, result=result,
            execution_receipt_sha256=sha256(root/'receipt.json'),
            stderr_tail=(root/'stderr.txt').read_text()[-4000:] if not terminal['accepted'] else None)
    return value


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--executed', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    value = collect(a.executed)
    if a.check:
        if json.loads(a.output.read_text()) != value:
            raise ValueError('archive differs')
    else:
        with a.output.open('x') as f:
            json.dump(value, f, indent=2)
            f.write('\n')
    print('Saved workflow archive PASS')
