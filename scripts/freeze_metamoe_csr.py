"""New R3 identities; never edit R2 or select new inputs before all smoke gates."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write

CHANGED = ['act/config/config.py', 'act/back_end/hybridz_tf/hybridz_tf.py',
           'act/back_end/moe/route_a.py', 'act/back_end/moe/class_separated_top1.py',
           'scripts/metamoe_paired_execution_r2.py']
ADDED = ['act/back_end/hybridz_tf/sparse_budget.py', 'scripts/metamoe_csr_execution.py',
         'scripts/diagnose_metamoe_csr.py', 'scripts/freeze_metamoe_csr.py',
         'scripts/audit_metamoe_csr.py', 'tests/test_sparse_resource_policy.py',
         'tests/test_metamoe_csr_execution.py']


def freeze(confirmatory=False):
    started = time.monotonic()
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise ValueError('commit reviewed implementation before execution freeze')
    parent_path = Path('configs/recent_moe/metamoe_paired_smoke_r2.json')
    cfg = json.loads(parent_path.read_text())
    changed = {}
    for file, old in cfg['files'].items():
        new = sha256(file)
        if old != new:
            if file not in CHANGED:
                raise ValueError('unapproved identity drift: '+file)
            changed[file] = {'before': old, 'after': new}
    cfg['files'].update({file: sha256(file) for file in CHANGED+ADDED})
    cfg['files'][str(parent_path)] = sha256(parent_path)
    cfg['parent_config_sha256'] = sha256(parent_path)
    cfg['explicit_source_rebinding'] = changed
    cfg['hybridz'] = {'sparse_resource_policy': 'csr_bytes_v1', 'sparse_representation_bytes': 2**31}
    cfg['group_rss_limit_bytes'] = 8*2**30
    suffix = 'confirmatory' if confirmatory else 'smoke'
    cfg['protocol'] = f'metamoe_csr_r3_{suffix}'
    cfg['output_root'] = f'/data1/Kane/MOE/baseline_runs/metamoe_csr_20260922_r3_{suffix}'
    cfg['execution_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    cfg['resource_contract'] = '2GiB conservative CSR representation admission, 8GiB sampled sum of own process-group RSS; neither is proof of peak-memory bound'
    if confirmatory:
        gates = {
            'docs/metamoe_csr_diagnostic_20260922_r3.json': ('passed', True),
            'docs/metamoe_csr_smoke_review_20260922_r3.json': ('execution_control_pass', True),
            'docs/metamoe_csr_smoke_replay_20260922_r3.json': ('audit', 'INDEPENDENT_ORIGINAL_MODEL_REPLAY_PASS')}
        smoke = Path('configs/recent_moe/metamoe_csr_smoke_r3.json')
        smoke_cfg = json.loads(smoke.read_text())
        for field in ('files', 'repositories', 'environment', 'python', 'checkpoint', 'repo',
                      'backend_repo', 'hybridz', 'group_rss_limit_bytes', 'seconds', 'epsilon', 'margin'):
            if cfg[field] != smoke_cfg[field]:
                raise ValueError('execution policy differs from completed smoke: '+field)
        reports = {}
        for file, (field, expected) in gates.items():
            gate = json.loads(Path(file).read_text())
            if gate.get(field) != expected:
                raise ValueError('formal execution gate closed: '+file)
            binding = gate.get('config_sha256', gate.get('result', {}).get('config_sha256'))
            if binding != sha256(smoke):
                raise ValueError('gate bound to wrong smoke')
            cfg['files'][file] = sha256(file)
            reports[file] = gate
        review = reports['docs/metamoe_csr_smoke_review_20260922_r3.json']
        replay = reports['docs/metamoe_csr_smoke_replay_20260922_r3.json']
        summary_hash = sha256(Path(smoke_cfg['output_root'])/'summary.json')
        if review['summary_sha256'] != summary_hash or replay['summary_sha256'] != summary_hash:
            raise ValueError('review and witness replay not same terminal roster')
        expected_replays = {(r['id'], r['arm'], r['result_sha256']) for r in review['rows'] if r['status'] == 'UNSAFE_REPLAYED'}
        observed_replays = [(r['id'], r['arm'], r['result_sha256']) for r in replay['rows']]
        if set(observed_replays) != expected_replays or len(observed_replays) != len(expected_replays):
            raise ValueError('incomplete/duplicate original-model replay')
        # Gates above run BEFORE importing model/data or creating a cohort.
        import sys
        import numpy as np
        import torch
        from metamoe_paired_model import load_full
        torch.set_num_threads(2)
        model = load_full(cfg['repo'], cfg['checkpoint'], cfg['files'][cfg['checkpoint']])
        sys.path.insert(0, str(Path(cfg['repo'])/'src/Formal_Neural_Network_Verification/alpha-beta-crown'))
        from create_vnnlib_specs import load_dataset
        parent = json.loads(Path('configs/recent_moe/metamoe_full_intake_r2.json').read_text())
        tensors = Path('/data1/Kane/MOE/baseline_data/metamoe_csr_20260922_r3_confirmatory')
        tensors.mkdir(parents=True, exist_ok=False)
        cfg['requests'] = []
        for dataset, offset in [('CIFAR10', 0), ('MNIST', 10)]:
            images, labels, _ = load_dataset(dataset, parent['data_root'], 1000)
            count = 0
            for index, (image, label) in enumerate(zip(images, labels)):
                if index == 0:
                    continue
                x = image.unsqueeze(0).double()
                with torch.no_grad():
                    output = model(x)[0]
                if not torch.isfinite(output).all():
                    raise ValueError('undefined selection')
                if int(output.argmax(1)) != label+offset:
                    continue
                rid = f'{dataset.lower()}_{index}'
                file = tensors/f'{rid}.npz'
                np.savez(file, center=x.numpy(), lower=(x-2/255).clamp(-10, 10).numpy(), upper=(x+2/255).clamp(-10, 10).numpy())
                cfg['files'][str(file)] = sha256(file)
                cfg['requests'].append({'id': rid, 'dataset': dataset, 'index': index, 'label': label+offset,
                                        'clean_prediction': int(output.argmax(1)), 'tensor_file': str(file)})
                count += 1
                if count == 10:
                    break
            if count != 10:
                raise ValueError('first1000 scan insufficient; no extension')
        cfg['selection'] = 'first10/domain clean-correct raw order, scan1000, exclude old index0, no route/bound filtering'
    cfg['preparation_seconds'] = time.monotonic()-started
    path = Path(f'configs/recent_moe/metamoe_csr_{suffix}_r3.json')
    write(path, cfg)
    print('FROZEN_NOT_EXECUTED', path, sha256(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirmatory', action='store_true')
    args = parser.parse_args()
    freeze(args.confirmatory)
