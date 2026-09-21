"""Run in the isolated workflow environment; freeze before any real update."""
import json
from pathlib import Path
import subprocess
from robust_experts_workflow_control import compose
from recent_moe_deployment import sha256, git_identity


if __name__ == '__main__':
    base = Path('/data1/Kane/MOE')
    repo = base / 'baselines/robust_experts_compat_20260921_r1'
    original = base / 'baselines/recent_moe_20260921/robust_experts'
    root = base / 'baseline_runs/robust_experts_workflow_20260921_r1'
    data = base / 'baseline_data/robust_experts_20260921_r3'
    env = base / 'baseline_runs/robust_experts_workflow_env_20260921.json'
    native, cfg, composed = compose(repo, root, data)
    source = git_identity(original)
    if source['status'] or source['head'] != 'ed22e81fabc3c3196b6bcd352ee83042473cdfbf':
        raise ValueError('author source identity')
    patch = Path('configs/recent_moe/robust_experts_optional_syncbn_r1.patch')
    if (subprocess.check_output(['git', '-C', str(repo), 'diff'], text=True) != patch.read_text()
            or subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() != source['head']):
        raise ValueError('compatibility copy differs beyond reviewed optional SyncBN patch')
    data_manifest = data / 'manifest.json'
    record = json.loads(data_manifest.read_text())
    if record['status'] != 'PUBLIC_DATA_INTEGRITY_PASS' or record['published_torchvision_md5'] != 'eb9058c3a382ffc7106e4002c42a8d85':
        raise ValueError('CIFAR100 public identity not complete')
    for name, h in record['files'].items():
        if sha256(data / 'cifar-100-python' / name) != h:
            raise ValueError('extracted data changed')
    controls = base / 'baseline_runs/robust_experts_config_controls_20260921_r1'
    control_receipt = json.loads((controls / 'receipt.json').read_text())
    if (control_receipt['status'] != 'COMPLETED' or
            sha256(controls / 'stderr.txt') != control_receipt['stderr_sha256'] or
            not (controls / 'stderr.txt').read_text().rstrip().endswith('OK')):
        raise ValueError('configuration/terminal controls did not pass')
    files = [repo / key for key in source['tracked_sha256'] if key.endswith(('.py', '.yaml'))]
    files += [env, patch, data_manifest, controls / 'receipt.json', controls / 'stderr.txt',
        Path('scripts/robust_experts_workflow_control.py'), Path(__file__),
        Path('scripts/recent_moe_deployment.py'), Path('scripts/recent_moe_env_inventory.py')]
    files += [data / 'cifar-100-python' / name for name in ['train', 'test', 'meta']]
    manifest = {'schema': 1, 'protocol': 'robust_experts_native_pgd_workflow_control_r1',
        'repo': str(repo), 'original_repo': str(original), 'author_commit': source['head'],
        'output_root': str(root), 'data_root': str(data), 'environment': str(env),
        'python': str(base / 'envs/robust-experts-workflow-cpu-20260921/bin/python'),
        'files': {str(p): sha256(p) for p in files}, 'composed_config': composed, 'total_seconds': 600,
        'source_patches': ['optional ordinary-BN-only SyncBN import, archived separately'],
        'execution_compatibility': ['Hydra1.3.2/Python3.12 vs author Hydra1.1',
            'CPU Torch2.9.1/Lightning1.9.5', 'native swallowed data_dir assigned after construction',
            'all remote logging disabled; local CSV/checkpoint only'],
        'limits': 'one native batch each training/validation/test and each PGD20/APGD20, batch2, workers0',
        'frozen_math': 'native E4/k1/layer4 ConvMoE, SGD.1/momentum.9/wd5e-4, PGD7 eps.03137 alpha.00784313725',
        'not_claimed': ['paper-model training', 'clean/robust accuracy', 'full Lightning resume', 'domain certification'],
        'failure_rule': 'preserve full failure, no retry or sample/attack alteration within R1'}
    out = Path('configs/recent_moe/robust_experts_workflow_r1.json')
    with out.open('x') as f:
        json.dump(manifest, f, indent=2, allow_nan=False)
        f.write('\n')
    print(sha256(out))
