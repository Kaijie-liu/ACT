"""Archive source-bound full-architecture controls; no performance claims."""
import argparse
import json
from pathlib import Path
import subprocess

from recent_moe_deployment import git_identity, sha256
from recent_moe_env_inventory import inventory


def collect():
    root = Path('/data1/Kane/MOE')
    original = root / 'baselines/recent_moe_20260921/robust_experts'
    compat = root / 'baselines/robust_experts_compat_20260921_r1'
    original_id = git_identity(original)
    if original_id['head'] != 'ed22e81fabc3c3196b6bcd352ee83042473cdfbf' or original_id['status']:
        raise ValueError('original source changed')
    diff = subprocess.check_output(['git', '-C', str(compat), 'diff', '--'], text=True)
    patch = Path('configs/recent_moe/robust_experts_optional_syncbn_r1.patch')
    if diff != patch.read_text():
        raise ValueError('compatibility copy differs from reviewed patch')
    env_file = root / 'baseline_runs/moe_author_cpu_env_20260921.json'
    recorded = json.loads(env_file.read_text())
    if inventory(list(recorded)) != recorded:
        raise ValueError('frozen new environment changed')
    records = []
    for name in ['robust_experts_arch_compat_20260921_r1', 'robust_experts_arch_compat_20260921_r2',
                 'robust_experts_intake_20260921_r1', 'rome_intake_20260921_r1']:
        folder = root / 'baseline_runs' / name
        receipt = json.loads((folder / 'receipt.json').read_text())
        for stream in ['stdout', 'stderr']:
            if sha256(folder / f'{stream}.txt') != receipt[f'{stream}_sha256']:
                raise ValueError('log changed')
        value = None
        if receipt['status'] == 'COMPLETED':
            text = (folder / 'stdout.txt').read_text()
            value = json.JSONDecoder().raw_decode(text[text.index('{'):])[0]
            if 'route_events' in value:
                for p, h in value['source_hashes'].items():
                    if sha256(p) != h:
                        raise ValueError('trace source changed')
                if (not value['full_forward_bitwise_equal'] or not value['input_gradient_bitwise_equal']
                        or not value['source_mutation_rejected'] or not value['train_mode_rejected']
                        or value['box_verification']['status'] != 'UNSUPPORTED'):
                    raise ValueError('trace control failed or overclaims verification')
                value = {k: v for k, v in value.items() if k not in {'source_hashes', 'route_events'}} | {
                    'gate_calls': len(value['route_events']),
                    'gate_layers': [e['layer'] for e in value['route_events']],
                    'mechanisms': sorted({e['mechanism'] for e in value['route_events']}),
                    'source_inventory_bound_in_stdout': True}
        elif name != 'robust_experts_arch_compat_20260921_r1':
            raise ValueError('unexpected new control failure')
        records.append({'attempt': name, 'status': receipt['status'], 'result': value,
            'execution_seconds': receipt['execution_including_preflight_seconds'],
            'total_with_postflight_seconds': receipt['total_with_postflight_seconds'],
            'record_hashes': {str(p): sha256(p) for p in sorted(folder.iterdir()) if p.is_file()}})
    return {'audit': 'PASS', 'scope': 'source/record and control checks; not paper replication or domain proofs',
        'original_robust_source_unchanged': True, 'compatibility_patch_sha256': sha256(patch),
        'cpu_environment': recorded, 'controls': records,
        'retained_setup_failure': {'operation': 'offline conda clone', 'reason': 'required package archives absent',
            'partial_directory': str(root / 'envs/moe-author-adapters-py312-20260921'),
            'replaced_by_separate_fresh_venv_not_modified_in_place': True}}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--check', action='store_true')
    args = p.parse_args()
    value = collect()
    if args.check:
        if value != json.loads(args.output.read_text()):
            raise ValueError('archive differs')
    else:
        with args.output.open('x') as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write('\n')
    print('Author adapter controls archive PASS')
