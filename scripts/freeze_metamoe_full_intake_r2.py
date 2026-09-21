import json
from pathlib import Path
from recent_moe_deployment import sha256

if __name__ == '__main__':
    original = Path('configs/recent_moe/metamoe_full_intake_r1.json')
    cfg = json.loads(original.read_text())
    cfg.update(protocol='metamoe_full_class_separated_intake_r2',
        python='/data1/Kane/MOE/envs/moe-author-cpu-py312-20260921/bin/python',
        output_root='/data1/Kane/MOE/baseline_runs/metamoe_full_intake_20260921_r2',
        parent_config=str(original), parent_config_sha256=sha256(original),
        environment_inventory='/data1/Kane/MOE/baseline_runs/moe_author_cpu_env_20260921.json')
    cfg['environment_inventory_sha256'] = sha256(cfg['environment_inventory'])
    cfg['files'].update({p: sha256(p) for p in ['scripts/metamoe_full_intake_r2.py',
        'scripts/audit_metamoe_full_intake.py', cfg['environment_inventory']]})
    out = Path('configs/recent_moe/metamoe_full_intake_r2.json')
    with out.open('x') as f:
        json.dump(cfg, f, indent=2)
        f.write('\n')
    print(sha256(out))
