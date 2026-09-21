"""Freeze SAME component controls with unfused eval-BN export; no new samples."""
import json
from pathlib import Path
from metamoe_component_control import validate_freeze
from recent_moe_deployment import sha256


if __name__ == '__main__':
    parent = Path('configs/recent_moe/metamoe_component_control_r2.json')
    cfg = json.loads(parent.read_text())
    validate_freeze(cfg)
    cfg.update(protocol='metamoe_unfolded_component_cpu_control_r3',
               parent_config=str(parent), parent_config_sha256=sha256(parent),
               output_root='/data1/Kane/MOE/baseline_runs/metamoe_component_20260922_r3',
               compatibility_change='No Conv/BN folding or ONNX simplification; ORT graph optimizations disabled; eval running stats preserved',
               scope='same CIFAR/MNIST index0 component deployment, NOT original exporter or full MoE comparison',
               export_contract='opset12, eval, no constant folding; BN node count must match; original conformance tolerance unchanged')
    for name in ['scripts/metamoe_unfolded_export.py', 'scripts/metamoe_component_unfolded_r3.py',
                 'scripts/freeze_metamoe_unfolded_r3.py', 'scripts/audit_metamoe_component_control.py',
                 'tests/test_metamoe_unfolded_export.py']:
        cfg['execution_files'][name] = sha256(name)
    validate_freeze(cfg)
    if Path(cfg['output_root']).exists():
        raise ValueError('result directory exists')
    target = Path('configs/recent_moe/metamoe_component_control_r3.json')
    with target.open('x') as stream:
        json.dump(cfg, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('FROZEN_NOT_EXECUTED', sha256(target))
