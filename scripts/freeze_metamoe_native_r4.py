"""Separate native PyTorch front end; unchanged author solver and two requests."""
import json
from pathlib import Path
from metamoe_component_control import validate_freeze
from recent_moe_deployment import sha256


if __name__ == '__main__':
    parent = Path('configs/recent_moe/metamoe_component_control_r3.json')
    cfg = json.loads(parent.read_text())
    validate_freeze(cfg)
    cfg.update(protocol='metamoe_native_pytorch_component_control_r4',
               parent_config=str(parent), parent_config_sha256=sha256(parent),
               output_root='/data1/Kane/MOE/baseline_runs/metamoe_component_20260922_r4',
               compatibility_change='Native backend Customized loader; no ONNX or BN folding; exact wrapper-removal probes',
               export_contract='NO_EXPORT; same original float32 checkpoint module and unchanged backend settings')
    for name in ['scripts/metamoe_native_model.py', 'scripts/metamoe_component_native_r4.py',
                 'scripts/freeze_metamoe_native_r4.py', 'scripts/audit_metamoe_native_r4.py']:
        cfg['execution_files'][name] = sha256(name)
    if Path(cfg['output_root']).exists():
        raise ValueError('result exists')
    with Path('configs/recent_moe/metamoe_component_control_r4.json').open('x') as stream:
        json.dump(cfg, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('FROZEN_NOT_EXECUTED')
