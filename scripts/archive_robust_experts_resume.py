"""Independent saved tensor/state reread after native continuation."""
import json
from pathlib import Path
import torch
from audit_dual_rs_training_control import canonical
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write


if __name__ == '__main__':
    torch.set_num_threads(2)
    path = Path('configs/recent_moe/robust_experts_resume_r1.json')
    cfg = json.loads(path.read_text())
    root = Path(cfg['output_root'])
    for p, h in cfg['files'].items():
        if sha256(p) != h:
            raise ValueError('execution source changed')
    states, batches = {}, {}
    for phase in ['reference', 'resume']:
        where = root / phase
        info = json.loads((where / 'result.json').read_text())
        if info['status'] != 'COMPLETED' or info['config_sha256'] != sha256(path):
            raise ValueError('phase not bound/completed')
        for file, digest in {**info['checkpoint_hashes'], **info['batch_hashes']}.items():
            if sha256(file) != digest:
                raise ValueError('checkpoint changed')
        states[phase] = torch.load(where / 'checkpoints/epoch01.ckpt', weights_only=False, map_location='cpu')
        batches[phase] = torch.load(where / 'batch_epoch1.pt', weights_only=True)
    equality = {key: canonical(states['reference'][key]) == canonical(states['resume'][key])
                for key in ['state_dict', 'optimizer_states', 'lr_schedulers', 'epoch', 'global_step',
                            'reproduction_rng', 'reproduction_binding']}
    equality['next_augmented_batch_and_rng'] = canonical(batches['reference']) == canonical(batches['resume'])
    stored = json.loads((root / 'audit.json').read_text())
    terminal = json.loads((root / 'terminal.json').read_text())
    receipt = json.loads((root / 'receipt.json').read_text())
    if not all(equality.values()) or equality != stored['equal'] or not terminal['accepted']:
        raise ValueError('continuation/audit/terminal disagreement')
    if receipt['status'] != 'COMPLETED' or not receipt['source_unchanged']:
        raise ValueError('outer/source gate')
    for name in ['stdout', 'stderr']:
        if sha256(root / f'{name}.txt') != receipt[f'{name}_sha256']:
            raise ValueError('log changed')
    write('docs/robust_experts_resume_archive_20260922_r1.json', {
        'audit': 'INDEPENDENT_SAVED_STATE_REREAD_PASS', 'equality': equality,
        'config_sha256': sha256(path), 'terminal': terminal,
        'record_hashes': {str(p): sha256(p) for p in root.rglob('*') if p.is_file()},
        'scope': stored['scope'], 'paper_training_complete': False})
