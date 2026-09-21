"""Saved-only certification archive, retaining counts and confidence scope."""
import argparse
import json
from pathlib import Path
from recent_moe_deployment import sha256


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reaudit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    config = Path('configs/recent_moe/dual_rs_certification_execution_r1.json')
    cfg = json.loads(config.read_text())
    root = Path(cfg['output_root'])
    audit = json.loads((root / 'audit.json').read_text())
    reread = json.loads(a.reaudit.read_text())
    strip = lambda r: {k: v for k, v in r.items() if k != 'audit_seconds'}
    if strip(audit) != strip(reread) or audit['status'] != 'COUNT_IDENTITY_AUDIT_PASS':
        raise ValueError('independent audit mismatch')
    terminal = json.loads((root / 'outer_terminal.json').read_text())
    receipt = json.loads((root / 'receipt.json').read_text())
    if not terminal['accepted'] or receipt['status'] != 'COMPLETED' or not receipt['source_unchanged']:
        raise ValueError('outer/source completion missing')
    for file, h in {**cfg['files'], **audit['files']}.items():
        if sha256(file) != h:
            raise ValueError('identity changed')
    for name in ['stdout', 'stderr']:
        if sha256(root / f'{name}.txt') != receipt[f'{name}_sha256']:
            raise ValueError('log identity')
    value = {'schema': 'dual_rs_certification_archive_v1', 'execution_commit': '0687fb6e6',
             'config_sha256': sha256(config), 'audit': audit, 'outer_terminal': terminal,
             'worker_complete': json.loads((root / 'worker_complete.json').read_text()),
             'saved_reaudit': {'path': str(a.reaudit), 'sha256': sha256(a.reaudit)},
             'stages': {f'input{i:05d}_{stage}': json.loads((root / f'input{i:05d}_{stage}.json').read_text())
                        for i in cfg['test_indices'] for stage in ['selector', 'classifier']},
             'record_hashes': {str(f): sha256(f) for f in root.iterdir() if f.is_file()},
             'grade': 'PROBABILISTIC_RS_NATIVE_NUMERICAL', 'deterministic_formal_SAFE': False,
             'paper_accuracy_estimate': False}
    with a.output.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('Certification archive PASS')
