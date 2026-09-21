"""Saved-record deployment audit, NOT reproof of gradients or robustness."""
import argparse
import json
from pathlib import Path
from recent_moe_deployment import sha256, git_identity


def audit():
    base = Path('/data1/Kane/MOE/baseline_runs')
    names = ['rome_public_listing_20260921_r1', 'rome_public_download_20260921_r1',
             'rome_trained_intake_20260921_r1', 'rome_prediction_intake_20260921_r2',
             'rome_author_eval_help_20260921_r1']
    records = []
    for name in names:
        root = base / name
        value = json.loads((root / 'receipt.json').read_text())
        for kind in ['stdout', 'stderr']:
            if sha256(root / (kind + '.txt')) != value[kind + '_sha256']:
                raise ValueError('saved log changed')
        expected = 'ERROR' if name == 'rome_trained_intake_20260921_r1' else 'COMPLETED'
        if value['status'] != expected or not value['source_unchanged']:
            raise ValueError('terminal/source mismatch: ' + name)
        records.append({'name': name, 'status': value['status'],
                        'receipt_sha256': sha256(root / 'receipt.json'),
                        'execution_seconds': value['execution_including_preflight_seconds'],
                        'total_with_postflight_seconds': value['total_with_postflight_seconds']})
    def logged_json(name):
        # Pinned ACT import emits four text lines, followed by one JSON object.
        text = (base / name / 'stdout.txt').read_text()
        return json.loads(text[text.index('{'):])
    first = logged_json(names[2])
    second = logged_json(names[3])
    if (first['status'] != 'STATE_MAPPING_BLOCKED' or len(first['missing']) != 48
            or first['unexpected'] or second['status'] != 'CONTROL_PASS'
            or sorted(first['missing']) != second['missing_auxiliary_tensors']
            or not second['synthetic_auxiliary_full_model_and_gradient_equal']
            or second['gate_calls'] != 24 or not second['strict_prediction_load']):
        raise ValueError('inference-only mapping evidence incomplete')
    manifest_path = Path('/data1/Kane/MOE/baseline_weights/rome_20260921/manifest.json')
    manifest = json.loads(manifest_path.read_text())
    if sha256(manifest['path']) != manifest['local_sha256'] or second['checkpoint_sha256'] != manifest['local_sha256']:
        raise ValueError('checkpoint identity')
    for path, h in second['source_hashes'].items():
        if sha256(path) != h:
            raise ValueError('author source changed')
    identity = git_identity('/data1/Kane/MOE/baselines/recent_moe_20260921/rome')
    if identity['status']:
        raise ValueError('original author checkout not clean')
    return {'status': 'SAVED_RECORD_AUDIT_PASS', 'records': records,
            'checkpoint': manifest, 'trained_intake': second,
            'trusted_source_assessment': 'global_proj only feeds returned auxiliary diversity, never gate_scores/LoRA/logits',
            'limitations': ['Not a paper-accuracy reproduction', 'Not resumed-training state',
                'Concrete and gradient equality is a control, not full-domain proof',
                'Original code-default s4,b6,alpha=rank, not reconstructed paper recipe',
                'Local SHA identifies download; no publisher checksum verified'],
            'formal_SAFE': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    value = audit()
    if a.check:
        if json.loads(a.output.read_text()) != value:
            raise ValueError('archive changed')
    else:
        with a.output.open('x') as f:
            json.dump(value, f, indent=2)
            f.write('\n')
    print('PASS')
