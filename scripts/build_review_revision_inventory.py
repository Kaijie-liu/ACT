"""Freeze/check revised manuscript identities without replacing old manifests."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
START = '282e0507b94ac2f6105096e43ea7c0f7a0d38bc5'
OLD = ROOT/'docs/external_ai_review_manifest_20260921.json'
OUTPUT = ROOT/'docs/review_revision_inventory_20260921_r2.json'
EXTRAS = (
    'docs/CODEX_HANDOFF.md', 'docs/EXTERNAL_AI_REVIEW.md', 'docs/EXTERNAL_AI_REVIEW_PROMPT.md',
    'docs/external_ai_review_response_20260921.md',
    'docs/main_table_input_composition_20260921.md', 'docs/main_table_input_composition_20260921.json',
    'docs/reviewer_artifact_readiness_20260921.md', 'docs/reviewer_artifact_readiness_20260921.json',
    'scripts/audit_moe_input_composition.py', 'scripts/test_moe_input_composition.py',
    'scripts/review_artifact_readiness.py', 'scripts/test_review_handoff.py',
    'scripts/build_review_revision_inventory.py', 'scripts/test_review_revision.py',
    'scripts/validate_review_response.py',
    'docs/review_revision_inventory_20260921.json', 'docs/review_response_validation_20260921.json',
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def record(path):
    p = Path(path)
    data = p.read_bytes()
    return {'path': str(p.relative_to(ROOT)), 'sha256': sha(data), 'bytes': len(data)}


def build():
    old = json.loads(OLD.read_bytes())
    if OLD.read_bytes() != subprocess.check_output(['git', 'show', START+':'+str(OLD.relative_to(ROOT))], cwd=ROOT):
        raise ValueError('original review manifest changed')
    protected = []
    for group in old['files'].values():
        for r in group:
            data = subprocess.check_output(['git', 'show', old['scientific_baseline_commit']+':'+r['path']], cwd=ROOT)
            if sha(data) != r['sha256'] or len(data) != r['bytes']:
                raise ValueError('historical baseline identity mismatch')
            if not r['path'].startswith('paper/') or r['path'] in (
                    'paper/results/main_tables.md', 'paper/appendices/source_proof_history.md'):
                if record(ROOT/r['path']) != r:
                    raise ValueError('protected scientific evidence changed: '+r['path'])
                protected.append(r)
    # No implementation/experimental configuration changes anywhere under act.
    if subprocess.check_output(['git', 'diff', START, '--', 'act'], cwd=ROOT):
        raise ValueError('ACT implementation/config/results changed')
    papers = [record(p) for p in sorted((ROOT/'paper').rglob('*.md'))]
    return {'schema': 'REVIEW_REVISION_INVENTORY_V1', 'starting_head': START,
            'historical_manifest': record(OLD), 'historical_scientific_baseline': old['scientific_baseline_commit'],
            'old_manifest_files_verified_from_git': old['file_count'],
            'current_manuscript_file_count': len(papers), 'current_manuscript': papers,
            'historical_appendix_policy': 'Retained chronology/preserved blocks; current README qualification governs old terminology.',
            'current_response_materials': [record(ROOT/p) for p in EXTRAS],
            'protected_unchanged_scientific_materials': protected,
            'act_changes': False, 'third_party_human_review_completed': False,
            'source_complete_positive_claim': False, 'input98_followup': 'STOP_INPUT98_FOLLOWUP',
            'external_release_performed': False,
            'validation_receipt': 'docs/review_response_validation_20260921_r2.json',
            'validation_receipt_note': 'Post-inventory execution receipt, intentionally not self-hashed here.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check', action='store_true')
    args = p.parse_args()
    result = build()
    if args.check:
        if json.loads(OUTPUT.read_bytes()) != result:
            raise ValueError('current revision inventory drift')
    else:
        with OUTPUT.open('x') as f:
            json.dump(result, f, indent=2, sort_keys=True)
            f.write('\n')
    print(json.dumps({'current_manuscript_files': len(result['current_manuscript']),
                      'original_manifest_files': result['old_manifest_files_verified_from_git'],
                      'protected_materials': len(result['protected_unchanged_scientific_materials']),
                      'check': args.check}))


if __name__ == '__main__':
    main()
