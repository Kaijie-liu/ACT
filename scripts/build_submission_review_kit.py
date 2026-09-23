"""Create a new local, table-only review kit; no release or model execution."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    'paper/review_main.tex', 'paper/results/submission_tables.tex', 'paper/results/main_tables.md',
    'scripts/rebuild_moe_main_tables.py', 'scripts/render_submission_tables.py',
    'scripts/check_submission_review_kit.py',
    'docs/SUBMISSION_REVIEW.md', 'docs/submission_references_20260921.md',
    'docs/main_table_input_composition_20260921.md', 'docs/main_table_input_composition_20260921.json',
    'docs/main_table_source_applicability_20260921.md',
    'docs/metamoe_las_followup_archive_20260923_r1.json',
    'docs/metamoe_las_followup_result_20260923_r1.md',
    'docs/competition_guarantee_disposition_20260923.md',
    'docs/external_ai_review_response_20260921.md',
    'docs/portable_conv_proof_v1_review.json', 'docs/full_source_v1_review.json',
    'docs/property_ranges_v1_analysis.json',
    'act/pipeline/moe/results/schedule_confirmation_100_review_20260914_r1.json',
    'act/pipeline/moe/results/external_pair_comparison_review_20260914_r1.json',
    'act/pipeline/moe/results/relation_ablation_review_20260914_r1.json',
    'act/pipeline/moe/results/conv_full_v2_review_20260915.json',
    'docs/general_evidence_execution_v1_results.json',
)
LIMITS = {
    'new_solver_calls': 0, 'new_model_forwards': 0,
    'independent_network_reproof': False, 'clean_environment_empirical_rerun': False,
    'human_review_completed': False, 'public_or_anonymous_release': False,
    'raw_models_inputs_or_proofs_bundled': False,
}


def build(root, output):
    # Read every source before creating output; failures never overwrite prior kits.
    sources = {}
    for name in FILES:
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('symlink/outside source')
        sources[name] = path.read_bytes()
    output.mkdir(parents=False, exist_ok=False)
    records = []
    for name, data in sorted(sources.items()):
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)
        records.append({'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    manifest = {
        'schema': 'MOE_SUBMISSION_REVIEW_KIT_V1',
        'evidence_grade': 'ARCHIVED_ACCOUNTING_ONLY',
        'build_parent_head': head,
        'identity_note': 'File hashes bind the captured working files; parent HEAD alone does not identify uncommitted additions.',
        'files': records, 'limitations': LIMITS,
        'scope': 'Integrity and archived-table reconstruction; no raw-run or proof recheck.',
    }
    data = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    with (output / 'manifest.json').open('xb') as stream:
        stream.write(data)
    return {'manifest_sha256': hashlib.sha256(data).hexdigest(), 'files': len(records),
            'total_bytes': len(data) + sum(r['bytes'] for r in records), 'output': str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(ROOT, args.output), sort_keys=True))


if __name__ == '__main__':
    main()
