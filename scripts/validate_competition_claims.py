"""Saved-evidence manuscript validation; never executes models or bound queries.

Produces a NEW receipt and local relocated table-only kit/PDF. The old R5
inventory is checked at its own Git commit, not mistaken for today's files.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
START = '083d4cb0c4ba820784beb02f8f4dd067c143bf7e'
PROTECTED = (
    'paper/results/main_tables.md',
    'docs/main_table_source_applicability_20260921.md',
    'docs/main_table_source_applicability_20260921.json',
    'docs/main_table_input_composition_20260921.json',
    'docs/metamoe_las_followup_archive_20260923_r1.json',
    'docs/metamoe_las_followup_analysis_20260923_r1.json',
    'docs/review_revision_inventory_20260921_r5.json',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--work', required=True, type=Path)
    parser.add_argument('--pdflatex', required=True, type=Path)
    args = parser.parse_args()
    for path in (args.output, args.work):
        if path.exists() or not path.resolve().is_relative_to(ROOT.parent):
            raise ValueError('requires NEW output under the research workspace')
    args.work.mkdir(parents=False, exist_ok=False)
    result = dict(schema='COMPETITION_CLAIMS_VALIDATION_V1', status='FAILED_RETAINED',
                  starting_head=START, records=[], new_model_forwards=0,
                  new_solver_calls=0, new_source_propagations=0,
                  strict_certificate_added=False, human_review_completed=False,
                  empirical_rerun=False, protected=[])

    def run(command, cwd=ROOT, timeout=60):
        begin = time.monotonic()
        try:
            done = subprocess.run(list(map(str, command)), cwd=cwd, capture_output=True,
                                  text=True, timeout=timeout)
            record = dict(command=list(map(str, command)), cwd=str(cwd),
                          returncode=done.returncode, stdout=done.stdout, stderr=done.stderr)
        except subprocess.TimeoutExpired as exc:
            record = dict(command=list(map(str, command)), cwd=str(cwd), returncode=None,
                          stdout=str(exc.stdout or ''), stderr=str(exc.stderr or ''), timeout=True)
        record['seconds'] = time.monotonic() - begin
        result['records'].append(record)
        print(str(command[-1]), record['returncode'], flush=True)
        if record['returncode'] != 0:
            raise ValueError('failed check; receipt and work preserved')
        return record

    try:
        if subprocess.check_output(['git', 'diff', START, '--', 'act', 'configs'], cwd=ROOT):
            raise ValueError('experimental implementation/config/results changed')
        for name in PROTECTED:
            raw = (ROOT / name).read_bytes()
            if raw != subprocess.check_output(['git', 'show', START + ':' + name], cwd=ROOT):
                raise ValueError('protected evidence changed: ' + name)
            result['protected'].append(dict(path=name, sha256=hashlib.sha256(raw).hexdigest()))
        old_name = 'docs/review_revision_inventory_20260921_r5.json'
        old = json.loads((ROOT / old_name).read_bytes())
        commit = subprocess.check_output(['git', 'log', '-1', '--format=%H', '--', old_name],
                                         cwd=ROOT, text=True).strip()
        old_records = old['current_manuscript'] + old['current_response_materials'] + old['protected_unchanged_scientific_materials']
        for r in old_records:
            raw = subprocess.check_output(['git', 'show', commit + ':' + r['path']], cwd=ROOT)
            if len(raw) != r['bytes'] or hashlib.sha256(raw).hexdigest() != r['sha256']:
                raise ValueError('historical R5 identity: ' + r['path'])
        result['historical_r5'] = dict(commit=commit, verified_records=len(old_records))
        for name in ('test_submission_review.py', 'test_review_handoff.py',
                     'test_moe_main_source.py', 'test_manuscript_source_contract.py',
                     'test_moe_input_composition.py', 'test_source_audit_replay.py',
                     'test_rebuild_moe_main_tables.py'):
            flags = ['-S'] if name == 'test_rebuild_moe_main_tables.py' else ['-I', '-S']
            run([sys.executable, '-B', *flags, ROOT / 'scripts' / name])
        # Keep all seven non-snapshot arithmetic/claim checks. R5 snapshot
        # identity itself was checked against Git above, not silently skipped.
        for test in ('test_composed_counts_and_both_directions',
                     'test_witness_arithmetic_separately_reconstructed',
                     'test_parent_unchanged_and_no_new_execution_claim',
                     'test_primitive_risks_not_claimed_resolved',
                     'test_unqualified_positive_count_phrases_removed_from_active_text',
                     'test_portable_check_not_upgraded_or_released',
                     'test_response_and_manuscript_local_links'):
            run([sys.executable, '-B', '-I', '-S', ROOT / 'scripts/test_review_revision.py', 'Revision.' + test])
        for name in ('rebuild_moe_main_tables.py', 'render_submission_tables.py'):
            run([sys.executable, '-B', '-I', '-S', ROOT / 'scripts' / name, '--check'])
        # Actual saved-tensor reread is a separate source audit, not a forward
        # pass or verification rerun. Uses the existing ACT decoding environment.
        run([sys.executable, '-B', ROOT / 'scripts/check_source_audit_replay.py'], timeout=180)
        run([sys.executable, '-B', ROOT / 'scripts/audit_moe_input_composition.py', '--check'], timeout=180)
        exported = json.loads(run([sys.executable, '-B', '-I', '-S',
                                  ROOT / 'scripts/build_submission_review_kit.py',
                                  '--output', args.work / 'kit'])['stdout'])
        result['export'] = exported
        moved = args.work / 'relocated-kit'
        shutil.copytree(args.work / 'kit', moved)
        result['relocated_check'] = json.loads(run([
            sys.executable, '-B', '-I', '-S', moved / 'scripts/check_submission_review_kit.py',
            '--manifest-sha256', exported['manifest_sha256']], cwd=args.work)['stdout'])
        build = args.work / 'pdf'
        build.mkdir()
        for _ in range(2):
            run([args.pdflatex, '-no-shell-escape', '-halt-on-error', '-interaction=nonstopmode',
                 '-output-directory', build, 'review_main.tex'], cwd=moved / 'paper')
        log = (build / 'review_main.log').read_text(errors='replace')
        if any(s in log for s in ('undefined references', 'Citation `', 'Overfull \\hbox', 'Overfull \\vbox')):
            raise ValueError('unresolved citation/reference or overfull PDF; retained')
        pdf = build / 'review_main.pdf'
        result['pdf'] = dict(path=str(pdf), bytes=pdf.stat().st_size,
                             sha256=hashlib.sha256(pdf.read_bytes()).hexdigest())
        result['status'] = 'PASS'
    except Exception as exc:
        result['error'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        paths = sorted(p for p in (ROOT / 'paper').rglob('*') if p.suffix in ('.md', '.tex'))
        paths += [ROOT / 'docs/competition_guarantee_disposition_20260923.md',
                  ROOT / 'scripts/validate_competition_claims.py']
        result['current_revision'] = [dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size,
                                         sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write('\n')
    if result['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
