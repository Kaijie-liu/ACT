"""Bounded local manuscript/kit validation; does not run models or proofs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='NEW receipt, preserve failures')
    parser.add_argument('--work', required=True, type=Path, help='NEW local work directory')
    parser.add_argument('--pdflatex', required=True, type=Path, help='already installed executable; never install')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('receipt exists')
    args.work.mkdir(parents=False, exist_ok=False)
    records = []
    result = {'schema': 'SUBMISSION_REVIEW_VALIDATION_V1', 'status': 'FAILED_RETAINED',
              'records': records, 'work': str(args.work), 'starting_head': subprocess.check_output(
                  ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'new_solver_calls': 0, 'new_model_forwards': 0, 'new_source_propagations': 0,
              'new_output_bounds': 0, 'human_review_completed': False,
              'clean_install_or_empirical_rerun': False, 'public_or_anonymous_release': False,
              'input98_followup': 'STOP_INPUT98_FOLLOWUP'}

    def run(command, cwd=ROOT, timeout=60):
        begin = time.monotonic()
        try:
            p = subprocess.run([str(c) for c in command], cwd=cwd, capture_output=True,
                               text=True, timeout=timeout)
            record = dict(command=[str(c) for c in command], cwd=str(cwd), returncode=p.returncode,
                          stdout=p.stdout, stderr=p.stderr, timeout=False)
        except subprocess.TimeoutExpired as exc:
            record = dict(command=[str(c) for c in command], cwd=str(cwd), returncode=None,
                          stdout=str(exc.stdout or ''), stderr=str(exc.stderr or ''), timeout=True)
        record['seconds'] = time.monotonic() - begin
        records.append(record)
        print(Path(str(command[-1])).name, record['returncode'], flush=True)
        if record['returncode'] != 0:
            raise RuntimeError('failed command; retained in receipt')
        return record

    try:
        for name in ('test_submission_review.py', 'test_review_handoff.py', 'test_moe_main_source.py',
                     'test_manuscript_source_contract.py', 'test_moe_input_composition.py',
                     'test_review_revision.py', 'test_rebuild_moe_main_tables.py'):
            flags = ['-S'] if name == 'test_rebuild_moe_main_tables.py' else ['-I', '-S']
            run([sys.executable, '-B', *flags, ROOT / 'scripts' / name])
        for name in ('rebuild_moe_main_tables.py', 'render_submission_tables.py',
                     'build_review_revision_inventory.py'):
            run([sys.executable, '-B', '-I', '-S', ROOT / 'scripts' / name, '--check'])
        record = run([sys.executable, '-B', '-I', '-S', ROOT / 'scripts/build_submission_review_kit.py',
                      '--output', args.work / 'kit'])
        exported = json.loads(record['stdout'])
        result['export'] = exported
        moved = args.work / 'relocated-kit'
        shutil.copytree(args.work / 'kit', moved)
        checked = run([sys.executable, '-B', '-I', '-S', moved / 'scripts/check_submission_review_kit.py',
                       '--manifest-sha256', exported['manifest_sha256']], cwd=args.work)
        result['relocated_check'] = json.loads(checked['stdout'])
        build = args.work / 'pdf'
        build.mkdir()
        for _ in range(2):
            run([args.pdflatex, '-no-shell-escape', '-halt-on-error', '-interaction=nonstopmode',
                 '-output-directory', build, 'review_main.tex'], cwd=moved / 'paper')
        log = (build / 'review_main.log').read_text(errors='replace')
        if any(s in log for s in ('undefined references', 'Citation `', 'Overfull \\hbox', 'Overfull \\vbox')):
            raise ValueError('unresolved reference/citation or overfull layout; inspect retained log')
        pdf = build / 'review_main.pdf'
        result['pdf'] = {'path': str(pdf), 'bytes': pdf.stat().st_size,
                         'sha256': hashlib.sha256(pdf.read_bytes()).hexdigest()}
        result['status'] = 'PASS'
    except Exception as exc:
        result['error'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        inv = ROOT / 'docs/review_revision_inventory_20260921_r5.json'
        result['revision_inventory_sha256'] = hashlib.sha256(inv.read_bytes()).hexdigest() if inv.exists() else None
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write('\n')
    if result['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
