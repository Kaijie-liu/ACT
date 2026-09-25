"""Stdlib-only integrity/accounting check, NOT a neural-network proof checker.

Run with an independently obtained manifest hash. The interpreter and this
checker remain trusted. A valid hash is identity, not semantic correctness.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import time

ROOT = Path(__file__).resolve().parents[1]
LIMITATIONS = {
    'new_solver_calls': 0, 'new_model_forwards': 0,
    'independent_network_reproof': False, 'clean_environment_empirical_rerun': False,
    'human_review_completed': False, 'public_or_anonymous_release': False,
    'raw_models_inputs_or_proofs_bundled': False,
}


def require(value, message):
    if not value:
        raise ValueError(message)


def load_json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(data, object_pairs_hook=unique, parse_constant=constant)


def validate_inventory(root, expected):
    require(re.fullmatch('[0-9a-f]{64}', expected) is not None, 'invalid external identity')
    path = root / 'manifest.json'
    require(path.is_file() and not path.is_symlink(), 'missing/linked manifest')
    data = path.read_bytes()
    require(len(data) <= 1024 * 1024, 'oversized manifest')
    require(hashlib.sha256(data).hexdigest() == expected, 'manifest identity mismatch')
    manifest = load_json(data)
    require(manifest['schema'] == 'MOE_SUBMISSION_REVIEW_KIT_V1', 'schema')
    require(manifest['evidence_grade'] == 'ARCHIVED_ACCOUNTING_ONLY', 'evidence grade')
    require(manifest['limitations'] == LIMITATIONS, 'unsupported guarantee upgrade')
    files = manifest['files']
    require(type(files) is list and 1 <= len(files) <= 100, 'inventory size')
    seen = set()
    total = 0
    for record in files:
        name = record['path']
        require(type(name) is str and '\\' not in name and '\x00' not in name, 'invalid path')
        rel = PurePosixPath(name)
        require(not rel.is_absolute() and '..' not in rel.parts and '.' not in name.split('/')
                and str(rel) == name and name != 'manifest.json', 'outside/noncanonical path')
        require(name not in seen, 'duplicate file')
        seen.add(name)
        require(type(record['bytes']) is int and 0 <= record['bytes'] <= 16 * 1024 * 1024, 'file size')
        require(type(record['sha256']) is str and re.fullmatch('[0-9a-f]{64}', record['sha256']), 'file digest')
        p = root / rel
        require(all(not q.is_symlink() for q in (p, *p.parents) if q.is_relative_to(root)), 'symlink')
        require(p.resolve().is_relative_to(root.resolve()) and p.is_file(), 'missing/outside file')
        require(p.stat().st_size == record['bytes'], 'size mismatch')
        content = p.read_bytes()
        require(hashlib.sha256(content).hexdigest() == record['sha256'], 'file identity mismatch')
        total += len(content)
        require(total <= 64 * 1024 * 1024, 'kit size')
    entries = list(root.rglob('*'))
    require(not any(p.is_symlink() for p in entries), 'extra symlink')
    actual = {str(p.relative_to(root)) for p in entries if not p.is_dir()}
    require(actual == seen | {'manifest.json'}, 'extra/missing file inventory')
    return manifest, total + len(data)


def check(root, expected):
    started = time.monotonic()
    manifest, size = validate_inventory(root, expected)
    # Only after external identity and ALL files have been checked, load the
    # bound table renderer. It imports no model, solver, or raw-data decoder.
    name = root / 'scripts/render_submission_tables.py'
    require(str(name.relative_to(root)) in {r['path'] for r in manifest['files']}, 'missing renderer')
    spec = importlib.util.spec_from_file_location('submission_tables', name)
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    require(renderer.render(root) == (root / 'paper/results/submission_tables.tex').read_text(),
            'stale review-paper table')
    closure_path = root / 'scripts/summarize_proof_closure.py'
    require(str(closure_path.relative_to(root)) in {r['path'] for r in manifest['files']}, 'missing proof-frontier recount')
    closure_spec = importlib.util.spec_from_file_location('proof_closure', closure_path)
    closure = importlib.util.module_from_spec(closure_spec)
    closure_spec.loader.exec_module(closure)
    frontier = closure.check(root)
    return {'status': 'PASS', 'evidence_grade': 'ARCHIVED_ACCOUNTING_ONLY',
            'manifest_sha256': expected, 'files': len(manifest['files']), 'bytes': size,
            'seconds': time.monotonic() - started, 'table_rows': 21,
            'additional_metamoe_rows': 2,
            'additional_proof_frontier_calls': frontier['real_summary']['calls'],
            'limitations': LIMITATIONS,
            'meaning': 'Captured-file integrity and archived accounting only; no independent network reproof.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    # Avoid interpreter-generated files in an otherwise immutable kit even
    # when the user did not supply -B. -I -S is recommended for isolation.
    import sys
    sys.dont_write_bytecode = True
    print(json.dumps(check(ROOT, args.manifest_sha256), sort_keys=True))


if __name__ == '__main__':
    main()
