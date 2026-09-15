"""Package the existing conv98 pre-F0 proof; no network, model or solver queries."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import time
import zipfile

from portable_proof.runtime import compact, digest, original_bytes

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/moe/results/conv_pre_f0_rational_20260915_r2/rank24_monolithic'


def extract(path, names=None):
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    selected = []
    for node in ast.parse(text).body:
        if names is None:
            keep = not isinstance(node, ast.FunctionDef) or node.name != 'propose'
        else:
            keep = isinstance(node, ast.FunctionDef) and node.name in names
        if keep:
            selected.append(''.join(lines[node.lineno - 1:node.end_lineno]))
    return '\n\n'.join(selected) + '\n'


def build(destination, source=SOURCE):
    start = time.monotonic()
    destination = Path(destination).resolve()
    if not destination.is_relative_to(Path('/data1/Kane/MOE')):
        raise ValueError('output must remain in authorized workspace')
    destination.mkdir(parents=True, exist_ok=False)
    sources = {}
    def put(name, data):
        p = destination / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data if isinstance(data, bytes) else data.encode())
    def source_text(relative, names=None):
        p = ROOT / relative
        sources[relative] = digest(p.read_bytes())
        return extract(p, names)
    for package in ('act', 'act/back_end', 'act/back_end/solver'):
        put(f'code/{package}/__init__.py', '')
    for module in ('lp_certificate', 'sparse_lp_certificate', 'check_hz_lp_export', 'check_rational_mccormick'):
        rel = f'act/back_end/solver/{module}.py'
        put('code/' + rel, source_text(rel))
    helper = source_text('scripts/check_conv_request_sign_lp.py', {'expected_scope', 'property_vector', 'check_reuse'})
    agg = source_text('scripts/check_conv_pre_f0_r2.py', {'order_bounds', 'aggregate'})
    trusted = ['network_input_to_HZ_interval_sources_and_expert_binding',
               'membership_pair_guard_lowering', 'router_infeasibility_exclusions']
    put('code/portable_request.py', 'from fractions import Fraction\nimport math\nimport itertools\n'
        + 'TRUSTED=' + repr(trusted) + '\n' + helper + '\n' + agg)
    for rel, target in [('portable_proof/runtime.py', 'code/runtime.py'), ('portable_proof/launcher.py', 'verify.py')]:
        p = ROOT / rel
        sources[rel] = digest(p.read_bytes())
        put(target, p.read_bytes())
    license_path = ROOT / 'LICENSE'
    if license_path.exists():
        put('LICENSE', license_path.read_bytes())
    manifest = json.loads((source / 'manifest.json').read_bytes())
    # Follow only proof references, not job paths or historical execution logs.
    refs = {'manifest.json': digest((source / 'manifest.json').read_bytes())}
    def discover(obj):
        if isinstance(obj, dict):
            if 'file' in obj and 'sha256' in obj:
                if Path(obj['file']).name != obj['file']:
                    raise ValueError('nonlocal reference')
                previous = refs.setdefault(obj['file'], obj['sha256'])
                if previous != obj['sha256']:
                    raise ValueError('conflicting reference')
            for v in obj.values():
                discover(v)
        elif isinstance(obj, list):
            for v in obj:
                discover(v)
    discover(manifest)
    logical = {}
    unique = set()
    logical_bytes = 0
    stored_bytes = 0
    array_refs = 0
    with zipfile.ZipFile(destination / 'evidence.zip', 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        def intern(obj):
            nonlocal stored_bytes
            data = compact(obj)
            sha = digest(data)
            if sha not in unique:
                archive.writestr(sha, data)
                unique.add(sha)
                stored_bytes += len(data)
            return sha
        def encode(obj):
            nonlocal array_refs
            if isinstance(obj, list):
                if len(obj) >= 32:
                    array_refs += 1
                    return {'$array': intern(obj)}
                return [encode(v) for v in obj]
            if isinstance(obj, dict):
                if '$array' in obj:
                    raise ValueError('reserved transport key')
                return {k: encode(v) for k, v in obj.items()}
            return obj
        for name, sha in sorted(refs.items()):
            p = source / name
            if p.is_symlink():
                raise ValueError('symlink source')
            data = p.read_bytes()
            obj = json.loads(data)
            if digest(data) != sha or original_bytes(obj) != data:
                raise ValueError('source hash/serialization mismatch: ' + name)
            logical_bytes += len(data)
            logical[name] = {'original_sha256': sha, 'root': intern(encode(obj))}
    statement = {'request': manifest['request'], 'routes': manifest['routes'],
                 'sources': {k: manifest[k] for k in ('common_facts', 'joint_source', 'router_source')}}
    files = {str(p.relative_to(destination)): digest(p.read_bytes()) for p in destination.rglob('*') if p.is_file()}
    meta = {'schema': 'PORTABLE_PRE_F0_REQUEST_V1', 'statement': statement,
            'manifest': {'file': 'manifest.json', 'sha256': refs['manifest.json']},
            'logical_files': logical, 'files': files, 'checker_sources': sources,
            'expected_result': json.loads((source / 'independent.json').read_bytes()),
            'trust_boundary': 'Given stored HZ and trusted source/guard/route lowering; NOT a deployed floating-point proof.'}
    put('bundle.json', original_bytes(meta))
    report = {'bundle_sha256': digest((destination / 'bundle.json').read_bytes()),
              'statement_sha256': digest(compact(statement)), 'logical_files': len(logical),
              'logical_original_bytes': logical_bytes, 'unique_object_count': len(unique),
              'deduplicated_uncompressed_bytes': stored_bytes, 'large_array_references': array_refs,
              'bundle_bytes': sum(p.stat().st_size for p in destination.rglob('*') if p.is_file()),
              'source_directory_bytes': sum(p.stat().st_size for p in source.iterdir() if p.is_file()),
              'pack_seconds': time.monotonic() - start, 'new_solver_calls': 0}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    a = parser.parse_args()
    print(json.dumps(build(a.destination), indent=2, sort_keys=True))
