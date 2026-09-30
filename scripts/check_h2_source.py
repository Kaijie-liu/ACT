"""Generate NEW fixed H2 synthetic records or recheck without any solver.

The cooperative per-arm deadline excludes fixture creation, imports and archive
publication. No hard-budget, portable bundle, real-model or timing claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scoped_source.endpoint_source_audit import audit_case
from source_enclosure.format import identity

PROTOCOL_SHA256 = '154e6171c8c2ffad14b6394ae19419827f22184214522a64a27cd3a727a7180a'
SOURCES = [
    'configs/h2_source_controls_20260930.json',
    'scoped_source/endpoint_source_build.py', 'scoped_source/endpoint_source_check.py',
    'scoped_source/endpoint_source_controls.py', 'scoped_source/endpoint_source_tests.py',
    'scoped_source/endpoint_source_audit.py', 'scoped_source/endpoint_build.py',
    'scoped_source/endpoint_check.py', 'scoped_source/endpoint_controls.py',
    'scoped_source/sparse_controls.py', 'scoped_source/sparse_check.py',
    'scoped_source/sparse_ir.py', 'scoped_source/graph.py',
    'source_enclosure/format.py', 'router_source/checker.py', 'upstream_source/checker.py',
    'act/back_end/solver/lp_certificate.py', 'act/back_end/solver/sparse_lp_certificate.py',
    'scripts/check_h2_source.py',
]


def dump(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2); stream.write('\n')


def sources():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}


def generate(directory):
    if not directory.is_relative_to(ROOT.parent/'baseline_runs'):
        raise ValueError('new local control directory must be within MOE/baseline_runs')
    from scoped_source.endpoint_source_controls import cases, protocol, weighted_negative_point
    from scoped_source.endpoint_source_build import build
    from act.back_end.solver.lp_certificate import propose
    config = protocol(); selected = cases()
    directory.mkdir()  # refuse any existing run, including partial/failed attempts
    manifest = {'schema': 'H2_SOURCE_RUN_MANIFEST_V1', 'protocol': config,
                'implementation_files': sources(),
                'cases': [{'name': name, 'source_sha256': identity(doc),
                           'reuse': [[list(pair), j] for pair, j in reuse]} for name, doc, reuse in selected]}
    dump(directory/'manifest.json', manifest)
    for name, doc, reuse in selected:
        dump(directory/(name+'.source.json'), doc)
        for mode in config['arms']:
            try:
                package = build(doc, expected_source_sha256=identity(doc), mode=mode, reuse_keys=reuse,
                                deadline=time.monotonic()+config['maximum_seconds_per_inprocess_arm'], proposer=propose)
                dump(directory/(name+'.'+mode+'.json'), package)
                if name == 'weighted_sign' and mode == 'mccormick':
                    dump(directory/'mc_negative_point.json', weighted_negative_point(doc, package))
            except Exception as error:
                dump(directory/(name+'.'+mode+'.error.json'), {'type': type(error).__name__, 'error': str(error)})
                raise


def audit(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    if (set(manifest) != {'schema', 'protocol', 'implementation_files', 'cases'} or
            manifest['schema'] != 'H2_SOURCE_RUN_MANIFEST_V1' or identity(manifest['protocol']) != PROTOCOL_SHA256 or
            manifest['implementation_files'] != sources()):
        raise ValueError('frozen protocol/implementation mismatch')
    if [c['name'] for c in manifest['cases']] != manifest['protocol']['cases']:
        raise ValueError('frozen case roster')
    rows = []; files = {'manifest.json': hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest()}
    def read(name):
        content = (directory/name).read_bytes(); files[name] = hashlib.sha256(content).hexdigest()
        return json.loads(content)
    for case in manifest['cases']:
        name = case['name']; doc = read(name+'.source.json')
        packages = {mode: read(name+'.'+mode+'.json') for mode in manifest['protocol']['arms']}
        if any(p['reuse_requested'] != case['reuse'] for p in packages.values()):
            raise ValueError('frozen reuse selection')
        negative = read('mc_negative_point.json') if name == 'weighted_sign' else None
        rows.append({'case': name, **audit_case(doc, packages, expected_source_sha256=case['source_sha256'], negative_point=negative)})
    return {'schema': 'H2_SOURCE_AUDIT_V1', 'protocol_sha256': PROTOCOL_SHA256,
            'implementation_files': manifest['implementation_files'], 'files': files, 'cases': rows,
            'real_requests': 0, 'hard_budget_supervision': False, 'runtime_comparison': False,
            'deployed_float_SAFE': False,
            'scope': 'Declared synthetic real graphs, exact stored binary64 coefficients; native program correspondence remains trusted.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run', type=Path); group.add_argument('--check', type=Path)
    parser.add_argument('--archive', type=Path, help='Write a new compact docs report on run; compare on check.')
    args = parser.parse_args(); directory = (args.run or args.check).resolve()
    if args.run: generate(directory)
    report = audit(directory)
    if args.archive:
        path = args.archive.resolve()
        if args.run:
            if not path.is_relative_to(ROOT/'docs'): raise ValueError('compact archive must be in docs')
            dump(path, report)
        elif json.loads(path.read_text()) != report:
            raise ValueError('compact result or original file identity differs')
    print(json.dumps({'status': 'PASS', 'cases': [
        {'name': r['case'], 'arms': {k: {'status': v['checked']['status'], 'positive': v['checked']['positive'],
                                      'required': v['checked']['required']} for k, v in r['arms'].items()}}
        for r in report['cases']]}, indent=2))


if __name__ == '__main__': main()
