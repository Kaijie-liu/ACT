"""Resource-only S1 continuation; original numerical executable is unchanged."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from scoped_proof.io import load, save, sha
from architecture_s1_execution import Artifacts, digest, read_record, require
from architecture_s1_study import all_requests, resources

ORIGINAL = Path('/data1/Kane/MOE/executions/architecture_s1_evaluation_20261030_r1')
CAP = 512*2**20


def tree_identity(root):
    rows = []
    for p in sorted(Path(root).rglob('*')):
        require(not p.is_symlink(), 'no alias in prior evidence')
        if p.is_file():
            rows.append((str(p.relative_to(root)), p.stat().st_size, sha(p)))
    return dict(sha256=digest(rows), files=len(rows), bytes=sum(r[1] for r in rows))


def remaining_requests(requests, root):
    """Errors and partial attempts with terminal ledgers are spent, never retried."""
    known = {r['id'] for r in requests}
    ledgers = {p.name.removesuffix('.ledger.json.gz'):p for p in Path(root).glob('*.ledger.json.gz')}
    require(set(ledgers) <= known, 'unknown historical request')
    for name, path in ledgers.items():
        row = read_record(path)
        require(row['id'] == name, 'historical ledger identity')
    for name in known-set(ledgers):
        require(not (Path(root)/name).exists(), 'unsettled attempt cannot be retried')
    return [r for r in requests if r['id'] not in ledgers]


def locate_requests(requests, roots):
    """One authoritative location per request; reject a retry rather than choose."""
    result = {}
    for i, root in enumerate(roots):
        for p in Path(root).glob('*.ledger.json.gz'):
            name = p.name.removesuffix('.ledger.json.gz')
            require(name not in result, 'duplicate request across continuations')
            result[name] = i
    require(set(result) <= {r['id'] for r in requests}, 'unknown aggregate request')
    return result


def validate(spec):
    c = spec['continuation']
    old = load(c['manifest'], c['manifest_sha256'])
    require('continuation' not in old, 'one approved S1 continuation')
    # The whitelist changes only capacity, the remaining roster, and its binding.
    unchanged = {k:v for k,v in spec.items() if k not in
        ('artifact_bytes', 'request_catalog_sha256', 'continuation')}
    expected = {k:v for k,v in old.items() if k not in ('artifact_bytes', 'request_catalog_sha256')}
    require(unchanged == expected, 'no numerical, input, method or deadline change')
    require(spec['artifact_bytes'] == CAP and old['artifact_bytes'] == 128*2**20,
        'approved aggregate512MiB only')
    prior = Path(c['root'])
    require(tree_identity(prior) == c['tree'], 'frozen prior evidence unchanged')
    account = load(prior/'artifact_account.json')
    require(account['used'] == c['prior_charged_bytes'] and account['used'] <= account['cap'],
        'retain all prior storage charges')
    requests = all_requests(old['models'])
    require(digest(requests) == old['request_catalog_sha256'], 'original full denominator')
    remaining = remaining_requests(requests, prior)
    queue = read_record(prior/'queue_complete.json.gz')
    require(queue['status'] == 'PREFIX_STOP' and queue['settled'] == len(requests)-len(remaining)
        and queue['not_executed'] == len(remaining) and queue['missing_request_cost'] == 0,
        'terminated and fully accounted old prefix')
    require(digest(remaining) == spec['request_catalog_sha256'] and len(remaining) == c['remaining'],
        'fixed order, only unexecuted requests')
    require(queue['summed_request_seconds'] == c['prior_request_seconds'], 'old time retained')
    require(Path(c['numerical_execution']) == ORIGINAL, 'original numerical executable')
    return old, requests, remaining


def freeze(path, prior, manifest, control):
    old = load(manifest); requests = all_requests(old['models'])
    remaining = remaining_requests(requests, prior)
    queue = read_record(prior/'queue_complete.json.gz')
    receipt = load(control)
    require(receipt['status'] == 'PASS', 'continuation controls')
    spec = copy.deepcopy(old)
    spec['artifact_bytes'] = CAP
    spec['request_catalog_sha256'] = digest(remaining)
    spec['continuation'] = dict(root=str(prior.resolve()), manifest=str(manifest.resolve()),
        manifest_sha256=sha(manifest), tree=tree_identity(prior),
        prior_charged_bytes=load(prior/'artifact_account.json')['used'],
        prior_request_seconds=queue['summed_request_seconds'], remaining=len(remaining),
        numerical_execution=str(ORIGINAL), authorization='2026-10-30 user: 批准共享上限 512 MiB',
        control=str(control.resolve()), control_sha256=sha(control),
        controller_sources={p:sha(ROOT/p) for p in ('scripts/architecture_s1_continuation.py',
            'scripts/architecture_s1_audit.py')},
        scope='counts describe full S1; request_catalog binds only unexecuted remainder; no retry')
    validate(spec)
    save(path, spec)
    print('FROZEN_CONTINUATION', len(remaining), sha(path))


def run(root, manifest):
    began = time.monotonic()
    require(ROOT != Path('/data1/Kane/MOE/ACT') and (ROOT/'.frozen_commit').is_file(),
        'committed immutable continuation controller')
    resource.setrlimit(resource.RLIMIT_AS, (8*2**30, 8*2**30))
    spec = load(manifest); _, _, remaining = validate(spec); c = spec['continuation']
    require(all(sha(ROOT/p) == h for p,h in c['controller_sources'].items()), 'controller freeze')
    require(sha(c['control']) == c['control_sha256'], 'control receipt')
    require(all(sha(ORIGINAL/p) == h for p,h in spec['source_files'].items()), 'original numerical source')
    require((ORIGINAL/'.frozen_commit').read_text().strip() == 'd139f9335ad9e780d42a12212f85784e4702316c',
        'original numerical commit')
    import genbab_cpu_ready as backend
    require(digest(backend.source_identity()) == spec['backend_sources_sha256'], 'unchanged engine')
    for model in spec['models']:
        require(sha(model['checkpoint']) == model['checkpoint_sha256'], 'unchanged weights')
    observation = resources(spec['workers'])
    require(observation['ordinary_available_disk_bytes'] >= 2*2**30, '2GiB disk headroom')
    require(observation['load'][0]+spec['workers'] <= observation['allowed_cpus'], 'CPU headroom')
    require(observation['memory']['MemAvailable'] >= (8*spec['workers']+4)*2**30, 'memory headroom')
    # Carry forward reservations and old failed evidence; never grant a fresh512MiB.
    store = Artifacts.create(root, initial_bytes=c['prior_charged_bytes']+Path(manifest).stat().st_size,
        cap=CAP)
    store.write(root/'start.json.gz', dict(manifest=str(manifest), manifest_sha256=sha(manifest),
        commit=(ROOT/'.frozen_commit').read_text().strip(), numerical_execution=str(ORIGINAL),
        resources=observation, preparation_seconds=time.monotonic()-began, pid=os.getpid(),
        prior_charged_bytes=c['prior_charged_bytes']))
    # Import the actual original module by file, not an edited numerical worker
    # or a monkeypatch. Its __file__ launches the original frozen worker source.
    module_spec = importlib.util.spec_from_file_location('s1_original_queue', ORIGINAL/'scripts/architecture_s1_execution.py')
    original = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(original)
    summary = original.finite_queue(root, remaining, manifest, workers=spec['workers'], seconds=300,
        timing_reserve=tuple(spec['timing_reserve']))
    print(json.dumps(summary), flush=True)
    return 0 if summary['status'] == 'COMPLETE' else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--freeze', type=Path); p.add_argument('--prior', type=Path)
    p.add_argument('--manifest', type=Path, required=True); p.add_argument('--control', type=Path)
    p.add_argument('--run', type=Path)
    a = p.parse_args()
    if a.freeze:
        freeze(a.freeze, a.prior, a.manifest, a.control)
    elif a.run:
        raise SystemExit(run(a.run, a.manifest))
    else:
        p.error('freeze or run required')
