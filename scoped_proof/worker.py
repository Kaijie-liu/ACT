"""Owned single-phase workers. All loading, construction and LP work is charged."""
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time
from source_enclosure.format import identity
from scoped_proof.io import ROOT, load, save, sha, tick, Events
from scoped_proof.evidence import bind_source, roster, context, aggregate, lower_bound


def intake(spec, deadline):
    for name, digest in spec['sources'].items():
        tick(deadline)
        if sha(ROOT/name) != digest: raise ValueError('frozen implementation changed: '+name)
    if 'environment' in spec:
        env = spec['environment']
        if sys.version != env['python'] or sha(sys.executable) != env['executable_sha256'] or any(importlib.metadata.version(k) != v for k,v in env['packages'].items()):
            raise ValueError('frozen Python/dependency identity')
    if 'source_protocol' in spec:
        if sha(ROOT/spec['source_protocol']['path']) != spec['source_protocol']['sha256']:
            raise ValueError('frozen source protocol changed')
    for key in ('checkpoint', 'input'):
        if sha(spec[key]['path']) != spec[key]['sha256']: raise ValueError('frozen '+key+' changed')
    tick(deadline)
    import torch
    from act.back_end.moe.factory import OutputMoEFactoryConfig, build_output_moe
    from scoped_source.capture import capture
    torch.set_num_threads(2); torch.set_default_dtype(torch.float64)
    payload = torch.load(spec['checkpoint']['path'], map_location='cpu', weights_only=True)
    if payload['format'] != 'act-output-moe-v1': raise ValueError('unsupported checkpoint format')
    # Allocate float64 BEFORE strict load: never round stored doubles through float32.
    model = build_output_moe(OutputMoEFactoryConfig(**payload['factory_config'])).cpu().double().eval()
    model.load_state_dict(payload['state_dict'], strict=True)
    saved = torch.load(spec['input']['path'], map_location='cpu', weights_only=True)
    center = saved['center']  # Old endpoints, labels, matrices and bounds are not used.
    s = spec['scope']
    doc = capture(model, center, label=s['label'], radius=s['radius'], margin=s['margin'], clip=s['clip'], deadline=deadline)
    bind_source(doc, s); tick(deadline)
    return doc


def propose(lp, seconds):
    """One untrusted HiGHS LP attempt; finite duals can be checked at any status."""
    import numpy as np
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix
    from fractions import Fraction
    from upstream_source.checker import csr
    def vector(values): return [float(Fraction(v)) for v in values]
    def matrix(key):
        m = lp[key]; csr(m, [len(lp['b' if key == 'A' else 'h']), len(lp['c'])])
        return csr_matrix((vector(m['data']), m['indices'], m['indptr']), shape=m['shape']) if m['shape'][0] else None
    result = linprog(vector(lp['c']), A_ub=matrix('A'), b_ub=vector(lp['b']) or None,
        A_eq=matrix('E'), b_eq=vector(lp['h']) or None,
        bounds=list(zip(vector(lp['lower']), vector(lp['upper']))), method='highs', options={'time_limit': seconds})
    metadata = {'solver_status': int(result.status), 'message': str(result.message), 'native_limit_seconds': seconds}
    y, z = result.ineqlin.marginals, result.eqlin.marginals
    if y is None or z is None or not np.isfinite(y).all() or not np.isfinite(z).all(): return None, metadata
    cert = {'lp_sha256': identity(lp), 'inequality_dual': [min(0., float(v)) for v in y],
        'equality_dual': [float(v) for v in z]}
    # No primal objective or optimal-status flag licenses a positive verdict.
    return cert, metadata


def work(phase, root, deadline):
    invocation = load(root/'invocation.json'); spec = load(root/'spec.json', invocation['spec_file_sha256'])
    tick(deadline); events = Events(root, phase, invocation['started_monotonic'])
    s = spec['scope']; token = invocation['invocation']
    if phase == 'intake':
        doc = events.call('hash_load_capture_bind', lambda: intake(spec, deadline))
        events.call('serialize_source', lambda: save(root/'source.json', doc))
    elif phase == 'construct':
        from scoped_source.build import build
        doc = events.call('read_source', lambda: load(root/'source.json'))
        digest = bind_source(doc, s)
        bundle = events.call('source_guard_output_construction', lambda: build(doc, expected_source_sha256=digest, deadline=deadline))
        events.call('serialize_construction', lambda: save(root/'construction.json', bundle))
    elif phase == 'source_check':
        from scoped_source.check import check
        doc, bundle = events.call('read_source_and_construction', lambda: (load(root/'source.json'), load(root/'construction.json')))
        result = events.call('independent_source_check', lambda: check(doc, bundle, expected_source_sha256=bind_source(doc, s), deadline=deadline))
        events.call('serialize_source_check', lambda: save(root/'source_check.json', result))
    elif phase == 'propose':
        from full_source.obligations import materialize
        doc, bundle = events.call('read_source_and_construction', lambda: (load(root/'source.json'), load(root/'construction.json')))
        source_hash = bind_source(doc, s); bundle_hash = identity(bundle)
        expected = roster(s); by_pair = {tuple(p['pair']): p for p in bundle['pairs']}
        (root/'candidates').mkdir(exist_ok=False); files = {}
        for i, obligation in enumerate(expected):
            tick(deadline); entry = by_pair[tuple(obligation['pair'])]
            row = next(r for r in entry['obligations']['rows'] if r['competitor'] == obligation['competitor'])
            lp = events.call('materialize_'+str(i), lambda: materialize(entry['base'], row))
            allowance = (deadline-time.monotonic())/(len(expected)-i)
            if not math.isfinite(allowance) or allowance <= 0: raise TimeoutError('no proposal time remaining')
            cert, metadata = events.call('native_lp_'+str(i), lambda: propose(lp, allowance))
            # Even a valid late dual remains partial evidence, not an accepted result.
            candidate = {'schema': 'SCOPED_LP_CANDIDATE_V1',
                'context': context(s, source_hash, bundle_hash, token, i, obligation, lp),
                'certificate': cert, 'status': 'CANDIDATE' if cert is not None else 'NO_CANDIDATE', 'native': metadata}
            name = f'{i:04d}.json'
            files[name] = events.call('serialize_candidate_'+str(i), lambda: save(root/'candidates'/name, candidate))
            tick(deadline)
        events.call('serialize_proposal_completion', lambda: save(root/'proposal_complete.json',
            {'invocation': token, 'request_sha256': identity(s), 'required': len(expected), 'files': files}))
    elif phase == 'aggregate':
        doc, bundle = events.call('read_source_and_construction', lambda: (load(root/'source.json'), load(root/'construction.json')))
        candidates = {}; folder = root/'candidates'
        for p in sorted(folder.glob('*.json')) if folder.exists() else []:
            if p.name != f'{int(p.stem):04d}.json': raise ValueError('unexpected candidate filename')
            index = int(p.stem)
            if index in candidates: raise ValueError('duplicate candidate index')
            candidates[index] = load(p); tick(deadline)
        complete = (root/'proposal_complete.json').is_file()
        if complete:
            manifest = load(root/'proposal_complete.json')
            if (manifest['invocation'] != token or manifest['request_sha256'] != identity(s) or
                manifest['required'] != len(roster(s)) or set(manifest['files']) != {f'{i:04d}.json' for i in range(len(roster(s)))}):
                raise ValueError('proposal completion inventory')
            for name, info in manifest['files'].items():
                if sha(folder/name) != info['sha256'] or (folder/name).stat().st_size != info['bytes']:
                    raise ValueError('candidate changed after publication')
                tick(deadline)
        result = events.call('fresh_full_source_and_bound_check', lambda: aggregate(s, doc, bundle, candidates,
            invocation=token, proposal_complete=complete, deadline=deadline))
        record = events.call('serialize_full_evidence_check', lambda: save(root/'evidence_check.json', result))
        small = {k: result[k] for k in ('status','request_sha256','invocation','required','checked_bounds','positive_bounds',
            'proposal_complete','complete_output_positive_proof','native_float_proof','route_changing_established')}
        small.update(evidence_check=record, missing=len(result['missing']), nonpositive=len(result['nonpositive']))
        events.call('serialize_candidate_terminal', lambda: save(root/'result_candidate.json', small))
    else: raise ValueError('unknown worker phase')
    tick(deadline)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase'); parser.add_argument('root', type=Path)
    parser.add_argument('--deadline', type=float, required=True); a = parser.parse_args()
    if a.phase in ('source_check', 'aggregate'):
        if not sys.flags.no_site: raise ValueError('independent check requires python -S')
        def forbid(event, args):
            if event == 'import' and args[0].split('.')[0] in ('torch','numpy','scipy','highspy','gurobipy','act'):
                raise ImportError('checker may not load model/solver')
            if event.startswith(('subprocess.', 'socket.')) or event in ('os.system','os.fork','os.exec'):
                raise PermissionError('checker external execution')
        sys.addaudithook(forbid)
    work(a.phase, a.root.resolve(), a.deadline)
