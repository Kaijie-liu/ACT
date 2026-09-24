"""Synthetic-only, owned process for complete-cost construction/proof controls."""
import argparse
import sys
import time
from pathlib import Path

from scoped_proof.io import Events, load, save, tick
from source_enclosure.format import identity


def work(phase, root, deadline):
    plan = load(root / 'plan.json')
    if plan['fixture'] not in ('prunable', 'tied') or plan['mode'] not in ('exhaustive', 'frontier'):
        raise ValueError('synthetic fixture/mode only')
    events = Events(root, phase, plan['started_monotonic'])
    if phase == 'build':
        from checked_route_frontier.fixtures import timing
        from scoped_proof.evidence import context, reconstruct_lp, roster
        doc = events.call('generate_synthetic_source', lambda: timing(plan['fixture']))
        digest = identity(doc)
        if digest != plan['expected_source_sha256']:
            raise ValueError('frozen synthetic source changed')
        if plan['mode'] == 'exhaustive':
            from scoped_source.build import build
            bundle = events.call('all_experts_all_pairs', lambda: build(doc,
                expected_source_sha256=digest, deadline=deadline))
        else:
            from checked_route_frontier.build import prefix, propose_final_affine, finish
            pre = events.call('router_source', lambda: prefix(doc,
                expected_source_sha256=digest, deadline=deadline))
            routes = events.call('fixed_final_affine_duals', lambda: propose_final_affine(doc, pre, deadline=deadline))
            bundle = events.call('checked_frontier_and_lazy_construction', lambda: finish(doc, pre, routes,
                expected_source_sha256=digest, deadline=deadline))
        scope = {k: v for k, v in doc['request'].items()
                 if k not in ('top_k', 'gate', 'tie_policy', 'training')}
        pairs = {tuple(p['pair']): p for p in bundle['pairs']}
        bh = identity(bundle)
        candidates = {}
        for index, obligation in enumerate(roster(scope)):
            tick(deadline)
            pair = pairs.get(tuple(obligation['pair']))
            if pair is None:
                continue
            row = next(v for v in pair['obligations']['rows'] if v['competitor'] == obligation['competitor'])
            lp = events.call('output_materialization_' + str(index),
                             lambda: reconstruct_lp(pair['base'], row))
            candidates[str(index)] = {'schema': 'SCOPED_LP_CANDIDATE_V1', 'status': 'CANDIDATE',
                'context': context(scope, digest, bh, plan['invocation'], index, obligation, lp),
                'certificate': {'lp_sha256': identity(lp), 'inequality_dual': ['0'] * len(lp['b']),
                                'equality_dual': ['0'] * len(lp['h'])}}
        events.call('serialize_source', lambda: save(root / 'source.json', doc))
        events.call('serialize_construction', lambda: save(root / 'construction.json', bundle))
        events.call('serialize_candidates', lambda: save(root / 'candidates.json', candidates))
    elif phase == 'check':
        doc, bundle, candidates = events.call('load_all_evidence', lambda:
            (load(root/'source.json'), load(root/'construction.json'), load(root/'candidates.json')))
        if plan['mode'] == 'exhaustive':
            from scoped_proof.evidence import aggregate
        else:
            from checked_route_frontier.evidence import aggregate
        scope = {k: v for k, v in doc['request'].items()
                 if k not in ('top_k', 'gate', 'tie_policy', 'training')}
        result = events.call('fresh_source_routes_and_all_output_checks', lambda: aggregate(scope, doc, bundle,
            {int(k): v for k, v in candidates.items()}, invocation=plan['invocation'],
            proposal_complete=True, deadline=deadline))
        events.call('serialize_check', lambda: save(root/'check.json', result))
    else:
        raise ValueError('phase')
    tick(deadline)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('build', 'check'))
    p.add_argument('root', type=Path)
    p.add_argument('--deadline', required=True, type=float)
    a = p.parse_args()
    if not sys.flags.no_site:
        raise ValueError('synthetic controls require python -S')
    def forbid(event, args):
        if event == 'import' and args[0].split('.')[0] in ('torch', 'numpy', 'scipy', 'act', 'highspy'):
            raise ImportError('synthetic check cannot import model/solver')
        if event.startswith(('subprocess.', 'socket.')) or event in ('os.system', 'os.fork', 'os.exec'):
            raise PermissionError('no nested executor')
    sys.addaudithook(forbid)
    work(a.phase, a.root.resolve(), a.deadline)
