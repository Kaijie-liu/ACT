"""Bound candidate intake for one whole request; no historic proof reuse."""
from itertools import combinations
from scoped_proof.evidence import bind_source, roster
from scoped_proof.io import load, sha, tick
from source_enclosure.format import identity

SCHEMA='RESIDUAL_PROOF_POLICY_V1'
MODES=('pairwise','shared')
PHASES=('intake','route_propose','route_check','construct','source_check','propose','aggregate')

def policy(spec):
    p=spec['proof_policy']
    if (set(p)!={'schema','mode','router_proposal','parse_cache','checker_cache'} or
            p['schema']!=SCHEMA or p['mode'] not in MODES or
            p['router_proposal']!='FINAL_AFFINE_DUAL_ONLY' or
            p['parse_cache'] is not False or p['checker_cache'] is not False):
        raise ValueError('unregistered residual proof policy')
    return p

def phases(spec):
    policy(spec)
    return PHASES

def route_roster(scope):
    return [{'higher':j,'lower':i} for j in range(scope['experts'])
            for i in range(scope['experts']) if i!=j]

def file_record(path):
    if path.is_symlink(): raise ValueError('evidence symlink')
    return {'sha256':sha(path),'bytes':path.stat().st_size}

def route_context(spec, inv, digest, prefix_hash, index):
    context={'mode':policy(spec)['mode'],'invocation':inv['invocation'],
        'request_sha256':identity(spec['scope']),'source_sha256':digest,
        'prefix_sha256':prefix_hash,'index':index}
    if context['mode']=='pairwise':
        context.update(route_roster(spec['scope'])[index])
    return context

def route_inputs(root, deadline, *, require_complete=True):
    tick(deadline)
    inv=load(root/'invocation.json');spec=load(root/'spec.json',inv['spec_file_sha256'])
    mode=policy(spec)['mode'];doc=load(root/'source.json');digest=bind_source(doc,spec['scope'])
    prefix=load(root/'router_prefix.json')
    if prefix['source_sha256']!=digest:raise ValueError('prefix source identity')
    ph=identity(prefix);count=len(route_roster(spec['scope'])) if mode=='pairwise' else 1
    files={};rows=[]
    for p in sorted((root/'route_candidates').glob('*.json')):
        tick(deadline)
        try:index=int(p.stem)
        except ValueError as exc:raise ValueError('route filename') from exc
        if p.name!=f'{index:04d}.json' or not 0<=index<count:raise ValueError('route candidate index')
        entry=load(p);want=route_context(spec,inv,digest,ph,index)
        if (set(entry)!={'schema','context','candidate'} or entry['schema']!='RESIDUAL_ROUTE_FILE_V1' or
                entry['context']!=want):raise ValueError('route candidate run/source binding')
        if mode=='pairwise' and any(entry['candidate'][k]!=want[k] for k in ('higher','lower')):
            raise ValueError('route direction')
        files[p.name]=file_record(p);rows.append(entry['candidate'])
    complete=(root/'route_complete.json').exists()
    context={'mode':mode,'invocation':inv['invocation'],'request_sha256':identity(spec['scope']),
        'source':file_record(root/'source.json'),'prefix':file_record(root/'router_prefix.json'),
        'files':files,'required_files':count,'required_margins':len(route_roster(spec['scope']))}
    if complete:
        if (load(root/'route_complete.json')!={'schema':'RESIDUAL_ROUTE_COMPLETE_V1',**context} or
                set(files)!={f'{i:04d}.json' for i in range(count)}):
            raise ValueError('route completion inventory')
    elif require_complete:raise ValueError('incomplete route publication')
    context['completion']=file_record(root/'route_complete.json') if complete else None
    tick(deadline)
    candidates=rows if mode=='pairwise' else (rows[0] if rows else None)
    return doc,prefix,candidates,context

def accepted_routes(root, deadline):
    doc,prefix,candidates,context=route_inputs(root,deadline)
    receipt=load(root/'route_check.json')
    if (set(receipt)!={'schema','context','frontier'} or
            receipt['schema']!='RESIDUAL_ROUTE_CHECK_V1' or receipt['context']!=context):
        raise ValueError('route check reception binding')
    return doc,prefix,candidates,receipt

def retained_roster(scope, bundle):
    """Use unchanged global pair/property indices, never reindex a reduced list."""
    all_pairs = [list(p) for p in combinations(range(scope['experts']), 2)]
    decisions = bundle['frontier']['pairs']
    if [d['pair'] for d in decisions] != all_pairs:
        raise ValueError('complete route ledger required')
    kept = {tuple(d['pair']) for d in decisions if d['status'] == 'RETAINED'}
    if not kept or [p['pair'] for p in bundle['pairs']] != [p for p in all_pairs if tuple(p) in kept]:
        raise ValueError('retained construction inventory')
    return [(i, row) for i, row in enumerate(roster(scope)) if tuple(row['pair']) in kept]


def output_inputs(root, scope, bundle, token, deadline):
    expected = retained_roster(scope, bundle)
    allowed = {i for i, _ in expected}
    candidates, files = {}, {}
    for p in sorted((root/'candidates').glob('*.json')):
        tick(deadline)
        try:
            index = int(p.stem)
        except ValueError as exc:
            raise ValueError('output filename') from exc
        if p.name != f'{index:04d}.json' or index not in allowed or index in candidates:
            raise ValueError('output index/duplicate/excluded obligation')
        candidates[index] = load(p)
        files[p.name] = file_record(p)
    complete = (root/'proposal_complete.json').exists()
    if complete:
        want = {'schema': 'RESIDUAL_OUTPUT_COMPLETE_V1', 'invocation': token,
                'request_sha256': identity(scope), 'bundle_sha256': identity(bundle),
                'original_required': len(roster(scope)), 'retained_indices': [i for i, _ in expected], 'files': files}
        if load(root/'proposal_complete.json') != want or set(candidates) != allowed:
            raise ValueError('complete output proposal inventory')
    tick(deadline)
    return candidates, complete
