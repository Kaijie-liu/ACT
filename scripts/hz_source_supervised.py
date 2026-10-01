"""Finite CPU source policy; parent checks receipts, owned checker does math."""
from fractions import Fraction as F
from itertools import combinations
import time

from scoped_proof.io import ROOT, load, sha, tick
from source_enclosure.format import identity
from scripts.hz_propagation_supervised import finite, required_hash
from scripts import hz_endpoint_supervised as base
from scripts.run_hz_source_controls import FILES as SOURCE_FILES

CONFIG = 'configs/hz_source_supervision_20261001.json'
CONFIG_SHA = '72aa15da113cca053631eb06c7bf92b8d17abfb74ac1890acc39dcac473f71d0'
REPORT_SHA = '4e5d9e19f702cd67970609943fcbda41c58ec28f88b10876c8beb65ea36d1327'
DONE = 'CHECKED_HZ_SOURCE_CPU_EXECUTION'
WORKER = 'scripts.hz_source_worker'
LIMIT = 4*2**20
FILES = tuple(dict.fromkeys((*SOURCE_FILES, *base.FILES, CONFIG,
    'docs/hz_source_supervision_design_20261001.md', 'docs/hz_source_connection_20261001_r2.json',
    'scripts/hz_source_supervised.py', 'scripts/hz_source_worker.py',
    'scripts/test_hz_source_supervised.py', 'scripts/run_hz_source_supervision.py')))
cleanup_confirmed = base.cleanup_confirmed
cutoff, visible, admitted, release_gate = base.cutoff, base.visible, base.admitted, base.release_gate


def protocol():
    cfg = load(ROOT/CONFIG, CONFIG_SHA)
    load(ROOT/cfg['reference_protocol'], cfg['reference_protocol_sha256'])
    load(ROOT/cfg['reference_report'], REPORT_SHA)
    return cfg


def sources(): return {name: sha(ROOT/name) for name in FILES}


def spec(case='weighted_sign', control=''):
    cfg = protocol()
    if case not in cfg['cases'] or control not in ('', *cfg['faults']) or control and case != 'weighted_sign':
        raise ValueError('fixed source control roster')
    return {'schema': 'HZ_SOURCE_CPU_SPEC_V1', 'case': case, 'control': control,
            'protocol_sha256': CONFIG_SHA, **cfg['cases'][case]}


def validate_spec(s, permit):
    if permit is not None or s != spec(s['case'], s['control']): raise ValueError('source spec binding')


def budget(s): return protocol()['faults'][s['control']][0] if s['control'] else 30


def source_doc(root, inv, digest=None):
    doc = load(root/'prefix_source.json', digest, LIMIT)
    if identity(doc) != inv['spec']['source_sha256']: raise ValueError('frozen declaration identity')
    return doc


def roster(doc): return [list(p) for p in combinations(range(doc['request']['experts']), 2)]


def lowering_prefixes(root, doc, refs, *, complete):
    pairs = roster(doc)
    if not refs or list(refs) != [f'lowering_{i:02}.json' for i in range(len(refs))] or len(refs) > len(pairs)+2:
        raise ValueError('lowering prefix inventory')
    previous = None
    for i, (name, digest) in enumerate(refs.items()):
        p = load(root/name, required_hash(digest), LIMIT)
        count = max(0, i-1)
        if (set(p) != {'schema', 'source_sha256', 'input', 'router', 'pairs', 'pending_pairs'}
                or p['schema'] != 'HZ_LOWERING_PREFIX_V1' or p['source_sha256'] != identity(doc)
                or (p['router'] is None) != (i == 0) or [q['pair'] for q in p['pairs']] != pairs[:count]
                or p['pending_pairs'] != pairs[count:]): raise ValueError('lowering prefix progress/roster')
        if previous is not None and (p['input'] != previous['input'] or p['pairs'][:len(previous['pairs'])] != previous['pairs']
                or i > 1 and p['router'] != previous['router']): raise ValueError('completed source prefix changed')
        previous = p
    if complete and len(refs) != len(pairs)+2: raise ValueError('source incomplete before endpoint proposals')
    if {p.name for p in root.glob('lowering_*.json')} != set(refs): raise ValueError('unaccounted lowering prefix')
    return previous


def package_prefixes(root, doc, lowering, refs, package):
    expected = roster(doc)
    if not refs or len(refs) > len(expected)+1 or list(refs) != [f'endpoint_{i:02}.json' for i in range(len(refs))]:
        raise ValueError('endpoint package prefix sequence')
    previous = None
    for i, (name, digest) in enumerate(refs.items()):
        p = load(root/name, required_hash(digest), LIMIT)
        if (set(p) != {'schema','source_sha256','input','router','pairs','endpoint_request','proof'}
                or p['schema'] != 'CHECKED_HZ_SOURCE_ENDPOINT_V1' or p['source_sha256'] != identity(doc)
                or any(p[k] != lowering[k] for k in ('input','router','pairs'))):
            raise ValueError('endpoint does not bind complete source prefix')
        req, proof = p['endpoint_request'], p['proof']
        if (req['context'] != {'request':identity(doc),'domain':identity(doc['request']),'guard':'ALL_TIE_LEGAL_PAIRS'}
                or [q['pair'] for q in req['pairs']] != expected
                or set(proof) != {'schema','request_sha256','pairs'}
                or proof['schema'] != 'GUARDED_HZ_ENDPOINT_PROOF_V1' or proof['request_sha256'] != identity(req)
                or [q['pair'] for q in proof['pairs']] != expected
                or any(set(q) != {'pair','candidates'} or (q['candidates'] is not None) != (j < i)
                       for j,q in enumerate(proof['pairs']))): raise ValueError('candidate prefix coverage/binding')
        if previous is not None and (req != previous['endpoint_request']
                or proof['pairs'][:i-1] != previous['proof']['pairs'][:i-1]):
            raise ValueError('completed endpoint candidate changed')
        previous = p
    if previous != package: raise ValueError('final package differs from prefix')
    if {p.name for p in root.glob('endpoint_*.json')} != set(refs): raise ValueError('unaccounted endpoint prefix')


def payload(root, inv, digest):
    p = load(root/'produce.json', required_hash(digest), LIMIT)
    if (set(p) != {'invocation','spec_sha256','source_file_sha256','source_prefixes','package_prefixes','package','cost_seconds'}
            or p['invocation'] != inv['invocation'] or p['spec_sha256'] != identity(inv['spec'])):
        raise ValueError('producer identity/schema')
    doc = source_doc(root, inv, required_hash(p['source_file_sha256']))
    lower = lowering_prefixes(root, doc, p['source_prefixes'], complete=True)
    package_prefixes(root, doc, lower, p['package_prefixes'], p['package'])
    execution_coverage(p, inv['spec'])
    stage = load(root/'produce_stage.json', limit=LIMIT)
    if stage.get('output_sha256') != digest: raise ValueError('parent producer anchor')
    costs = p['cost_seconds']
    if set(costs) != {'create','source','prepare','propose'}: raise ValueError('source cost inventory')
    for value in costs.values(): finite(value)
    if sum(costs.values()) > stage['seconds']: raise ValueError('source costs exceed producer execution')
    return p


def execution_coverage(p, s):
    req = p['package']['endpoint_request']
    n = len(req['pairs'])
    if (len(p['package_prefixes']) != n+(not s['partial'])
            or n*len(req['properties']) != s['duties']
            or sum(len(pair['batch']['queries']) for pair in req['pairs']) != s['endpoints']):
        raise ValueError('frozen normal/partial duty coverage')


def produced(root, inv, stage):
    payload(root, inv, stage['output_sha256']); return True


def aggregate_rows(p, agg):
    package = p['package']; request = package['endpoint_request']; proof = package['proof']
    if (agg['source_sha256'] != package['source_sha256'] or agg['package_sha256'] != identity(package)
            or agg['source_lowering_checked'] is not True
            or any(agg[k] is not False for k in ('deployed_float_SAFE','hard_budget_supervision','portable_distribution'))
            or agg['remaining_trust'] != ['declaration_corresponds_to_intended_program','exact_checker_implementation']):
        raise ValueError('source/checker guarantee binding')
    translated = {k:agg[k] for k in ('required','positive','checked_endpoints','missing_endpoints','results','status')}
    translated.update(request_sha256=identity(request), proof_sha256=identity(proof), source_complete=False,
                      deployed_float_SAFE=False, hard_budget_supervision=False)
    if agg['status'] == 'CHECKED_POSITIVE_DECLARED_REAL_SOURCE':
        translated['status'] = 'CHECKED_POSITIVE_GIVEN_GUARDED_HZ_AND_GATE'
    return base.aggregate_rows({'request':request,'proof':proof}, translated)


def check(root, inv, digest, deadline):
    p = payload(root, inv, digest); tick(deadline)
    doc = source_doc(root, inv, p['source_file_sha256'])
    from scoped_source.hz_source_check import check as mathematical_check
    agg = mathematical_check(doc, p['package'], expected_source_sha256=inv['spec']['source_sha256'], deadline=deadline)
    flat = aggregate_rows(p, agg); tick(deadline)
    return {'invocation':inv['invocation'], 'payload_sha256':digest,
            'result':{'results':flat,'aggregation':agg}}


def receive(root, inv, inputs):
    p = payload(root, inv, inputs['produce'])
    checked = load(root/'check.json', required_hash(inputs['check']), LIMIT)
    if (set(checked) != {'invocation','payload_sha256','result'} or checked['invocation'] != inv['invocation']
            or checked['payload_sha256'] != inputs['produce'] or set(checked['result']) != {'results','aggregation'}):
        raise ValueError('parent checker output binding')
    agg = checked['result']['aggregation']
    if checked['result']['results'] != aggregate_rows(p, agg): raise ValueError('flattened endpoint identity')
    missing = len(p['package']['endpoint_request']['pairs'][-1]['batch']['queries']) if inv['spec']['partial'] else 0
    if (agg['required'] != inv['spec']['duties'] or agg['missing_endpoints'] != missing
            or agg['checked_endpoints'] != inv['spec']['endpoints']-missing):
        raise ValueError('frozen checker coverage')
    return {'status':DONE,'invocation':inv['invocation'],'inputs':inputs,'checked':checked['result'],
            'obligations_complete':agg['missing_endpoints']==0,
            'declared_source_positive':agg['status']=='CHECKED_POSITIVE_DECLARED_REAL_SOURCE',
            'complete_moe_proof':False,'real_model_proof':False,'deployed_float_SAFE':False,'cuda_execution':False}


def decision(root, inv, stages):
    records = {s['phase']:s for s in stages}
    if any(not cleanup_confirmed(s) for s in stages): return 'CLEANUP_INCOMPLETE',None,None
    release = release_gate(root,inv,records) if 'release' in records else None
    for s in stages:
        if s['status'] != 'COMPLETED': return s['status'],release,None
    if 'admit' in records: admitted(root,inv,records['admit'])
    if 'produce' in records and not (release or {}).get('release_confirmed'):
        return 'CPU_RELEASE_UNCONFIRMED',release,None
    if list(records) != ['admit','produce','release','check','receive']: return 'ERROR',release,None
    inputs = {k:records[k]['output_sha256'] for k in ('produce','release','check')}
    accepted = receive(root,inv,inputs)
    if load(root/'receive.json',records['receive']['output_sha256'],LIMIT) != accepted:
        raise ValueError('receiver receipt identity')
    return DONE,release,accepted


def recheck_prefix(root, inv, deadline):
    """Offline diagnostic only: completed graph pieces, not partial-source SAFE."""
    lower_paths = sorted(root.glob('lowering_*.json'))
    endpoint_paths = sorted(root.glob('endpoint_*.json'))
    if not (root/'prefix_source.json').exists():
        if lower_paths or endpoint_paths: raise ValueError('published prefix lacks source')
        return None
    if endpoint_paths and not lower_paths:
        raise ValueError('published endpoint lacks lowering prefix')
    doc = source_doc(root, inv)
    refs = {p.name:sha(p) for p in lower_paths}
    if not refs: return {'source_declared':True,'source_graphs_checked':0,'checked_endpoints':0,'output_accepted':False}
    p = lowering_prefixes(root,doc,refs,complete=False)
    from scoped_source.graph import validate, clock
    from scoped_source.hz_source_check import checked_state, network, route_entry, gate_range, check as math_check
    from source_enclosure.check import check_box
    r,lo,hi = validate(doc,inv['spec']['source_sha256'],clock(deadline))
    checked_state(p['input']); check_box(lo,hi,p['input']); count=0
    if p['router'] is not None:
        router,_ = network(doc,'router',p['input'],p['router'],'router',deadline); count+=1
        for pair in p['pairs']:
            a,b = pair['pair']; entry = pair['entry']
            route_entry(p['input'],router,entry,(a,b),r['experts'])
            for name,e in [('a',a),('b',b)]:
                network(doc,f'expert{e}',entry,pair[name],f'pair{a}-{b}/expert{e}',deadline); count+=1
            if pair['gate_evidence'] != gate_range(router,(a,b)): raise ValueError('source prefix gate')
    ep = {q.name:sha(q) for q in endpoint_paths}
    result = None
    if ep:
        if p['pending_pairs']: raise ValueError('endpoint evidence precedes full source')
        package = load(root/next(reversed(ep)),limit=LIMIT)
        package_prefixes(root,doc,p,ep,package)
        result = math_check(doc,package,expected_source_sha256=inv['spec']['source_sha256'],deadline=deadline)
    return {'source_declared':True,'source_graphs_checked':count,
            'checked_endpoints':0 if result is None else result['checked_endpoints'],
            'output_accepted':False,'source_prefix_not_online_acceptance':True,'aggregation':result}
