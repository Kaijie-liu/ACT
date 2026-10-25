"""Read-only source catalog for stage4; no tensors/models/solvers are loaded.

The output is a diagnostic source inventory, NOT an execution freeze. A final
call order/budget must include the completed1b cohort before numerical probes.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    path = Path(path)
    data = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    return json.loads(data)


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def unrank_history(ordinal, experts):
    counts = [math.comb(e, 2) for e in experts]
    if type(ordinal) is not int or not 0 <= ordinal < math.prod(counts):
        raise ValueError('pending ordinal outside full history catalog')
    result = []
    for e, count in reversed(list(zip(experts, counts))):
        result.append(list(itertools.combinations(range(e), 2))[ordinal % count])
        ordinal //= count
    return [list(p) for p in reversed(result)]


def source_chain(root):
    result = []
    while True:
        root = Path(root).resolve()
        if root in result:
            raise ValueError('cyclic result ancestry')
        result.append(root)
        path = root/'continuation.json'
        if not path.exists():
            return result
        root = load(path)['parent_root']


def multilayer(root, *, cohort, exclude_replayed):
    audit_path = Path(root)/'audit.json'
    audit = load(audit_path)
    if audit['status'] != 'PASS_COMPLETE' or audit['unexecuted']:
        raise ValueError('completed audited cohort required')
    unsafe = {(x['model'], x['index']): x['sources'] for x in audit['unsafe_union']}
    selected = {(r['model'], r['index']): r for r in audit['rows']
        if r['arm'] == 'genbab_pruned' and not r['model'].startswith('L1')
        and r['status'] != 'POSITIVE'
        and (not exclude_replayed or (r['model'], r['index']) not in unsafe)}
    requests = {}
    for parent in reversed(source_chain(root)):
        for path in sorted(parent.glob('verify_*/request.json')):
            q = load(path)
            if q.get('arm') != 'genbab_pruned': continue
            key = q['model']['id'], q['sample']['index']
            if key not in selected: continue
            if key in requests and binding(requests[key][0])['sha256'] != binding(path)['sha256']:
                raise ValueError('ambiguous retained request source')
            requests[key] = (path, q)
    if set(requests) != set(selected):
        raise ValueError('missing declared request file')
    result = []
    for key, row in sorted(selected.items()):
        path, q = requests[key]; folder = path.parent
        architecture = q['model']['architecture']
        experts = [architecture['experts']]*architecture['layers']
        sources = [binding(path), binding(folder/'terminal.json')]
        outcome = folder/'outcome.json.gz'; progress = folder/'progress.json'
        history, ranges = None, None
        if outcome.exists():
            value = load(outcome)['value']['result']
            sources.append(binding(outcome))
            unproved = sorted((r['history'] for r in value['histories'] if r['outcome']['status'] != 'verified'))
            if unproved:
                history, kind = unproved[0], 'SAVED_FIRST_RETURNED_UNPROVED'
            else:
                kind = 'FULL_OUTCOME_NO_RETURNED_UNPROVED'
            if history is not None:
                needed = {tuple(map(tuple, history[:i])) for i in range(len(history))}
                records = [r for r in value['prefix_bounds'] if tuple(map(tuple, r['prefix'])) in needed]
                if len(records) == len(history):
                    ranges = dict(source=binding(outcome), field='value.result.prefix_bounds',
                                  prefix_count=len(records))
        else:
            if progress.exists(): sources.append(binding(progress))
            cov = row['coverage']
            if cov['returned_unproved'] == 0 and cov['returned'] == cov['verified']:
                ordinal = cov['pruned']+cov['verified']
                if ordinal < math.prod(math.comb(e, 2) for e in experts):
                    history = unrank_history(ordinal, experts)
                    active = cov.get('active') or {}
                    kind = ('SAVED_INFLIGHT_FIRST_HISTORY' if active.get('phase') == 'history_solver'
                            and active.get('prefix') == history else 'SAVED_CUT_FIRST_PENDING_NOT_RETURNED')
                else:
                    kind = 'SAVED_CUT_COVERAGE_COMPLETE_NO_PENDING_HISTORY'
            else:
                kind = 'EARLIER_UNPROVED_ID_MISSING'
        if history is None and ranges is not None:
            raise ValueError('range without history')
        result.append(dict(id=f'{cohort}:{key[0]}:{key[1]}', cohort=cohort, group='multilayer',
            model=key[0], index=key[1], baseline_status=row['status'],
            known_replayed_sources=unsafe.get(key, []), experts=experts,
            original_request=dict(model=q['model'], sample=q['sample'], selection_root=q['selection_root']),
            source_kind='multilayer', history=history, history_source_kind=kind,
            gate_range_source=ranges, range_reconstruction_needed=ranges is None,
            original_sources=sources, baseline_coverage=row['coverage'],
            note='saved-cut identity does not certify reachability or unrecorded later work'))
    return result, binding(audit_path)


def singlelayer(records_path, freeze_path):
    records = load(records_path)
    final, freeze = records['final_stage_result'], load(freeze_path)
    if (binding(final['complete'])['sha256'] != final['complete_sha256']
            or binding(freeze_path)['sha256'] != final['freeze_sha256']):
        raise ValueError('B2 final record/freeze binding')
    queries = {q['id']: q for q in freeze['queries']}
    known = {r['genbab_aggregate']: r['witness'] for r in records['saved_witness_crosscheck']['rows']}
    result = []
    for row in final['rows']:
        if row['kind'] != 'genbab_aggregate' or row['status'] == 'POSITIVE': continue
        q = queries[row['query_id']]; folder = Path(row['run_root'])
        spec_path = folder/'spec.json'; launch_path = folder/'launch.json'
        launch = load(launch_path)
        if launch['query']['item'] != q['item']:
            raise ValueError('B2 source item changed')
        spec = load(spec_path); request = spec['request']
        pairs = [list(p) for p in itertools.combinations(range(request['experts']), 2)]
        if request['pairs'] != pairs or request['properties'] == []:
            raise ValueError('B2 full pair/property catalog')
        sources = [binding(launch_path), binding(spec_path), binding(folder/'terminal.json')]
        returned = {}
        for path in sorted(folder.glob('pair_*_result.json')):
            r = load(path); pair = tuple(r['pair'])
            if pair in returned or list(pair) not in pairs:
                raise ValueError('B2 duplicate or unknown pair result')
            returned[pair] = (r, path)
        started = {tuple(load(p)['pair']): p for p in folder.glob('pair_*_started.json')}
        history, kind = None, 'NO_UNCLOSED_PAIR_IN_SAVED_RESULTS'
        for pair in pairs:
            r, path = returned.get(tuple(pair), (None, None))
            if r is not None and r['outcome']['status'] == 'verified':
                sources.append(binding(path)); continue
            history = [pair]
            if r is not None:
                kind = 'SAVED_FIRST_RETURNED_UNPROVED'; sources.append(binding(path))
            elif tuple(pair) in started:
                kind = 'SAVED_FIRST_STARTED_NO_RETURN'; sources.append(binding(started[tuple(pair)]))
            else:
                kind = 'SAVED_FIRST_UNVISITED_PAIR'
            break
        item = q['item']
        result.append(dict(id=f'B2:{item["model"]}:{item["dataset_index"]}', cohort='B2', group='singlelayer',
            model=item['model'], index=item['dataset_index'], baseline_status=row['status'],
            baseline_accepted=row['accepted'], known_saved_witness=known.get(row['query_id']),
            experts=[request['experts']], original_request=item, source_kind='output_mlp',
            properties=request['properties'], history=history, history_source_kind=kind,
            gate_range_source=None, range_reconstruction_needed=True, original_sources=sources,
            note='tool ERROR is not a mathematical negative; retained partial receipts do not upgrade original terminal'))
    return result, [binding(records_path), binding(freeze_path), binding(final['complete'])]


def inventory(stage1b=None):
    stage1a_record = load(ROOT/'docs/multilayer_stage1_records_20261022.json')['final_result_20261025']
    first, audit = multilayer(stage1a_record['result_root'], cohort='1a', exclude_replayed=True)
    if audit['sha256'] != stage1a_record['audit_sha256'] or len(first) != 55:
        raise ValueError('fixed1a hard-core denominator/audit')
    second, bindings = singlelayer(ROOT/'docs/b2_observed_records_20261021.json',
                                   ROOT/'docs/b2_observed_freeze_20261021.json')
    if len(second) != 24:
        raise ValueError('fixedB2 unresolved denominator')
    rows, sources = first+second, [audit, *bindings]
    if stage1b is not None:
        third, source = multilayer(stage1b, cohort='1b', exclude_replayed=False)
        rows += third; sources.append(source)
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate diagnostic object')
    return dict(schema='gate-decoupling-source-inventory-v1',
        status='SOURCE_INVENTORY_NOT_EXECUTION_FREEZE', stage1b_complete_included=stage1b is not None,
        source_files=[binding(__file__), *sources], objects=rows, object_count=len(rows),
        source_kinds=dict(Counter(r['history_source_kind'] for r in rows)),
        complete_range_sources=sum(r['gate_range_source'] is not None for r in rows),
        real_calls=0, new_model_loads=0, new_solver_calls=0,
        two_variant_calls_if_all_executed=2*len(rows), local_call_cap=250,
        scope='failure-selected provenance; no checkpoint/tensor contents loaded, no original result upgraded')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('output', type=Path)
    p.add_argument('--stage1b', type=Path); a = p.parse_args()
    if not a.output.resolve().is_relative_to(Path('/data1/Kane/MOE')):
        raise ValueError('output outside research workspace')
    value = inventory(a.stage1b)
    with a.output.open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({k:v for k,v in value.items() if k not in ('objects','source_files')}))
