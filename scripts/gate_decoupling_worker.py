"""Opt-in stage4 local worker; existing supervisor, one budget for both endpoints.

No queue is launched here. A final catalog/ordered call table and passed controls
must be frozen by the stage driver. V3 is always counterfactual; V2 closes only
the selected history. Neither partial receipts nor offline checking finish an
old complete request.
"""
import time
ENTRY = time.monotonic()
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from scoped_proof.io import load, save, sha, tick
from scoped_proof.owned_session import execute
from scoped_source.development_work import POLICY
import genbab_cpu_ready as backend

CAP = 128*2**20


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def size(folder):
    console = folder.with_suffix('.controller.log')
    return (sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
            + (console.stat().st_size if console.is_file() else 0))


class Journal:
    """Small append-only phase receipts: an interrupted ENTER stays visible."""
    def __init__(self, folder, deadline, limit):
        self.folder, self.deadline, self.limit = folder, deadline, limit
        self.ordinal = 0

    def put(self, name, value):
        tick(self.deadline)
        length = len(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
        require(size(self.folder)+length <= self.limit, 'remaining stage artifact allowance')
        result = save(self.folder/name, value)
        tick(self.deadline)
        return result

    def event(self, phase, state, **fields):
        self.ordinal += 1
        self.put(f'phase_{self.ordinal:03d}.json', dict(phase=phase, state=state,
            clock=time.monotonic(), **fields))

    def call(self, phase, function):
        entered = time.monotonic(); self.event(phase, 'ENTER')
        result = function()
        self.event(phase, 'RETURN', seconds=time.monotonic()-entered)
        return result


def endpoints(gate, domain, group, properties, deadline, *, prepare, verify, replay, journal):
    """Dependency injection is for fixed controls; production uses the same API.

    No endpoint gets a fresh time budget. All endpoints are attempted in fixed
    order while time remains, even after an ordinary unknown/falsified return.
    """
    require(group in ('V2', 'V3'), 'explicit diagnostic variant')
    variants = ('V2_LO', 'V2_HI') if group == 'V2' else ('V3_MIDPOINT',)
    records, witnesses = [], []
    for variant in variants:
        tick(deadline)
        graph, box = journal.call(variant+':construct', lambda: prepare(variant))
        outcome, attacks = journal.call(variant+':solve',
            lambda: verify(graph, box['lower'], box['upper'], properties, deadline))
        row = dict(variant=variant, source_id=domain.source_id,
            domain_sha256=gate.domain_digest(domain), outcome=outcome)
        # Validate raw-property and source bindings before publishing the row.
        gate.receive_diagnosis(domain, group, records+[row], property_count=properties,
            expected_source_id=domain.source_id, deadline=deadline)
        journal.put(variant+'.json', row); records.append(row)
        replayed = journal.call(variant+':replay', lambda: replay(attacks, box))
        journal.put(variant+'_replay.json', replayed); witnesses.extend(replayed)
        del graph, box, attacks
    aggregate = gate.receive_diagnosis(domain, group, records, property_count=properties,
        expected_source_id=domain.source_id, deadline=deadline)
    return dict(**aggregate, endpoints=records, full_model_replays=witnesses,
        original_witness_found=any(r['valid'] for r in witnesses))


def source_object(spec, call):
    inventory = load(spec['catalog']['path'], spec['catalog']['sha256'])
    matches = [o for o in inventory['objects'] if o['id'] == call['object_id']]
    require(len(matches) == 1, 'one bound hard-core object')
    require(call['variant'] in ('V2', 'V3'), 'registered V2/V3 call')
    return matches[0]


def check_received(folder, value, spec, call, deadline):
    """Recheck local scope/coverage without loading a model or numerical library."""
    obj = source_object(spec, call)
    source_id = digest(dict(object=obj, sources=spec['source_files'], backend=spec['backend_identity']))
    require(value['source_id'] == source_id and value['group'] == call['variant'], 'received object/variant binding')
    if obj['history'] is None:
        require(value['status'] == 'SOURCE_MISSING' and not value['closed']
            and not value['original_history_closed'] and not value['formal_SAFE']
            and not value['complete_MoE'] and not value['endpoints'], 'missing-source nonacceptance')
        return
    name = 'gate_decoupling_receipt_source'
    module_spec = importlib.util.spec_from_file_location(name, ROOT/'act/back_end/moe/gate_decoupling.py')
    gate = importlib.util.module_from_spec(module_spec); sys.modules[name] = gate
    module_spec.loader.exec_module(gate)
    saved = load(folder/'domain.json')
    domain = gate.GateDomain(saved['source_id'], tuple(map(tuple, saved['history'])),
        tuple(map(tuple, saved['margins'])), tuple(map(tuple, saved['intervals'])))
    require(domain.history == tuple(map(tuple, obj['history'])), 'received exact history')
    properties = (len(obj['properties']) if obj['source_kind'] == 'output_mlp'
        else obj['original_request']['model']['architecture']['classes']-1)
    require(value['property_count'] == properties and value['domain_sha256'] == gate.domain_digest(domain)
        and saved['domain_sha256'] == value['domain_sha256'], 'received domain/property binding')
    for row in value['endpoints']:
        require(load(folder/(row['variant']+'.json')) == row, 'saved endpoint/result identity')
    aggregate = gate.receive_diagnosis(domain, call['variant'], value['endpoints'], property_count=properties,
        expected_source_id=source_id, deadline=deadline)
    require(all(value[k] == v for k, v in aggregate.items()), 'local closure/scope recomputation')


def load_object(obj, torch):
    if obj['source_kind'] == 'multilayer':
        from multilayer_stage1 import subject, tensor_row
        request = obj['original_request']
        model = subject(request['model'])
        tensors = tensor_row(Path(request['selection_root']), request['sample'], torch)
    elif obj['source_kind'] == 'output_mlp':
        from b2_observed_worker import load_model
        model, tensors = load_model(obj['original_request'])
        tensors = dict(tensors)
        tensors['rows'] = torch.tensor([[float(v) for v in p['q']] for p in obj['properties']], dtype=torch.float64)
        tensors['thresholds'] = torch.tensor([-float(p['c']) for p in obj['properties']], dtype=torch.float64)
    else:
        raise ValueError('registered hard-core model family')
    return model, tensors


def original_ranges(obj, model, tensors, deadline, journal):
    source = obj.get('gate_range_source')
    if source is not None:
        binding = source['source']
        require(sha(binding['path']) == binding['sha256'], 'saved prefix bound object')
        with gzip.open(binding['path'], 'rt') as stream:
            value = json.load(stream)
        require(source['field'] == 'value.result.prefix_bounds', 'registered range field')
        records = value['value']['result']['prefix_bounds']
        wanted = {tuple(map(tuple, obj['history'][:i])) for i in range(len(obj['history']))}
        records = [r for r in records if tuple(map(tuple, r['prefix'])) in wanted]
        journal.put('ranges_saved.json', dict(source=source, records=records))
        return records, 'SAVED_ORIGINAL_PREFIX_CROWN'
    from act.back_end.moe.multilayer_external import SequentialTop2, router_upper_bounds
    if obj['source_kind'] == 'output_mlp':
        from act.back_end.moe.multilayer import RoutedLayer
        # Only exposes the original router for the empty prefix. No model swap
        # in the diagnostic graph or full-model replay.
        prefix_model = SequentialTop2([RoutedLayer(model.router, model.experts,
            k=2, mode='selected_softmax')]).eval()
    else:
        prefix_model = model
    records = []
    for depth in range(len(obj['history'])):
        prefix = obj['history'][:depth]
        before = time.monotonic()
        upper = journal.call(f'range:{depth}', lambda: router_upper_bounds(prefix_model,
            prefix, tensors['center'], tensors['lower'], tensors['upper'], deadline))
        row = dict(prefix=prefix, upper=[[i, j, v] for (i, j), v in sorted(upper.items())],
            seconds=time.monotonic()-before, guarantee='WHOLE_BOX_CROWN_NUMERICAL_UPPER')
        journal.put(f'range_{depth}.json', row); records.append(row)
    return records, 'REBUILT_ORIGINAL_PREFIX_CROWN_CHARGED'


def diagnose(obj, group, spec, deadline, journal):
    for binding in obj['original_sources']:
        tick(deadline)
        require(sha(binding['path']) == binding['sha256'], 'original request/history evidence binding')
    source_id = digest(dict(object=obj, sources=spec['source_files'], backend=spec['backend_identity']))
    journal.put('source.json', dict(object=obj, source_id=source_id, group=group))
    if obj['history'] is None:
        return dict(status='SOURCE_MISSING', closed=False, original_history_closed=False,
            complete_MoE=False, formal_SAFE=False, reason='first unclosed history identity unavailable',
            source_id=source_id, group=group, endpoints=[])
    from multilayer_stage1 import setup
    torch = journal.call('native_setup', lambda: setup(deadline))
    environment = journal.call('backend_import', backend.probe)
    require(digest(environment['sources']) == spec['backend_identity'], 'same external engine snapshot')
    journal.put('environment.json', dict(python=environment['python'], torch=environment['torch'],
        modules=environment['modules'], source_identity=spec['backend_identity']))
    from act.back_end.moe import gate_decoupling as gate
    from b2_observed_worker import verify_graph
    from act.back_end.moe.class_separated_top1 import validate_replay
    model, tensors = journal.call('load_object', lambda: load_object(obj, torch))
    history = gate.canonical_history(obj['history'], obj['experts'])
    records, origin = journal.call('gate_ranges',
        lambda: original_ranges(obj, model, tensors, deadline, journal))
    domain = gate.domain_from_prefix_bounds(history, obj['experts'], records,
        source_id=source_id, expected_source_id=source_id)
    journal.put('domain.json', dict(source_id=source_id, history=domain.history,
        margins=domain.margins, intervals=domain.intervals, domain_sha256=gate.domain_digest(domain),
        origin=origin, evidence_grade='CROWN_SIGMOID_NUMERICAL_POLICY_NOT_EXACT_ENCLOSURE'))
    pixels = tensors['center'][0].numel()
    def prepare(variant):
        box = gate.extend_box(tensors['center'], tensors['lower'], tensors['upper'], domain,
            variant=variant, expected_source_id=source_id)
        graph = gate.build_graph(model, history, tensors['rows'], tensors['thresholds'], domain,
            variant=variant, input_features=pixels, expected_source_id=source_id)
        return graph, box
    def replay(attacks, box):
        rows = []
        if attacks is None:
            return rows
        width = box['lower'][0].numel()
        require(attacks.numel() % width == 0, 'extended candidate dimensions')
        for point in attacks.detach().cpu().double().reshape(-1, width):
            tick(deadline)
            candidate = point[:pixels].reshape_as(tensors['center'])
            valid = validate_replay(model, candidate, tensors['lower'], tensors['upper'],
                tensors['rows'], tensors['thresholds'])
            rows.append(dict(valid=valid, original_input=candidate.flatten().tolist() if valid else None,
                dropped_independent_gates=width-pixels, scope='ORIGINAL_DYNAMIC_MODEL_REPLAY'))
        return rows
    result = endpoints(gate, domain, group, len(tensors['rows']), deadline,
        prepare=prepare, verify=verify_graph, replay=replay, journal=journal)
    return dict(**result, source_id=source_id, domain_sha256=gate.domain_digest(domain),
        range_origin=origin, property_count=len(tensors['rows']), history=history)


def worker(folder):
    request = load(folder/'request.json'); deadline = request['worker_deadline']
    backend.limits(deadline)
    journal = Journal(folder, deadline, request['artifact_limit'])
    # Preserve structured bounds/phase/errors; discard non-evidence solver chatter.
    fd = os.open(os.devnull, os.O_WRONLY); os.dup2(fd, 1); os.dup2(fd, 2); os.close(fd)
    try:
        spec = load(request['manifest'], request['manifest_sha256'])
        require(spec['schema'] == 'gate-decoupling-stage4-v1' and spec['policy'] == POLICY, 'frozen stage4 policy')
        require(all(sha(ROOT/p) == h for p, h in spec['source_files'].items()), 'execution source identity')
        require(spec['genbab_config'] == backend.CONFIG, 'unchanged engine configuration')
        call = spec['calls'][request['call_index']]
        require(digest(call) == request['call_sha256'], 'frozen call table')
        obj = journal.call('source_binding', lambda: source_object(spec, call))
        value = diagnose(obj, call['variant'], spec, deadline, journal)
        journal.put('outcome.json', dict(value=value, request_sha256=sha(folder/'request.json'),
            clock=time.monotonic(), worker_seconds=time.monotonic()-ENTRY))
        journal.put('published.json', dict(sha256=sha(folder/'outcome.json'), clock=time.monotonic()))
    except BaseException as error:
        # Reserve already deducted by controller; errors never yield acceptance.
        journal.limit += 65536
        if time.monotonic() < deadline:
            journal.put('error.json', dict(kind=type(error).__name__, message=str(error),
                traceback=traceback.format_exc(), clock=time.monotonic()))
        return 124 if isinstance(error, TimeoutError) else 1
    return 0


def invoke(folder, manifest, call_index, *, stage_deadline):
    """One registered local call; no batching, retries or per-endpoint timeout."""
    from b1_observed_batch import session_ok
    began = time.monotonic(); end = began+300
    require(end <= stage_deadline, 'insufficient remaining stage time; suffix unexecuted')
    resource.setrlimit(resource.RLIMIT_AS, (8*2**30, 8*2**30))
    spec = load(manifest)
    require(spec['schema'] == 'gate-decoupling-stage4-v1' and spec['policy'] == POLICY, 'stage4 freeze required')
    require(ROOT != Path('/data1/Kane/MOE/ACT') and (ROOT/'.frozen_commit').is_file(), 'immutable execution copy')
    require(all(sha(ROOT/p) == h for p, h in spec['source_files'].items()), 'frozen execution sources')
    control = load(spec['control']['path'], spec['control']['sha256'])
    require(control['status'] == 'PASS', 'frozen integration control receipt')
    call = spec['calls'][call_index]
    remaining = CAP-size(folder.parent)
    require(remaining > 256*1024, 'remaining cumulative stage artifact allowance')
    folder.mkdir(); (folder/'tmp').mkdir(); (folder/'cache').mkdir()
    journal = Journal(folder, end, remaining)
    journal.put('request.json', dict(manifest=str(manifest), manifest_sha256=sha(manifest),
        call_index=call_index, call_sha256=digest(call), started=began, hard_deadline=end,
        worker_deadline=end-6, artifact_limit=remaining-192*1024))
    env = dict(PATH='/usr/bin:/bin', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
        CUDA_VISIBLE_DEVICES='', TMPDIR=str(folder/'tmp'), XDG_CACHE_HOME=str(folder/'cache'))
    receipt = execute([str(backend.ENVIRONMENT/'bin/python'), '-I', '-B', __file__, str(folder)],
        run_deadline=end-4, cleanup_deadline=end-2, env=env, rss_limit=None, development_policy=POLICY)
    journal.put('session.json', receipt)
    accepted, value, status = False, None, 'ERROR'
    try:
        if session_ok(receipt) and (folder/'published.json').is_file():
            publication = load(folder/'published.json')
            outcome = load(folder/'outcome.json', publication['sha256'])
            require(outcome['request_sha256'] == sha(folder/'request.json'), 'received request/result identity')
            accepted = outcome['clock'] < end-6 and publication['clock'] < end-6
            status = outcome['value']['status'] if accepted else 'TIMEOUT'
            if accepted:
                value = outcome['value']
                check_received(folder, value, spec, call, end)
        elif receipt.get('returncode') == 124 or 'TIMEOUT' in receipt['status'] or time.monotonic() >= end-6:
            status = 'TIMEOUT'
    except (ValueError, KeyError, TypeError, OSError) as error:
        accepted, value, status = False, None, 'ERROR'
        journal.put('reception_error.json', dict(kind=type(error).__name__, message=str(error)))
    phases = sorted(folder.glob('phase_*.json'))
    row = dict(object_id=call['object_id'], variant=call['variant'], status=status, accepted=accepted,
        last_phase=load(phases[-1]) if phases else None, evidence=value,
        session_status=receipt['status'], returncode=receipt['returncode'], hard_budget=300,
        through_reception_seconds=time.monotonic()-began,
        files={p.name: sha(p) for p in folder.iterdir() if p.is_file()})
    journal.put('terminal.json', row)
    row.update(seconds=time.monotonic()-began, terminal_sha256=sha(folder/'terminal.json'))
    require(row['seconds'] <= 300 and size(folder.parent) <= CAP, 'complete cost and artifact accounting')
    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('folder', type=Path)
    raise SystemExit(worker(parser.parse_args().folder))
