"""Optional M-A history callback: hard120s V2, then remaining coupled GenBaB.

Runs inside an existing complete-request worker/session. It is not a new batch
supervisor and does not select models, inputs or histories. Default dispatcher
behavior is unchanged until a caller explicitly supplies this callback.
"""
import time
ENTRY = time.monotonic()
import argparse
import json
import math
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import gate_decoupling_worker as W
from scoped_proof.io import load, save, sha, tick
from scoped_proof.owned_bounded import execute
from scoped_source.development_work import POLICY

T1 = 120.


def identity(source_request, history, records, backend_identity):
    return W.digest(dict(source=source_request, history=history, prefix_bounds=records,
                         backend=backend_identity, method='M_A_V2_ENDPOINTS_V1'))


def scoped_domain(G, source_request, history, records, sid, backend_identity, deadline):
    conditional = any(r.get('guarantee') == 'GUARD_CONDITIONAL_CROWN_NUMERICAL_UPPER' for r in records)
    if conditional:
        from moe_aware_frontend import audit_result
        audit_result(dict(prefix_bounds=records, histories=[]), source_request, backend_identity, deadline)
    return G.domain_from_prefix_bounds(history,
        [source_request['model']['architecture']['experts']]*len(history), records,
        source_id=sid, expected_source_id=sid, allow_conditional=conditional)


def receive_phase(folder, request, value, domain, deadline):
    from moe_aware_diagnostics import gate_module
    G = gate_module()
    domain = G.GateDomain(domain.source_id, domain.history, domain.margins, domain.intervals)
    tick(deadline)
    W.require(value['source_id'] == request['source_id'] == domain.source_id
        and value['domain_sha256'] == G.domain_digest(domain), 'same cascade domain')
    W.require(value['property_count'] == request['properties'] and value['history'] == request['history'],
              'same complete history and all properties')
    for row in value['endpoints']:
        W.require(load(folder/(row['variant']+'.json')) == row, 'bound endpoint result')
    aggregate = G.receive_diagnosis(domain, 'V2', value['endpoints'],
        property_count=request['properties'], expected_source_id=request['source_id'], deadline=deadline)
    W.require(all(value[k] == v for k, v in aggregate.items()), 'both endpoints, never partial local success')
    tick(deadline)


def phase_worker(folder):
    request = load(folder/'phase_request.json'); deadline = request['worker_deadline']
    W.backend.limits(deadline)
    journal = W.Journal(folder, deadline, request['artifact_limit'])
    fd = os.open(os.devnull, os.O_WRONLY); os.dup2(fd, 1); os.dup2(fd, 2); os.close(fd)
    try:
        W.require(request['schema'] == 'gate-cascade-phase-v1' and request['policy'] == POLICY,
                  'registered bounded phase')
        from multilayer_stage1 import setup
        torch = journal.call('native_setup', lambda: setup(deadline))
        environment = journal.call('backend_import', W.backend.probe)
        W.require(W.digest(environment['sources']) == request['backend_identity'], 'same coupled engine')
        sid = identity(request['source_request'], request['history'], request['prefix_bounds'], request['backend_identity'])
        W.require(sid == request['source_id'], 'parent history/range identity')
        obj = dict(source_kind='multilayer', original_request=request['source_request'])
        model, tensors = journal.call('load_same_source', lambda: W.load_object(obj, torch))
        from act.back_end.moe import gate_decoupling as G
        from act.back_end.moe.class_separated_top1 import validate_replay
        from b2_observed_worker import verify_graph
        experts = [len(layer.experts) for layer in model.layers]
        history = G.canonical_history(request['history'], experts)
        domain = scoped_domain(G, request['source_request'], history, request['prefix_bounds'], sid,
                               request['backend_identity'], deadline)
        W.require(G.domain_digest(domain) == request['domain_sha256']
            and len(tensors['rows']) == request['properties'], 'same gate ranges and all output properties')
        pixels = tensors['center'][0].numel()
        def prepare(variant):
            box = G.extend_box(tensors['center'], tensors['lower'], tensors['upper'], domain,
                              variant=variant, expected_source_id=sid)
            graph = G.build_graph(model, history, tensors['rows'], tensors['thresholds'], domain,
                variant=variant, input_features=pixels, expected_source_id=sid)
            return graph, box
        def replay(attacks, box):
            result = []
            if attacks is None: return result
            width = box['lower'][0].numel()
            W.require(attacks.numel() % width == 0, 'complete extended candidate')
            for point in attacks.detach().cpu().double().reshape(-1, width):
                tick(deadline)
                candidate = point[:pixels].reshape_as(tensors['center'])
                valid = validate_replay(model, candidate, tensors['lower'], tensors['upper'],
                                        tensors['rows'], tensors['thresholds'])
                result.append(dict(valid=valid, original_input=candidate.flatten().tolist() if valid else None,
                    dropped_independent_gates=width-pixels, scope='ORIGINAL_DYNAMIC_MODEL_REPLAY'))
            return result
        result = W.endpoints(G, domain, 'V2', request['properties'], deadline,
            prepare=prepare, verify=verify_graph, replay=replay, journal=journal)
        value = dict(**result, source_id=sid, history=request['history'],
            property_count=request['properties'], domain_sha256=G.domain_digest(domain))
        journal.put('outcome.json', dict(value=value, request_sha256=sha(folder/'phase_request.json'),
            clock=time.monotonic(), worker_seconds=time.monotonic()-ENTRY))
        journal.put('published.json', dict(sha256=sha(folder/'outcome.json'), clock=time.monotonic()))
    except BaseException as error:
        journal.limit += 65536
        if time.monotonic() < deadline:
            journal.put('error.json', dict(kind=type(error).__name__, message=str(error),
                traceback=traceback.format_exc(), clock=time.monotonic()))
        return 124 if isinstance(error, TimeoutError) else 1
    return 0


def run_phase(folder, source_request, history, records, properties, backend_identity,
              deadline, artifact_limit, *, phase_seconds=T1, phase_started=None):
    """Owner-session child group; no new session can escape the outer cutoff."""
    from act.back_end.moe import gate_decoupling as G
    began = time.monotonic() if phase_started is None else phase_started
    W.require(math.isfinite(began) and began <= time.monotonic(), 'original phase start')
    end = min(deadline, began+phase_seconds)
    W.require(math.isfinite(phase_seconds) and 0 < phase_seconds <= T1, 'phase cannot extend120s')
    tick(deadline)
    sid = identity(source_request, history, records, backend_identity)
    experts = [source_request['model']['architecture']['experts']]*len(history)
    G.canonical_history(history, experts)
    # Infinite/missing numerical margins cannot define a useful V2 domain.
    # Preserve the coupled fallback; malformed identities/duplicate sources
    # remain errors, not hidden optimizer failures.
    mapped = {tuple(map(tuple, r['prefix'])): r for r in records}
    W.require(len(mapped) == len(records) == len(history), 'one source for each original prefix')
    for depth, (a, b) in enumerate(history):
        raw = mapped[tuple(history[:depth])]['upper']
        pairs = {(i, j): v for i, j, v in raw}
        W.require(len(pairs) == len(raw), 'unique original ordered margin rows')
        if any(pairs.get(key) is None or not math.isfinite(pairs[key]) for key in ((a, b), (b, a))):
            return dict(status='GATE_RANGE_UNAVAILABLE', accepted=False, evidence=None,
                        seconds=time.monotonic()-began, source_id=sid)
    domain = scoped_domain(G, source_request, history, records, sid, backend_identity, end)
    if end-time.monotonic() <= 6:
        return dict(status='NO_PHASE_BUDGET', accepted=False, evidence=None,
                    seconds=time.monotonic()-began, source_id=sid)
    folder.mkdir()
    journal = W.Journal(folder, end, artifact_limit)
    request = dict(schema='gate-cascade-phase-v1', policy=POLICY, source_request=source_request,
        history=[list(p) for p in history], prefix_bounds=records, properties=properties,
        source_id=sid, domain_sha256=G.domain_digest(domain), backend_identity=backend_identity,
        request_deadline=deadline, phase_started=began, phase_deadline=end,
        worker_deadline=end-5, artifact_limit=artifact_limit-65536)
    journal.put('phase_request.json', request)
    # The outer session owns this group; RLIMIT_AS is inherited and also set in
    # the child before native imports. Keep the caller's registered thread env.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    receipt = execute([str(W.backend.ENVIRONMENT/'bin/python'), '-I', '-B', __file__, str(folder)],
        folder/'native.log', run_deadline=end-4, cleanup_deadline=end-2, env=env,
        rss_limit=None, development_policy=POLICY, owner_session=os.getsid(0))
    journal.put('session.json', receipt)
    W.require(receipt['cleanup_status'] == 'LEADER_REAPED_NO_LIVE_GROUP',
              'phase ownership/cleanup required before coupled fallback')
    value, accepted, status = None, False, receipt['status']
    if status == 'COMPLETED':
        publication = load(folder/'published.json')
        outcome = load(folder/'outcome.json', publication['sha256'])
        W.require(outcome['request_sha256'] == sha(folder/'phase_request.json'), 'phase request binding')
        if outcome['clock'] < end-5 and publication['clock'] < end-5:
            value = outcome['value']
            receive_phase(folder, request, value, domain, end)
            accepted = True; status = value['status']
        else:
            status = 'TIMEOUT'
    elif receipt['returncode'] == 124:
        status = 'TIMEOUT'
    elif status != 'TIMEOUT':
        raise RuntimeError('V2 phase tool error; retain evidence before repair')
    result = dict(status=status, accepted=accepted, evidence=value, source_id=sid,
        seconds=time.monotonic()-began, phase_deadline=end, request_deadline=deadline,
        session_sha256=sha(folder/'session.json'))
    journal.put('phase_terminal.json', result)
    result['seconds'] = time.monotonic()-began
    W.require(time.monotonic() < end and W.size(folder) <= artifact_limit,
              'phase reception/cleanup/terminal within same allowance')
    return result


class GateCascade:
    """Callback for verify_histories; no changes to canonical routing coverage."""
    def __init__(self, model, tensors, source_request, folder, backend_identity,
                 artifact_remaining, *, phase_runner=run_phase, coupled=None):
        self.model, self.tensors = model, tensors
        self.source = {k: source_request[k] for k in ('model', 'sample', 'selection_root')}
        self.folder, self.backend_identity = Path(folder), backend_identity
        self.remaining, self.phase_runner, self.coupled = artifact_remaining, phase_runner, coupled
        self.calls = []

    def __call__(self, history, prefix_records, deadline):
        from act.back_end.moe.multilayer_external import GuardedHistory
        from b2_observed_worker import verify_graph
        import torch
        tick(deadline); began = time.monotonic()
        wanted = {tuple(history[:d]) for d in range(len(history))}
        records = [r for r in prefix_records if tuple(map(tuple, r['prefix'])) in wanted]
        phase = self.phase_runner(self.folder/f'v2_history_{len(self.calls):04d}',
            self.source, history, records, len(self.tensors['rows']), self.backend_identity,
            deadline, self.remaining(), phase_started=began)
        tick(deadline)
        value = phase['evidence'] if phase['accepted'] else None
        attacks = None
        if value:
            points = [r['original_input'] for r in value['full_model_replays'] if r['valid']]
            if points:
                attacks = torch.tensor(points, dtype=torch.float64).reshape(-1, *self.tensors['center'].shape[1:])
        meta = dict(method='M_A_HARD120_V2_THEN_REMAINING_COUPLED', phase_status=phase['status'],
            phase_seconds=phase['seconds'], source_id=phase['source_id'], used_fallback=False,
            phase_folder=f'v2_history_{len(self.calls):04d}')
        if value and (value['closed'] or attacks is not None):
            properties = len(self.tensors['rows'])
            outcome = dict(status='verified' if value['closed'] else 'unknown',
                actual_raw_spec=[([[float(i == j) for j in range(properties)]], [0.]) for i in range(properties)],
                cascade=meta, seconds=time.monotonic()-began, evidence_grade='GENBAB_NUMERICAL_FILTER')
        else:
            graph = GuardedHistory(self.model, history, self.tensors['rows'], self.tensors['thresholds']).eval()
            tick(deadline)
            outcome, attacks = (self.coupled or verify_graph)(graph, self.tensors['lower'], self.tensors['upper'],
                len(self.tensors['rows']), deadline)
            tick(deadline)
            outcome = dict(outcome, cascade=dict(meta, used_fallback=True), seconds=time.monotonic()-began)
        self.calls.append(dict(history=[list(p) for p in history], **outcome['cascade'], seconds=outcome['seconds']))
        return outcome, attacks


def audit_files(folder, result, source_request, deadline):
    """Structural reception of cascade evidence, not an independent bound proof."""
    from moe_aware_diagnostics import gate_module
    G = gate_module()
    expected = {k: source_request[k] for k in ('model', 'sample', 'selection_root')}
    for relative, digest in result['cascade_files'].items():
        tick(deadline)
        W.require(not Path(relative).is_absolute() and '..' not in Path(relative).parts,
                  'owned relative phase artifact')
        W.require(sha(folder/relative) == digest, 'phase artifact/result identity')
    for row in result['histories']:
        meta = row['outcome'].get('cascade')
        if meta is None or meta['used_fallback'] or meta['phase_status'] != 'POSITIVE':
            continue
        child = folder/meta['phase_folder']; request = load(child/'phase_request.json')
        W.require(request['source_request'] == expected and request['history'] == row['history'],
                  'same model/input and history in cascade result')
        W.require(request['source_id'] == meta['source_id'] == identity(expected, row['history'],
            request['prefix_bounds'], request['backend_identity']), 'same range source')
        receipt = load(child/'session.json'); publication = load(child/'published.json')
        value = load(child/'outcome.json', publication['sha256'])
        W.require(receipt['status'] == 'COMPLETED' and receipt['cleanup_status'] == 'LEADER_REAPED_NO_LIVE_GROUP'
            and value['request_sha256'] == sha(child/'phase_request.json')
            and value['clock'] < request['worker_deadline'] and publication['clock'] < request['worker_deadline'],
            'owned on-time phase evidence')
        domain = scoped_domain(G, expected, row['history'], request['prefix_bounds'],
            request['source_id'], request['backend_identity'], deadline)
        receive_phase(child, request, value['value'], domain, deadline)
        W.require(value['value']['closed'], 'every skipped coupled query has complete endpoint closure')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('folder', type=Path)
    raise SystemExit(phase_worker(parser.parse_args().folder))
