"""Default-off live Closed HZ and sparse append ownership in one apply scope."""

from contextlib import contextmanager
import gc
import hashlib
import json
import weakref

from experiments.neural_hz_20260831 import c9_live_runtime_v1 as base
from experiments.neural_hz_20260831.c9_integrated_suffix_v1 import lift as original_lift
from experiments.neural_hz_20260831.c24_dense_emission_v1 import lift as fresh_lift
from experiments.neural_hz_20260831.c24_checked_overlay_v1 import build as build_overlay
from experiments.neural_hz_20260831.c23_sparse_phase_overlay_v1 import Overlay
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import check_append
from experiments.neural_hz_20260831.c25_live_binding_v1 import bind
from experiments.neural_hz_20260831.c10_fused_rows_v1 import WorkPool
from experiments.neural_hz_20260831.c5_functional_transaction_v1 import measured_build
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest

SelectedRejected = base.SelectedRejected
EXTRA = {'closed_binding', 'phase_ownership'}
_PHASE_ISSUER = object()
_PHASE_BINDINGS = weakref.WeakKeyDictionary()


class _PhaseBinding:
    """Immutable construction binding, not a base-feasibility certificate."""
    __slots__ = ('__weakref__',)
    def __new__(cls, issuer):
        if issuer is not _PHASE_ISSUER: raise ValueError('phase binding requires actual construction')
        return super().__new__(cls)
    def __reduce__(self):
        raise TypeError('phase runtime bindings require an independently authenticated archive')


def _phase_stamp(owned):
    return (owned['post_sha256'], owned['event_sha256'], owned['report']['source_closed_identity'],
        hashlib.sha256(json.dumps((owned['report'], owned['construction']), sort_keys=True, allow_nan=False).encode()).hexdigest())


def append_live(closed, actual):
    """Read ACTUAL newly constructed native rows, before accepting apply result."""
    closed.validate()
    check_append(closed.hz, actual)
    pre = closed.hz
    ne, nl = actual.n_eq - pre.n_eq, actual.n_ineq - pre.n_ineq
    first = closed.report['radix_uid_base'] + 16384
    pool = WorkPool(closed.report['total_work_upper'], closed.report['largest_branch_work_upper'])
    nnz = int(actual.Ac.nnz - pre.Ac.nnz + actual.Auc.nnz - pre.Auc.nnz)
    pool.charge('actual_appended_CSR_slices', 8 * nnz + 4 * (ne + nl))
    overlay, report = build_overlay(closed,
        [(actual.Ac[pre.n_eq:], first), (actual.Auc[pre.n_ineq:], first + ne)],
        pool=pool, enabled=True)
    report.update(actual_native_rows_used=True, new_EQ_rows=ne, new_INEQ_rows=nl,
        new_phase_executed=True, copied_dense_phase_owner_vector=False,
        whole_generation_plus_event_work=pool.whole_base + pool.used,
        branch_generation_plus_event_work=pool.branch_base + pool.used,
        append_construction_work=pool.used, append_construction_work_parts=dict(pool.parts),
        source_closed_identity=closed.fingerprint(), formal_gain=0)
    return {'base': overlay.base, 'events': overlay.events, 'old_uid_ceiling': first,
        'post_hz': actual, 'post_sha256': source_digest(actual),
        'event_sha256': hashlib.sha256(overlay.events.tobytes()).hexdigest(), 'report': report}


def phase_overlay(state):
    owned = state['phase_ownership']
    if type(owned) is not dict or set(owned) != {'base', 'events', 'old_uid_ceiling', 'post_hz',
            'post_sha256', 'event_sha256', 'report', 'construction', 'receipt'}:
        raise ValueError('missing/unregistered live phase ownership')
    receipt = owned['receipt']
    if type(receipt) is not _PhaseBinding or receipt not in _PHASE_BINDINGS or _PHASE_BINDINGS[receipt] != _phase_stamp(owned):
        raise ValueError('live phase construction binding changed, including resealing')
    closed = state['lifted']
    closed.validate()
    if (owned['base'] is not closed.owners
            or owned['report']['source_closed_identity'] != closed.fingerprint()
            or owned['old_uid_ceiling'] != closed.report['radix_uid_base'] + 16384
            or source_digest(owned['post_hz']) != owned['post_sha256']
            or hashlib.sha256(owned['events'].tobytes()).hexdigest() != owned['event_sha256']):
        raise ValueError('live phase HZ/base/append ownership binding changed')
    result = Overlay(owned['base'], owned['events'], owned['old_uid_ceiling'])
    result.validate()
    return result


def numeric_roots(state):
    if not EXTRA <= set(state):
        raise ValueError('closed live state lacks explicit proof/phase records')
    result = base.numeric_roots({k:v for k,v in state.items() if k not in EXTRA})
    result['closed_binding'] = state['closed_binding']
    if state['phase_ownership'] is not None:
        phase_overlay(state)
        owned = state['phase_ownership']
        result['phase_ownership'] = {k:v for k,v in owned.items() if k != 'receipt'}
        result['phase_construction_binding'] = _PHASE_BINDINGS[owned['receipt']]
    return result


@contextmanager
def installed(*, enabled=False, proof_bytes=None, expected_proof_sha256=None,
              before=None, ready=None, consumed=None, emit=None):
    if not enabled:
        yield
        return
    if (type(proof_bytes) is not bytes or hashlib.sha256(proof_bytes).hexdigest() != expected_proof_sha256):
        raise ValueError('missing/changed independently anchored live proof')
    if base.lift is not original_lift:
        raise ValueError('closed runtime requires the unmodified registered lift hook')

    def report(name, **values):
        if emit is not None: emit({'event': name, **values})

    def entering(state):
        state.update(closed_binding=None, phase_ownership=None)
        if before is not None: before(state)

    def built(state):
        draft = state['lifted']
        retired = [weakref.ref(n[k]) for n in draft.nodes for k in ('support','needed','slots','exponents')]
        draft_ref = weakref.ref(draft)
        closed, binding = bind(draft, proof_bytes, expected_proof_sha256=expected_proof_sha256, enabled=True)
        state['lifted'] = closed
        del draft
        gc.collect()
        if draft_ref() is not None or any(ref() is not None for ref in retired):
            raise ValueError('live proof binding retained the unpublished construction graph')
        binding.update(all_new_graph_fields_physically_retired=True, retired_field_arrays=len(retired))
        state['closed_binding'] = binding
        report('c25_closed_live_bound', **binding)
        if ready is not None: ready(state)

    def applied(state, fact):
        if state['consumer_construction'] is None or state['phase_ownership'] is not None:
            raise ValueError('selected closed scope lacks exactly one native ReLU construction')
        actual = state['tf']._sparse_hz_cache.get(state['layer'].id)
        if actual is None:
            raise ValueError('actual native post-ReLU cache publication is missing')
        owned, measurement = measured_build(lambda: append_live(state['lifted'], actual))
        owned['construction'] = measurement
        receipt = _PhaseBinding(_PHASE_ISSUER)
        _PHASE_BINDINGS[receipt] = _phase_stamp(owned)
        owned['receipt'] = receipt
        state['phase_ownership'] = owned
        phase_overlay(state)
        report('c25_actual_native_phase_ownership', **owned['report'], construction=measurement)
        if consumed is not None: consumed(state, fact)

    base.lift = fresh_lift
    try:
        with base.installed(enabled=True, before=entering, ready=built, consumed=applied, emit=emit):
            yield
    finally:
        base.lift = original_lift
