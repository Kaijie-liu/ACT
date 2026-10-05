"""New checked native custody for immutable local-source + sparse splice maps.

Externally anchored complete proof text is an admission check, never an
eligibility selector. No archived numeric HZ or legacy Closed token is used.
Authentication scans are separately paid, timed and reported as in C32.
"""
from dataclasses import dataclass, fields as dataclass_fields
import hashlib
import json
import time
import weakref
import numpy as np
from experiments.neural_hz_20260831.c65_physical_archive_v1 import FIELDS, fingerprint
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout
from experiments.neural_hz_20260831.c62_local_equations_v1 import SCHEMA as LOCAL
from experiments.neural_hz_20260831.c68_local_splice_v1 import SCHEMA as JOURNAL
from experiments.neural_hz_20260831.c73_outer_query_v1 import GuardedLocalSpliceJournal
from experiments.neural_hz_20260831.c70_native_proof_v1 import digest, entries
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool


def anchored(raw, expected, schema):
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('independent complete proof text changed')
    record = json.loads(raw)
    if record.get('schema') != schema:
        raise ValueError('exact proof schema mismatch')
    return record


def load_transfer(raw, expected):
    p = anchored(raw, expected, 'c74_complete_source_native_transfer_v1')
    if (p.get('complete_C73_archive_authenticated') is not True
            or p.get('complete_independent_inverse_restored') is not True
            or p.get('full_LIVE_admission') is not False or p.get('formal_gain') != 0):
        raise ValueError('complete independently restored component proof required')
    return p


_KEY = object()
_ISSUED = weakref.WeakKeyDictionary()


class _Receipt:
    __slots__ = ('__weakref__',)
    def __new__(cls, key):
        if key is not _KEY: raise ValueError('only complete binder issues custody')
        return super().__new__(cls)
    def __reduce__(self): raise TypeError('runtime custody is not a portable proof')


def _shape(obj, cls):
    if set(vars(obj)) != {f.name for f in dataclass_fields(cls)}:
        raise ValueError('unregistered runtime state field')


def _issue(obj):
    stamp, report = obj._check()
    obj.receipt = _Receipt(_KEY)
    _ISSUED[obj.receipt] = stamp
    return obj, report


def _validate(obj):
    if type(obj.receipt) is not _Receipt or obj.receipt not in _ISSUED:
        raise ValueError('missing newly issued runtime custody')
    stamp, report = obj._check()
    if stamp != _ISSUED[obj.receipt]: raise ValueError('bound runtime state changed')
    return report


@dataclass
class SourceState:
    original_fields: dict
    proof_bytes: bytes
    expected_proof_sha256: str
    authentication: list
    receipt: object = None

    def __getattr__(self, name):
        if name in FIELDS: return self.original_fields[name]
        raise AttributeError(name)

    def _check(self):
        _shape(self, SourceState)
        started = time.monotonic(); pool = WorkPool(256_000_000)
        record = anchored(self.proof_bytes, self.expected_proof_sha256,
                          'c65_source_bound_physical_proof_record_v1')
        proof = record['proof']
        if (record['native_or_LIVE_admission'] or record['formal_gain'] != 0
                or proof.get('schema') != 'c65_complete_original_rows_local_inverse_owner_proof_v1'
                or proof.get('original_boxes_and_all_new_inverse_boxes_proved') is not True
                or proof.get('certified_unchanged_owner_rows_and_all_changed_actual_rows_proved') is not True
                or self.report['new_lineage_schema'] != LOCAL):
            raise ValueError('complete local-source/box/owner proof required')
        layout = numeric_layout(self.original_fields, pool)
        pool.charge('c74_complete_source_authentication', int(layout.resident_entries) + 1024)
        identity = fingerprint(self.original_fields)
        if identity != record['physical_identity']:
            raise ValueError('fresh full source differs from independent proof')
        report = dict(kind='complete_source', identity=identity, work=pool.used,
                      parts=dict(pool.parts), elapsed_s=time.monotonic()-started,
                      full_hash_scans_separate_from_generation=True)
        self.authentication.append(report)
        return (identity, self.expected_proof_sha256), report

    def validate(self): return _validate(self)

    def numeric_roots(self):
        self.validate()
        return dict(original_source_fields=self.original_fields,
                    source_proof_bytes=self.proof_bytes,
                    source_authentication=self.authentication)


def admit_source(fields, raw, *, expected_sha256, enabled=False):
    if not enabled: return None
    return _issue(SourceState(fields, raw, expected_sha256, []))


def phase_image(source, view, transfer):
    """Full reference-wire image made from FRESH phase arrays, with no copy.

    The inherited offline marker belongs to the reference hash serialization;
    it never claims these live arrays were loaded or supplied by that packet.
    """
    h = source.hz
    out = dict(transfer['packet_header'])
    out.update(pre_c=h.c, pre_Gc=h.Gc, pre_Gb=h.Gb,
        frame_id=h.frame_id, source_n_cont=h.n_cont, source_n_bin=h.n_bin,
        old_n_cont=source.old_n_cont, old_n_eq=source.old_n_eq,
        logical_n_cont=source.logical_n_cont,
        first_uid=source.report['radix_uid_base']+16384)
    out.update({name: getattr(view, name) for name in
        ('eq_c','eq_b','eq_rhs','le_c','le_b','le_rhs','c','Gc','Gb')})
    return out


@dataclass
class NativeState:
    source: SourceState
    hz: object
    lineage: GuardedLocalSpliceJournal
    events: np.ndarray
    actual_phase_image: dict
    transfer_proof_bytes: bytes
    expected_transfer_sha256: str
    construction_report: dict
    authentication: list
    receipt: object = None

    def _check(self):
        _shape(self, NativeState)
        started = time.monotonic(); pool = WorkPool(256_000_000)
        transfer = load_transfer(self.transfer_proof_bytes, self.expected_transfer_sha256)
        checked = self.source.validate(); j = self.lineage
        pool.charge('c74_complete_native_phase_journal_events_authentication',
            entries(self.hz) + sum(a.size for a in j.numeric_roots().values())
            + 4 * sum(getattr(v, 'size', 0) for v in self.actual_phase_image.values())
            + 4 * len(self.events) + len(self.transfer_proof_bytes) + 2048)
        if (type(j) is not GuardedLocalSpliceJournal or j.schema != JOURNAL or j.source_schema != LOCAL
                or j.eq_roots is not self.source.eq_roots or j.eq_scales is not self.source.eq_scales
                or j.old_n_cont != self.source.old_n_cont or j.old_n_eq != self.source.old_n_eq
                or j.source_n_cont != self.source.hz.n_cont
                or self.hz.frame_id != self.source.hz.frame_id or not self.hz.exact
                or self.events.dtype != np.dtype(np.uint64) or self.events.ndim != 1
                or checked['identity'] != transfer['source_identity']
                or self.source.expected_proof_sha256 != transfer['source_proof_sha256']):
            raise ValueError('native/source/local-journal custody mismatch')
        hashes = dict(new_HZ_sha256=source_digest(self.hz),
                      journal_identity=digest(vars(j)),
                      packet_identity=digest(self.actual_phase_image),
                      event_sha256=hashlib.sha256(self.events.tobytes()).hexdigest())
        if any(transfer[k] != value for k, value in hashes.items()):
            raise ValueError('complete actual phase/HZ/journal/event proof mismatch')
        stamp = (checked['identity'], self.expected_transfer_sha256,
                 *hashes.values(), digest(self.construction_report))
        report = dict(kind='complete_native', work=pool.used, parts=dict(pool.parts),
            elapsed_s=time.monotonic()-started, **hashes,
            full_source_and_native_content_bound=True,
            full_hash_scans_separate_from_generation=True,
            full_LIVE_admission=False, concrete_witness=False, formal_gain=0)
        self.authentication.append(report)
        return stamp, report

    def validate(self): return _validate(self)

    def numeric_roots(self):
        self.validate()
        return dict(original_source_fields=self.source.original_fields, hz=self.hz,
            journal=vars(self.lineage), actual_phase_image=self.actual_phase_image,
            phase_events=self.events, source_proof_bytes=self.source.proof_bytes,
            transfer_proof_bytes=self.transfer_proof_bytes,
            construction_report=self.construction_report,
            source_authentication=self.source.authentication,
            native_authentication=self.authentication)

    def reconstruct_fraction(self, continuous, *, pool):
        self.validate()
        return self.lineage.reconstruct_fraction(self.hz, continuous, pool=pool)


def admit_native(*, enabled=False, **fields):
    if not enabled: return None
    return _issue(NativeState(**fields, authentication=[]))
