"""Explicit circuit-source/native custody from complete independent text proofs."""
from dataclasses import dataclass
from types import SimpleNamespace
import hashlib
import time
import numpy as np
from experiments.neural_hz_20260831.c74_native_binding_v1 import anchored,_issue,_validate,_shape
from experiments.neural_hz_20260831.c65_physical_archive_v1 import FIELDS
from experiments.neural_hz_20260831.c91_physical_archive_v1 import fingerprint
from experiments.neural_hz_20260831.c91_physical_circuit_v1 import SCHEMA as CIRCUIT
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c99_circuit_consumer_v2 import CircuitSource
from experiments.neural_hz_20260831.c99_circuit_journal_v2 import CircuitJournal,SCHEMA as JOURNAL,reconstruct
from experiments.neural_hz_20260831.c70_native_proof_v1 import digest,entries
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c26_tagged_transplant_v1 import Plan


def load_transfer(raw,expected):
    p=anchored(raw,expected,'c100_complete_circuit_native_transfer_v1')
    if (not p.get('complete_C99_archive_authenticated') or not p.get('complete_independent_inverse_restored')
        or p.get('full_LIVE_admission') is not False or p.get('formal_gain')!=0):
        raise ValueError('complete changed-source/native/circuit inverse proof required')
    return p


class SourceState(SimpleNamespace):
    """Actual complete state plus once-published shared fields, no query proxy."""
    def __init__(self,state,raw,expected):
        super().__init__(**state['fields'])
        self.circuit_state=state;self.consumer=CircuitSource(state)
        self.proof_bytes=raw;self.expected_proof_sha256=expected
        self.authentication=[];self.receipt=None

    def _check(self):
        started=time.monotonic();pool=WorkPool(256_000_000)
        if (set(vars(self))!=FIELDS|{'circuit_state','consumer','proof_bytes','expected_proof_sha256','authentication','receipt'}
            or self.circuit_state['schema']!=CIRCUIT
            or self.consumer.state is not self.circuit_state
            or any(getattr(self,n) is not self.circuit_state['fields'][n]
                or getattr(self.consumer,n) is not self.circuit_state['fields'][n] for n in FIELDS)):
            raise ValueError('complete flat source fields/state identity differs')
        p=anchored(self.proof_bytes,self.expected_proof_sha256,'c91_complete_physical_circuit_proof_v1')
        proof=p['proof']
        if (p['native_or_LIVE_admission'] or p['formal_gain']!=0
            or not proof.get('all_original_maps_and_other_predicates_preserved')
            or not proof.get('independent_complete_owner_delta_proved')
            or not proof.get('every_fresh_native_literal_matches_original_theorem')):
            raise ValueError('complete original and new circuit source/box/owner proof required')
        layout=numeric_layout(self.circuit_state,pool)
        pool.charge('c100_complete_source_authentication',int(layout.resident_entries)+1024+8*len(FIELDS)
            +len(self.circuit_state['original_source_proof'])+len(self.circuit_state['original_circuit_proof']))
        identity=fingerprint(self.circuit_state)
        if identity!=p['identity']:raise ValueError('fresh complete circuit source differs from original proof')
        report=dict(kind='complete_circuit_source',identity=identity,work=pool.used,parts=dict(pool.parts),
            elapsed_s=time.monotonic()-started,full_hash_scans_separate_from_generation=True)
        self.authentication.append(report)
        return ('c100_circuit_source',identity,self.expected_proof_sha256),report

    def validate(self):return _validate(self)

    def numeric_roots(self):
        self.validate()
        return dict(complete_circuit_source=self.circuit_state,source_proof_bytes=self.proof_bytes,
            source_authentication=self.authentication)


def admit_source(state,raw,*,expected_sha256,enabled=False):
    if not enabled:return None
    return _issue(SourceState(state,raw,expected_sha256))


def phase_image(source,view,transfer):
    h=source.hz;out=dict(transfer['packet_header'])
    out['coordinate_injection']=tuple(out['coordinate_injection'])
    out.update(pre_c=h.c,pre_Gc=h.Gc,pre_Gb=h.Gb,frame_id=h.frame_id,source_n_cont=h.n_cont,
        source_n_bin=h.n_bin,old_n_cont=source.old_n_cont,old_n_eq=source.old_n_eq,
        logical_n_cont=source.logical_n_cont,first_uid=source.report['radix_uid_base']+16384)
    out.update({name:getattr(view,name) for name in
        ('eq_c','eq_b','eq_rhs','le_c','le_b','le_rhs','c','Gc','Gb')})
    return out


def journal_image(journal):
    return dict(local=vars(journal.local),circuit_tails=journal.circuit_tails,schema=journal.schema)


@dataclass
class NativeState:
    source: SourceState
    hz: object
    lineage: CircuitJournal
    events: np.ndarray
    actual_phase_image: dict
    transfer_proof_bytes: bytes
    expected_transfer_sha256: str
    construction_report: dict
    authentication: list
    receipt: object=None

    def _check(self):
        _shape(self,NativeState);started=time.monotonic();pool=WorkPool(256_000_000)
        p=load_transfer(self.transfer_proof_bytes,self.expected_transfer_sha256)
        source=self.source.validate();j=self.lineage
        if (type(j) is not CircuitJournal or j.schema!=JOURNAL or j.state is not self.source.circuit_state
            or j.local.eq_roots is not self.source.eq_roots or j.local.eq_scales is not self.source.eq_scales
            or j.local.source_n_cont!=self.source.hz.n_cont or self.hz.frame_id!=self.source.hz.frame_id
            or not self.hz.exact or self.events.dtype!=np.uint64 or self.events.ndim!=1
            or source['identity']!=p['source_identity'] or self.source.expected_proof_sha256!=p['source_proof_sha256']):
            raise ValueError('actual circuit native/source/journal binding differs')
        pool.charge('c100_complete_native_phase_journal_events_authentication',entries(self.hz)
            +sum(a.size for a in j.local.numeric_roots().values())+len(j.circuit_tails)
            +4*sum(getattr(a,'size',0) for a in self.actual_phase_image.values())
            +4*len(self.events)+len(self.transfer_proof_bytes)+2048)
        hashes=dict(new_HZ_sha256=source_digest(self.hz),journal_identity=digest(journal_image(j)),
            packet_identity=digest(self.actual_phase_image),event_sha256=hashlib.sha256(self.events.tobytes()).hexdigest())
        if any(p[k]!=v for k,v in hashes.items()):raise ValueError('fresh complete phase/native/circuit image differs')
        stamp=('c100_circuit_native',source['identity'],self.expected_transfer_sha256,*hashes.values(),digest(self.construction_report))
        report=dict(kind='complete_circuit_native',work=pool.used,parts=dict(pool.parts),elapsed_s=time.monotonic()-started,
            **hashes,full_source_and_native_content_bound=True,full_hash_scans_separate_from_generation=True,
            full_LIVE_admission=False,concrete_witness=False,formal_gain=0)
        self.authentication.append(report);return stamp,report

    def validate(self):return _validate(self)

    def numeric_roots(self):
        self.validate()
        return dict(complete_circuit_source=self.source.circuit_state,hz=self.hz,journal=journal_image(self.lineage),
            actual_phase_image=self.actual_phase_image,phase_events=self.events,source_proof_bytes=self.source.proof_bytes,
            transfer_proof_bytes=self.transfer_proof_bytes,construction_report=self.construction_report,
            source_authentication=self.source.authentication,native_authentication=self.authentication)

    def reconstruct_fraction(self,continuous,*,pool):
        self.validate();p=load_transfer(self.transfer_proof_bytes,self.expected_transfer_sha256)
        plans=[Plan(**{**v,'tail':tuple(v['tail'])}) for v in p['complete_component_proof']['plans']]
        return reconstruct(self.source.consumer,self.hz,self.lineage,plans,continuous,pool=pool,enabled=True)['full_point']


def admit_native(*,enabled=False,**fields):
    if not enabled:return None
    return _issue(NativeState(**fields,authentication=[]))
