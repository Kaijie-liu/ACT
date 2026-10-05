"""Whole-source and actual spliced-HZ binding to independent complete proofs.

Pre-source maps carry reversible tags, NOT a still-valid old Closed receipt.
The complete normalized semantic map is hashed using sparse tag patches;
neither old nor normalized full map arrays are copied. Hash authentication is
reported separately from generation, as for C25/C27. No feasibility shortcut.
"""

from dataclasses import dataclass,fields as dataclass_fields
import hashlib
import json
import struct
import time
import weakref
import numpy as np

from experiments.neural_hz_20260831.c27_source_image_v1 import verify as source_verify
from experiments.neural_hz_20260831.c32_fresh_lineage_v1 import SPLICE,REDIRECT,MASK,ReversibleLineage
from experiments.neural_hz_20260831.c22_uid_runs_v1 import row_for_uid
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c23_sparse_phase_overlay_v1 import Overlay


def semantic_fingerprint(fields,lineage,*,pool):
    """C26's ENTIRE lineage fingerprint without functional map copies."""
    lineage.validate()
    edits={}
    pool.charge('semantic_sparse_tag_patch',32*len(lineage.columns))
    for col in lineage.columns:
        at=lineage.old_n_eq+int(col)-lineage.old_n_cont
        raw=int(lineage.eq_roots[at])
        if raw<SPLICE:raise ValueError('missing actual splice tag')
        edits[at]=raw&(SPLICE|((1<<48)-1))
    for raw in lineage.retired:
        uid=int(raw)>>20
        main=row_for_uid(fields['uid_slabs'],uid,pool=pool)
        if main is None:continue
        pool.charge('semantic_sparse_redirect_patch',32)
        at=lineage.old_n_eq+main;value=int(lineage.eq_roots[at])
        if not REDIRECT<=value<SPLICE or at in edits:
            raise ValueError('missing/disjoint actual MAIN consumer redirect')
        edits[at]=REDIRECT|(value&MASK)
    n=len(edits)
    pool.charge('semantic_sparse_patch_sort',4*n*max(1,(n-1).bit_length())+64)
    h=hashlib.sha256(str((lineage.old_n_cont,lineage.old_n_eq)).encode())
    h.update(b'eq_roots')
    data=memoryview(lineage.eq_roots).cast('B');cursor=0
    for at,value in sorted(edits.items()):
        h.update(data[cursor:8*at]);h.update(struct.pack('=q',value));cursor=8*(at+1)
    h.update(data[cursor:])
    for name in ('eq_scales','columns','retired','tails'):
        h.update(name.encode());h.update(memoryview(getattr(lineage,name)).cast('B'))
    return h.hexdigest()


def load_transfer(raw,expected_sha):
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest()!=expected_sha:
        raise ValueError('independently anchored complete splice transfer proof changed')
    proof=json.loads(raw)
    if (proof.get('schema')!='c32_independent_C31_C30_transfer_v1' or proof.get('completed') is not True
            or proof.get('full_C31_source_math_and_report_checked') is not True
            or proof.get('full_C30_HZ_UID_box_reconstruction_checked') is not True
            or proof.get('all_pre_HZ_map_owner_UID_bits_equal') is not True
            or proof.get('formal_gain')!=0):
        raise ValueError('incomplete independent full splice proof chain')
    return proof


@dataclass
class SplicedState:
    original_fields: dict
    hz: object
    lineage: ReversibleLineage
    events: np.ndarray
    old_uid_ceiling: int
    source_proof_bytes: bytes
    source_proof_sha256: str
    transfer_proof_bytes: bytes
    transfer_proof_sha256: str
    construction_report: dict
    receipt: object=None

    def _check(self):
        started=time.monotonic()
        if set(vars(self))!={f.name for f in dataclass_fields(SplicedState)}:
            raise ValueError('unregistered spliced state payload')
        proof=load_transfer(self.transfer_proof_bytes,self.transfer_proof_sha256)
        source=source_verify(self.original_fields,self.source_proof_bytes,
            expected_proof_sha256=self.source_proof_sha256)
        pool=WorkPool(256_000_000)
        semantic=semantic_fingerprint(self.original_fields,self.lineage,pool=pool)
        overlay=Overlay(self.original_fields['owners'],self.events,self.old_uid_ceiling)
        overlay.validate()
        post=source_digest(self.hz)
        event_sha=hashlib.sha256(memoryview(self.events).cast('B')).hexdigest()
        if (self.source_proof_sha256!=proof['new_source_proof_sha256']
                or source['complete_original_source_image_sha256']!=proof['new_closed_identity']
                or source_digest(self.original_fields['hz'])!=proof['pre_HZ_sha256']
                or post!=proof['actual_spliced_HZ_sha256'] or semantic!=proof['semantic_lineage_sha256']
                or event_sha!=proof['actual_phase_events_sha256']
                or self.old_uid_ceiling!=self.original_fields['report']['radix_uid_base']+16384
                or self.lineage.eq_roots is not self.original_fields['eq_roots']
                or self.lineage.eq_scales is not self.original_fields['eq_scales']
                or self.hz.frame_id!=self.original_fields['hz'].frame_id or not self.hz.exact):
            raise ValueError('actual spliced matrix/source/lineage/incidence does not match full proof')
        stamp=(source['complete_original_source_image_sha256'],post,semantic,event_sha,
            hashlib.sha256(json.dumps(self.construction_report,sort_keys=True,allow_nan=False).encode()).hexdigest())
        return stamp,dict(complete_original_source_image=source,complete_semantic_lineage_sha256=semantic,
            complete_new_HZ_sha256=post,semantic_authentication_work=pool.used,
            semantic_authentication_work_parts=dict(pool.parts),full_map_arrays_copied=False,
            complete_source_splice_authentication_elapsed_s=time.monotonic()-started,
            full_hash_byte_scans_are_separate_from_generator_work=True,
            new_splice_base_feasibility_proved=False,formal_gain=0)

    def validate(self):
        if type(self.receipt) is not _Receipt or self.receipt not in _ISSUED:
            raise ValueError('new splice lacks issued complete proof binding')
        stamp,report=self._check()
        if stamp!=_ISSUED[self.receipt]:raise ValueError('bound actual splice or construction report changed')
        return report

    def numeric_roots(self):
        self.validate()
        return dict(original_source_fields=self.original_fields,hz=self.hz,
            **self.lineage.numeric_roots(),phase_events=self.events,old_uid_ceiling=self.old_uid_ceiling,
            source_proof_bytes=self.source_proof_bytes,transfer_proof_bytes=self.transfer_proof_bytes,
            construction_report=self.construction_report,checked_splice_record=_ISSUED[self.receipt])

    def reconstruct_fraction(self,continuous,*,pool):
        self.validate()
        return self.lineage.reconstruct_fraction(self.hz,continuous,pool=pool)


_KEY=object()
_ISSUED=weakref.WeakKeyDictionary()


class _Receipt:
    __slots__=('__weakref__',)
    def __new__(cls,key):
        if key is not _KEY:raise ValueError('only complete new splice checker issues bindings')
        return super().__new__(cls)
    def __reduce__(self):raise TypeError('runtime splice bindings are not portable certificates')


def admit(*,enabled=False,**fields):
    if not enabled:return None
    state=SplicedState(**fields)
    stamp,report=state._check()
    state.receipt=_Receipt(_KEY);_ISSUED[state.receipt]=stamp
    return state,report


def export(state):
    state.validate()
    return {k:v for k,v in vars(state).items() if k!='receipt'}
