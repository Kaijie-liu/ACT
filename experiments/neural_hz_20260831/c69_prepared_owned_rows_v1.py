"""Prepared emission with ownership at the REAL one-use store boundary.

No old emit hook is assumed to see direct C29 fitting rows. All uid and radix
relocation events follow the actual storage path. Construction-only head lists
are explicitly discarded before the unchanged C24 alias rewrite.
"""

from experiments.neural_hz_20260831.c69_prepared_row_v1 import _Encoder
from experiments.neural_hz_20260831.c17_packed_ownership_v1 import word


class PreparedOwnedEncoder(_Encoder):
    def __init__(self,*args,pool,ledger,radix_uid_base,**kwargs):
        super().__init__(*args,head_pool=pool,**kwargs)
        self.pool,self.ledger,self.radix_uid_base=pool,ledger,int(radix_uid_base)
        self.eq_uids,self.ineq_uids=[],[]
        self.pending_uid=None;self.in_auxiliary=False
        self.logical_rows=self.logical_coefficients=0
        self.heads_discarded=False

    def charge(self,work):
        # Preserve BOTH the global coupled pool and the independent radix
        # reserve. The C29 base encoder otherwise charges only its local reserve.
        self.pool.charge('radix',work)
        super().charge(work)

    def encode_uid(self,uid,cc,cv,bc,bv,rhs,**kwargs):
        word(uid)
        if self.pending_uid is not None or self.heads_discarded:
            raise ValueError('nested or post-discard logical row emission')
        self.pool.charge('prepared_logical_input_counters',4)
        self.pending_uid=uid
        try:
            result=super().encode(cc,cv,bc,bv,rhs,**kwargs)
            self.logical_rows+=1
            self.logical_coefficients+=len(cv)+len(bv)
            return result
        finally:
            self.pending_uid=None

    def auxiliary(self,*args,**kwargs):
        if self.in_auxiliary or self.pending_uid is None or self.heads_discarded:
            raise ValueError('nested/unowned/post-discard radix emission')
        self.in_auxiliary=True
        try:return super().auxiliary(*args,**kwargs)
        finally:self.in_auxiliary=False

    def _store_prepared(self,cc,bc,payload,*,inequality=False):
        if self.pending_uid is None or self.heads_discarded:
            raise ValueError('physical prepared row lacks current logical UID')
        uid=self.radix_uid_base+len(self.def_rows) if self.in_auxiliary else self.pending_uid
        word(uid)
        self.pool.charge('ownership_physical_row_label',1)
        if self.in_auxiliary:
            self.ledger.relocate_radix(cc,self.pending_uid,uid)
        index,scale=super()._store_prepared(cc,bc,payload,inequality=inequality)
        labels=self.ineq_uids if inequality else self.eq_uids
        if index!=len(labels):raise ValueError('physical row/UID label ordering diverged')
        labels.append(uid)
        return index,scale

    def discard_unpublished_heads(self):
        if self.pending_uid is not None or self.in_auxiliary or self.heads_discarded:
            raise ValueError('head retirement outside unpublished emission boundary')
        self.pool.charge('prepared_head_metadata_retirement',16)
        physical=len(self.eq)+len(self.ineq)
        if (len(self.eq_heads)!=len(self.eq) or len(self.ineq_heads)!=len(self.ineq)
                or len(self.eq_uids)!=len(self.eq) or len(self.ineq_uids)!=len(self.ineq)):
            raise ValueError('incomplete emitted physical/head/UID population')
        attempts=self.pool.parts.get('prepared_row_decision_and_head',0)//16
        if attempts!=self.logical_rows+len(self.def_rows)+self.packed_rows:
            raise ValueError('complete prepared attempts do not match actual radix topology')
        result=dict(schema='c31_actual_prepared_owned_encoding_v1',
            logical_input_rows=self.logical_rows,logical_input_coefficients=self.logical_coefficients,
            actual_preparation_attempts=attempts,physical_rows_before_alias=physical,
            actual_omitted_post_magnitude_coefficients=self.omitted_post_magnitude_elements,
            credited_logical_coefficients=self.logical_coefficients,
            extra_radix_coefficients_not_credited=self.omitted_post_magnitude_elements-self.logical_coefficients,
            discarded_head_entries=physical,head_metadata_retained_after_alias=False,
            direct_store_canonical_UID_hooks=True,default_off=True,formal_gain=0)
        if result['extra_radix_coefficients_not_credited']<0:
            raise ValueError('claimed omitted coefficient work exceeds actual physical emission')
        self.eq_heads=self.ineq_heads=None
        self.heads_discarded=True
        result.update(schema='c69_finite_input_inverse_bound_prepared_encoding_v1',
            omitted_post_finite_coefficient_checks=self.omitted_post_magnitude_elements,
            original_input_finite_and_complete_inverse_checks_retained=True,
            generic_RHS_finite_check_retained=True)
        return result


def make_encoder(*args,enabled=False,**kwargs):
    if not enabled:return None
    return PreparedOwnedEncoder(*args,**kwargs)
