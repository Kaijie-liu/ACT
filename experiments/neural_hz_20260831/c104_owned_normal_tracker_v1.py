"""Precision statistics only during a fresh exact owned encoder's emission."""
import numpy as np
from experiments.neural_hz_20260831.c104_prepared_owned_rows_v1 import PreparedOwnedEncoder
from experiments.neural_hz_20260831.c64_birth_quotient_v1 import BirthTracker as OriginalTracker


class BirthTracker(OriginalTracker):
    def __init__(self,encoder,old_nc,old_eq,output_slots):
        encoder.pool.charge('c65_fresh_normal_producer_binding',32)
        if (type(encoder) is not PreparedOwnedEncoder or encoder.eq or encoder.ineq
                or encoder.logical_rows or encoder.heads_discarded or encoder.pending_uid is not None):
            raise ValueError('fresh empty exact owned source encoder required')
        super().__init__(encoder,old_nc,old_eq,output_slots)

    def _owned_odd(self,row,positions):
        # This takes a row of our own bound producer, NOT an external numeric
        # vector or a serialized normality flag. No row is rewritten before
        # C31 discards its unpublished heads, after which this method rejects.
        self.pool.charge('c65_owned_normal_domain_read',8)
        encoder=self.encoder
        if (encoder.heads_discarded or encoder.pending_uid is not None or encoder.in_auxiliary
                or not 0<=row<len(encoder.eq)):
            raise ValueError('normal statistics outside original owned emission')
        values=encoder.eq[row][1][positions]
        # _prepare + _store_prepared already established and bound finite
        # nonzero normal [2^-20,2^40] coefficients to these actual row buffers.
        # The exponent shift/mask, two comparisons, OR and any are truly absent.
        bits=values.view(np.uint64)
        mantissa=(bits&np.uint64((1<<52)-1))|np.uint64(1<<52)
        divisor=mantissa&(~mantissa+np.uint64(1))
        return mantissa//divisor

    def collect(self,row,dyadic=False):
        cc,cv,bc,bv,rhs=self.encoder.eq[row]
        self.pool.charge('c63_routed_incidence_row',16+3*len(cc))
        self.inspected_rows+=1;self.inspected_coefficients+=len(cc)
        positions=np.flatnonzero(self.raw[cc])
        if not len(positions):return
        ids=cc[positions];self.external_rows+=1;self.external_occurrences+=len(ids)
        if dyadic:
            self.pool.charge('c64_source_dyadic_maximum',32+8*len(cc)+2*len(ids))
            np.maximum.at(self.maxima,ids,1);self.dyadic_maximum_occurrences+=len(ids)
        else:
            self.pool.charge('c65_owned_normal_external_maximum',32+8*len(cc)+10*len(ids))
            np.maximum.at(self.maxima,ids,self._owned_odd(row,positions))
        self.hits.append((row,positions))

    def fold(self,*args,**kwargs):
        observe=kwargs.pop('observe',None)
        er,es,dq,report=super().fold(*args,observe=None,**kwargs)
        report.update(schema='c65_owned_normal_birth_precision_quotient_v1',
            normal_domain_source='fresh_C97_prepared_owned_rows_before_head_retirement',
            normal_reader_events=self.pool.parts.get('c65_owned_normal_domain_read',0)//8,
            arbitrary_vector_or_archive_normality_flags_accepted=False)
        if observe:observe('c65_complete_owned_quotient',report)
        return er,es,dq,report
