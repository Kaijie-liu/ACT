"""Exact normalized carrier sharing and necessary native costs; not a writer."""
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
import math
from experiments.neural_hz_20260831.c56_gauged_carrier_v1 import digit_rows
from experiments.neural_hz_20260831.c59_full_consumer_census_v2 import Census as ForwardCensus


def recipe(m,k,pool):
    """Independently eliminate actual native row operands for one root sign."""
    if not isinstance(m,int) or m<=0 or not m&1 or not 53<m.bit_length()<=512 or not 0<=k<60:
        raise ValueError('canonical long mantissa and bounded carrier arity required')
    digits=tuple(digit_rows((m,0),min(53,60-k)))
    pool.charge('c60_independent_Fraction_digit_recurrence',128*(1+len(digits)))
    previous=F(0)
    for i,(digit,width) in enumerate(digits):
        a=max(0,width+k-20)
        pivot=math.ldexp(1.,a)
        parent=math.ldexp(1.,a-width) if i else 0.
        coefficient=math.ldexp(float(digit),a-width-k) if digit else 0.
        for value in (pivot,parent,coefficient):
            if value and (not math.isfinite(value) or not 2**-20<=abs(value)<=2**40):
                raise ValueError('native digit operand outside unchanged window')
        if (F(pivot)!=F(2)**a or (i and F(parent)!=F(2)**(a-width))
                or F(coefficient)!=digit*F(2)**(a-width-k)):
            raise ValueError('rounded native digit operand')
        current=(F(parent)*previous+F(coefficient))/F(pivot)
        if not 0<=current*2**k<=1:raise ValueError('carrier intermediate box is not redundant')
        previous=current
    if previous!=F(m,2**(m.bit_length()+k)):
        raise ValueError('native digit recurrence changed exact mantissa')
    return digits


class CarrierCensus(ForwardCensus):
    def __init__(self,roots,weights,removed,pool):
        super().__init__(roots,weights,removed,pool)
        self.plans={};self.legacy_keys=set();self.recipes={}
        self.native=Counter(group_occurrences=0,long_coefficient_terms=0,auxiliary_continuous=0,auxiliary_nnz=0)
        self.arities=Counter();self.native_shifts=Counter();self.plan_digest=hashlib.sha256()
        self.consumer_digest=hashlib.sha256()

    def inspect_row(self,updates,untouched_values,binary_values,rhs,*,kind,index):
        low,high,original_shift=super().inspect_row(updates,untouched_values,binary_values,rhs,kind=kind,index=index)
        self.pool.charge('c60_complete_long_group_scan',16*len(updates)+64)
        buckets={}
        for root,(m,e) in sorted(updates.items()):
            if abs(m).bit_length()>53:buckets.setdefault((abs(m),e),[]).append((root,1 if m>0 else -1))
        self.pool.charge('c60_canonical_group_order',len(buckets)*max(1,len(buckets).bit_length()))
        multipliers=[];row_delta=0
        for (m,e),terms in sorted(buckets.items()):
            terms=tuple(terms);n=len(terms);k=(n-1).bit_length()
            sign=terms[0][1];normalized=tuple((root,s*sign) for root,s in terms)
            if (len({root for root,s in normalized})!=n or any(s not in (-1,1) for root,s in normalized)
                    or normalized[0][1]!=1):raise ValueError('canonical distinct signed-root carrier required')
            key=(m,normalized)
            self.pool.charge('c60_exact_carrier_key_and_digest',64+16*n)
            if original_shift is not None:self.legacy_keys.add((m,e+original_shift,terms))
            self.native['group_occurrences']+=1;self.native['long_coefficient_terms']+=n
            row_delta+=1-n
            if key not in self.plans:
                shape=(m,k)
                if shape not in self.recipes:self.recipes[shape]=recipe(m,k,self.pool)
                digits=self.recipes[shape]
                self.pool.charge('c60_new_carrier_exact_cost',64+16*len(digits))
                aux=len(digits)
                nnz=sum(1+(i>0)+(n if d else 0) for i,(d,w) in enumerate(digits))
                self.plans[key]=(k,aux,nnz)
                self.native['auxiliary_continuous']+=aux;self.native['auxiliary_nnz']+=nnz
                self.arities[str(n)]+=1
                self.plan_digest.update(json.dumps([key,k,aux,nnz]).encode())
            power=m.bit_length()+e+k
            self.pool.charge('c60_all_consumer_multiplier_intervals',32)
            low=max(low,-20-power);high=min(high,40-power)
            multipliers.append((sign,power))
        shift=None if low>high else min(max(0,low),high)
        self.native['native_gauge_incompatible_rows']+=shift is None
        self.native['native_shift_differs_from_original_profile']+=shift!=original_shift
        self.native['predicate_carrier_replacement_nnz_delta']+=row_delta
        if shift is not None:self.native_shifts[str(shift)]+=1
        self.consumer_digest.update(json.dumps([kind,index,low,high,shift,multipliers]).encode())
        return low,high,shift

    def finish(self,hz,observe=None):
        base=super().finish(hz,observe)
        self.pool.charge('c60_complete_native_profile_record',4096)
        aux=self.native['auxiliary_continuous'];nnz=self.native['auxiliary_nnz']
        after=base['symbolic_predicate_nnz']+nnz+self.native['predicate_carrier_replacement_nnz_delta']
        added=2*nnz+3*aux
        preallocation=256*(aux+len(self.plans))+32*nnz+128
        flags=dict(all_native_consumer_gauges=self.native['native_gauge_incompatible_rows']==0,
            auxiliary_cap=aux<=16384,complete_added_entry_cap=added<=131072,
            symbolic_native_predicate_nnz_strict=after<base['original_predicate_nnz'],
            unchanged_C56_preallocation_work_fits_16M=preallocation<=16_000_000)
        result=dict(schema='c60_complete_normalized_carrier_native_cost_v1',counts=dict(self.native),
            normalized_carriers=len(self.plans),exponent_sign_sensitive_C56_keys=len(self.legacy_keys),
            independent_native_recurrence_shapes=len(self.recipes),arity_histogram=dict(self.arities),
            native_shift_histogram=dict(self.native_shifts),native_carrier_identity_sha256=self.plan_digest.hexdigest(),
            native_consumer_identity_sha256=self.consumer_digest.hexdigest(),added_native_numeric_entries=added,
            auxiliary_CSR_numeric_byte_lower_bound=12*nnz+20*aux+4,
            symbolic_native_predicate_nnz=int(after),symbolic_native_continuous=base['symbolic_retained_continuous']+aux,
            unchanged_C56_preallocation_work_lower_bound=preallocation,
            unchanged_C56_complete_scan_lower_bound=base['unchanged_C56_full_scan_lower_bound'],
            necessary_flags=flags,all_necessary_native_flags_pass=all(flags.values()),
            full_native_HZ_physical_or_source_first_proved=False,formal_gain=0)
        if result['counts']['long_coefficient_terms']!=base['counts']['derived_mantissa_over53']:
            raise ValueError('native plan omitted a long actual coefficient')
        if observe:observe(dict(event='complete_native_carrier_cost_profile',**result))
        base['native_carrier_profile']=result
        return base
