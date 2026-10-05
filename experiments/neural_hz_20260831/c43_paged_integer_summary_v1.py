"""Exact integer summary with bounded pages and saturated histogram counters.

One byte per occupied PAGE SLOT, not a Python dictionary entry per integer.
Values>=16 retain exactly the exported16+ category; total occurrences and
distinctness remain exact. This is diagnostic metadata, not HZ ownership.
"""
from collections import Counter
import numpy as np

PAGE=256;MAX_SLOTS=64_000_000


def category(n):
    return None if n==0 else '1' if n==1 else '2' if n==2 else '3' if n==3 else '4-7' if n<8 else '8-15' if n<16 else '16+'


class Summary:
    def __init__(self,pool):
        self.pool=pool;self.pages={};self.occurrences=0;self.distinct=0
        self.small_occurrences=0;self.small_distinct=0;self.bins=Counter()
    def add(self,value):
        if type(value) is not int:raise ValueError('integer literal, not memo index or bool, required')
        self.pool.charge('paged_integer_lookup_and_summary_update',16)
        page,slot=divmod(value,PAGE)
        if page not in self.pages:
            if (len(self.pages)+1)*PAGE>MAX_SLOTS:raise MemoryError('unchanged64M numeric metadata entry cap')
            self.pool.charge('paged_integer_owner_allocation',PAGE+64)
            self.pages[page]=np.zeros(PAGE,dtype=np.uint8)
        a=self.pages[page];old=int(a[slot]);new=min(old+1,16)
        if not old:
            self.pool.charge('paged_integer_first_value',16);self.distinct+=1
        self.occurrences+=1
        if -5<=value<=256:
            self.small_occurrences+=1;self.small_distinct+=not old
        else:
            before,after=category(old),category(new)
            if before!=after:
                if before:self.bins[before]-=1
                self.bins[after]+=1
        a[slot]=new
    def report(self):
        self.pool.charge('paged_integer_final_summary',256)
        return dict(integer_literal_occurrences=self.occurrences,distinct_integer_literal_values=self.distinct,
            small_integer_range=[-5,256],small_range_integer_occurrences=self.small_occurrences,
            small_range_distinct_values=self.small_distinct,
            outside_small_range_integer_occurrences=self.occurrences-self.small_occurrences,
            outside_small_range_distinct_values=self.distinct-self.small_distinct,
            outside_small_range_repeated_occurrences=self.occurrences-self.small_occurrences-self.distinct+self.small_distinct,
            outside_small_range_value_multiplicity_histogram=dict(sorted((k,v) for k,v in self.bins.items() if v)),
            paged_counter_pages=len(self.pages),paged_counter_numeric_slots=PAGE*len(self.pages),
            paged_counter_numeric_bytes=PAGE*len(self.pages),individual_counts_above16_not_retained=True,
            complete_exported_frequency_categories_exact=True)
