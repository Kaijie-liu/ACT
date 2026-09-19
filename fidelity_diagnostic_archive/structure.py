"""Saved-source inventory only: no optimization or basis reconstruction."""
from collections import Counter
from fractions import Fraction


def lp_numbers(lp):
    for name in ('c','lower','upper','h','b'):
        for i,v in enumerate(lp[name]):yield f'{name}[{i}]',v
    yield 'offset',lp['offset']
    for name in ('E','A'):
        for i,v in enumerate(lp[name]['data']):yield f'{name}.data[{i}]',v


def input_bit_inventory(lp):
    count=0;max_num=max_den=0;num_location=den_location=None
    for location,stored in lp_numbers(lp):
        # Same exact interpretation of finite stored binary64 values as checker.
        v=Fraction(stored);n=abs(v.numerator).bit_length();d=v.denominator.bit_length()
        count+=1
        if n>max_num:max_num,num_location=n,location
        if d>max_den:max_den,den_location=d,location
    return {'numeric_entries':count,'max_numerator_bits':max_num,'max_denominator_bits':max_den,
            'numerator_location':num_location,'denominator_location':den_location,
            'scope':'stored original LP constants only; not intermediate arithmetic or exact solution size'}


def basis_inventory(mapping):
    if mapping is None or mapping['status']!='MAPPED_HINT_ONLY':return None
    h=mapping['hint']
    return {'basic_counts':dict(Counter(c['kind'] for c in h['basic_columns'])),
            'anchor_counts':dict(Counter(c['column']['kind']+':'+c['at'] for c in h['anchors'])),
            'retained_original_rows':len(h['rows']),
            'scope':'untrusted hint structure; not feasibility'}
