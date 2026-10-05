# SPDX-License-Identifier: AGPL-3.0-or-later
"""Producer-owned signed unit/dyadic keys; no native-row readback or cache."""


def routes(forms, keep_v, *, pool):
    """Full original-column identities, exact power ratios, one tile namespace."""
    ordered=sorted(keep_v)
    pool.charge('c111_complete_existing_form_keys',
                sum(32+8*len(forms[key]) for key in ordered))
    groups={};descriptors={};independent=[]
    for key in ordered:
        terms=sorted(forms[key])
        if (len({col for col,_,_ in terms})!=len(terms)
            or any(sign not in (-1,1) for _,sign,_ in terms)):
            independent.append(key)
            continue
        _,first_sign,first_power=terms[0]
        canonical=tuple((col,sign*first_sign,power-first_power) for col,sign,power in terms)
        groups.setdefault(canonical,[]).append(key)
        descriptors[key]=(first_sign,first_power)
    routing={key:(key,1,0) for key in independent}
    representatives=[]
    for members in groups.values():
        representative=min(members,key=lambda key:(-descriptors[key][1],key))
        rep_sign,rep_power=descriptors[representative]
        representatives.append((min(members),representative))
        for key in members:
            sign,power=descriptors[key]
            routing[key]=(representative,sign*rep_sign,power-rep_power)
    representatives += [(key,key) for key in independent]
    representatives=[rep for _,rep in sorted(representatives)]
    return representatives,routing
