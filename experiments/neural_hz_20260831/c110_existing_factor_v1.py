# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exact complete-form quotient; isolated prototype, no source/LIVE admission."""
from fractions import Fraction as F
import math
import struct
import numpy as np

from experiments.neural_hz_20260831.c86_complete_tile_hz_v1 import (
    native_row, project_outputs, extend_actual)
from experiments.neural_hz_20260831.c109_shared_partial_v1 import packed as original_packed


def bits(value):
    return struct.unpack('q', struct.pack('d', value))[0]


def literal(word):
    return struct.unpack('d', struct.pack('q', int(word)))[0]


def definition(row, base):
    """Read a whole literal definition, never a support hash or rounded ratio."""
    slot = row['slot']
    pairs = row['coefficients']
    if len({c for c, _ in pairs}) != len(pairs):
        raise ValueError('canonical coalesced definition required')
    terms = {c: F(v) for c, v in pairs}
    if any(not v or c < 0 or c > slot for c, v in terms.items()):
        raise ValueError('nonzero topological actual coefficients required')
    pivot = terms.pop(slot)
    rhs = F(row['rhs'])
    if pivot <= 0 or abs(rhs)+sum(map(abs, terms.values()), F(0)) > pivot:
        raise ValueError('positive pivot and redundant normalized box required')
    if any(not F(2)**-20 <= abs(F(v)) <= F(2)**40 for _, v in pairs):
        raise ValueError('original complete native coefficient window required')
    root = all(c < base for c in terms)
    if not root or len(terms) < 2:
        return root, None
    expression = tuple((c, -v/pivot) for c, v in sorted(terms.items()))
    lead = expression[0][1]
    key = (tuple((c, v/lead) for c, v in expression), rhs/pivot/lead)
    return root, (key, lead)


def discover(aux, base, *, pool):
    """One invocation is one source/frame namespace; no cross-source cache."""
    pool.charge('c110_actual_definition_and_normal_form',
                128*sum(len(r['coefficients']) for r in aux)+64*len(aux))
    groups, roots = {}, []
    for i, row in enumerate(aux):
        if row['slot'] != base+i:
            raise ValueError('complete contiguous original generated namespace required')
        root, form = definition(row, base)
        if root:
            roots.append(row['slot'])
        if form is not None:
            key, lead = form
            groups.setdefault(key, []).append((row['slot'], lead))
    aliases = {}
    for members in groups.values():
        representative, amplitude = min(members, key=lambda p: (-abs(p[1]), p[0]))
        for slot, lead in members:
            if slot == representative:
                continue
            scale = lead/amplitude
            value = float(scale)
            if not math.isfinite(value) or F(value) != scale or not 0 < abs(scale) <= 1:
                raise ValueError('exact finite dominated inverse scale required')
            aliases[slot] = (representative, value)
    return aliases, roots, dict(input_only_rows=len(roots),
        eligible_multi_parent_rows=sum(map(len, groups.values())),
        duplicate_groups=sum(len(g) > 1 for g in groups.values()),
        candidate_aliases=len(aliases),
        group_size_histogram={str(n):sum(len(g)==n for g in groups.values())
                              for n in sorted({len(g) for g in groups.values()})})


def packed(problem):
    result = original_packed(problem)
    nc = problem['n_cont']
    if 'kept_original_aux' in problem:
        old_nc = nc+len(problem['quotient_inverse'])
        result['source_ids'] = np.array([*range(problem['base']),
            *problem['kept_original_aux'],
            *(old_nc+b for b in problem['source']['binary_ids'])], np.int64)
    result['quotient_inverse'] = np.array(problem.get('quotient_inverse', ()), np.int64).reshape(-1, 3)
    return result


def size(arrays):
    owners = {id(a): a for a in arrays.values()}
    return dict(bytes=sum(a.nbytes for a in owners.values()),
        entries=sum(a.size for a in owners.values()), nnz=arrays['data'].size,
        rows=arrays['row_lb'].size, variables=arrays['var_lb'].size,
        arrays={k:dict(bytes=a.nbytes, entries=a.size, dtype=str(a.dtype)) for k,a in arrays.items()})


def quotient(problem, *, pool, enabled=False):
    if not enabled:
        return None
    base, aux = problem['base'], problem['aux']
    if problem['n_cont'] != base+len(aux) or len(aux) > 16384:
        raise ValueError('bounded complete generated-factor namespace required')
    aliases, roots, report = discover(aux, base, pool=pool)
    report.update(formal_gain=0, source_runtime_LIVE_admitted=False,
                  mathematical_numeric_accepted=False, removed_factors=0)
    if not aliases:
        report['reason'] = 'no_complete_proportional_definitions'
        return report, problem
    root_set = set(roots)
    kept = [s for s in roots if s not in aliases]
    kept += [r['slot'] for r in aux if r['slot'] not in root_set]
    mapping = {old:base+i for i,old in enumerate(kept)}
    pool.charge('c110_complete_remap_and_all_consumer_exact_rewrite',
        256*sum(len(r['coefficients']) for r in [*aux,*problem['outputs']])
        +64*(len(kept)+len(aliases)))

    def rewrite(row):
        merged = {}
        for col, value in row['coefficients']:
            rep, scale = aliases.get(col, (col, 1.))
            target = mapping[rep] if rep >= base else rep
            merged[target] = merged.get(target, F(0))+F(value)*F(scale)
        old_slot = row.get('slot')
        slot = mapping[old_slot] if old_slot is not None and old_slot >= base else old_slot
        new = native_row(sorted(merged.items()), F(row['rhs']), slot=slot)
        new['gauge'] += row['gauge']
        return new

    candidate = dict(problem, aux=[rewrite(aux[s-base]) for s in kept],
        outputs=[rewrite(r) for r in problem['outputs']], n_cont=base+len(kept),
        kept_original_aux=tuple(kept),
        quotient_inverse=tuple((s,mapping[rep],bits(scale))
            for s,(rep,scale) in sorted(aliases.items())))
    for r in candidate['aux']:
        definition(r, base)
    before, after = size(packed(problem)), size(packed(candidate))
    pool.charge('c110_complete_numeric_admission_ledger',4*(before['entries']+after['entries']))
    if not all(after[k] < before[k] for k in ('nnz','bytes','entries')):
        report.update(reason='complete_numeric_cost_not_strictly_lower',
                      proposed_before=before, proposed_after=after)
        return report, problem
    report.update(mathematical_numeric_accepted=True, removed_factors=len(aliases),
                  reason='exact_complete_form_quotient')
    return report, candidate


def reconstruct(original, candidate, arrays, point):
    """Use only the exported kept-ID and omitted-factor inverse payload."""
    if len(point) != candidate['n_cont']:
        raise ValueError('complete candidate continuous point required')
    result = [None]*original['n_cont']
    for old, value in zip(arrays['source_ids'][:candidate['n_cont']], point, strict=True):
        result[int(old)] = F(value)
    for old, representative, word in arrays['quotient_inverse']:
        result[int(old)] = F(literal(word))*F(point[int(representative)])
    if any(v is None or abs(v)>1 for v in result):
        raise ValueError('complete original normalized inverse required')
    return result


def prove_and_measure(old, new, expected, report, *, pool):
    """Universal literal identity for every factor, plus actual two-phase points."""
    if old['source'] is not new['source'] or old['base'] != new['base']:
        raise ValueError('same original source/frame/owner object required')
    base = old['base']
    before, after = packed(old), packed(new)
    for problem in (old, new):
        if project_outputs(problem['aux'], problem['outputs'], base) != expected:
            raise ValueError('complete output differs from independent spatial polynomial')
    old_selectors = [dict(coefficients=((r['slot'],1.),),rhs=0.,gauge=0) for r in old['aux']]
    old_forms = project_outputs(old['aux'], old_selectors, base)
    exported = {int(old_id):(i,F(1)) for i,old_id in enumerate(after['source_ids'][:new['n_cont']])}
    exported.update({int(s):(int(rep),F(literal(word))) for s,rep,word in after['quotient_inverse']})
    inverse_selectors = [dict(coefficients=((exported[r['slot']][0],exported[r['slot']][1]),),
                              rhs=0.,gauge=0) for r in old['aux']]
    inverse_forms = project_outputs(new['aux'], inverse_selectors, base)
    if old_forms != inverse_forms:
        raise ValueError('some old generated-factor polynomial is not reconstructed')
    output_ids = set(map(int,old['source']['output_ids'].flat))
    for binary in (-1,1):
        point = [F((i%3)-1,4) for i in range(base)]
        point[0] = F(binary,2)
        point[int(old['source']['old_inverse'][0,0])] = point[0]/2
        for poly in expected:
            output = next(c for c in poly if c in output_ids)
            point[output] = -(poly.get(-1,F(0))+sum((v*point[c] for c,v in poly.items()
                if c not in (-1,output)),F(0)))/poly[output]
        extended = extend_actual(new['aux'], point)
        recovered = reconstruct(old,new,after,extended)
        if recovered != extend_actual(old['aux'],point):
            raise ValueError('literal inverse differs from original unique extension')
        for problem,values in ((old,recovered),(new,extended)):
            for r in [*problem['aux'],*problem['outputs']]:
                if sum((F(v)*values[c] for c,v in r['coefficients']),F(0)) != F(r['rhs']):
                    raise ValueError('some complete original or rewritten equation failed')
            for equality,cc,bc,rhs in problem['source']['source_rows']:
                value = sum((F(v)*values[c] for c,v in cc),F(0))+sum(F(v)*binary for c,v in bc)
                if (equality and value!=F(rhs)) or (not equality and value>F(rhs)):
                    raise ValueError('original source EQ/INEQ/binary meaning differs')
    b,a = size(before),size(after)
    pool.charge('c110_final_complete_comparison_ledger',4*(b['entries']+a['entries']))
    report.update(before=b,after=a,nnz_delta=a['nnz']-b['nnz'],
        byte_delta=a['bytes']-b['bytes'],entry_delta=a['entries']-b['entries'],
        exact_all_output_polynomials=True,exact_all_old_factor_inverse_polynomials=True,
        actual_inverse_both_binary_phases=True,all_source_EQ_INEQ_retained=True,
        all_normalized_boxes_proved=True,unchanged_source_identity=True,
        strict_complete_numeric_win=all(a[k]<b[k] for k in ('nnz','bytes','entries')),
        work=pool.used,python_objects_in_numeric_byte_claim=False,
        full_source_LIVE_and_runtime_payment_proved=False)
    return report,before,after
