# SPDX-License-Identifier: AGPL-3.0-or-later
"""Default-off exact sink projection census; never mutates or admits an HZ."""
from collections import Counter, defaultdict
from fractions import Fraction as F
import math

from experiments.neural_hz_20260831.c86_complete_tile_hz_v1 import native_row


def bounded(value):
    value = F(value)
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > 512:
        raise ValueError('exact rational width exceeds512')
    return value


def read_row(row, pool):
    entries = row['coefficients']
    pool.charge('c114_native_headers_and_rationals', 128*len(entries)+64)
    if not entries or len({c for c, _ in entries}) != len(entries):
        raise ValueError('complete nonempty coalesced native row required')
    result = {}
    for col, val in entries:
        if type(col) is not int or col < 0 or not math.isfinite(val) or not 2.**-20 <= abs(val) <= 2.**40:
            raise ValueError('native column or coefficient window invalid')
        result[col] = bounded(val)
    if not math.isfinite(row['rhs']):
        raise ValueError('finite native RHS required')
    return result, bounded(row['rhs'])


def definition(row, values, rhs):
    slot = row['slot']
    if slot not in values:
        raise ValueError('unique defining pivot absent')
    pivot = values[slot]
    if (pivot <= 0 or pivot.numerator & (pivot.numerator-1)
            or pivot.denominator & (pivot.denominator-1)):
        raise ValueError('positive power-of-two pivot required')
    parents = {c:v for c,v in values.items() if c != slot}
    if any(c >= slot for c in parents):
        raise ValueError('complete triangular defining row required')
    if abs(rhs)+sum(map(abs,parents.values()),F(0)) > pivot:
        raise ValueError('factor box is not proved redundant')
    return pivot, parents


def native(values, rhs, old, pool):
    pool.charge('c114_exact_native_conversion',128*len(values)+64)
    checked = native_row(sorted(values.items()),rhs,slot=old['slot'])
    shift = checked['gauge']
    scale = F(2)**shift
    if ({c:F(v)/scale for c,v in checked['coefficients']} != values
            or F(checked['rhs'])/scale != rhs):
        raise ValueError('independent literal equation check failed')
    checked['gauge'] += old['gauge']
    return checked


def group_result(slots, consumers, rows, definitions, pool):
    """Independent groups may overlap consumers: NEVER sum their gain."""
    selected = set(slots)
    old_nnz = sum(len(rows[s]['coefficients']) for s in slots)
    old_nnz += sum(len(rows[s]['coefficients']) for s in consumers)
    multiplicities = []
    generated = 0
    for consumer in consumers:
        row = rows[consumer]
        count = sum(len(definitions[c][1]) if c in selected else 1
                    for c,_ in row['coefficients'])
        pool.charge('c114_multiplicity_lower_bound',16*count)
        generated += count
        counts = Counter()
        for col,_ in row['coefficients']:
            if col in selected:
                counts.update(definitions[col][1].keys())
            else:
                counts[col] += 1
        multiplicities.append(sum(v == 1 for v in counts.values()))
    lower = sum(multiplicities)
    result = dict(slots=list(slots),consumers=list(consumers),factors=len(slots),
        old_local_nnz=old_nnz,new_nnz_lower=lower,lower_by_consumer=multiplicities,
        generated_terms=generated,strict_nnz_reduction_proved=False,
        candidate_only=True,actual_eliminations=0)
    if lower >= old_nnz:
        result['reason'] = 'necessary_nnz_lower_bound'
        return result
    projected = []
    for consumer in consumers:
        row = rows[consumer]
        count = sum(len(definitions[c][1]) if c in selected else 1
                    for c,_ in row['coefficients'])
        pool.charge('c114_complete_exact_expansion',128*count+64)
        values = {}; rhs = bounded(row['rhs'])
        for col,coefficient in row['coefficients']:
            coefficient = F(coefficient)
            if col not in selected:
                values[col] = bounded(values.get(col,F(0))+coefficient)
                continue
            pivot,parents,offset = definitions[col]
            multiplier = bounded(coefficient/pivot)
            rhs = bounded(rhs-multiplier*offset)
            for parent,value in parents.items():
                values[parent] = bounded(values.get(parent,F(0))-multiplier*value)
        values = {c:v for c,v in values.items() if v}
        try:
            projected.append(native(values,rhs,row,pool))
        except (ValueError,OverflowError) as exc:
            result.update(reason='exact_native_guard',guard=str(exc))
            return result
    new_nnz = sum(len(r['coefficients']) for r in projected)
    result.update(new_local_nnz=new_nnz,nnz_delta=new_nnz-old_nnz)
    if new_nnz >= old_nnz:
        result['reason'] = 'exact_nnz_not_smaller'
        return result
    pool.charge('c114_exact_proof_serialization',16*sum(
        2*len(r['coefficients'])+3 for r in projected))
    result.update(reason='independent_group_exact_reduction',
        strict_nnz_reduction_proved=True,projected_native_rows=projected,
        inverse_definition_slots=list(slots),unique_inverse_and_box_proved=True,
        packet_only_byte_delta_formula=12*(new_nnz-old_nnz)-88*len(slots),
        packet_only_entry_delta_formula=2*(new_nnz-old_nnz)-13*len(slots),
        complete_source_physical_gate_proved=False)
    return result


def analyse(auxiliary, outputs, base, *, pool, aliases=(), enabled=False):
    if not enabled:
        return None
    if type(base) is not int or base < 0:
        raise ValueError('original continuous namespace required')
    all_rows = [*auxiliary,*outputs]
    if ([r['slot'] for r in auxiliary] != list(range(base,base+len(auxiliary)))
            or len({r['slot'] for r in all_rows}) != len(all_rows)
            or any(not 0 <= r['slot'] < base for r in outputs)):
        raise ValueError('complete unique input/output/auxiliary namespace required')
    values = {}; definitions = {}
    for row in all_rows:
        coefficients,rhs = read_row(row,pool)
        if any(c >= base+len(auxiliary) for c in coefficients):
            raise ValueError('column outside complete continuous packet namespace')
        values[row['slot']] = (coefficients,rhs)
        if row['slot'] >= base:
            pivot,parents = definition(row,coefficients,rhs)
            definitions[row['slot']] = (pivot,parents,rhs)
    mapping = {a['old']:(a['representative'],bounded(a['scale'])) for a in aliases}
    if len(mapping) != len(aliases):
        raise ValueError('duplicate supplied alias')
    for old,(representative,scale) in mapping.items():
        if (old not in definitions or representative not in definitions
                or representative in mapping or not 0 < abs(scale) <= 1):
            raise ValueError('independent complete root alias required')
        op,parents,offset = definitions[old]
        rp,rparents,roffset = definitions[representative]
        pool.charge('c114_supplied_alias_identity',128*(len(parents)+len(rparents)+2))
        if (any(c >= base for c in (*parents,*rparents))
                or {c:v/op for c,v in parents.items()} != {c:scale*v/rp for c,v in rparents.items()}
                or offset/op != scale*roffset/rp):
            raise ValueError('supplied alias is not an exact full native identity')
    effective = {}
    for row in all_rows:
        slot = row['slot']
        if slot in mapping:
            continue
        if not mapping or not any(c in mapping for c,_ in row['coefficients']):
            effective[slot] = row
            continue
        coefs,rhs = values[slot]
        pool.charge('c114_alias_native_rewrite',128*len(coefs)+64)
        rewritten = {}
        for col,value in coefs.items():
            new,scale = mapping.get(col,(col,F(1)))
            rewritten[new] = bounded(rewritten.get(new,F(0))+value*scale)
        rewritten = {c:v for c,v in rewritten.items() if v}
        effective[slot] = native(rewritten,rhs,row,pool)
        if slot >= base:
            newvalues = {c:F(v) for c,v in effective[slot]['coefficients']}
            newrhs = F(effective[slot]['rhs'])
            pivot,parents = definition(effective[slot],newvalues,newrhs)
            definitions[slot] = (pivot,parents,newrhs)
    incidence = defaultdict(list)
    for slot,row in effective.items():
        pool.charge('c114_complete_consumer_incidence',32*len(row['coefficients']))
        for col,_ in row['coefficients']:
            if col >= base and col != slot:
                incidence[col].append(slot)
    candidates = []; groups = defaultdict(list); excluded = Counter()
    for slot,(pivot,parents,rhs) in definitions.items():
        if slot in mapping:
            continue
        pool.charge('c114_sink_group_inventory',64)
        if not any(c >= base for c in parents):
            excluded['input_root_or_constant'] += 1
            continue
        consumers = tuple(sorted(incidence[slot]))
        if not consumers or any(c >= base for c in consumers):
            excluded['not_live_output_sink'] += 1
            continue
        groups[consumers].append(slot)
        candidates.append(dict(slot=slot,parents=len(parents),uses=len(consumers),
            no_collision_single_nnz_delta=(len(consumers)-1)*len(parents)-len(consumers)-1))
    findings = [group_result(tuple(sorted(slots)),consumers,effective,definitions,pool)
                for consumers,slots in sorted(groups.items())]
    positive = [g for g in findings if g['strict_nnz_reduction_proved']]
    report = dict(original_auxiliary_rows=len(auxiliary),original_output_rows=len(outputs),
        original_nnz=sum(len(r['coefficients']) for r in all_rows),
        effective_auxiliary_rows=len(auxiliary)-len(mapping),
        effective_nnz=sum(len(r['coefficients']) for r in effective.values()),
        independently_checked_prior_root_aliases=len(mapping),eligible_sinks=len(candidates),
        parent_width_histogram=dict(sorted(Counter(c['parents'] for c in candidates).items())),
        consumer_count_histogram=dict(sorted(Counter(c['uses'] for c in candidates).items())),
        excluded=dict(excluded),groups=len(findings),group_size_histogram=dict(sorted(Counter(g['factors'] for g in findings).items())),
        group_reasons=dict(Counter(g['reason'] for g in findings)),individually_positive_groups=len(positive),
        positive_group_factors=sum(g['factors'] for g in positive),
        positive_groups_may_share_consumers=True,combined_reduction_not_proved=True,
        factors=candidates,group_certificates=findings,actual_eliminations=0,
        new_source_or_LIVE_admission=False,formal_gain=0)
    # Native floats/ints only: the unchanged closed physical ledger need not
    # accept Fraction. All original rows remain held by the caller; peak tracing
    # also accounts for the complete rational workspace above.
    return report,dict(effective_native_rows=list(effective.values()),report=report)
