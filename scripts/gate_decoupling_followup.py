"""Finite sixth-revision selection/reconstruction, without a numerical engine.

Reconstruction selects the first CROWN-unexcluded history in canonical order.
It does NOT recover a lost old unproved query or prove route reachability.
"""
import itertools
import math
import time

MODE = 'REBUILT_FIRST_UNEXCLUDED_NOT_OLD_UNPROVED_ID'


def tick(deadline):
    if not math.isfinite(deadline) or time.monotonic() >= deadline:
        raise TimeoutError('shared reconstruction deadline')


def supplement_objects(objects, old_rows):
    """Keep original order; only source-missing pairs or wholly unexecuted pairs."""
    by = {(r['object_id'], r['variant']): r for r in old_rows}
    if len(by) != len(old_rows):
        raise ValueError('duplicate historical call')
    expected = {(o['id'], v) for o in objects for v in ('V2', 'V3')}
    if not set(by) <= expected:
        raise ValueError('foreign historical object')
    selected = []
    for obj in objects:
        rows = [by.get((obj['id'], v)) for v in ('V2', 'V3')]
        if all(r is None for r in rows) or all(r and r['status'] == 'SOURCE_MISSING' for r in rows):
            selected.append(obj)
        elif any(r is None or r['status'] == 'SOURCE_MISSING' for r in rows):
            raise ValueError('partial old pair requires explicit accounting')
    return selected


def frozen_t1(positives):
    values = sorted(float(r['seconds']) for r in positives)
    if len(values) != 38 or any(not math.isfinite(v) or not 0 <= v <= 300 for v in values):
        raise ValueError('exact original38 positive full-call costs')
    rank = math.ceil(.9*len(values))
    return dict(n=len(values), rule='nearest_rank_ceil_0.90n_cap120', rank=rank,
                p90_seconds=values[rank-1], cap_seconds=120, T1_seconds=min(120., values[rank-1]))


def _bounds(row, prefix, experts):
    if row['prefix'] != [list(p) for p in prefix] or row['guarantee'] != 'WHOLE_BOX_CROWN_NUMERICAL_UPPER':
        raise ValueError('prefix CROWN binding')
    values = {(i, j): v for i, j, v in row['upper']}
    expected = {(i, j) for i in range(experts) for j in range(experts) if i != j}
    if len(values) != len(row['upper']) or set(values) != expected:
        raise ValueError('complete unique router difference rows')
    if any(v is not None and (not isinstance(v, (float, int)) or not math.isfinite(v)) for v in values.values()):
        raise ValueError('nonfinite router upper bound must be explicit missing')
    return values


def first_unexcluded(experts, bound, deadline):
    """Canonical DFS, strict exclusion preserves ties; all work shares deadline."""
    if not experts or any(type(e) is not int or e < 2 for e in experts):
        raise ValueError('top2 expert dimensions')
    records = []
    def walk(prefix):
        tick(deadline)
        depth = len(prefix)
        if depth == len(experts):
            return prefix
        row = bound(prefix, deadline)
        tick(deadline)
        values = _bounds(row, prefix, experts[depth]); records.append(row)
        for pair in itertools.combinations(range(experts[depth]), 2):
            tick(deadline)
            excluded = any(values[i, j] is not None and values[i, j] < -1e-9
                           for i in pair for j in range(experts[depth]) if j not in pair)
            if not excluded:
                answer = walk((*prefix, pair))
                if answer is not None:
                    return answer
        return None
    history = walk(())
    return dict(mode=MODE, history=None if history is None else [list(p) for p in history],
                prefix_records=records, old_unproved_identity_recovered=False,
                route_reachability_proved=False)


def check_selection(selection, experts, source_id, deadline):
    if selection['source_id'] != source_id:
        raise ValueError('reconstructed source identity')
    rows = iter(selection['prefix_records'])
    def recorded(prefix, end):
        tick(end)
        try:
            return next(rows)
        except StopIteration as error:
            raise ValueError('missing reconstruction prefix') from error
    rebuilt = first_unexcluded(experts, recorded, deadline)
    if next(rows, None) is not None or any(selection.get(k) != v for k, v in rebuilt.items()):
        raise ValueError('canonical reconstruction receipt changed')
    return rebuilt['history']
