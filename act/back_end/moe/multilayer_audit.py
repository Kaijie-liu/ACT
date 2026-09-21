"""Independent structural coverage audit (stdlib only; NOT solver reproof)."""
import hashlib
import itertools
import json
import math


def audit_multilayer_result(result, *, expected_request_id, replay=None):
    """Reconstruct every legal history and property obligation from the catalog.

    The caller binds the expected request identity. Lowering and native solver
    bounds remain trusted; this checker rejects missing/forged *structure*, not
    arbitrary lies in a solver's numeric bound. UNSAFE requires original replay.
    """
    issues = []
    def require(ok, message):
        if not ok:
            issues.append(message)
    try:
        require(result['schema'] == 'multilayer-histories-v1', 'schema')
        identity = result['identity']
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        require(digest == expected_request_id == result['request_id'], 'request binding')
        require(result['source_complete'] is False, 'source-complete overclaim')
        require(result['semantics'] == 'real_arithmetic_stored_coefficients_ANY_LEGAL_TOPK', 'semantics')
        require(result['numerical_policy'] == identity['policy'], 'policy binding')
        require(result['properties'] == identity['properties'] and type(result['properties']) is int
                and result['properties'] > 0, 'property count binding')
        sites = result['sites']
        require(sites == identity['sites'] and bool(sites), 'site catalog binding')
        require(len({s['name'] for s in sites}) == len(sites), 'call-site uniqueness')
        count = 1
        for s in sites:
            require(type(s['experts']) is int and type(s['k']) is int and 1 <= s['k'] <= s['experts'], 'site dimensions')
            require(s['mode'] in {'hard_top1','selected_softmax','raw_epsilon','raw','nonzero_ste'}, 'site semantics')
            count *= math.comb(s['experts'], s['k'])
        require(result['expected_histories'] == count, 'history count')
        observed = set()
        for record in result['records']:
            h = tuple(tuple(v) for v in record['history'])
            require(len(h) == len(sites), 'partial history')
            require(h not in observed, 'duplicate history')
            observed.add(h)
            for selected, s in zip(h, sites):
                require(len(selected)==s['k'] and tuple(sorted(set(selected)))==selected
                        and all(type(i) is int and 0 <= i < s['experts'] for i in selected), 'illegal route set')
            if record['status'] in {'ACCEPTED','EXCLUDED'}:
                require(record.get('phase')=='complete', 'unfinished accepted history')
                needed = []
                for j,(s,selected) in enumerate(zip(sites,h)):
                    if s['mode']=='raw_epsilon':
                        needed.append((s['name'],selected,j+1))
                    elif s['mode']=='nonzero_ste':
                        needed.extend((s['name'],(i,),j+1) for i in selected)
                actual=[]
                for d in record['definedness']:
                    actual.append((d['site'],tuple(d['selected']),d['prefix_sites']))
                    require(d['accepted'] is True and math.isfinite(d['lower']) and math.isfinite(d['upper'])
                            and d['lower'] <= d['upper'] and (d['lower']>0 or d['upper']<0), 'undefined gate')
                require(actual==needed, 'missing or wrong-prefix definedness')
                if record['status']=='EXCLUDED':
                    require(record.get('feasibility')=='infeasible', 'unsupported history exclusion')
                    require(record['properties']==[], 'excluded property mismatch')
                else:
                    props=record['properties']
                    require([p['row'] for p in props]==list(range(result['properties'])), 'incomplete properties')
                    for p in props:
                        require(p['status']=='optimal' and p['solver_status']==0 and
                                p['lower'] is not None and math.isfinite(p['lower']) and
                                p['lower']>identity['policy']['safe_positive_margin'], 'unaccepted bound')
        require(len(observed) <= count, 'too many histories')
        if result['complete']:
            # Valid distinct full histories with the exact cardinality cover
            # the Cartesian product; no need to materialize an enormous grid.
            require(len(observed)==count and all(r.get('phase')=='complete' for r in result['records']), 'incomplete coverage')
        status=result['status']
        if status=='POSITIVE':
            require(result['complete'] is True and len(observed)==count, 'positive missing history')
            require(any(r['status']=='ACCEPTED' for r in result['records']) and
                    all(r['status'] in {'ACCEPTED','EXCLUDED'} for r in result['records']), 'undischarged history')
            require(result['evidence_grade']=='HZ_POLICY_ACCEPTED', 'positive grade')
        elif status=='UNSAFE_REPLAYED':
            require(replay is not None and bool(replay(result['witness'])), 'full-model replay missing/invalid')
            require(result['evidence_grade']=='FULL_MODEL_REPLAY', 'unsafe grade')
        else:
            require(status in {'UNKNOWN','TIMEOUT','ERROR','UNSUPPORTED'} and result['evidence_grade']=='NONE', 'terminal grade')
        require(math.isfinite(result['seconds']) and result['seconds'] >= 0, 'cost')
        if status in {'POSITIVE','UNSAFE_REPLAYED'}:
            require(result['seconds'] < identity['budget_seconds'], 'late positive')
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        issues.append('malformed evidence: '+str(exc))
    return {'status':'PASS' if not issues else 'FAIL','issues':issues,
            'scope':'identity/coverage/acceptance structure; not independent numeric reproof'}
