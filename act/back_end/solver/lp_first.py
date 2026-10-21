"""Optional same-matrix LP sign query; numerical policy, not a new abstraction.

Only integrality is relaxed. All native calls spend one original absolute
deadline; a positive incumbent or a non-optimal LP is never accepted.
"""
import math
import time


class LPFirstDeadline(TimeoutError):
    pass


def solve(*, solver, objective, integrality, bounds, constraints, offset,
          tolerance, lp_seconds, deadline, attempts, clock=time.monotonic):
    import numpy as np
    from act.back_end.solver.solver_hz import (
        HZ_NUMERICAL_POLICY, _certified_solver_lower_bound, hz_outward_slack,
    )
    if not math.isfinite(lp_seconds) or lp_seconds<=0:
        raise ValueError('positive finite LP sub-budget')

    def call(mode, flags, cap):
        start=clock();remaining=deadline-start
        if remaining<=0:raise LPFirstDeadline('original F0 deadline before '+mode)
        limit=min(cap,remaining)
        row=dict(mode=mode,attempted=True,returned=False,status=None,
                 seconds_limit=limit,accepted_positive=False)
        attempts.append(row)
        try:
            result=solver(c=objective,integrality=flags,bounds=bounds,constraints=constraints,
                options={'presolve':True,'time_limit':limit,
                         'mip_rel_gap':HZ_NUMERICAL_POLICY.mip_relative_gap})
        except TimeoutError:
            raise
        except Exception as exc:
            row['error']=type(exc).__name__
            if mode=='LP_CONTINUOUS_RELAXATION':return None
            raise
        finally:
            row['seconds']=clock()-start
        row.update(returned=True,status=int(getattr(result,'status',-1)))
        if clock()>=deadline:raise LPFirstDeadline('original F0 deadline after '+mode)
        return result

    if not np.any(integrality):
        return call('ORIGINAL_LP',integrality,math.inf),integrality
    continuous=np.zeros_like(integrality)
    result=call('LP_CONTINUOUS_RELAXATION',continuous,lp_seconds)
    accepted=_certified_solver_lower_bound(result,continuous) if result is not None else None
    if accepted is not None:
        value=accepted[0]
        lower=float(np.nextafter(offset+value-hz_outward_slack(offset,value),-np.inf))
        attempts[-1]['corrected_lower']=lower
        if lower>tolerance:
            attempts[-1]['accepted_positive']=True
            return result,continuous
    return call('ORIGINAL_MILP',integrality,math.inf),integrality
