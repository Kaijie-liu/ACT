"""Default-off common strong CPU proposal source for the A2 comparison.

No new optimizer/checker, accepted bound, fallback, or production dispatch.
HiGHS solves one objective per invocation. All exact claim construction remains
in the existing metered producers and original-source receivers.
"""
from scoped_source.hz_request_store import require
from scoped_source.check_gate_margin_rows import MARGIN_PROFILE, freeze
from scoped_source.check_native_margin_support import stored
from scoped_source.rowwise_native import propose_duals

PROVIDER = "HIGHS_RAW_MARGINALS_V1"


def columns(base, exact, *, work, query_seconds, observe):
    require(type(exact) is list and 1 <= len(exact) <= 8, "finite exact LP objectives")
    require(callable(observe), "caller-owned native intent sink")
    ys, zs, outcomes = [], [], []
    for index, (center, row) in enumerate(exact):
        work.tick()
        work.visit("coefficient_visits", len(row)+1)
        lp = dict(base, c=list(map(stored, row)), offset=stored(center))
        raw, identity = freeze(lp, work, profile=MARGIN_PROFILE)
        def event(item):
            work.tick()
            observe(dict(item, objective_index=index, provider=PROVIDER))
            work.tick()
        # Raw candidate extraction never calls legacy unmetered _evaluate or
        # rowwise_bound. Scope-specific exact checks below the caller still run.
        proposal = propose_duals(lp, work=work, query_seconds=query_seconds, observe=event)
        require(freeze(lp, work, profile=MARGIN_PROFILE)[0] == raw,
                "exact LP changed during strong proposal")
        candidate = proposal["candidate"]
        require(candidate["lp_sha256"] == identity, "strong proposal original LP binding")
        ys.append(candidate["inequality_dual"]); zs.append(candidate["equality_dual"])
        outcomes.append(proposal["metadata"])
    work.tick()
    return ys, zs, dict(provider=PROVIDER, native_candidate_calls=len(exact),
                        objective_slots=len(exact), status0_returns=len(exact),
                        outcomes=outcomes, native_kernel_Work="NOT_COUNTED_AS_PYTHON_VISITS",
                        hard_budget_supervision=False)
