# Execution freeze for the new twenty-input evidence study

User authorized cohort integration, freezing and launch on2026-09-16. This
execution supersedes only the previous "freeze but do not launch yet" gate;
it does not change the frozen selection, model, mathematical code or numerical
rules. Selection SHA256:
`db3043fb124703e8123e5326eda853dc0d67d45e9104ca316487a631603daea7`.

`evidence_cohort/` is separate from all method-source freeze namespaces.
Twenty inputs, three cyclically interleaved arms, original matched V2,
generic evidence and original CROWN:60 requests. Every request has one300s
clock including outer publication, interpreter startup, source/model loading,
all phases and terminal serialization. The driver passes the *outer* start
timestamp to the unchanged phase APIs. The work watchdog expires at298s to
reserve2s for terminal publication. The V2 internal5s reserve, LP cap60s,
80s checking reserve and every method configuration remain unchanged.

The outer watchdog covers the complete driver, including serialization and
packaging, not only native calls. Owned descendants are identified through
Linux PID/start-time/ancestry, stopped before cleanup and killed even if they
have created their own sessions. No unrelated process group is killed. A
deadline result remains TIMEOUT even if a late candidate package exists.
Partial stages are explicitly censored; unavailable durations are not zeros.

The global route-complexity lock is held throughout execution and final audit.
One CPU request runs at a time with one numerical thread, no visible GPU,
and nice+10. Resource checks require16GiB RAM,5GiB disk and load/logical-core
<=.5; resource waiting is recorded separately, polled every30s and capped at
24h. No other user's job is interrupted. Startup requires clean feature HEAD,
live remote equality, method/selection hashes and a fresh clean reconstruction.
No resume, repeat launch, sample replacement or effect-dependent extension.

Rows are durable and ordered. ERROR stops the queue; UNKNOWN and TIMEOUT
continue. A supervisor exception records an aborted job separately from the
completed-terminal list and genuinely unattempted suffix. Raw failed and
partial directories remain. The runtime carries PID, active job and pending
roster, including during resource waiting. Abrupt machine loss cannot be
relabelled completion; no automatic rerun is authorized by this protocol.

After execution, a fresh process reconstructs the terminal roster and audits
each request in a separate process capped at600s. It checks frozen requests,
artifact inventories, source/phase identity, on-time status, observed common
facts, complete package obligations and full-model UNSAFE replay. Completed
evidence requests also undergo fresh rational aggregation and portable
`python -I -S` checks. These archival costs are separate, cannot rescue an
original timeout, and an audit failure prevents a completed-cohort claim.
No claim is inferred from partial journals or missing snapshots.

The final table preserves the three distinct positive grades:
HZ_POLICY_ACCEPTED, CHECKED_RATIONAL_CONDITIONAL and CROWN_NUMERICAL_FILTER.
It reports states, missing vs nonpositive evidence, timeouts, replayed UNSAFE,
positive intersections/gains/losses, all-terminal costs, phase/censored costs,
proof sizes and single/multiple/unavailable route strata. Common source facts
must match wherever both observations survive; missing observations are not
equality. A positive/full-model witness conflict is an audit failure.

Statistics retain all three arms in each of20 input blocks, with the already
registered10,000 bootstrap draws/seed20260916. Only a fully audited60-request
run gets paired cohort statistics. Unequal proof contracts are explicit; a
zero-width zero interval is not equivalence and no aggregate "formal SAFE"
column mixes the arms. Trust in upstream network→HZ, guards and exclusions
is unchanged. There is no backend tuning, production-gate relaxation or free
cross-arm reuse of witnesses/facts.

Commands (ACT environment):

```
python -m evidence_cohort.controls
python -m evidence_cohort.prepare
# after commit/push of source, controls, reconstruction and execution freeze:
python -m evidence_cohort.run --launch
```

The detached supervisor writes to
`data/moe/results/general_evidence_launch_20260916_v1/supervisor.log` and
`data/moe/results/general_evidence_cohort_20260916_v1/`. It performs final audit
automatically but does not auto-commit raw data/proofs or announce a scientific
success. Results are archived separately after completion.
