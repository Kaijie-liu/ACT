# C100v3 — correct pre/post payload boundary before actual execution

C100v2 full116files/2548tests passed40.414s. During its preparation, read-only
review found a deterministic runtime adapter defect inherited from v1:
it equated the PRE-splice payload upper bound to the strictly lower POST-splice
exact inventory. Full C99 already records21883402pre versus21882407post;
the995 difference is5*199 because two coefficients and one EQ RHS are removed
per splice. The native payload cap must use the independently proved post bill,
then actual dispatch/traffic must equal it. No tariff or bound is weakened.

The supervisor PID1115212 was explicitly interrupted with SIGINT before the
original-network worker started; its subprocess.run terminated the preparation
child and its finally block preserved exit.json/log/test/source hashes. This is
an operator-aborted version, not successful preparation or a timeout. No v2
source/result was modified. Full original input-memory preparation is still
unqualified and repeated only in the new corrected version.

V3 changes only the runtime writer adapter to use the post-splice bill, tested
on ordinary positive/negative and EQ/INEQ cases. It retains v2's complete fresh
original-input proof, full original model and C99 state and stage-lifetime
measurement. All117files/2550tests must pass<=60s. Same240s preparation/worker,
45s ordinary MILP with base feasibility,256M/200M,64M,both1GiB,CPU1/GPU0,
shared reserves, source authentication/full LIVE/native/inverse/witness gates.
All frozen archives/defaults/production and1870/all13/E0 scores remain unchanged.
