# Gate weighted facts finite implementation and controls

This approved stage tests a default-off computation-saving rule, not a new abstract
domain or measured model speedup. The user resolved the later goal-text conflict in
favor of this finite M1 implementation and control stage on 2026-10-20. G1-G6 remain
open. Real-request, native optimizer and GPU execution are outside this approval.

## Conditional rule and difference from existing reuse

For a canonical pair `(a,b)`, let `p(z)=q·z+c`. Given valid membership-domain
lower bounds `La <= p(Ea(x))`, `Lb <= p(Eb(x))` and the first expert's weight
`lambda in [l,h] subset [0,1]`, pair membership implies both expert membership
conditions. Normalized weights give

```
p(F(x)) = lambda*p(Ea(x)) + (1-lambda)*p(Eb(x))
        >= min(l*La+(1-l)*Lb, h*La+(1-h)*Lb).
```

The constant `c` is included exactly once by this identity. A reverse-direction gate
is transformed with exact rational endpoints `[1-h,1-l]`. A signed bound may be
negative and is NEVER an expert SAFE fact. The old positive-fact fast path is unchanged;
M1 can additionally discharge a property with a negative expert fact and a sufficiently
small weight on that expert. It does not restore gate-input coupling.

The producer uses an exact rational interval vertex sum, with the unchanged
`1e-9 + 1e-9*scale` correction and downward `nextafter`. The checker independently
recomputes the given expert interval using center/radius algebra and uses
`Lb + lambda*(La-Lb)` for the two endpoint sums. Accepted values must remain below
that rational bound and strictly above the original `1e-7` margin. Stored and
intermediate rational limits remain 4096/8192 bits.

Scope binds request, model-state and input-domain identities, frame, expert, full
property `(q,c)`, membership rule and numerical policy. Online gate capture retains
the existing conditioned-router object identity, frame, width and pair checks.
Saved-row audit reconstructs expert sources from the Tier 1 record and gate sources
from the request record; missing, mismatched or incomplete output obligations do
not establish SAFE. Checks apply to recorded positive rows even in UNKNOWN/UNSAFE.

This is **HZ numerical-policy acceptance conditional on given interval and gate
sources**, not an independently proved network or sigmoid range. Replacing all
trusted gate records consistently is outside the downstream arithmetic check.
Input containment, lowering, support and float sigmoid provenance remain upstream
assumptions. Exact combination does not repair historical source gaps.

## Integration and unchanged behavior

Enable only with `f0.gate_weighted_facts=true`, existing scoped reuse, and the
unchanged scheduled 300-second/.25 staged or matched-monolithic configuration.
Default configs are not changed. Tier 1 terminal results do not construct M1 sources.
The rule runs after the already scheduled pair propagation and gate range, before
F0 construction. It adds no support query and does not move propagation earlier.
Missing facts, nonpositive combinations or unsupported gate status use the original
F0 path. Malformed bindings fail closed, never yielding a positive result.

This version conservatively requires complete optimal gate support and recorded
zero gap. A missing gap (including some otherwise valid continuous native queries)
falls back rather than throwing a new error or issuing another query. Therefore the
historical 402/752 estimate cannot be assumed to be this implementation's hit rate.

Staged discharges individual pair/property rows. Monolithic removes only proved
pair/property obligations from its disjunction and takes the minimum over reused and
residual branches. Old all-positive facts can still skip all pair work. Full tie-legal
coverage and all output properties are still required for a complete result.

Counts distinguish generated/check-passing facts from actual F0 calls. A monolithic
fact generated before an early termination may never enter the final partition.
`f0_solve_attempted` increments only after obtaining the original solve budget,
immediately before invoking the solver; returned/status-0 counts are separate.
Partial records remain censored. Skipping a query can lose its incidental witness
for another property: zero SAFE/UNSAFE/solved loss needs a future measured comparison.

## Frozen finite controls

One exclusive directory: `/data1/Kane/MOE/results/gate_weighted_facts_20261020_r1`.
One admission and one batch; source hashes and exact case order are in the companion
freeze JSON. No resume, replacement cases, retries, higher limits or extra samples.
The 19 fixed groups are:

1. Default off and configuration rejection.
2. Signed negative expert fact with a positive weighted combination.
3. Canonical and reverse gate direction.
4. Nonzero property constant and changed output dimension.
5. Equal-margin tie at one-half; endpoint weights zero/one as **given-weight
   algebraic degeneracies only**, not gates realizable by finite softmax.
6. Unit gate range and two negative facts return to F0.
7. Request/model/input/property/frame/policy/gate/tie scope mutations.
8. Expert/property/coefficient/constant/source/bound/grade mutations; NaN and both infinities.
9. Gate source/direction/range/frame mutations and forged accepted bound; NaN and both infinities.
10. Same-frame foreign conditioned-router object and wrong width.
11. Incomplete, missing-gap and fast-fallback support preserves original fallback.
12. Captured source alias mutation and source recheck rejection.
13. Missing and duplicate expert sources.
14. Injected request-clock expiry before work and before acceptance.
15. Old positive fact equality and both function paths skipping gate/F0.
16. Staged function differential with two pairs and two properties.
17. Monolithic function differential on the same fixed obligations.
18. Saved nested/flat source audit, altered source and missing-obligation rejection.
19. Stub native exceptions and deadline partial-result rejection.

Controls use standard-library fixed sources and unchanged function ASTs with explicit
local dependency doubles. They do not monkeypatch global modules or run Torch,
NumPy, SciPy, a checkpoint or any optimizer. Pure old interval/LP checking is retained.
The staged fixture expects four versus two stub solves; monolithic expects two versus
one. Both remain UNKNOWN because the other output property is unproved. These are
deterministic call-wiring checks, NOT real solve savings or end-to-end model comparison.
The old positive fixture expects no propagation/gate/F0 in either function path.
Full package imports and real solver integration remain untested by this authorization.

## Budget and cost boundary

The existing `owned_session.execute` runs one isolated worker. There is no new
supervision mechanism or observer repair chain. Whole batch cap is 300 seconds from
the controller's first timestamp, including imports, source-hash checks, bounded
admission, worker creation/loading, fixed construction/checking, serialization,
reception, saved-record recheck, cleanup and terminal export. Worker deadline is
entry+240, supervision entry+250, cleanup entry+270; all reserves are WITHIN 300.
Admission is one check within 60 seconds, requiring 4 GiB available and 128 MiB free.
OS/clock/interpreter and existing pidfd observer remain explicit trusted components.

The interpreter's time before the script's first timestamp, final stdout flush
duration and process exit are not included in the observed terminal-file export
duration; `full_cold_start_seconds=null`. The hard deadline remains armed through
stdout flush and final exit; a late/nonzero process exit does not pass. This small control
is not used for a performance claim or to change real-request timing contracts.
Injected deadline cases check acceptance ordering, not a 300-second soak or a new
validation of the unchanged supervisor.

Per managed process AS is hard limited to 8 GiB. The existing session observer stops
at 2 GiB sampled worker-plus-controller RSS; this is not an instantaneous aggregate
hard memory bound. Parent RSS is also sampled before/after result loading, after
saved-record audit and before terminal export. Total retained artifacts are capped at 128 MiB, with bounded
member reads and exclusive atomic publication. Source files, stdlib startup and
filesystem/OS structures are not counted as newly generated artifacts.
The final cap check reserves 32 KiB for externally retained stdout/stderr and a
second terminal-file length for atomic publication. The fixed worker member is
separately bounded to 1 MiB; these reservations do not raise the 128 MiB cap.

One M1 Work ledger covers all control cases; saved-record reception continues its
counts, enforcing a combined 8M cap rather than granting another allowance. The Work
metric covers M1 capture/arithmetic/check visits, not every interpreter instruction,
existing legacy check operation or real HZ propagation. All such execution is still
charged to wall time. This is not an eight-expert capacity test.

## Acceptance and stop rule

Success requires all fixed groups, saved positive-fact reconstruction, original
fallback/coverage checks, timely complete reception, clean owned-session termination,
no descriptor leak and unchanged caps. Any unexpected failure saves the available
prefix and stops this one authorization; no repair-and-rerun under the same freeze.
Offline audit may describe a failed prefix but cannot promote a late/missing result.

After acceptance, archive the compact records and cost boundary, commit and push.
Do not infer measured request speedup, restored gate coupling, new strict certificates
or completion of G1. The next decision would be a separately authorized/frozen small
real-request development protocol with whole-request time, native call counts and
SAFE/UNSAFE/solved gain-loss; no real run is included here.
