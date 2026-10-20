# Native margin source integration control

This A1 stage tests whether the new margin-support interface preserves the
original shared and private HZ factors, all ten output classes and both router
rows through native CPU candidate generation and independent exact reception.
It is an interface gate, not an effectiveness comparison or a complete MoE proof.
The starting implementation is `ca8515cd38712ff769e66a03149cda5c75fe6f18`.

## Frozen source and queries

One synthetic given-HZ source has 3,072 input outputs and shared continuous
factors, one shared binary factor, A-private widths (1 continuous, 1 binary),
and B-private widths (2 continuous, 2 binary). The unchanged joint P therefore
has 3,075 continuous and four binary factors. The 22 outputs retain both
ten-class experts and router rows for canonical pair (0,2). Router constraints
map to equality and inequality row1 of the entry, rather than an implicit
identity map. All centers, constraints and maps are fixed by the hashed fixture
code. This is not the registered eight-expert complete topology.

Two properties use a dense signed ten-class q with constant1/8, and classes8/9
with constant−1/4. Each uses the same four (t,nu) objectives:
(0,0), (1/2,−1/4), (1,1/4), (1/2,0). Two existing CPU candidate calls generate
eight columns, each with the unchanged128 updates/129 evaluations. No positive
bound is required. Native RHS float conversion proposes candidates only;
original rational RHS values remain authoritative for exact residual checking.

Seven fixed source groups cover full dimensions, exact coordinate projection,
native candidates, source/scope/router-map/direction mutations, property and
candidate mutations, malformed CSR and live aliases, and capacity/Work/bit/
deadline/partial evidence controls. Expected rejections are PASS; any unexpected
exception or accepted mutation stops the batch. Completed prefixes survive.
There is no retry, source replacement, extra query or repair execution.

The controller also checks six deterministic observation cases: missing RSS
remains live and unknown, legacy missing-RSS rejection, identity mutation,
known-RSS legacy differential, terminal identity and expired observation.
After native reception it runs one relocated stdlib recheck, then exactly two
process faults: explicit exit7 and a sleeping child killed at its deadline.
No older successful toy/control roster is repeated.

## Shared budget and policy

The explicit opt-in `DEV_MARGIN_20261021_V1` uses the existing owned-session
supervisor. Every managed process inherits an 8GiB RLIMIT_AS before imports;
this is not aggregate physical-memory enforcement. RSS is telemetry, missing
values remain unknown, and Linux wait4 usage is recorded separately from sampled
session-plus-controller memory. Identity, exit, pidfd cleanup, exact arithmetic
and evidence acceptance remain mandatory. Legacy execution defaults are unchanged.

The controller entry starts the single300s budget. Native worker alarm195s,
supervision205s/cleanup220s, portable recheck260s/cleanup270s and final fault
cleanup by285s are nested deadlines, not new budgets. Import, capture, views,
candidates, exact checks, prefix serialization, relocation, receipt checking,
cleanup and terminal publication are inside that interval. Interpreter time
before entry and the final stdout/exit tail are not falsely reported as measured
full cold start; outer command exit is separately required for acceptance.
There is one fresh admission, at most60s, included and separately reported.

Work accumulates through producer, controller and relocated receiver; it is not
an8M gate. The control explicitly adds8,000,001 artificial accounting visits,
reported separately from source/candidate work, to distinguish the new policy
from the unchanged legacy Work gate. Stored4096-bit and intermediate8192-bit
limits remain. No claim of speed or algorithmic work reduction uses this load.

All new batch artifacts together are at most128MiB. Each publication reserves
its temporary copy, a1MiB terminal/portable reserve, and one additional copy of
all run artifacts for archival. Portable receipt export is at most1MiB. Final
archive accounting includes the retained run and new repository records.

## Reception and portability

Reception binds the original fixture, five scope identities, canonical pair,
nonidentity router row maps, both complete properties and all eight objectives.
It independently reconstructs source composition and rational objectives before
checking dual signs, all rows and exact residual compensation. Missing a property
or objective cannot become an accepted partial proof.

The relocated copy executes with `python -I -S`: only the standard-library
checker, fixture/receiver code and source/property/candidate records are present.
No checkpoint, dataset, ACT package, numerical solver or historical result
directory is read. The parent verifies copied code hashes before launch; this
trusted frozen launch is not a claim that arbitrary future checker-code edits
are self-authenticating. The same deadline and Work counters continue; relocation
is online, not a post-deadline repair.

Acceptance requires all fixed cases, both online rechecks, expected fault
terminals, cleanup, FD balance, timely publication and outer exit0. Result files
alone cannot override failed process execution. Offline audit preserves the
original online verdict. Receipts remain
`CHECKED_SUPPORT_ON_GIVEN_HZ_CONTINUOUS_RELAXATION`, with scalar aggregation and
complete request both false. Binary relaxation and upstream source assumptions
remain explicit. G1–G6 stay OPEN; real-development usage stays0.

## Execution identity

The immutable [freeze](margin_native_freeze_20261021.json) binds this protocol,
the implementation, fixture, checker, supervisor and portable file roster.
Result directory: `/data1/Kane/MOE/results/margin_native_source_20261021_r1`.
Freeze is committed and pushed before the sole attempt under standing authority.
No trained source, GPU, dependency installation, M1b retry or observer repair
campaign is authorized by this batch.
