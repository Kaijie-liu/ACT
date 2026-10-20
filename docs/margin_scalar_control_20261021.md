# Sparse margin scalar integration: one finite A1 control

This stage connects the native shared-source supports to the already specified
rational scalar rule. It does not change the candidate search, sigmoid enclosure,
production verifier, resource policy or numerical acceptance threshold.
Starting HEAD is `69fec94e9524528bde31edc992893acfe8559a2f`.
The completed native-source/RSS gate and13 toy groups are not rerun.

## Operator and guarantee

The property view reconstructs b, d and m from the original22-output HZ,
retaining all ten classes, constants and shared/private factor identities.
The plan uses the original factor box for the margin interval and the existing
sign/quarter-Lipschitz sigmoid enclosure. For each anchor t it queries zero
tilt and, optionally, the fixed closest-to-zero weighted-median tilt. The median
is now computed on sparse nonzero margin coordinates; other coordinates remain
in every support and residual check. This removes the toy16-factor limitation,
not a mathematical check or a proof-resource gate.

For a closed cell C=[x,y], checked support lower bounds L(t,nu) imply

    A_t(C) = max_nu { L(t,nu) + min(-nu*x, -nu*y) }.

Every feasible point whose margin lies in C has anchor objective at least
A_t(C). Nonnegative interpolation between the two anchor bounds, minimized
over the checked sigmoid enclosure of C, therefore bounds the weighted
objective. The minimum over the complete closed grid covers the original P.
The producer uses a slope formula; the checker independently uses the two
endpoint-weighted sums, after reconstructing every original source and dual
residual. Binary factors are explicitly relaxed to [-1,1].

The sigmoid enclosure is sound but loose even at a nonzero point. This stage
does not quantify anchor, finite-tilt, support-optimality or scalar-optimality
gaps, and does not claim that more cells remove the sigmoid error. It produces
`CHECKED_MARGIN_LOWER_ON_GIVEN_HZ_CONTINUOUS_RELAXATION`, not network-level SAFE,
route reachability, or a source-complete certificate.

## Fixed roster

Reuse only the deterministic source descriptors from `margin_native_cases.py`,
not its old run or old receipts. A fresh `A1_SCALAR_20261021_*` scope binds one
source:3,072 input rows,3,075 continuous and four binary factors, ten classes per
expert, canonical pair(0,2), and nonidentity retained router-row maps.
Use dense q_i=(-1)^i(i+1)/16 and c=1/8, eight uniform closed scalar cells.
Analytically, m=1/2+x0/4+3*x2/8+3*z_shared/16, giving margin[-5/16,21/16],
anchors[27/64,53/64] and median tilt−1/128 at both anchors.

Exactly two native full scalar calls run in fixed order: zero then median.
They have respectively two and four objective slots, unchanged128 updates each.
There are no additional native queries. A third scalar-only record uses the
median call's two zero-tilt certificates. Only this matched-certificate record
is used to assert finite-tilt monotonicity; the separately generated native
zero/median difference is descriptive and may be negative. Unequal slots and
fixed execution order preclude a speedup claim.

Nine fixed mutations must be rejected: plan mode, grid cut, missing cell,
duplicated cell, cell gate, cell lower, aggregate lower, missing support and
wrong property binding. A valid record is also rejected by an expired Work
continuation; its60ms deliberate wait is charged, no live deadline is reset.
Worker, controller and relocated `python -I -S` independently check all three
records against the source/property and externally fixed mode/cell count.
The relocated tree contains only standard-library checkers and these fixtures;
no ACT, numerical library, checkpoint, training data or old result is loaded.

## Execution and stopping contract

Use the already controlled owned-session supervisor and
`DEV_MARGIN_20261021_V1`: one300s batch, per-managed-process8GiB RLIMIT_AS,
RSS/ru_maxrss telemetry only, unchanged exact bit limits, continued Work with
no8M rejection, and128MiB cumulative artifacts including one archival copy.
There is one fresh admission≤60s, included and separately reported.
Worker alarm200s, supervised run210s/cleanup220s, portable run260s/cleanup280s
are nested portions of the same deadline. Capture, imports, candidates,
planning, construction, independent rechecks, faults, serialization, copying,
reception, cleanup and terminal export are charged. Arm timings are nested
scopes, not independent end-to-end measurements. Interpreter pre-entry and
final stdout/exit tail are not reported as measured full cold start; the outer
exit is required separately. No new supervisor or observation controls.

PASS requires complete fixed outputs, exact reception agreement, all expected
rejections, identity/exit/cleanup evidence, balanced FDs and timely publication.
A nonpositive bound is permitted; no positivity or performance endpoint is
invented for this integration gate. Any unexpected failure stops and preserves
the prefix. No repair/retry, additional source, extra query or model run.
New directory: `/data1/Kane/MOE/results/margin_scalar_source_20261021_r1`.
Freeze file binds code trees, source/roster, portable files and resource values;
commit/push precede the once-only invocation. G1–G6 remain OPEN and real-stage
usage remains0. A2 must separately test effectiveness against strong slicing
on matched sources/budgets, not interpret this gate as research success.
