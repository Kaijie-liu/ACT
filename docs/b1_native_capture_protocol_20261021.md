# B1 native capture — one synthetic integration batch

This stage closes one necessary attribution gap: saving the actual matrix,
objective and raw point used by an F0 query before production drops them.
It is **not an algorithm improvement, network-source proof, LP feasibility
certificate or complete-request experiment**. B1's guarded upstream capture,
complete obligation roster and prospective input/query manifest remain separate
unfinished interfaces. No real-source slot is consumed here.

## Interface and trusted boundary

`diagnostic_capture=None` is the unchanged default in `hz_minimize_output`,
`solve_weighted_top2_f0` and `solve_monolithic_weighted_top2_f0`. Enabled capture
copies the original stored binary64 CSR buffers, variable bounds, integrality,
objective offset and raw native return. It does not coalesce duplicate entries,
resolve a problem again or feed a changed point back into the decision.
The scalar F0 HZ and input map are stored separately from the lowered problem.
Optional supplied upstream HZs are not independently authenticated propagation.

The single-use observer gives callers immutable bytes, not mutable solver-array
aliases. All captures share one byte ledger and existing Work/deadline. Size is
preflighted before buffer copies; metadata has a bounded traversal. Lowering and
solver failures have explicit missing-problem records. Capture errors propagate
outside the numerical solver's ordinary UNKNOWN handler. The staged catch list
also preserves this typed error; no complete-request factory is enabled yet.

The stdlib receiver checks external scope/hash, schema, source and matrix
dimensions, CSR indices, vector types/bounds, property identity, target row and
constant, binary map and monolithic branch/selector objective map. It verifies
the online deadline is unchanged. Stored CSR duplicate summation for objective
consistency is binary64, not a rational proof of the lowering. It **does not**
check candidate feasibility, optimality, all disjunction rows' construction,
guard/route coverage, F0 construction or network-to-HZ conversion. Raw NaN/Inf
candidate bits can be retained under `RAW_UNCHECKED`; they are never a witness.
Rehashed structural mutations must fail, but a coherent forged source cannot
be rejected without an independent upstream source contract. Scope hashes are
caller-supplied bindings, not a substitute for that contract.

## Frozen denominator

One batch, at most8 native calls, in this exact order:

| Slots | Objects | Required outcome |
| --- | --- | --- |
|1–2|same positive scalar HZ, capture off/on|identical decision and stored offset|
|3–4|same two-branch disjunction, off/on|identical UNKNOWN, complete local/selector maps|
|5–6|same infeasible scalar HZ, off/on|identical UNKNOWN, absent raw point retained|
|7–8|pair and monolithic, local deadline fault injected at capture entry after solve|typed rejection, no complete capture|

The supplied fixture has one continuous and one binary factor, an equality,
nonzero center and a ten-coordinate property identity. It is an F0-shaped
scalar fixture, **not** an expert/router network or a proof of that property's
projection. No checkpoint, dataset, GPU, training, tilt strategy or dependency
change is used. Each native call has at most5s solver time within the one batch
deadline, not its own300s allowance. Events distinguish attempted slots,
returned decisions, post-native capture rejection and zero-variable shortcuts.

The manifest binds29 additional deterministic structural/fault controls:
missing candidate/matrix/branch/problem; wrong dtype/source dimension/offset/
objective/branch/scope/property/deadline/selector/call flag; expired reception;
first and cumulative record capacities; oversized ASCII and Unicode metadata;
malformed encoding and repeated observer; pair/monolithic lowering failures;
supplied solver-failure/backend-missing/point-missing returns; nonfinite raw
candidate, post-capture alias pollution and zero-variable off/on shortcut.
Supplied return fixtures test the capture interface; they are **not** additional
solver calls or observed solver failures. No global monkeypatch is used.

## Budget, reception and decision

One fresh admission, bounded by60s and included in this batch's clock, requires
4GiB MemAvailable and128MiB free disk. Reuse the existing owned-session
supervisor and `DEV_MARGIN_20261021_V1`: total300s from the driver's earliest
Python timestamp, per-process8GiB RLIMIT_AS, RSS telemetry only, cumulative
Work through producer/online receiver/relocated receiver, and128MiB including
prefixes, relocation and one archive. AS is not group physical memory, and
`wait4.ru_maxrss` is not simultaneous group RSS. Pre-interpreter startup and
the driver's final write/exit tail are not mislabeled full cold-start cost.

Worker cutoff200s, supervised run210s/cleanup220s; relocated stdlib receiver
run260s/cleanup280s; terminal must finish before300s. These are reservations
within the same deadline, not replenished budgets. One native worker completes
the fixed calls; `python -I -S -B` rechecks copied records with no ACT/numerical
imports and the same deadline/continued Work. This is structural reception,
not independent arithmetic or network proof.

Record full batch elapsed, each call's inclusive time and contained capture
time, worker runtime, online reception, relocation and owned-session cleanup.
Contained costs must not be summed twice; off/on synthetic timings do not
establish performance gains. Final audit requires all slots/controls/files,
matching on/off results,10 structural records, continued Work, clean owned
termination, FD balance, all identity/capacity gates and unchanged deadline.
Expected fault rejection is PASS. Any unexpected execution or acceptance
failure preserves the prefix and stops this attempt; no repaired retry or
replacement control is authorized by this freeze. No B1 real request launches
merely because these controls pass.
