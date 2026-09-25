# Read-only exact source representation R1

Separate optional engineering control, requested 2026-09-25. Parent:
`parsed_source_reuse_result_20260925_r1.md`. Old copy-cache negative result is
sealed. No change to production defaults, source lowering, guards, properties,
numerical gates, query inventory, order optimization or source serialization.

## Single intervention and trust boundary

The prior invocation-local parser retains full JSON snapshot, hash AND byte
identity, scope binding, bounded LRU, deadline and cleanup bytecode. Only its
internal stored representation and return operation change. Validated sparse
rows/dictionaries become privately owned mapping proxies; list-like data become
read-only sequences; rationals become assignment-protected Fraction subclasses.
Hits borrow that representation, rather than rebuilding containers and every
Fraction. Slices/concatenation are detached lists so the original projection
checker may safely edit its local slice. The four checker predicates and parent
aggregation use unchanged bytecode with private globals. Router and network
checking are untouched. No verdict, route exclusion or bound is cached.

This is read-only data under the trusted Python runtime, NOT isolation from
hostile `object.__setattr__`, `ctypes`, reflection or malicious threads. It is
not authority to skip checking a subsequent source. Every hit checks full
current content; every invocation checks all obligations again. Original
network/source assumptions and native numerical limitations remain unchanged.

## Required controls, before timing

Identity/hash collision/wrong scope/transplanted entry; mutable input rebinding;
container, factor-ID and Fraction-slot poisoning; arithmetic and comparison
differentials; detached slices; mutable producer aliases; malformed sparse
matrices; missing and changed properties after warm lookup; ties, multiple
pairs, dimensions and zero-width controls; reentry, close, eviction, oversized
fallback. Re-run the original checker on the same source. Relocate the saved
control and check with fresh `python -S`, forbidding producers/model/solvers.
Add shared-clock cutoff during sealing/receipt, partial data, exceptions,
missing and late candidates, RSS termination, method mutation and whole-cost
checks. Run 33 new + 207 prior controls; freeze exact sources after they pass.

Development test-oracle corrections are retained: the first alias test used
`deepcopy` of parsed Fractions, which shares their objects; a second oracle
reparse still shared input factor-ID lists. Both failed as test-oracle aliases,
not as view corruption. Correct oracle reparses an independent JSON snapshot.
No performance result was examined to make these corrections.

## Finite comparison after controls and committed freeze

Only the two already archived synthetic source/construction objects from
`source_cost_supervised_r1.json`; no regeneration, new model/input or solve.
Three contemporaneous arms: `none` (original parse), `copy` (old exact cache),
`readonly` (new view). Three repeats per object; rotated order
none/copy/readonly, copy/readonly/none, readonly/none/copy: exactly 18 calls.
Each uses a fresh process/cache, one 300s total clock, 2s terminal reserve,
8GiB sampled owned group+parent RSS and two threads. No retries, replacement,
extra sizes or repeats. Partial/error/timeouts remain in the denominator.

Bind source/construction bytes, expected original checker result, fresh
invocation, all transitive frozen source files, Python and execution method.
All import/load/hash/snapshot/seal/borrow/check/receipt/publication/cleanup cost
is charged. Timer names inherited from R1: `freeze` means seal; `copy` means
constant-time borrow in the readonly arm. Record component times and retained
entries/payload/cells (not total Python memory). Compare entire checker and
outer segment, not hits or parser time alone. Offline audit is separate cost.

Saved-only independent audit rechecks original predicates without either new
parser/producer/solver, validates every terminal/cost/identity, and replays in
a fresh `python -S` process. This demonstrates agreement on checked source
CONSTRUCTIONS; there are zero new output lower-bound certificates.

## Decision fixed before execution

Report all three arms, per-object medians/ranges and all paired differences.
No speed claim from a component alone or from censored/early UNKNOWN returns.
This finite local study supports at most a source-check implementation effect,
NOT end-to-end real MoE speedup, a strict certificate or an ISSTA acceptance
claim. If whole checking has no benefit, stop this variant; no capacity tuning
or removed checks. Even a positive effect remains opt-in and requires a
separately scoped complete source-generation integration before real requests.
Sealed 98/4088/4096/4098/4099, historical 23 source gaps and venue boundaries
are unchanged. Do not let microbenchmark optimization replace MoE research.
