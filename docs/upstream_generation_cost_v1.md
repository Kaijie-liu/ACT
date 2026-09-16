# Evidence-generation cost analysis: saved records only

Starting HEAD e968c35c0af424ff4203189c488301c0ddfb7124; clean feature branch.
This analysis uses the sealed eight-request experiment and frozen code. No new
input, checkpoint load, LP solve, proof replay, timeout extension, or execution
change. [Derived JSON](upstream_generation_cost_v1.json) includes every query
gap, clock identities, static path counts, file sizes and source hashes.
[Parent results](upstream_portable_v1_execution_results.md) remain unchanged.

## 1. What the clocks actually measure

For each completed proposal subprocess we reconstruct the exact partition:

    phase = prefix + sum(query windows) + sum(inter-query gaps) + suffix

The prefix begins at the recorded subprocess phase entry; a query window starts
immediately before appending/saving its PENDING journal entry. The stored window
ends after `propose()` returns and the certificate file has been saved, BEFORE
the completed journal is written and the certificate reference is hashed.
Source: `moe_evidence/generate.py::propose_all.solve`.

| Portion | Operations included | Can current logs separate their times? |
|---|---|---|
| Query window | initial journal write; sparse validation and float conversion; SciPy/HiGHS call; exact residual evaluation and self-check; certificate serialization | No |
| Between queries | completed journal write; certificate hashing; post-solve source/construction and dual check; manifest write; next record read/decode and precheck; for weighted queries also rational construction/export writing | No |
| Prefix/suffix | startup/imports, transport validation, loop preparation or final checks, reserve handoff, manifest/reference I/O, process exit | No |

`query_log.seconds` is therefore NOT native solver time. In particular,
`sparse_lp_certificate.propose()` performs exact `evaluate()`, then `check()`
(which performs another `evaluate()`) before it returns. The caller subsequently
checks the bound again, outside that window. The final isolated checker is a
later phase and is not included in the proposal-phase numbers below.

## 2. Closed accounting, all eight requests

| Input | Arm | Proposal phase(s) | Query windows(s) | Outside windows(s) |
|---|---|---:|---:|---:|
|207|double|150.611|60.177|90.434|
|207|single|150.754|60.141|90.614|
|209|single|175.155|70.381|104.773|
|209|double|174.049|70.039|104.011|
|211|double|134.774|50.616|84.158|
|211|single|134.176|50.555|83.621|
|214|single|99.009|41.006|58.003|
|214|double|99.806|41.207|58.599|
|**Total**||**1118.334**|**444.122**|**674.212**|

**60.29% lies outside query windows.** The other39.71% still contains exact
arithmetic, preparation and I/O, so it is only an upper envelope for native
solver time, not its measured share. This arithmetic describes the work that
actually ran; it is not a prediction of extra proof coverage after optimization.

Most outside-window time is BETWEEN queries:~52.9–100.6s/request, versus
~0.86s prefix and~2.25–4.22s suffix. Thus process startup alone is not an
adequate explanation. Each gap mixes previous-query postchecking with next-query
preparation, so it must not be attributed entirely to the next LP or to checking.

All212 entered queries recorded PROPOSED, none UNAVAILABLE or PENDING. There
is no observed failed native solve in these logs. Work stopped when the80s
reserve was reached in six requests, not because the recorded queries reported
native time-limit failure.211 completed the loop in both arms. This does not
prove all unattempted LPs would finish quickly or produce positive bounds.

## 3. Repeated exact work: confirmed call structure, not timed attribution

The actual upstream path calls the original modules directly. The optional
exact-matrix cache is bound inside the downstream `check_manifest()` adapter;
it is NOT used by upstream `propose_all()`, `build()` or `sparse.propose()`.

For every successfully proposed sparse LP:

1. A source-to-LP or rational-McCormick reconstruction check runs before solve.
2. Inside propose, sparse inputs are parsed/converted and native linprog runs;
   exact residual evaluation runs once to form the claim and again for self-check.
3. After propose, source/construction is reconstructed again and the certificate
   is evaluated exactly again before its manifest reference is committed.

For the212 completed proposals, this implies **636 exact sparse dual
evaluations in generation**, excluding the later precheck/isolated check.
These are counts inferred from the frozen successful code path, not profiler
samples. Additional preflight work can occur before a grant is refused. Never
read636 as636 seconds or as636 unnecessary checks: their acceptance roles must
be preserved unless a new correctness contract explicitly replaces one.

The structural checkers repeatedly parse the same six source CSR matrices and
two LP constraint matrices into Fractions, reconstruct projected objectives and
constraints, and serialize canonical identities. The rational builder also
reconstructs source constraints and both property projections per weighted
property. This gives a concrete candidate for request-scoped immutable parsed
source reuse, but no new cache or check removal is implemented here.

## 4. Serialization/data volume: evidence of repetition, not a disk bottleneck

Known completed paths read at least the following logical support-export bytes
per arm:207≈809.6MB,209≈612.7MB,211≈410.2MB,214≈505.2MB(decimal units).
These exports were created during CAPTURE, not all written during proposal;
their decoding/reconstruction occurs during proposal. A last unentered
preflight may add more reads, hence these are lower bounds.

For209, each arm reads the same~16.9MB joint source six times for weighted
properties(~101.5MB) and writes~242.4MB of weighted export records. For211,
each arm reads the same~11.7MB source nine times(~105.2MB), writing~248.2MB
of weighted exports. Each weighted record embeds its source again. All
weighted exports observed correspond to entered queries; no unentered weighted
export was found in this run. Small certificate files total only~1.5–2.6MB
per request, so they do not explain the bulk logical bytes.

Logical bytes are NOT physical disk I/O: page cache may serve the reads, while
JSON decoding, Fraction conversion, canonical hashing and allocation still
consume CPU. Current clocks cannot separate disk wait, serialization, exact
checking and construction. Those fields are explicitly null in the JSON;
we do not fabricate separate percentages from file size or file modification
times. Repeated JSON representations also do not justify dropping source hashes.

## 5. A scheduling barrier separate from computational cost

Frozen `propose_all()` has two global loops: ALL router/difference support
queries first, then ANY weighted objective. It does not start a weighted
property as soon as its own ranges have been checked.

| Input (both arms) | All residual obligations | Four range certificates present at handoff | Weighted certificates generated |
|---|---:|---:|---:|
|207|18|15|0|
|209|9|9|6|
|211|9|9|9|
|214|27|7|0|

The availability column requires the recorded certificates for router lower/
upper and that property's difference lower/upper. It is a conservative metadata
condition, not a fresh proof replay, guarantee of a positive product bound or
proof of the complete request.207 had all ranges for one pair and several
properties of its other pair;214 had ranges for seven properties of its first
pair. The all-supports-first barrier explains why none had even entered the
weighted loop. It does not establish that interleaving would produce SAFE:
all remaining legal pairs/properties would STILL require coverage, and time
spent on weighted queries could displace other necessary support queries.

209 did enter weighted verification but stopped with three missing outputs;
its six checked outputs were already nonpositive.211 finished all nine but
all were nonpositive. Hence better scheduling/completion alone is not proven
to yield any new positive certificate.214 additionally has large capture and
transport costs documented in the parent study. Do not label all cases as
solver failure or use these counts to reclassify the sealed results.

## 6. Recommended bounded next work (not executed)

Priority is an opt-in, identity-bound reuse of immutable decoded source and
exact CSR parses in GENERATION, preserving per-query property/range/source,
construction and dual checks and the sole authoritative final checker. Existing
tail caching is a design reference, not evidence this upstream change is safe
or will save a measured number of seconds. Controls must include cache pollution,
cross-request identity, changed matrices/properties and uncached differential.

Before claiming a detailed cost split, a future controlled version needs
separate nested timers for read/decode, canonical identity, source/LP parsing,
construction, native linprog, exact claim evaluation, postcheck, serialization
and publication. This is an observability requirement, not permission to rerun
this sealed study or add profiling time to its300s budget. Native solver and
exact-check timing should never again share a label called solver_seconds.

Global support-order interleaving is a SEPARATE algorithmic ablation. Do not
combine it with parser/source reuse and then attribute all benefit to caching.
Do not remove checks, change mathematical domains/gates, expand samples, or
extend the reserve on the basis of this read-only analysis.211's complete
nonpositive evidence remains a separate proof-strength limitation.

## Reproduction and validation

`upstream_cost_analysis/analyze.py` validates the frozen sources and all parent
archival hashes, uses monotonic recorded clocks and file stat sizes, and writes
a new immutable derived artifact. It never imports/calls model execution,
proposal or proof-check routines. Two unit controls verify clock partitioning,
reject overlapping/negative/censored/duplicate query windows, and distinguish
range availability from a weighted certificate. A fresh analysis reconstruction
must equal the saved JSON exactly. No old files or outcomes are modified.
