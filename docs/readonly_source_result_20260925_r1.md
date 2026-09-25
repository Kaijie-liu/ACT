# Read-only checked source data: copying removed, modest local net benefit

**Keep experimental/opt-in, not a production default or a new certificate.**
The18 frozen saved-source checks completed with identical original-checker
conclusions. Read-only borrowing removes recursive defensive materialization,
but the net improvement over direct parsing is small: full checker medians
improve about1.6% and2.6% on the two synthetic objects. One small-object repeat
is slower than both alternatives. This is not evidence of general speedup or
of completing the sealed real MoE requests within300s.

## Identity, controls and audit

- Implementation:`3f13d07d7`; committed freeze/launch:`5ddfa5db675ec4a1a4aba56f736b35ea7ff34948`.
- Config:`configs/backend_controls/readonly_source_r1.json`, SHA256
  `dbbcc3bfb9b740609aee19d91656ab6f314a0e1ec7147657c2120dccf29dbbf6`.
- Controls:`readonly_source_controls_20260925_r1.json`:240PASS
  (33new+207prior),90.181485s;611source bindings unchanged.
- Saved-only audit:`readonly_source_audit_20260925_r1.json`; fresh python-S
  replay:`readonly_source_replay_20260925_r1.json`:both PASS,0issues,
  18/18complete,54cost mutations rejected. Audit costs0.192665/0.205900s
  are separate, not silently charged to or subtracted from online cost.
- Raw archive:`data/moe/results/readonly_source_20260925_r1`;
  236files/494,048bytes bound in audit, plus separately bound parent source
  objects. Raw data not committed; compact reports/source identities are.

The audit forbids producers, solvers, both cached parsers and the new view.
It independently replays the ORIGINAL source checker on each saved object,
then validates every result, method, terminal and complete cost record. This
is differential/structural checking, not independent full network certification.

## What changed, and what did not

The same invocation-scoped content-bound parser now stores privately owned
mapping proxies, read-only list-compatible sequences and assignment-protected
Fraction coefficients. Every hit still snapshots and binds the complete current
JSON bytes. It borrows sealed data rather than allocating fresh containers and
Fractions. Slices/concatenation produce detached lists; unchanged projection
checks may safely edit them. Original checker predicate bytecode, source/guard/
factor/property binding, route coverage and acceptance rules are retained.

Identity collision, transplanted scope, mutated input, malformed CSR, cached
missing duties, container/Fraction-slot aliases, producer aliases, arithmetic
and no-cache differential tests passed. Relocated fresh python-S checks and
actual cutoff/partial/error/late/RSS/receipt controls passed. The protocol
records two development test-oracle alias mistakes (deepcopy(Fraction), then
borrowed factor-ID lists) and their corrections, before timing was inspected.

Read-only here means normal Python mutation is blocked under the trusted
runtime. It is NOT protection from hostile runtime reflection, ctypes or
object.__setattr__. No acceptance fact, route verdict or lower bound is cached.
No saved data from one invocation becomes proof authority for another.

## Finite contemporaneous comparison

Two prior saved synthetic constructions; none/copy/readonly x3rotated repeats
each. All18calls retained, no retry or expansion. Same300s total/2sreserve,
sampled8GiB,2threads. Whole cost includes startup/import, source read/hash,
snapshot, representation creation/access, all checks, receiver, publication and
owned cleanup. No regeneration, propagation, native solve, dataset or checkpoint.

Small:12original duties,12retained constructions. Medium:252original duties,
225checked exclusions+27retained constructions. These are construction checks,
NOT positive output bounds; new complete real certificates remain zero.

### Medians of three, seconds

| Object / metric | No cache | Old copy cache | Read-only view |
|---|---:|---:|---:|
| Small full checker |0.047690|0.053103|0.046946|
| Medium full checker |0.129698|0.140404|0.126301|
| Small whole supervised segment |0.146241|0.179186|0.144979|
| Medium whole supervised segment |0.252489|0.253036|0.250013|

Read-only checker median is11.6%/10.0% lower than the old copy cache, but only
1.6%/2.6% lower than no cache. Whole-segment medians versus no cache improve
only0.9%/1.0%. These are ratios of medians, NOT medians of paired ratios or
population confidence claims. Import is inside this checker segment for every
arm; do not splice this table with the previous study's differently placed
import timer or overwrite that negative result.

Read-only vs no-cache full checker improves in5/6paired repeats; vs old copy
also5/6. Small repeat2 is worse by0.003858s vs none and0.000077s vs copy.
Read-only vs none whole segment improves in4/6; vs copy in5/6. All differences
are in the audit, including the losing observations.

Small whole ranges:none0.143272–0.147377,copy0.143742–0.179300,
readonly0.144791–0.178971s. The wide overlap/variation forbids interpreting the
small object's copy-cache median gap as a stable overall acceleration.
Medium whole ranges:none0.252401–0.254216,copy0.252202–0.256421,
readonly0.248872–0.251780s. Three repeats and subsecond synthetic objects do
not establish real-model throughput or tail behavior.

### Representation mechanism

| Component median, seconds | Small copy | Small readonly | Medium copy | Medium readonly |
|---|---:|---:|---:|---:|
| Current source snapshot |0.005066|0.005013|0.009859|0.010325|
| Exact parsing |0.012563|0.012540|0.033716|0.036581|
| Freeze / seal once |0.003313|0.005562|0.009854|0.011296|
| Fresh copy / borrow |0.007582|0.000014|0.020087|0.000008|
| Total parser interface |0.028978|0.023655|0.073825|0.058867|

The old `copy` timer measures constant-time borrow in the new arm. Sealing is
more expensive here, but it replaces repeated materialization. Total parser
interface is smaller; the full checker still incurs all the remaining network,
router, guard, projection, matrix/property comparisons and runtime overhead.
Separate component medians need not sum; each call's actual cost closes.

Copy and readonly have identical counts:small60lookups/24parses/36hits,
medium30/15/15. No cache parses60/30. Peak entries24/15, serialized payload
265,393/884,176bytes, retained cells14,640/34,168. These are NOT total Python
memory or proof-size reductions. No bound/value changed to obtain the timing.

## Decision and next boundary

The specific representation hypothesis has a useful but modest engineering
signal: repeated defensive copying can be removed without dropping the tested
checks, and local net checker cost is slightly lower than direct parsing.
Freeze this result; do not expand repeats/sizes, tune cache limits, remove full
content binding or enable it globally to manufacture a larger effect.

Keep this as an opt-in interface candidate. A justified next step is a separate
complete source-generation/checking integration control under the same total
clock, retaining all propagation, construction, serialization and checking cost.
It is not automatically authorization for another real-request run. Any later
experiment needs its own identity and controls, not a replay of sealed
98/4088/4096/4098/4099 or a changed historical main table.

This does not close the source/output proof gap, establish superiority over an
external verifier, or meet an ISSTA acceptance criterion by itself. The primary
contribution remains relationship-preserving complete MoE verification;
microbenchmark success must not replace decisive output evidence.
