# Exact matrix parsing reuse V1: completed controls and offline timing

Completed 2026-09-16. Implementation, controls and timing protocol were committed
and pushed at `b3de034cbe5d12d7abf8fbb4056abba88f715bf1` BEFORE the timing launch.
Read [the frozen protocol](exact_matrix_cache_v1.md),
[controls](exact_matrix_cache_v1_controls.json) and
[six-run timing receipt](exact_matrix_cache_v1_timing.json).
This is an optional checker adapter, not a change to a frozen verifier study.

## Correctness gate

41/41 controls PASS, zero failures/errors/skips, before timing. These include
11 cache controls plus 30 existing general-evidence/handoff/accounting controls.
Coverage includes mutable-input contamination, immutable parsed objects, forced
digest collisions, request isolation, eviction, malformed CSR, rehashed semantic
mutations after cache warming, complete-result no-cache differential, deadline
failure and a stdlib-only isolated `python -I -S` analytic check.

Only exact CSR parsing is reused. Source, request, property, factor/frame,
guard, range, dual and full-obligation checks still execute at every reference.
No numerical lower bound, source validation or successful verdict is cached.
All original source/method/execution freezes passed before and after controls;
all controlled checker/source hashes remained identical after timing.
The original production and portable-bundle implementations were not edited.

## Frozen observed costs

The first archived saved precheck, rank0/input114, was fixed before timing.
There were no solver calls, new network verification requests, checkpoint loads
or dataset loads in this timing run. One CPU thread, nice10, no GPU; fresh
processes in reference/uncached/cached/cached/uncached/reference order.
All six completed, none retried or substituted; supervisor wall time339.709s.

| Checker mode | Two checker times (s) | Median checker (s) | Median full process (s) | Maximum process RSS (MiB) |
|---|---:|---:|---:|---:|
| Original reference | 55.022 / 54.863 | 54.942 | 55.982 | 850.324 |
| Adapter, cache disabled | 78.225 / 77.493 | 77.859 | 78.904 | 906.773 |
| Adapter, cache enabled | 33.899 / 33.904 | 33.902 | 34.961 | 1127.680 |

Relative to the original reference, cached median checker time is38.30% lower
(ratio1.621); full-process time is37.55% lower. Relative to the uncached
adapter, checker ratio is2.297. Report BOTH: the uncached adapter is slower
than the original, so its ratio alone overstates practical improvement over
the pre-existing implementation. The extra uncached adapter work includes
canonical snapshots and both row/map representations; no detailed causal
time breakdown was measured here.

Peak process RSS increases277.355MiB,32.62%, relative to the reference. This
is a time/memory tradeoff, not a memory optimization. Both cached runs report
290 lookups,267 hits,23 parses,0 evictions and0 oversized bypasses. Retention
peaks are23 matrices,37,578,405 canonical payload bytes and1,138,205 cells;
all live cache counters are zero at request return. The configured limits
bound retained representation counts/bytes, not total RSS or transient memory.

## Results and independent accounting review

Every complete returned result equals the unchanged saved precheck exactly:
`UNKNOWN_NONPOSITIVE`,3 positive and6 nonpositive obligations. Canonical full
result SHA256 is
`987cff13fc4b2df9e3fb8469823c2d74d6f950cce44efaa18bae8f2dcd580ec0`.
The original request remains TIMEOUT; no result was promoted and no additional
SAFE was obtained. Faster checking does not make its six nonpositive bounds
positive or remove any upstream trusted lowering assumptions.

After execution, a separate jq accounting pass reconstructed all three
two-run checker/process medians and maximum RSS values from the six rows,
checked order/count/terminal/result invariants and cache clearing. Shell hash
checks revalidated every control-bound source, the archived parent and the
original request/terminal; the raw summary and committed compact timing JSON
are byte-identical. This is an accounting/identity review, not a second
independent implementation of the underlying rational proof checker.

Bound artifacts:

- Controls SHA256: `4a414c10886f225c2282c2c75b1c4e899fec2651e04d5cd25426dee4468a80e0`.
- Timing JSON/raw summary SHA256: `292bbc896e3ba8279555a27416b81dd0d0fb99373c4cd8d79fea046abed86355`.
- Parent archive SHA256: `67cf5602723221df1b6ac1d10a04363de8b71bd385722c8a87f272353b9a21fe`.
- Raw logs and per-process JSON remain under
  `data/moe/results/exact_matrix_cache_benchmark_20260916_v1` and are not committed.

## Disposition

This single saved request with two runs per mode gives a positive engineering
signal for exact parse reuse, not a population estimate, new certificate or
end-to-end verifier speedup. Timings are unprofiled; do not compare against the
previous102.85s cProfile run. Shared-server variation and memory effects remain.
Do not repeat the immutable controls-receipt or benchmark writer in its existing
directory. Original20-input negatives and all failed historical attempts remain.

Next separately scoped step: package the optional cache in a new identity-bound
portable checker/driver, then test relocation, timeout/partial evidence and
whole-budget accounting before considering any new real-request evaluation.
Do not silently adopt it into a frozen execution or change its numerical gates.
No further cohort or proof-search run was launched by this stage.
