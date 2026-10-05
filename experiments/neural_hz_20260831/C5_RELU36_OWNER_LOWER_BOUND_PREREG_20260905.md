# ReLU36 whole-state strict-reduction certificate by a reference lower bound

The V1 live boundary run is closed with numerical equivalence passed and
`live expanded preflight cap exceeded`. Its evidence SHA-256 is
`c00a8daed2725fdacc935d522304bc0e8182f70fac7ad4b419c0afb1e9fd10c8`.
Both 38-row requests, all four branches, 9,939,968 cumulative products, and
all unchanged construction/work gates passed. Do not reinterpret the failed
aggregate reference allocation as a completed physical measurement.

## Proof, unchanged metric and comparator

Let C be the COMPLETE registered candidate state at a fixed boundary. Let R
be the same `phase_selective_expanded_v1` reference as before: identical HZ,
phase/bias/factor/predicate metadata, input/model state, roots, cache retention,
and identity sharing, with every implicit Conv replaced by its exact CSR.
For one uniquely reachable operator selected deterministically by maximum
logical expanded nnz (first registered occurrence breaks ties), materialize
its exact reference CSR W with the same independent native builder. Verify
every row's columns and coefficient bytes against the implicit operator.
One replacement by identity is necessarily strongly reachable in R. Never
add W to the candidate or drop any candidate roots to improve the comparison.

The existing owner-aware numeric metric is monotone under strong-root
inclusion: bytes are sums over the union of backing allocations; CSR data
entries are unions of accepted exact/disjoint data spans. Hence

    M(C) < M({W} union retained_outputs) <= M(R)

proves the SAME strict whole-state inequality for both bytes and entries.
It is stronger than comparing a candidate subcomponent: the LEFT side still
contains every registered root and retained output. Other reference roots
can only increase its metric; their omitted size is never presented as a
measured total. The native CSR constructor allocates fresh data/index/indptr
buffers, so the replacement introduces no conflicting original-root aliases.
Unknown ownership/layout, coefficient mismatch, missing reachability or an
insufficient bound rejects. Do not try a second witness or a larger budget.

This is a validation-method change, not a changed representation comparator,
resource threshold, runtime algorithm, phase rule, score or promotion gate.
The old all-at-once 64M reference preflight remains unchanged and its rejection
remains archived. The new proof materializes only ONE <=64M witness and bounds
its construction with the same 1 GiB measurement, inside the same 16 GiB and
240-second diagnostic limits. It does not allocate an oversized reference.
Record the aggregate logical reference nnz separately, without treating it as
physical allocation. All candidate numerical requests still precede oracles.

## Tests and real qualification

Before the fresh live V2 request, test the inequality against complete exact
reference ledgers on small Conv graphs, including shared operator/source
identities, output masks, stride/dilation/groups, and retained predicates.
Test inclusion/alias deduplication, source immutability, actual coefficient
tampering, witness absence, fixed-cap rejection before allocation, and a
large retained candidate root that makes the strict bound fail. Keep all
existing runtime, ordered equivalence, transaction and ownership tests.

Fresh prefix and target are identical to ReLU36 live V1. Reuse its entire
candidate/oracle transaction; replace only the physical reference proof with
the registered lower bound, report all four COMPLETE candidate boundaries,
and explicitly name the right side `baseline_lower_bound`. Do not claim a
complete reference measurement, terminal solve, speed gate, family retention,
or new score. Require unchanged incoming live-root fingerprints.

Exclusive results `results/c5_relu36_live_20260905_v2/` and evidence
`evidence/c5_relu36_live_20260905_v2.json`, with inherited source/config hashes,
raw tests/logs/snapshots and automatic exit. Passing this numeric-storage
certificate plus all exactness/resource checks discharges the ReLU36 boundary
prerequisite; deeper integration still needs a separate frozen experiment.
