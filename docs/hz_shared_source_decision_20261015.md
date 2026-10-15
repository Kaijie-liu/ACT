# Shared-source capacity decision: stop representation-only admission work

**The approved shared-source implementation is complete. Pure packing or source
pass fusion cannot admit the registered shape under the unchanged contract.**
This static follow-up binds clean `a210f8224ad0c0472fe2151c881b8b5e3b3a2fe2` on
`feat/moe-route-verification`. It creates no source, candidate, worker, model,
solver or GPU execution. Five new arithmetic/binding tests and four prior ledger
tests pass. It does not add to the previously reported 156 implementation controls.

The [machine-readable decision](hz_shared_source_decision_20261015.json),
[calculator](../scripts/assess_shared_source_decision.py) and
[tests](../scripts/test_shared_source_decision.py) bind the implementation and
two frozen shape inventories. The previous
[implementation results](hz_shared_source_controls_20261015.md) and
[actual-lifetime ledger](hz_shared_source_cost_20261015.json) remain unchanged.

## Why removing representation overhead is insufficient

Keep two fresh, independent source-equation checks: production and online
portable replay. Each visits every affine weight position, including zeros.
For `P=6,961,368` stored affine scalars and `A=3,288` biases:

    source equation visits >= 2(P-A) = 13,916,160 > 8,000,000.

This gives producer construction, Snapshot packing, parsing, pages, source norms,
candidates and both consumers zero cost. Biases have deliberately been removed
from this weight-position count. Unlike the previous Snapshot-only floor, this
one persists after removing all packing overhead. It is a conditional lower
bound for this complete checking schedule and counting contract, not an observed
run or a lower bound for every possible MoE proof algorithm.

The current consumers also merge a residual and compute two full norms for
every endpoint with complete evidence. Each operation has a coefficient visit
and an exact addition for every factor. Granting all buffer accesses, pullbacks,
private factors and scale-group final sums zero cost, two consumers require at
least `2*6*n*E` additional Work, with `n=3072` input factors:

| Complete endpoint roster | Source equation floor | Access-free consumer floor | Combined floor |
| --- | ---: | ---: | ---: |
| 252 endpoints | 13,916,160 | 9,289,728 | 23,205,888 |
| 504 endpoints | 13,916,160 | 18,579,456 | 32,495,616 |

These are not full upper bounds or proposed quotas. A partial or missing-candidate
request may stop earlier, but cannot establish complete positive closure.

Membership is a separate refusal. Pair guard views alone have `28*3072=86,016`
input-output aliases. All mandatory view uses total **126,632**, above **65,536**.
The implemented sequence sharing saves physical storage; the reader still checks
and charges expanded logical memberships. Counting only unique roots or stored
sequences instead would change the resource contract even if its numerical cap
were still called 65,536. Such a change is not authorized here.

## What the new storage representation does establish

For the bound topology, explicit local entries have the conservative bound

    l <= n + 2A + 12U = 48,048.

The terms retain the input basis, both possible affine compensation tails and
all conservative ReLU output/constraint terms. First-affine weights remain in
the original source and are referenced; later weights and guards use references.
This is not a claim about observed weight sparsity.

With `N<=19,496`, `r<=277,280`, `u<=136,232` stored sequence entries and
`F<=32,048`, the actual format has

    S = 36+h+32N+12(l+r)+4u+8F <= 13,717,764 bytes,

**conditional on** the metadata header passing its 8 MiB cap and coefficients
passing the existing binary64 encoding checks. That is at most 210 pages.
Source tensor payload plus this forest envelope is **69,433,284 bytes**.
It excludes the page index, source header, candidates, code, manifests, results,
logs, receipts, parsed objects and arithmetic temporaries. It neither proves all
metadata can be generated within its caps nor bounds the whole archive or RSS.
Expanded logical local/reference caps still apply despite physical compression.

Candidate evidence has a different format from earlier compressed-wire proposals.
Both proposed and explicit-zero roles still occupy full vectors. For per-pair
dual rows `M`, scalar width `w`, tile width `k`:

    tile bytes = 128 + k*(72 + 2*1040 + 2*M*w).

At `M<=2700`, 504 endpoints and 84 pair-local tiles of widths 8/8/2:

| Candidate scalar profile | Complete stream upper | Largest tile upper | Existing caps |
| --- | ---: | ---: | --- |
| binary64, `w=8` | 22,868,160 B | 362,944 B | 16 MiB minus 8,192 B stream; 4 MiB tile |
| general rational, `w=1040` | 2,831,559,360 B | 44,945,344 B | Same |

The general-profile single-endpoint envelope is already 5,618,280 B. These
oversized **upper bounds show the absence of a full-shape capacity guarantee**,
not that an observed request has overflowed. No actual source or dual dimensions
were loaded. Unlike the Work and membership floors, they do not prove inevitable
failure for every instance. Do not silently remove the zero role, round a general
candidate to binary64, or treat a profile restriction as an equivalent codec.

## Stop/continue decision and remaining authority

Stop using source-packing fusion alone as a strategy to fit the full request in
8M. Keep the default-off engineering result; do not start another tiny timing
loop or full-shape retry. Consumer optimization cannot erase the source-only
floor while the present two-check contract is fixed.

Do not revive the [72 residual-template proposal](hz_residual_template_capacity_20261013.md)
as an 8M remedy: its dense first computation was already insufficient. Merging
objective and dual before pullback was already described by the
[original DAG contract](hz_coefficient_dag_design_20261013.md); calling it new
would not establish innovation. Changing the later separate-pullback checks is
a separate mathematical/check-schedule decision, not free removal of overhead.

A next resource design must address three remaining items together:

1. Current shared implementation's **complete** phase-by-phase Work upper for
   each matched arm, including online replay and reception. Do not reuse older
   JSON-path bounds or derive quotas from the floors above.
2. A consistent resource vector: expanded memberships versus physical aliases,
   source chunking, candidate profile and both roles, metadata demand, files,
   members and total archive. A binary64-only proposal and a general-rational
   proposal must not share an unsupported 128 MiB capacity claim.
3. Full live-object/RSS accounting for source banks, graph/header parsing,
   Fraction temporaries, consumer arena and allocator lifetime. Sampled RSS
   termination and reference release do not provide a success upper bound.

The current model-scaled-policy authorization permits static drafting only. Any
changed counting meaning, quota, runtime check schedule, candidate encoding or
full/native/GPU execution needs its own explicit scope. No new quota is set by
this report. All mathematical checks, acceptance gates and the 300-second budget
remain unchanged. The registered shape stays refused; G1–G6 and the historical
scientific evidence boundaries remain open.

Three read-only AI reviews checked the counting, historical alternatives and
storage/lifetime interpretation. They are not independent human review or a
machine proof of the cost derivations. The tests check arithmetic, bindings and
classification, not the truth of arbitrary supplied bounds.

## Reproduce without numerical dependencies

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -I -S scripts/assess_shared_source_decision.py --check-report docs/hz_shared_source_decision_20261015.json
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -S -m unittest scripts.test_shared_source_decision scripts.test_shared_source_cost
```
