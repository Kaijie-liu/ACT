# Full model resource ledger and storage decision

The registered eight-expert topology now has a code-bound **Work counter upper
bound for the complete captured-source V3 path**, including independent source
checking and every output endpoint. It still has no complete resource admission
contract. An ordinary binary64 coefficient case already exceeds the unchanged
128 MiB archive envelope, and the native workspace and full resident-memory
bounds remain open. No runtime limit, algorithm or frozen experiment changed.

This follows the [model-scaled proposal](hz_model_scaled_resources_20261013.md).
The [recomputable ledger](hz_full_resource_ledger_20261014.json) binds 21 existing
code/metadata files and the [analysis script](../scripts/analyze_full_resource_ledger.py).
Starting branch was `feat/moe-route-verification`, clean HEAD
`eb911b09889f1ca74e494d10571189b5bdaf5b11`. No model, captured source, numerical
library, solver or GPU was loaded by the analysis. Small JSON syntax checks are
not source construction or verification queries.

## Scope and comparable internal paths

The bound is for the registered seed 1/2 topology: 3,072 inputs, router
3072–128–8, eight experts 3072–256–128–10, selected-softmax top-2, eval semantics.
It does not establish seed 0 topology or correspondence with a live checkpoint.
All 28 unordered pairs remain in the conservative coverage roster, with 252
properties, at most 504 distinct endpoints and 1,008 proposed/explicit-zero
candidates. This roster does not assert that every pair is reachable.

The selected internal accounting reference is the existing V3
[`produce(width=1)`](../act/back_end/moe/scaled_forest_endpoints.py) against
`produce(width=8)`. Both use the same source enclosure, properties, 128 updates,
129 evaluations, CPU float64, one thread and independent exact acceptance.
Width changes **both native candidate batching and exact dual consumption**.
This is a whole-endpoint batching comparison, not a consumer-only ablation.
Generated candidate trajectories need not coincide; a same-candidate exact
differential remains necessary for correctness. No real runner or experiment
for these two paths is frozen here, and neither replaces the strong native or
external baselines. A consumer-only study would require a separate orchestration
that decouples proposal width from checking width.

## Complete counted Work for captured sources

These are conservative bounds on the counters in
[`Work`](../scoped_source/hz_request_store.py), conditional on a conforming full
source and finite candidate path. They are not the measured cost of the trained
model, not a prediction of completion, and not a proposal to assign billions of
Work units. Existing structural/byte/bit/deadline gates may refuse much earlier.

| Whole endpoint batching | Tile upper | Row visits | Coefficient visits | Exact arithmetic calls | Total Work upper |
|---|---:|---:|---:|---:|---:|
| Width 1 | 504 | 26,635,776 | 2,161,406,048 | 4,273,366,725 | 6,461,408,549 |
| Width 8 | 84 | 15,909,396 | 1,298,519,408 | 4,273,366,725 | 5,587,795,529 |

Portable rechecking starts again from raw source and both raw candidates; its
upper totals are 6,415,178,465 and 5,544,103,505 respectively. It does not generate
native candidates or build the source forest, and it cannot retrospectively
complete an unfinished online request.

Let n=3072, W1=6684672, W2=273408, A=3288, U=3200, G=336. The parent structural
bounds are F=32048 factors, N=19496 nodes, L=6732720 local entries, R=277280
references, V=82 views, H=136232 root occurrences. Let P=28 pairs, C=10 classes,
D=252 duties, E=504 endpoints, M=2700 pair constraint rows, and
T=P ceil(18/width). S≤3073 distinct scales allows every input scale to differ;
the checker requires every auxiliary scale to equal one.

The router ancestor closure has Kr=399248 entries and Fr=4240 factors. A
router/two-expert/pair-guard closure has K=2051792 entries. These follow the
parent formulas restricted to the respective source graphs, without assuming
zero centers, no compensation, sparse weights or sparse duals. A pullback still
visits all N nodes; only coefficient and arithmetic work may use the ancestor
closure. Native matrices, in contrast, retain the **whole forest**. The registered
graphs have nonempty reference depth levels 2 and 3, so J≤2 native levels.

| Phase | Row visits upper | Coefficient visits upper | Arithmetic upper |
|---|---|---|---|
| Source construction | F+N+U | W1+W2 | n+W1+2W2+2A+9U+G+3L+R |
| Independent source check | 2V+H+N+F+U | F+L+R+W1+W2 | n+W1+2W2+A+9U+G+3L+R |
| Gate and obligation plan | 11+P+D+E+PN | P(Kr+Fr) | 1+P(2Kr+3Fr+3)+2CE |
| Native preparation | N | L+R+F | 0 |
| Objective preparation | E(N+2C) | EK | E(2K+12) |
| Native candidate generation | 3E+T(M+129+257J) | 4E+TM+EM | 0 |
| All-candidate acceptance | T(M+N)+E(32+2S) | 2EM+TK+3EF | 3EM+2EK+E(3F+4S+3) |
| Aggregation and proof generation | E+D+13 | 0 | 0 |

Construction/check formulas cover first-affine centers, compensation and scaled
norms, later-affine references, ReLU enclosure rows and all guards. The
[source producer](../scoped_source/scaled_references.py) and
[independent checker](../scoped_source/check_scaled_references.py) inherit the
reviewed [reference](../scoped_source/affine_references.py) and
[row construction](../scoped_source/source_dag.py) operations. For portable
checking, omit construction/native preparation/native generation and use E+D+9
for final aggregation, since it does not generate `proof()`.

[`Session.prepare`](../scoped_source/scaled_endpoints.py) still pulls each
objective back separately. Batched dual consumption shares coefficient visits,
not each column's exact multiply/add. This explains the equal arithmetic upper
in the two accounting paths; it is not a measured timing attribution.

**Work is not FLOPs or elapsed time.** Tensor parsing/hash/read, JSON encoding,
frozen containers, file publication and Torch tensor arithmetic consume time
and memory without incrementing the three Work counters. Width 8 still permits
10,836 adjoint and 10,752 forward calls on the global sparse bank. Native
arithmetic=0 in the table means zero `Work.exact` calls, not zero computation.
The accounting includes only the current publication-only event behavior, not
an arbitrary callback. A new live checkpoint capture or fallback needs its own
ledger; the current frozen worker merely copies an already captured source.

## A concrete archive encoding obstruction

The current first-affine representation stores every nonzero source weight as
`[factor_id, str(Fraction(weight))]`. Input factors are exactly 0 through 3071.
Consider the registered shape with all first-affine weights equal to binary64
1/2. This is a counterfactual encoding case, **not an observation of training
weights and not an additional source run**. Its dense local bank is forced
regardless of center, scale or compensation values.

The decimal digit sum for IDs 0..3071 is 11178. The 3072 weight entries with their
minimal array wrapper require `1 + 11178 + 9*3072 = 38827` bytes per row; any
compensation tail makes the full row larger. There are 2176 first-affine outputs.

| Necessary component | Logical bytes |
|---|---:|
| First-affine local arrays alone | 84,487,552 |
| Raw source binary64 payload | 55,715,520 |
| Single bundle lower bound | **140,203,072** |
| Retained entire archive cap | 134,217,728 |
| Excess before all other evidence | **5,985,344** |

Node wrappers, views, metadata, scales, proof, partial candidates and logs are
all omitted from that lower bound. The runner's extra 2 MiB publication reserve
would refuse sooner. If a new protocol retained the archived dense runner's
online-plus-relocated layout, these two components alone would double to
280,406,144 bytes. No new full-topology runner is frozen here. Thus unchanged
encoding cannot provide a universal 128 MiB archive completion contract for this
shape, even if Work and structural
limits were separately raised. This is not a claim that every actual checkpoint
overflows, or a lower bound on every possible verifier representation.

## Other encoding bounds and their limits

The captured source uses little-endian binary64 chunks, not decimal/hex strings.
Its raw payload plus a bounded 4 MiB manifest is at most 59,909,824 logical bytes.
Live capture's base64 and intermediate declaration are separate objects.

A finite binary64 rational needs at most 342 decimal characters: an integer
needs at most 309 digits, while a noninteger has at most a 16-digit reduced
numerator and a power-of-two denominator with at most 324 digits, plus sign and
separator. General checker input allows up to 2470 characters. For M≤2700 and
E≤504, counting the full proposed vector **and** explicit zero vector gives:

| Candidate value encoding | Width 8 dual elements upper | Full proof dual elements upper |
|---|---:|---:|
| Current finite binary64 producer | 7,538,400 | 474,919,200 |
| General stored-rational checker input | 53,503,200 | 3,370,701,600 |

These are upper bounds on the **dual-element component**, not complete proof
upper bounds or measured sizes. They exceed the 4 MiB tile/16 MiB proof caps,
but unlike the preceding constructive lower bound they do not establish that
the actual candidates will overflow. They cannot justify dropping the zero
candidate, assuming sparse duals or accepting incomplete properties.

The parent structural maxima give a 3,130,874,528-byte upper on forest value
components (local/reference/root/scale entries), before complete metadata.
Full forest/proof/archive uppers remain null. The archived dense runner retains source
and forest, partial tiles, full proof and a relocated copy; its accounting is
`2*source + 2*forest + 2*proof + partial_tiles + metadata/logs/code`, not one raw
parameter bank. No byte cap changed.

## Resident objects and missing memory contract

| Stage | Objects simultaneously live or copied |
|---|---|
| TensorRows intake | Manifest bytes and metadata, input intervals, weight/bias chunk buffers and one coefficient row; not the old whole-layer Fraction reader |
| Builder serialization | Numeric forest, string/list JSON document, complete JSON text and encoded bytes |
| Source check | Raw forest, decoded document, canonical re-encoding, parsed Fraction forest and view containers |
| Native preparation | Checked forest and raw bytes, Python row/column/value/edge lists, COO tensors, coalesce workspace, transpose and reference levels |
| Native tile | Objectives, dual iterates/best values, adjoint clones, residuals, forward and index-select temporaries; not 129 retained copies |
| Exact acceptance | Prepared objective dictionaries, incoming candidate document, alpha/dual/residual dictionaries, full-factor merges and scale groups |
| Full proof | All retained raw tiles, newly decoded full proof, complete canonical encoding |
| Portable check | Outer proof document, re-encoded bytes and second decoded proof, per-tile serialization plus independent source and endpoint checks |

Named native array shapes are bounded (N×8, F×8, M×8); one local COO's values and
two int64 index arrays alone are at most 161,585,280 bytes. This is not a peak-RSS
bound. Python/Fraction/allocator high-water lifetimes, Torch coalesce/sparse-mm
workspace and a charged live capture interface remain unbounded by this ledger.
Several size checks occur **after** creating the large encoded/decoded object;
`work.exact(q)` also sees an already evaluated temporary rational. The 8 GiB RSS
and 16 GiB address-space limits are stop rules, not promises of completion.

## Decision and next boundary

The Work subledger and comparison identity are now explicit. The complete
resource contract is **not** closed: archive encoding has a constructive
obstruction, peak memory lacks its required bound, and live capture is not the
current worker. New Work quota, absolute ceiling and time prediction remain
null. All old 8M/96M identities, 300 seconds, obligations, bit limits and exact
acceptance checks remain unchanged. No experiment or GPU retry is admitted.

Do not spend another control run trying to fix this by Work alone. The next
decision must separately address evidence encoding/storage, or explicitly
authorize a different byte/resource envelope with its own bounds. A compact
encoding/source-reference design would be a new representation/transport factor,
not a hidden change to this resource-only comparison. It must preserve complete
raw candidate binding, independent rechecking and partial-evidence recovery.
Peak native workspace and metadata/log bounds would still need to be addressed.
No stored failure, model or sealed input is removed or reopened.

## Verification

Eleven new static arithmetic/identity tests and nine parent tests pass in the
existing `act-py312` environment with site imports disabled. The final selected
regression also includes tensor-alias capacity (6), residual-template capacity
(6), manuscript source contract (6), review handoff (4), and project layout (9):
**51 tests pass**. These are static accounting, identity and record-preservation
controls, not 51 new algorithm or real-model experiments. Initial invocation
with bare `-I -m unittest` could not import the repository; the corrected command
explicitly adds its fixed path. No mathematical test or bound changed for that
launcher correction. These controls validate the formulas and refusal labels,
not trained-model completion or an independently checked network proof.

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -I -S -c 'import sys,unittest; sys.path.insert(0,"/data1/Kane/MOE/ACT"); unittest.main(module=None,argv=["unittest","scripts.test_full_resource_ledger","scripts.test_model_scaled_resources","-v"])'
```

Three read-only agent reviews separately checked Work formulas, storage/memory,
and reference/fairness scope. All passed; the storage review's qualifications
about compensation tails and the conditional two-copy archive layout are
incorporated above. This is not an independent human technical review.

The workspace/navigation audit passes: 619 recorded frozen source identities
unchanged, 4,632 history lines preserved, 431 navigation links and all 10 links
in this report checked. No experiment was started and no file was deleted.

G1–G6 remain OPEN. This static capacity decision does not change the archived
real-model comparisons, source limitations, or lack of measured CUDA benefit.
