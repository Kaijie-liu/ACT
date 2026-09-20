# Complete source-range pipeline: finite two-arm protocol

This version integrates `source_ranges_v1` into the **whole registered expert
trace and all fresh output obligations**. It is not another run of the sealed
`full_bounds` protocol. That attempt remains zero positive, seven checked
nonpositive and two missing bounds. No old matrix, dual, threshold or result
is edited or reused as a new accepting certificate.

## Hypothesis and fixed scope

The previous complete source enclosure used generator-box bounds at hidden
affine/ReLU transitions. This control asks whether a small, predetermined
set of independently checked, constraint-aware ranges can be generated,
consumed and checked within the same complete-request budget, and whether
they change the resulting output evidence. It does **not** assume this is
the unique cause of the old negative bounds.

Same stored convolutional request98, label0, four experts, ten classes,
selected-softmax top-2, sole tie-legal pair[1,2], represented2/255 box. Model
state `44572f06b657883fe169406d3cc260878d1cf41e762cbacdb15fe006da6487f5`.
This is the67.06% model and a single-route request, not high-accuracy or
route-changing evidence. No checkpoint selection, training, new input,
forward pass, gate tuning or native-float acceptance change.

Both experts have Conv/ReLU/Conv/ReLU/AvgPool/Flatten/Linear/ReLU/Linear.
Read-only metadata inspection confirms hidden Linear layer6 has64 rows and
512 input coordinates in each expert. **Only rows0 and1 of layer6, in each
expert**, may receive newly generated range facts. They were chosen as a
fixed prefix before running this protocol, not by positivity or tightness.
The other62 rows retain explicit generator-box fallbacks.

## Two arms and unchanged obligations

`range_off` rebuilds every input, compensated first-Conv/ReLU prefix, remaining
expert layer, shared/private join and all nine output LPs, with no range calls.
`range_on` performs the identical work, with at most four two-sided range
queries at the fixed roster. Each side gets one SciPy `highs` call capped3s;
at most eight range calls total. No retries or later row substitutions.

Both arms use SciPy1.16.3, NumPy2.3.5, SciPy-bundled HiGHS1.8.0, one thread,
CPU. No separate highspy engine, solver installation, precision fallback,
basis reuse or cross-arm shared source/fact/answer cache. Final output
proposals use the unchanged `full_bounds.worker`: competitors1..9 in order,
one call each at most16s, bounded by the shorter batch/total remainder.

The range proposer constructs positive and negative source-factor objectives
with every original constraint, relaxing binary factors to[-1,1]. Both signed
dual candidates must pass the existing exact sparse residual-box checker,
bound to this source, request, domain, layer, row and expression. Missing
either candidate means a visible `None` fallback; a supplied invalid fact or
inconsistent consumed range rejects the build. A half-published successful
side is retained but cannot authorize a two-sided range.

Range facts only change justified hidden affine ranges. Exact recentering
and defining equalities preserve the same factors and input relation; the
subsequent unchanged ReLU transfer consumes those affine bounds. Gate[0,1],
output McCormick formula, binary relaxation, all legal routes and all nine
classification properties remain required. Difference ranges and every
output LP are regenerated from the resulting **new** joint source. No old
positive or negative certificate is transplanted onto changed matrices.

## Batched parsing, not fewer checks

The producer and independent affine checker parse a layer source once, then
reconstruct every row. They are separate implementations; the checker never
calls the producer. Supplied range facts still get full source projection,
identity and both signed-dual checks; all inherited constraints, factors,
row counts, equalities and resulting source hashes remain checked. Fallback
rows do not need to construct an LP merely to sum generator magnitudes.

This avoids the old scalar control implementation's repeated whole-source
parsing for every fallback row. Exact state/proof/check-result differential
tests against that implementation cover partial facts, all fallbacks and
an existing binary source. Both real arms use this same batched path, so
the finite comparison is not a claim about parsing speedup.

## One complete budget per arm

Each arm has a separate300s budget, including subprocess startup and imports,
source validation/copy, all input/prefix/remaining propagation, range LP
construction/conversion/native solving/exact prechecks, delta serialization,
new output LP construction, output candidates, portable packing, complete
source and bound checking, inventory hashing and terminal publication.

| Phase | Cap, also limited by actual total remainder |
|---|---:|
| Complete build, including range generation | 100s |
| All new output dual proposals | 120s |
| Seal published proof inputs | 5s |
| Independent complete source and output check | Actual remainder |
| Reserved terminal publication | 2s |

The caps do not add to a larger budget. Every native call remains subject to
the owning subprocess's hard deadline. Build failure/timeout preserves its
partial trace and stops; no later obligations can be inferred. Output proposal
timeout may still seal/check published candidates within the remainder, with
missing obligations explicit. Exceptions cannot create positive terminals.
Late publication cannot establish a positive result. Only owned child process
groups may be killed.

Frozen historical **parameter/input/nominal-first-Conv capture** costs are
excluded. The input envelope, compensated prefix and all subsequent source
propagation are newly generated and charged. Thus this is a complete
stored-source proof pipeline, not production time including model loading.
An independent post-terminal relocated audit is separately charged and
reported, never subtracted from execution time. Per-layer ranges, propagation,
delta serialization and joint/output construction are subcosts of build;
native/conversion/check subcosts must not be double-counted.

Fixed arm order is off then on. One input, no repeated timing trials: the
result is a feasibility/representation control, **not a population estimate
or an unbiased speedup measurement**. Both arms execute regardless of the
first arm's scientific result, subject to the resource gate. Resource
unavailability is a visible not-started arm; no automatic retry or resume.

## Controls, checks and acceptance

The six new tests cover: exact old/new affine differentials; independently
matching source LPs; complete/partial/exception/invalid-sign/expired range
proposals; outer deadline and malformed partial execution; rejection of a
forged positive incomplete terminal; and the complete synthetic two-arm flow.
The latter generates all fresh obligations, checks both arms positive, and
checks again after moving the package and removing test-owned originals.
Four hash-rebound mutations (missing layer, wrong fact source, disabled policy
with facts, missing output) reject. These positives are synthetic controls,
not trained-model results. Test commands/results are in the control receipt.

The isolated `python -I -S verify_bounds.py --manifest-hash <hash>` checker
needs only the moved bundle and standard library. It freshly checks input
containment, affine compensation, all ReLU/factor transitions, exact routing
coverage, the join, every new McCormick construction, and every available
output dual. Checking does not load checkpoints, datasets, historical paths,
numerical solvers or producers.

Only nine strictly positive checked output bounds yield
`CHECKED_POSITIVE_DECLARED_REAL_MOE`. Missing evidence or any nonpositive
bound remains UNKNOWN. A nonpositive lower bound is neither a model
counterexample nor an exact feasible upper bound proving an intrinsic LP gap.
Declared graph/parameter correspondence to the intended program and checker
execution remain assumptions; native floating execution is not proved.

After preparation commit/push, execute once, then review and archive either
outcome without increasing rows or limits:

```sh
python -m range_pipeline.run
python -m range_pipeline.review data/moe/results/range_pipeline_conv98_20260920_v1
```

Report all nine outcomes per arm, range candidates/fallbacks, checked range
changes, source dimensions/identities, phase costs, package bytes and fresh
check costs. If no two-sided fact survives, that limits this fixed proposer,
not the mathematical value of ranges. If tighter ranges survive without a
positive request, report that separation directly. No automatic follow-up
search or scientific claim upgrade is licensed by completing this pipeline.
