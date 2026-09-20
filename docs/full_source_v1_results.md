# Full expert containment and all new output LP constructions checked

Frozen execution `a0410abd32649f1bdc38f58cc89e0f036c3e54b4` completed once.
Independent relocated review **PASS, 0 issues**. Read the
[frozen protocol and proof arguments](full_source_v1.md) and
[complete review](full_source_v1_review.json).
Raw local artifacts: `data/moe/results/full_source_conv98_20260920_v1`.

This is a complete extension of the checked input98 source prefix, not another
partial first-layer audit. Both experts reach their 10-dimensional outputs,
and all 9 necessary weighted classification LP constructions are independently
checked against the new joint state. **No positive output lower bound was
proposed or checked.** The result is not an additional SAFE certificate.

## Complete checked chain

The isolated checker rechecks all 7 original prefix steps and the 5 exact
router exclusions. It then checks 16 new steps, followed by the final expert
join and complete output-LP construction:

| Operation, per expert | Expert 1 | Expert 2 |
|---|---:|---:|
| Identity-coordinate lift, 4,096 outputs | 3,488 new factors | 1,501 new factors |
| Second Conv, 2,048 outputs | Checked | Checked |
| Second ReLU, active / inactive / unstable | 716 / 842 / 490 | 687 / 1,186 / 175 |
| Average pool, 512 outputs | Checked | Checked |
| CHW-preserving flatten | Checked | Checked |
| Hidden Linear, 64 outputs | Checked | Checked |
| Hidden ReLU, active / inactive / unstable | 0 / 27 / 37 | 3 / 45 / 16 |
| Final Linear, 10 outputs | Checked | Checked |

Every affine step is an exact sparse equality lift, not an interval reset.
All original common-input factors and prior constraints survive. The independent
checker recomputes the source polynomial, finite factor scale, defining equality
and factor allocation. Exact ReLU ranges/graphs and operator options are checked
separately. Append-only delta transport reduces repeated matrix storage but does
not skip any mathematical step or constraint check.

The new joint source contains 20 output coordinates, 24,464 continuous factors,
1,619 binary factors, 11,581 equalities and 3,242 inequalities. Exactly 3,072
input factors are shared; private factors remain disjoint. Total joint nnz is
548,634, including outputs and all constraints.

New joint identity:
`93b15af1a0ff299aa60a6be0525513947ad0640f51749471e596a8709db0e3f3`.

## Nine new obligations, zero inherited certificates

For the unchanged label 0 and sole legal pair [1,2], competitors 1 through 9
are all present. Each obligation uses both experts in the same factor frame.
Its difference interval is recomputed from the new coefficients; the gate range
is the frozen universal [0,1]. The checker verifies all four rational McCormick
planes, corner bounds, objective and complete common constraint mapping.

Each materialized LP has 26,085 variables, 11,581 equalities and 3,246
inequalities (four product planes added to the shared base). The 1,619 binary
factors are explicitly relaxed to continuous [-1,1]. The common source/base
matrices are stored once, with nine small property blocks, not nine duplicate
full matrices. The materialization adapter has a synthetic exact differential
against the existing rational builder and independent construction checker.

Base identity:
`96a1f576413532c0c3dd3914506ff0cef6e623a265dd92ba1b312842c160831a`.

All lower-bound-certificate fields are null. No historical LP dual, checked
positive bound, membership fact or old F0 matrix is used here. There were zero
native solver calls and zero network forward calls. One checkpoint load captures
the remaining original parameter bytes, matched to the pinned state inventory.

This closes the **declared graph → new complete expert enclosure → new LP
construction** chain for this fixed request. It does not show those LPs have
positive minima, nor that an LP relaxation is sufficient to prove the request.
It also does not retroactively repair the older conditional positive proof.

## Cost, relocation and controls

| Measurement | Value |
|---|---:|
| Build, internal (capture, new layers, serialization) | 23.4306 s |
| Build child including startup | 24.0204 s |
| Complete checker child | 42.6245 s |
| Total through terminal publication | **66.7383 s** |
| Separate relocated checker | **40.0274 s** |
| Separate review including copying/accounting | 40.2451 s |
| Bundle size | 149,395,256 bytes |
| Build peak RSS | 1,147,096 KiB |
| Fresh relocated check peak RSS | 530,836 KiB |

All execution work fits the unchanged 300 s budget. Per-layer timings and nnz
are retained in the review; they are nested in build time, not added again.
The original prefix generation is excluded, so this is **not** a production
end-to-end verification runtime or a speedup claim. Actual remaining-layer
propagation, parameter capture, checking and serialization are included.

39 tests pass: 7 new controls and 32 existing regressions. The moved synthetic
full chain works after its test-owned original directories are deleted; seven
hash-rebound semantic mutations reject missing layers, changed pool semantics,
wrong expert parameters, wrong shapes, missing properties, old certificates
and wrong joint maps. Supervisor controls reject timeout, exception and partial
output. The separately moved real package reproduces every state/LP identity
and exact result without loading the checkpoint, data, old directories or solver.

Manifest identity:
`ade6c1b34db6fa4923c9c8f13e6a547771aa0aa723fa32027b77c93a93bb295d`.

After copying the complete `relocated/` directory:

```sh
python -I -S /new/location/verify_full.py --manifest-hash ade6c1b34db6fa4923c9c8f13e6a547771aa0aa723fa32027b77c93a93bb295d
```

## Remaining scientific boundary

The checker establishes enclosure for declared real graphs on the pinned
represented box. Correspondence of the captured graph to the intended program
and correctness of checker/interpreter execution remain assumptions; actual
preprocessing, requested real epsilon-ball construction and native floating
execution are not proved. This is still input98, a single-route request on the
67.06% convolutional model, not high-accuracy or route-changing strict evidence.

Next: under a separately frozen finite budget, propose and independently check
fresh lower bounds for all nine LPs bound to this new source and aggregate only
if every required bound is positive. Preserve absent or nonpositive bounds as
unclosed, not UNSAFE. A negative candidate alone cannot diagnose an intrinsic
relaxation gap. No new sample, wider budget, model training or old-certificate
reuse is implied. Production acceptance and main experiment counts are unchanged.
