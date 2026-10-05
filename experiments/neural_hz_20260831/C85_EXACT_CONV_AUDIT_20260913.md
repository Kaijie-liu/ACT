# C85: complete Tiny weight transform passes its necessary exactness gate

Research evidence only. Formal1870/2413/all13 and E0 CIFAR25/Tiny36 unchanged;
no original network, actual new source-HZ, native solver or witness was run.
No production/default/history/commit/push change. Supervisor session71794 is
terminal exit0; every declared stage passed, with no source/provenance drift.

The known F(2,3) integer identity is checked on all9 kernel basis vectors times
all16 input basis vectors at all4 outputs:576 exact polynomial coefficients.
This is a complete bilinear coefficient proof, not a floating tolerance or
sampled point test. The published identity is not claimed as new research:
Lavin and Gray equations7-9, https://arxiv.org/pdf/1509.09308 .
Exact input-transform equations retain shared IDs, offsets, nonuniform radii
and padding. Their inverse reads the actual equations and proves redundant
new factor boxes. C86 separately completes channel-sum/output equations.

All16 focused tests passed under the frozen60s gate: collection+tests
2.704473569057882s. This C85 run did not rerun the inherited2006 tests; C86 now
reruns those together with all33 new C85/C86 tests,2039 total.

## Complete actual weights, not selected favorable kernels

The frozen Tiny original model contains19 Conv nodes; ALL14 matching
3x3/stride1/dilation1/group1 operators were inspected. Each is128x128 channels:
229376 complete3x3 kernels and3670016 transformed coefficients in total.
Original binary32 bits are decoded and transformed using bounded signed-int64
arithmetic. Every original kernel is independently recovered from actual
transformed entries. Maximum observed exponent alignment shift28 is within
the preregistered35 limit. All transformed coefficients are exact binary64;
zero precision or nonfinite failures. All coefficient-only row windows pass,
with maximum dyadic gauge13. No normalizing pivot/RHS/source scale is included
in that necessary window screen; it is NOT complete native-row admission.

Model SHA234b04b151d640f8fc859fab00729448ba533d8feb3679427cbadb94467ec776 and
universe SHAa8a0dc7504af2c6b89d099fd5c74f27aa5ae98458c5c151cbb6d0da2ef5c1f59
match before and after read-only execution. The full original and shape-inferred
models plus all selected numeric arrays remain strongly held during measurement.

Complete worker0.5274222707375884s,109992992 whole work and100183072 branch work
fit256M/200M. The exact transform/inverse price is100007936; all model headers,
576 basis coefficients, full spatial geometry count and9662464 exported array
entries are separately included. File hashing0.009998098015785217s is separately
reported; no all-CPU or verifier-runtime payment claim. Numeric payload68009984B
is an array sum, not a unique-owner or HZ ledger.

Measured from before original ONNX loading: resident growth133525504B and traced
peak91560364+127360 metadata=91687724B; BOTH unchanged1GiB gates pass. CPU1/GPU0,
AS16GiB,64M retained array entries,240s worker unchanged. Supervisor4.371653447s
includes tests and result retention. Fatal-only reporting; no sampling signal.

## Cost warning: no blanket whole-layer admission

The geometry-only independent-activation basis has seven14x14 and seven7x7
eligible operators. For EACH14x14 layer, ideal direct nnz26239488 becomes
13669888 but adds200704 transform factors. EACH7x7 layer becomes5920896 to
3893248 nnz and adds57600 factors, including boundary/odd output tiles.
Totals:225122688 to122941952 ideal equation nnz, with1808128 new factors.

Even122941952 coefficient entries exceed the64M entry cap before indices,
metadata or old HZ state. Therefore materializing all these unpruned operators
is NOT an admissible candidate. This is an analytic rejection of that blanket
plan, not a claimed measurement or rejection of actual demand-masked tiles.
Raw full-network operator nnz cannot be compared with C83's10959602 current
predicate nnz: the latter already has actual support, liveness and phase slicing.
The original current source graph, masks/scales, every retained inverse/cache,
native rows and complete physical ledger must determine any admissible rewrite.
Do not charge new factors outside a resource boundary or relabel radix work.

## Retained evidence

All output is in results/c85_exact_conv_20260913_v1. Full original/transformed
weights NPZ SHA52ea0f066d86becdfc5acf779d29c2cd52686c821a1c6aa78e9901663fedacc5;
result SHAa8697d35a1278f8b3c3cb5157bdf1e70dfed398c351257c9e30fe1217e603e3c.
Configuration, source manifest, exact test inventory, JUnit, event/fatal logs,
automatic result/exit and all hashes are retained. Formal gain0 throughout.
See C86_COMPLETE_TILE_HZ_AUDIT_20260913.md and C86_SOURCE_MASK_HANDOFF_20260913.md.
