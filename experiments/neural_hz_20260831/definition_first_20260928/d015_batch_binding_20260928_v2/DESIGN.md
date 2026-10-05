# D015 v2 — bind the ordinary symbolic batch to the single-input property

2026-09-28, redu-hz, commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac.
Default-off source diagnostic; no production edit, native admission or score.

## Evidence and the single delta

The frozen v1 passed all3725 tests/163 files in57.54913759790361s including
collection, without failure/skip. Its source worker then rejected the first
original CIFAR100-large input annotation before extracting coefficients:
`expected static batch1 RGB NCHW input`. It did not measure shielding hits.
The original protobuf tail records `batch_size,3,32,32` for modelInput.
v1 exit SHA256 is779eebbc5425469148db3011ded4d6628b1c6b6d94ed6117770a78c0d44c64ae.
All v1 source, tests, identities, failure logs and evidence stay untouched.

This is an ordinary source-binding omission, not a failed shielding theorem
or an extreme numerical case. v2 binds a nonempty symbolic BATCH dimension
to the explicitly requested single-sample property shape. Static batch1 is
unchanged; other static batches, absent/zero batch dimensions, symbolic
channel/spatial dimensions, multiple inputs and non-RGB inputs still reject.

The wrapper parses a fresh in-memory model, records the original dimension
oneof/value/symbol tuple, and changes ONLY its input batch type annotation to1
before reusing v1's frozen `_Reader`. Original model/spec bytes, graph nodes,
edges, weights, BN parameters, operation order and input identity do not
change. No model is saved or serialized back. The packet stores the original
raw-byte hash, original dimensions, symbol binding and scope explicitly.

Conv, pointwise affine, inference BN and ReLU in the admitted local grammar
have exactly their ordinary per-sample interpretation under N=1. The local
grammar contains no shape-dependent reshape, batch reduction or training BN;
such operators remain unsupported. This is a specialization to the property,
not a proof for arbitrary N or the entire model's remaining operators. The
caller must still parse the authenticated ORIGINAL VNNLIB and require exactly
3*H*W input declarations and both bounds for each; failure rejects the attempt
before certificate work. The full 400-property goal remains unchanged.

The primary [ONNX IR shape specification](https://onnx.ai/onnx/repo-docs/IR.html#static-tensor-shapes)
distinguishes integer dimensions, named dimension variables and unspecified
dimensions. Named variables share their value throughout the model. Our
mapping records that shared N=1 environment; it does not treat separate uses
of the same batch symbol as independent choices or claim whole-model typing.

## Unchanged mathematical experiment

The v1 [design](../d015_source_shielding_20260928/DESIGN.md) governs the exact
real-arithmetic source semantics, interval BN enclosure, source sharing,
reference-phase identity, composed baseline, pair conflicts and D014 negative
shielding rule. No second rule, positive-extraction path, label/solver menu,
attack, split or query is added. The v2 worker is mechanically copied from
the frozen v1 with only its RUN path, source-wrapper import and explicit
input_batch=1 argument changed. The interval kernel and all numeric work are
the frozen v1 implementation.

Population is IDENTICAL: three original models; the same lexicographically
first original spec per model; all direct next-ReLU Conv branches; all output
channels at four corners and integer center. All side consumers stay recorded.
No failed model/window is dropped. Outer-box crossing is only possible
instability, not evidence of both signs actually reachable. A positive source
certificate is not a concrete ADV, CERT, native rewrite or novel abstract domain.

## Costs, tests and closure

The old raw-byte parse reservation covers one protobuf parse as before; there
is no extra model serialization or second copy. Four original dimension
records and one binding mapping fit inside the unchanged4096 metadata reserve.
All wrapper allocations are in the worker's complete tracemalloc/RSS scope;
returned binding metadata is in the retained roots and encoding. Everything
else uses the v1 pre-operation scalar/container charging, including composed
baselines and quadratic pair generation/verification plus pair sorting.

CPU1/GPU0, AS16GiB, worker240s, whole256M/nested200M work,64M retained entries,
both1GiB memory gates and65536 summary reserve are unchanged. The complete
three-model source/coefficients/bounds/certificates plus serialized evidence
must fit together; timeout/unsupported/resource failure is terminal and saved.

The FULL v1 test population3725/163 is retained byte-for-byte, plus5 explicit
wrapper tests: default-off/explicit binding; unchanged static input; symbolic
batch equivalence and original-byte custody; rejected unknown/conflicting
dimensions; exact original-property input population after binding. Total
3730/164, same combined collection/execution60s. This does not rerun v1's
source attempt; it is a separately frozen v2 composition qualification.

Baseline remains1870/2413 and independent CIFAR25+Tiny36=61/400. The source
pilot does not establish the end-to-end usefulness or novelty required by the
active definition-first goal. All subsequent real-structure/shadow/family/
same-path full2413+400 and nonregression gates still apply.
