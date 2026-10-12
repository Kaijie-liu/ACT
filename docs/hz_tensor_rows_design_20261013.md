# Bounded captured tensor row intake

Frozen before implementation, 2026-10-13. This is a source-intake component,
not source-to-HZ lowering, trained-model admission or a numerical experiment.
It advances the capacity prerequisite in [the current goal](ALGORITHM_RESEARCH_GOAL.md).
The existing H2 chunked source format and its binary64 tensor/state identities
remain unchanged. Old source, store and numerical acceptance limits remain.

## Contract

Read a hash-bound `H2_CHUNKED_SOURCE_V1` manifest and its bounded little-endian
binary64 chunks. Validate the complete eval, selected-softmax top-2 graph,
parameter roles, ordered layers, input/domain/property identities and endpoint
dimensions before yielding affine rows. Support only existing Flatten `[1,-1]`,
non-inplace ReLU and biased Linear. No new convolution operator is implied.

Retain at most one bounded raw chunk per active weight/bias stream and one row
of exact rational coefficients, plus bounded graph metadata and input coordinates.
The new reader must not call the old whole-tensor or scalar-node reader for
weights. Bias is streamed too. Tensor hashes cover original bytes, including
signed zero; every value must be finite and is interpreted as its exact binary64
rational. Row length and all layer dimensions are at most 4096. Existing limits
of 1,000,000 elements per tensor, 1 MiB per chunk, 4 MiB per manifest and 256 MiB
per source remain; input coordinates are additionally bounded by 4096.

No completion receipt is available until every row, tensor hash and graph has
been exhausted and the manifest/member inventory rechecked. Partial consumption,
exceptions and deadline expiration never count as a checked source. A receipt
describes the bytes consumed under the frozen identity, not a guarantee that an
external process can never mutate files later. Every new traversal rechecks bytes.
Public descriptors and rows are immutable; request metadata is copied on access.
This interface does not certify a live checkpoint-to-declaration correspondence.

## Fixed controls

1. Small two-expert mixed Linear/ReLU graph: exact differential against the old
   scalar-node reader, with both large chunks and 40-byte cross-row chunks.
   Include negative zero, the smallest positive binary64 subnormal and zeros.
2. One registered-shape synthetic source: 3072 inputs, router `3072→2`, expert 0
   `3072→256→2`, expert 1 `3072→2`. Use deterministic dense dyadic coefficients,
   validate all 786432 coefficients in the first expert layer and every other
   coefficient/bias. Complete source bytes exceed 4 MiB. This is not the full
   registered eight-expert model or evidence that its HZ fits memory.
3. Refuse wrong manifest, state/tensor hash, role, shape, layer order, eval mode,
   Flatten/ReLU semantics, gate/ties, center/property identity; missing, extra,
   truncated, swapped and corrupted members; nonfinite coefficients and unused
   tensors. Mutations with a freshly bound manifest must still fail semantics.
4. Partial iteration, last-row-without-exhaustion, explicit deadline, chunk
   mutation during iteration, exposed-metadata alias attempts and independent
   replay. No cached value can waive a byte or identity check.
5. Copy the small source/reader and its standard-library dependencies to a new
   directory; check under `python -I -S`, without model/data/numerical packages.
   Independently recompute fixture coefficients and completion inventories.

These controls retain all raw reads and report chunk reads/bytes, coefficients,
rows, maximum raw chunk and row sizes, component time and total process time.
Resident-width accounting is an algorithmic bound, not measured process peak RSS.
No solver, checkpoint, training data, native numerical propagation or CUDA runs.

## Finite execution and decision

Use the existing `act-py312` Python and CPU only. Each complete control process
has a 300-second hard outer cap; the normal reader also receives one cumulative
deadline, never reset per row/chunk. At most two archived execution attempts for
correctness repairs, never for performance tuning. Keep failed prefixes. The
entire stage's retained raw evidence, including relocation, is capped at 64 MiB.
Unit controls and the one dense-shape traversal are finite, not a scaling grid.

Accept this component only after the fixed normal and refusal controls pass,
an independent read-only review, and archive/cost accounting. Then stop its timing
study. A failure gets a specific bounded repair or a stop, not a larger cap.

Next source-to-store integration remains a separate decision: current affine
producer/checker still require whole layers and 262144 nonzeros per layer, and
the store has a global 1048576 nonzero limit and eager row parsing. Capturing
live tensors still materializes base64. Streaming intake alone changes none of
these facts and closes none of G1–G6. Preserve the GPU refusal, all sealed inputs,
old results and all output obligations.
