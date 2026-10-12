# Complete V3 dense query capacity result

The once-only [frozen capacity query](hz_scaled_dense_design_20261013.md)
completed on 2026-10-13. Source construction, independent source checking,
native proposals, both-candidate exact checking and complete aggregation fit
the unchanged 300 seconds and 8M Work. The output remains
**UNKNOWN_NONPOSITIVE**, not SAFE or a counterexample. This closes capacity
for the fixed captured synthetic source, not the registered trained models.
All G1–G6 remain open.

## Source and execution

Protocol commit: `83f8f3c54`; execution code: `f6c6d4d43`.
The [compact independent audit](hz_scaled_dense_20261013.json) binds the complete
protocol/code/source/forest/proof/terminal identities. Archive:
`/data1/Kane/MOE/runs/hz_scaled_dense_20261013_r1`.

Source SHA:
`0745027b58f00c608a9db921204cb2feae58273600fee658d4c9ccc3d8c19dbc`.
The pre-existing 3072-input captured source has a two-output router, expert 0
with 3072–256–2 and ReLU, and expert 1 with 3072–2. Center zero, radius 1/255,
clip [-1,1], label 0 and selected-softmax top-2 are unchanged. No weights were
regenerated and no old matrices or bounds reused. One legal unordered pair
{0,1}, one classification property, two gate endpoints and four proposed/zero
candidates were all retained. This two-expert top-2 control cannot establish
changing route sets.

The complete forest contains 3,730 nodes, 802,784 local coefficients, 611
references, 3,569 factors and 11 views; its serialization is 12,041,896 bytes.
No expanded constraint matrix or dense differential oracle was constructed.

## Costs and checked result

| Measured quantity | Result |
|---|---:|
| Parent-observed query including reception | 24.989 s |
| Source construction and checking Work | 4,072,643 |
| Complete production Work | 7,351,435 / 8,000,000 |
| OS worker peak RSS | 914,264,064 bytes, 871.91 MiB |
| Parent sampled peak RSS | 899,596,288 bytes |
| Relocated stdlib check, parent observed | 11.260 s |
| Relocated checking Work | 4,915,948 |
| Separate independent archive audit | 11.308 s |
| Archive logical bytes including terminal | 37,856,616 |

The actual Work fits the reviewed conservative upper bound 7,963,310, which
did not assume early cancellation or zero duals. This slack does not authorize
extra properties or search. Six supervisor-only controls passed before execution
and again in the archive; the numerical query ran exactly once.

Production phases were imports 1.141 s, source copy 0.005 s, source build and
encoding 4.792 s, forest publication 0.006 s, source checking 8.897 s, complete
candidate/check path 9.749 s, and proof/binding publication 0.007 s. The
candidate/check path contains planning 0.028 s, native preparation 0.621 s,
objective adjoints 1.432 s, proposals 6.568 s, exact checking 1.094 s and
encoding. Do not add nested times twice. The archive took 37.178 s before
terminal publication, including controls and relocation; the separate audit
is additional. These are capacity costs, not a matched speedup experiment.

Both endpoints completed; every candidate contains all 99 equality and 198
inequality multipliers. The complete property lower bound is exactly
`-1638016559839389544708037/1180591620717411303424` (about −1387.454).
This is nonpositive, not a violating feasible point. It identifies neither a
unique looseness cause nor model unsafety. Do not tune this synthetic query
until it becomes positive.

## Independent checking and trust

The relocated bundle ran with `python -B -I -S`, without torch, NumPy, SciPy,
highspy, checkpoint, solver or historical run access. It reconstructed the
source and every proposed/zero endpoint bound, returning the same UNKNOWN.
The separate stdlib audit passed: source and candidate proofs, full code
inventory, all eight progress records, monotone Work, phase costs, raw candidate
publication and unchanged relocated bytes were checked. Timing/RSS are measured
records, not independently proved operating-system accounting.

New supervisor controls cover deadlines, exceptions, memory refusal, incomplete
receipts, interrupted JSON, and reception costs. Only a final interrupted prefix
may be recorded as unreadable on a noncomplete parent call. Work refusal keeps
the actual first over-cap count; it is not clamped. Oversized archives can only
be nonaccepting. These controls do not imply the numerical query failed.

The result explicitly retains `checkpoint_correspondence_checked=false` and
`native_float_proof=false`. A checked captured declaration is not correspondence
to an intended trained checkpoint or a native floating-point execution proof.
No CUDA polling, trained model, training, sealed input, dependency change or
external comparison occurred.

## Trained recipe refusal and next decision

The full synthetic capacity control is closed. Read-only compatibility checking
of the [registered seed 1/2 MLP recipe](hz_combined_capacity_20261012.md) gives
router 3072–128–8 and eight experts 3072–256–128–10. This is registered topology,
not a new checkpoint load; seed 0 keeps the earlier report's identity caveat.

The current V3 [first-affine producer](../scoped_source/scaled_references.py)
and independent [checker](../scoped_source/check_scaled_references.py) both
unconditionally visit every input weight, including zeros. Just these loops
require `2 × 3072 × (128 + 8 × 256) = 13,369,344 > 8,000,000` visits. This is
a necessary cost of this implementation, not an nnz estimate, measured trained
runtime or universal algorithmic lower bound. It already refuses the registered
whole request before later layers, exact arithmetic or any output queries.
GPU proposal acceleration cannot remove that pre-proposal source cost.

Next investigate source-linked tensor/reference consumption: can source-bound
weights be referenced without duplicating the first-affine bank and construction
visits, while retaining independent enclosure and complete residual checks?
First derive a full registered-recipe source **and all-query** cost contract,
including nonzero centers, compensation, all 28 pairs and nine properties.
Source-only savings do not admit execution. Stop the mechanism if the full
contract cannot fit; do not redefine Work, raise caps, delete checks, pick an
easier trained source or retry GPU. Tensor references are a research question,
not an established solution to full-model capacity or bound precision.
