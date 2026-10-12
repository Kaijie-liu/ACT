# Residual template capacity decision

2026-10-13, reviewed code `f5022871d5d3c458f02eb3961528a529c83244ce`.
**Stop the proposal that source tensor aliases plus 72 scoped residual templates
are sufficient to admit the registered eight-expert recipe.** The reuse algebra
is valid, but it does not eliminate the first computation of each dense residual
or the independent source arithmetic. Neither a complete 8M Work contract nor
an implementation admission follows.

This closes the joint-capacity question left by the
[standalone alias analysis](hz_tensor_alias_capacity_20261013.md). The
[bound calculation](hz_residual_template_capacity_20261013.json) pins that
analysis and its thirteen source/configuration/result files. It uses only the
registered seed-1/2 layer shapes. All counts below concern specified algorithms
and conditional cases, not measured trained-model costs or seconds.

## Source computation remains necessary

Let `N₁=6,684,672` be the first-affine weight positions, `Z` their nonzero count,
and `K` the positions with both a nonzero weight and a nonzero input interval
center. Grant the whole producer and native preparation for free. If one
independent checker retains its present per-value visitation and exact center
and absolute-value norm computation, its first-layer cost is still at least

```
N₁ + Z + K.
```

Thus `Z+K≤1,315,328` is necessary even before later layers or output checking.
For dense weights it is 13,369,344 with all centers zero, or 20,054,016 when all
corresponding centers are nonzero. Actual `Z,K` remain unknown; no checkpoint
was loaded. The counting includes only one norm and center computation, not
the producer's duplicate work or compensation.

Keeping constants as symbolic affine expressions can avoid computing rounded
centers and compensating them immediately. However, if checked ReLU ranges or
later support still demand the same exact `ΣWc` and `Σ|W|D`, lazy evaluation
moves their cost rather than removing it.

## Templates remove repeated pairs but not their first dense pullback

There are eight experts and nine properties, hence 72 proposed expert/property
templates. Consider a case permitted by the registered architecture in which
each first expert bank is dense and all 256 first-hidden adjoints are nonzero
for every property. This is not an observation about the frozen checkpoints
or their optimized duals, nor a claim that existing V3 storage caps admit that
bank. Such adjoints
can occur, for example, in a positive affine/ReLU chain with stable activations
and nonzero output-margin coefficients; the current algorithm has no invariant
excluding them.

Using the current exact multiply/add accounting gives:

| Necessary term in this case | Count |
|---|---:|
| First expert bank, `B=8×3072×256` | 6,291,456 |
| Coefficient–column products, `9B` | 56,623,104 |
| Exact multiply and add calls, `18B` | 113,246,208 |
| Coefficient visits with width eight, `2B` | 12,582,912 |
| Partial total | **125,829,120** |

Nine properties require two tiles per expert at maximum width eight. Smaller
tiles add visits; they do not remove the per-column arithmetic. The bound grants
source checking, zero candidates, router, later layers, endpoint combination,
native optimization, serialization and reception for free. Adding these costs
cannot establish a general 8M bound for this algorithm.

An even more optimistic necessary condition illustrates the sparsity assumption
that would otherwise be hidden. After only one visit to every first source weight,
the remaining budget can pay at most

```
floor((8,000,000 − 6,684,672)/(2×3072)) = 214
```

active first-row/template associations, out of 18,432 possible associations.
This assumes each active row has 3072 nonzero coefficients and makes every other
cost free. It is not a sufficient admission criterion. No observed candidate
sparsity justifies assuming it, and choosing a sparse candidate changes the
search family and potentially its proof strength.

The scoped residual formula from the previous report remains sound with checked
domain inclusion, row embeddings, shared/private identity and dual signs. The
failure here is the proposed capacity argument, not its soundness. Replacing
residual vectors with scalar bounds would be cheaper but would lose shared-input
cancellation and overlap existing scalar scoped-fact reuse; it is not the same
algorithmic result.

## A different enclosure needs an accuracy and cost contract

A concrete alternative is to keep `z=Wx+b` symbolically and independently check
coarse tensor magnitude ranges. From `|W_ij|≤M`, `|b_i|≤B` and `|x_j|≤U`, the
range `|z_i|≤fanin×M×U+B` is valid. This avoids the same per-entry rational center
and L1 accumulation, but changes the enclosure and may make many more ReLUs
unstable. Checking the source values, signs/exponents, finite ranges, graph,
factor identity and all output obligations is still required.

One scan of all registered weights, biases and inputs already covers
`6,958,080 + 3,288 + 3,072 = 6,964,440` values. If charged one visit each, only
1,035,560 Work remains before any other computation. This is a counting base,
not a complete upper bound or a proposed change to current Work semantics.

Likewise, an independently justified frontier cut can replace unexpanded
dependencies with outer intervals. It may bound computation by weakening the
relation that the project aims to exploit. Neither coarse ranges nor frontier
cuts are admitted here: current evidence supplies no complete budget and no
precision argument showing that these changes preserve useful output proofs.
There is no reason to start another small control cycle merely because a weaker
enclosure is easy to construct.

## Execution decision and next work

No source conversion, candidate query, checkpoint load or GPU call occurred.
Do not implement this capacity fix, repeat the dense synthetic query, add an
easier source, drop obligations/checks, relabel scalar work or enlarge a frozen
budget. The previous nonpositive result and all sealed inputs remain unchanged.

Two distinct decisions are now explicit:

1. A new representation would need a concrete accuracy/cost mechanism and its
   full registered-recipe ledger before control execution. The present template
   and coarse-range sketches are not such a contract.
2. A source-scaled resource policy would be a **new protocol**, not a correction
   of the old result. It requires user approval before changing the 8M limit.
   Merely raising that count would not prove completion within 300 seconds or
   improve any lower bound. The user has been asked whether to draft/evaluate
   this separate policy; no permission or real execution is assumed.

While that choice is pending, independent G6 semantic-comparison and evidence
work can continue from existing artifacts. It cannot be counted as completion
of G1–G5 or substitute for the missing algorithm/GPU/real-certificate results.
The total research goal remains active, with all six gates OPEN.

A manuscript correction accompanies this decision: the related-work section
no longer calls all newer baseline work deployment-only. It links the archived
Robust Experts training/full-test evaluation, Dual RS two-input pilot and
completed MetaMoE comparison, preserving their different guarantees and limits.
This is a reporting correction, not a new baseline result. The next G6 task is
an explicit function/domain/property/quantifier admission contract for each
work, not another table-rebuild or relocation control.

## Reproduction and review

Six new static tests pass, together with the six parent tests. They cover the
dense countercase, tile widths, conditional source counts, small arithmetic
enumeration, the unpaid symbolic alternative and parent identity mutation.
Six existing manuscript preservation, evidence-boundary and local-link tests
also pass after the reporting correction: 18 tests in total. The frozen result
tables and historical evidence remain unchanged.
Two read-only reviews independently checked the source and query reasoning.
They are not independent human review or an audit of a new network proof.

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -S -m unittest scripts.test_residual_template_capacity scripts.test_tensor_alias_capacity scripts.test_manuscript_source_contract -v
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -S -m scripts.analyze_residual_template_capacity --check-report docs/hz_residual_template_capacity_20261013.json
```

Only small analysis/code files are added. No raw archive or scientific evidence
is created, deleted or replaced.
