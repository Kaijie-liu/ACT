# Captured source affine and ReLU coefficient references

The [frozen source-construction component](hz_source_dag_design_20261013.md)
connects captured tensor rows to an independently checked coefficient forest.
A small complete source and the fixed dense affine/ReLU prefix pass. **This is
not a full large-model conversion, output certificate or GPU result.**

Protocol commit: `7dfb917fb`; parent: `b2057e75b`.
The [compact report](hz_source_dag_20261013.json) binds executed code, sources,
receipts and the final audit. Local archive:
`/data1/Kane/MOE/runs/hz_source_dag_20261013_r1`.

## Algorithm and source contract

The [producer](../scoped_source/source_dag.py) stores each affine coefficient
vector once. Active ReLU outputs alias it; unstable ReLU equalities refer to its
negative and store only fresh auxiliary and compensation terms. Right-hand
sides remain independent. All layer error factors precede output lifts; ReLU
auxiliaries precede output lifts and equality lifts. This is an extension of the
coefficient-reference representation, not a new relaxation or acceptance rule.

The [independent standard-library checker](../scoped_source/check_source_dag.py)
re-reads the original captured tensors, checks input containment, recomputes
affine error and ReLU construction, and checks every factor, intermediate view,
output row and constraint. An inactive output does not remove its ancestor from
the source inventory. Each pair has a separate guard view; different route
conditions are not conjoined. Complete source controls retain all unordered
top-2 pairs, without feasibility exclusions.

The checker calculates local L1 radii on the first parse. It reuses them only
after validating the exact source structure: a complete parent reference plus
disjoint fresh factors. This avoids scanning a dense parent again for ReLU
bounds. General overlapping references still require shared-factor combination.
The affine producer keeps only row indices and scalar pending lift corrections;
the checker keeps sparse compensation tails, not duplicate whole-layer matrices.

Native `sparse_hz_linear` supplies nominal coefficients. Its temporary input has
the parent output matrices but no copied constraints; original constraints stay
in the forest. The checker independently verifies the enclosure and does not
trust the native implementation's result. Whole forest storage and JSON
encoding/decoding still occupy bounded memory: this is not a claim of one-row
end-to-end memory or an independent proof of native floating-point execution.

## Results

| Fixed source | Lowered scope | Local entries | References | Factors | Result |
|---|---|---:|---:|---:|---|
| Small captured source | Router, both experts, all outputs and the pair | 49 | 1 | 22 | CHECKED_SOURCE_FOREST |
| Dense captured source | Input/router, expert0 Linear1 and ReLU2 | 796896 | 99 | 3825 | CHECKED_SOURCE_PREFIX |

The dense case uses the original 256×3072 first layer. Its ReLU has 256 outputs,
99 unstable equalities and 198 inequalities. The new ReLU adds 990 local entries;
its output/equality rows would contain 787282 coefficient occurrences when
expanded separately. These are storage/representation counts, not a measured
runtime speedup or a tighter bound. The forest is 36889704 bytes.

Both producer and checker drain all 802566 captured values, including 799494
parameters and 3072 input coordinates. Later tensors are identity-checked but
**not lowered** in the dense control. Its 256-output intermediate endpoint is
not the source's final two-class output. No property query or numerical solve
was appended to this control.

| Dense component cost | Seconds |
|---|---:|
| Generation and encoding, combined | 10.2269 |
| Native affine work, nested in generation | 0.3136 |
| Forest file write | 0.0160 |
| Independent parse and source check | 11.9251 |
| Whole worker | 22.1827 |
| Parent-observed call including startup | 22.6162 |

Generation and encoding were not separately timed in the original receipt;
encoding-only cost is recorded as `null`, not inferred or set to zero. The
native component is not added again. The four-call stage took 26.5685 seconds
before terminal publication; later audits are separate costs.

One cumulative Work object recorded **7202670 / 8000000** operations across
generation and checking. There was no per-layer reset or endpoint-policy
substitution. The counts cover used source coefficients, forest parsing and
exact operations; tensor byte decoding and the unused suffix drain are covered
by reader statistics and the common wall deadline, not represented as one Work
operation per decoded byte. The 300-second limit and existing representation
caps were unchanged.

## Controls and independent recheck

Seventeen source tests, three new archive-metadata tests and 49 existing
regressions pass: **69 tests in the final run**. Controls include active,
inactive and unstable ReLU; expert-first ReLU private scope; a three-expert
router `[x,-x,0]` with ties; all three guard views; nonzero affine error and
output/equality lifts; wrong bindings, factor/ref/rhs mutations, missing
intermediate and final rows, prefix promotion, damaged unlowered suffix,
deadline and work exhaustion. The small sources are also compared against the
old producer and independent checker, including all expanded rows, inherited
constraints and factor identities after the explicit layer-index mapping.

The small package was copied to a new directory and checked with `python -I -S`
using only captured source, forest and standard-library code. It loads no
checkpoint, data or numerical library. Final offline audit rechecks both
unchanged forests against their captured sources and verifies relocated bytes
and result. This checker is independent of construction, not a second formal
proof of the checker implementation.

Review repairs are preserved. The initial unit suite's only failure was an
incorrect test guess of more than 100 expanded mixed roots; the full comparison
had correctly checked 91. The assertion now fixes that exact inventory. Before
the frozen large run, review also removed dense pending copies, fixed expert
scope for a first-layer ReLU and captured the caller's prefix privately. Later
audit V2 pins source identities, protocol, archive limit, complete code roster,
nonnegative counters, cost consistency and relocation. R1 code/receipts and
audit V1 remain; **the large control was generated only once**. Audit V2 only
re-reads those bytes.

The archive is about 43.8 MB, below the 64 MiB cap. No old evidence or cache was
deleted; automatic temporary test directories were cleaned. No dependency was
installed, real model run, GPU retried or sealed input reopened.

## Decision and next boundary

Stop this prefix control. The captured-source-to-reference construction is now
implemented, but the forest does not yet feed the complete endpoint/support
consumer. Do not expand it silently into the old eager Store and declare that
capacity problem solved.

Next inspect the **existing** prefix and complete small forest to specify a
bounded source-forest-to-support mapping: preserve separate pair domains, all
intermediate-source obligations, global factor identity and original rhs. Check
later-layer and full-query capacity before admitting another execution; the
remaining 797330 counted operations are not automatically enough for a large
support query. Convolution and trained-source identity remain separate work.
No new width/timing sweep, real solve or physical GPU attempt is admitted by
these component results. All G1–G6 remain **OPEN**.
