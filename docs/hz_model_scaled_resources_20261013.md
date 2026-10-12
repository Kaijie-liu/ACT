# Model scaled resource policy proposal

The user approved drafting and statically assessing a new model-scaled resource
policy on 2026-10-13. This is **design approval only**, not a change to any live
limit, a control-execution freeze, or permission to load a trained model. The
300-second request budget, complete obligations and numerical acceptance rules
remain unchanged. Old 8M and 96M protocols/results keep their original identities.

The assessment finds that **raising Work alone cannot admit the registered
eight-expert model shape**. The current forest repeats input root lists across
views; the complete root roster necessarily exceeds its separate limit. A new
policy must specify a resource vector and a new bound identity, not patch one
global integer. Full Work, encoded-byte and resident-memory bounds remain open;
there is no justified numerical Work quota or completion-time prediction yet.

## Scope and checked metadata

Starting branch: `feat/moe-route-verification`, clean HEAD
`3ae08e36b8f2c4eb3ff4206ba6512b792b770f40`.
The [static assessment](hz_model_scaled_resources_20261013.json) binds the
registered seed 1/2 training topology, factory and relevant producer/checker
code. It does not establish seed 0 topology or checkpoint correspondence.
The [assessment script](../scripts/assess_model_scaled_resources.py) imports only
standard-library analysis code. No captured source, weights, input arrays,
network, optimizer or GPU are evaluated.

The fixed shape has 3,072 inputs, a 128-hidden-unit router, eight
256/128-hidden-unit experts and ten shared classes. Source semantics remain
eval, selected-softmax top-2 with all tie-legal unordered sets covered. The
current conservative endpoint path retains all 28 pairs; that roster does not
establish that all 28 are reachable or that a route flip occurs.

| Shape inventory | Count |
|---|---:|
| First affine weight positions | 6,684,672 |
| Later affine weight positions | 273,408 |
| Affine outputs and biases A | 3,288 |
| ReLU units U | 3,200 |
| Parameters plus input-center coordinates | 6,964,440 |
| Raw binary64 tensor payload only | 55,715,520 bytes |
| Pair guard rows | 336 |
| Complete classification obligations | 252 |
| Distinct endpoints | 252 to 504 |
| Proposed plus explicit zero candidates | Up to 1,008 |
| Pair-local width-eight tiles | Up to 84 |
| Dual values across both candidate roles | Up to 2,721,600 |

These are shape counts or conservative maxima, not actual trained sparsity,
unstable activations or candidate values. Raw tensor bytes exclude source
headers, forest, rational proof, temporary objects and archive copies.

## Resource compatibility

The factory has nine Flatten operations. Every pair view retains all input
output roots, even though those roots refer to shared nodes. Pair views alone
require 28 times 3,072 = 86,016 root entries. Including input, common, Flatten,
affine and ReLU output lists and the mandatory guard rows gives

    roots >= (28 + 2 + 9)*3072 + 3288 + 3200 + 336 = 126632.

Up to three constraint roots per ReLU give 136,232 total roots. The
[checker](../scoped_source/check_source_dag.py) counts roster occurrences, not
unique node IDs. This is an unconditional full-shape refusal under its 65,536
root limit, independently of Work, weights, centers or ReLU stability.

For a conservative structural envelope, allow two affine compensation factors
per output and seven factors/twelve local entries per ReLU. The latter permits
even the inequality-row lifts that exact unit coefficients do not need, so it
is deliberately loose. With n inputs, W1 first-layer weights, W2 later weights,
A affine outputs, U ReLUs and G guard rows:

    F <= n + 2*A + 7*U
    N <= n + A + 4*U + G
    L <= n + W1 + 2*A + 12*U
    R <= W2 + U + 2*G.

| Resource | Current limit | Bound from the registered shape | Interpretation |
|---|---:|---:|---|
| Work | 8,000,000 | At least 13,369,344 first-layer visits | Already insufficient; not the total cost |
| Total view root entries | 65,536 | 126,632 to 136,232 | Definite independent refusal |
| Factors | 32,768 | At most 32,048 | Structural count fits only |
| Active local factors | 16,384 | At most 32,048 | No worst-case capacity contract |
| Nodes | 65,536 | At most 19,496 | Structural count fits only |
| Views | 128 | Exactly 82 | Structural count fits only |
| Local nonzero entries | 1,048,576 | At most 6,732,720 | Actual count unknown; upper exceeds limit |
| Reference edges | 65,536 | At most 277,280 | Actual count unknown; upper exceeds limit |
| Forest bytes | 128 MiB | No complete encoded bound | Open |
| Candidate tile and full proof | 4 MiB and 16 MiB | No complete encoded bounds | Open |
| Entire archived stage | 128 MiB | No complete encoded bound | Open |
| Sampled RSS and address space | 8 GiB and 16 GiB | No complete live-object bound | Open |

An upper bound exceeding a cap is not proof that actual trained execution will
exceed it. By contrast, the first-layer visit floor and mandatory root count
already establish their respective refusals. All actual counts are left null.

## Proposed policy and binding

Keep the current V3 representation and mathematics as the starting point for a
**resource-policy-only** design. Root deduplication or a new tensor/reference
representation would be a separate algorithm factor, not a hidden part of this
policy comparison. No production representation is changed in this stage.

A future version must take an immutable policy from the trusted frozen request
protocol. Its identity binds source topology, complete property/pair inventory,
candidate roles, width/iteration rules, code versions, counters and all resource
limits. The source/proof package cannot grant itself larger limits. Production,
independent source checking, endpoint checking, supervisor reception and portable
checking must agree on this identity. Do not monkeypatch `POLICY`, relabel old
V3 evidence, use a per-tile reset, or splice old positive bounds into new matrices.

For each comparable internal method m, derive full-path Work upper bounds U_m
from its frozen implementation and model shape. The required phase inventory is
source construction, independent source checking, gate/obligation planning,
native preparation/candidate generation, objective pullback, exact dual checking
and support, and serialization/receipt/terminal handling. Keep exactly the
existing row-visit, coefficient-visit and exact-arithmetic counting meanings.
Operations not incrementing Work still consume the wall-clock budget.

The proposed shared quota rule is

    B(D) = 1000000 * ceil(max(8000000, U_reference(D), U_proposed(D))/1000000).

An independently frozen absolute Work ceiling also applies. A missing phase
upper bound yields **no quota**, and B above that ceiling refuses admission.
The [arithmetic controls](../scripts/test_model_scaled_resources.py) exercise
this rule but do not certify supplied derivations. Lower bounds, measured
prefixes and the previous synthetic timing cannot serve as U_m. Both U_m and
the absolute ceiling are currently unset; no arbitrary 96M or multiplier is used.

For the unchanged layout, proposed structural ceilings are the maximum of the
old cap and its shape upper bound: roots 136,232, active local factors 32,048,
local entries 6,732,720 and references 277,280. Factors 32,768, nodes 65,536 and
views 128 need no increase. These are **draft resource-vector values only**;
they are not runtime overrides. Root and node limits should be separate fields,
so allowing more aliases does not silently enlarge every node allocation.

Retain 4,096-bit stored rationals, 8,192-bit intermediate rationals, one CPU
thread, CPU float64 proposals, width at most eight, 128 updates/129 evaluations,
both candidate roles, all source checks and strict bound > 1/10000000. Do not
assume equal scales, zero centers, no compensation or sparse duals. Captured
source bytes, forest/proof/partial evidence, archive size and peak memory need
independent bounds. Their limits are not increased by the Work proposal.

The 300 seconds include startup, source capture/read, propagation/construction,
all proposals/fallbacks, exact acceptance, serialization, reception, cleanup and
terminal publication. The current fixed synthetic runner copies captured data;
it does not establish a charged live-model capture path. That interface remains
an explicit real-execution prerequisite. Current V3 candidate code is CPU-only;
older block-device controls do not authorize silently replacing it with CUDA.

Portable checking reconstructs from raw source and both candidates, with its own
full Work bound and separately reported cost. It cannot finish an incomplete
online request retrospectively. Any deadline, memory, bit, byte or Work failure
preserves its prefix and prevents a complete positive acceptance.

## Comparison and research claims

Old 8M results remain a historical capacity condition. Future algorithm arms
must use the same new policy and 300-second budget on the same source, properties
and guarantee level. Merely changing rejected-under-8M to completed-under-B is
capacity engineering, not algorithmic improvement. A single-method old/new-cap
contrast is a separate policy ablation; a full two-by-two design is needed only
if the algorithm/cap interaction itself is a claim.

For consumer-only attribution, a candidate reference is the same V3 source and
endpoint formulation with scalar versus batched consumption. Source enclosure
changes and native strong baselines must remain distinct comparisons, not be
called same-matrix speedups. Actual method IDs and fair common ledgers must be
fixed before an execution protocol; this document does not freeze that study.

GPU is a later separate factor: compare the same algorithm CPU/GPU with all
transfer, synchronization and independent CPU checks charged. External tools do
not share ACT's custom Work unit; align actual time, hardware opportunity, memory,
task and guarantee grades instead. Preserve strong author configurations.
No new certificate, accelerator gain, external advantage or G1–G6 completion
follows from this resource-policy design.

## Decision and next prerequisite

**DESIGN ONLY, NOT ADMISSIBLE.** Work-only adjustment is insufficient. The next
bounded deliverable is a complete per-phase Work and storage/lifetime ledger
for the registered full topology and an explicitly selected internal reference,
including worst-case permitted centers, scales and compensation. Do not rerun
small sources, source-only prefixes, sealed inputs or physical GPU attempts as
a substitute. If mandatory storage or memory remains over the retained envelope,
stop that policy proposal and state the particular resource change or separate
representation decision required; do not conceal it in a Work increment.

Only after the ledger closes can a separately reviewed implementation freeze
introduce a versioned policy interface and its deadline, identity, partial-evidence
and cost controls. An eventual full capacity call and real experiment each need
their own finite execution scope. No current protocol is edited or reopened.

## Checks

Nine new standard-library arithmetic/identity tests and eighteen parent and
manuscript preservation regressions pass. Two read-only code reviews independently
recomputed the report and repeated the nine new tests; all twenty existing source
anchors match. Navigation checking preserves all 619 frozen source identities.
These AI reviews do not substitute for independent human technical review.
The compact report is reproducible
with the existing environment and no site packages:

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -I -S -c 'import sys; sys.path.insert(0,"/data1/Kane/MOE/ACT"); from scripts.assess_model_scaled_resources import analyze; import json; from pathlib import Path; assert analyze()==json.loads(Path("/data1/Kane/MOE/ACT/docs/hz_model_scaled_resources_20261013.json").read_bytes()); print("PASS: static proposal only")'
```

These checks validate arithmetic, source identity and refusal boundaries. They
are not model execution, a performance experiment or independent network proof.
