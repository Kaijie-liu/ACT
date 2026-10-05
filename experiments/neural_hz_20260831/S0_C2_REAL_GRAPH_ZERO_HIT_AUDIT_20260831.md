# S0-C2 Real-Graph Zero-Hit Audit

Recorded on 2026-08-31 on branch `redu-hz` at repository commit
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`. This read-only audit applies the
frozen S0-C2-v1 rule to the actual ACT graph and lazy-cache evidence for the
fixed TinyImageNet iid143 ReLU36 target. It supersedes the earlier synthetic
shape assumption but does not modify the preregistration.

The result is deterministic: S0-C2-v1 is a structural zero-hit and is closed
with formal and E0 gain zero. No production attempt or target benchmark is
authorized for this version. The formal baseline remains exactly 1,870/2,413
and every one of the 13 family counts remains unchanged.

## Authoritative evidence

The source record is
`results/trial8_phase_selective__tinyimagenet_2024__iid143__relu63_v1.json`,
SHA-256
`f1f6abcdc87d96e9d8898d0f667d8087b2a72a9898f44d61b4e123426d45b274`.
Its real layer-28--36 graph is:

```text
ReLU28 -> Conv29 -> Scale30       (no successor)
                 \-> Bias31 -> ADD32

ADD32 -> Conv33 -> Scale34        (no successor)
                \-> Bias35 -> ReLU36

ADD32 also has a second successor, layer 40.
```

The exact rows are:

| id | kind | successors | lazy terms | lazy operator entries |
|---:|---|---|---:|---:|
| 28 | ReLU | 29 | - | - |
| 29 | Conv2D | 30,31 | 1 | 26,239,488 |
| 30 | Scale | none | 1 | 26,264,576 |
| 31 | Bias | 32 | 1 | 26,239,488 |
| 32 | Add | 33,40 | 4 | 80,273,920 |
| 33 | Conv2D | 34,35 | - | - |
| 34 | Scale | none | - | - |
| 35 | Bias | 36 | - | - |
| 36 | ReLU | 37 | 5 | 84,880,604 |

The return from Bias31 to exactly Conv29's operator-entry count, rather than
Scale30's larger count, independently proves that the Scale expression is not
the expression consumed by ADD32. The same sibling pattern holds at Conv33.

A separate read-only rebuild of iid143 inspected both graph edges and layer
variable IDs. It found:

```text
Conv29 out begins 773056
Scale30 in/out begin 773056 / 798144
Bias31  in begins 798144, but graph predecessor is Conv29

Conv33 out begins 873408
Scale34 in/out begin 873408 / 898496
Bias35  in begins 898496, but graph predecessor is Conv33
```

Thus the sequential variable program says `Conv -> Scale -> Bias`, while
`preds/succs` says Scale and Bias are siblings. The omitted maps are not
identity: all 25,088 Scale30 entries differ from one, with range
`[-0.04747119835380177, 2.1341762976063428]`; all 25,088 Scale34 entries
differ from one, with range
`[0.00014627055224732068, 0.02266037406504889]`.

The production sources used by this audit have SHA-256 values:

```text
d6cb15ec6b0dedef0d30bd0441f1dc98e02f6202bcb1e279c337d219c7c33d44  act/pipeline/verification/torch2act.py
8c2ebc9c3cfac78360af36770f54990704c2e031d4c42ddb58532e515c6762a9  act/back_end/hybridz_tf/tf_cnn.py
234b04b151d640f8fc859fab00729448ba533d8feb3679427cbadb94467ec776  TinyImageNet_resnet_medium.onnx
d105a0c7ca711eb46b9f20ab772a0564444569d06f6206bca0e95d5a19990cca  iid143 converted VNNLIB
```

## Root cause and frozen-rule result

TorchToACT expands BatchNorm into Scale followed by Bias, but maps the FX
BatchNorm node to the final Bias layer. During predecessor construction the
unmapped Scale is connected to the preceding Conv, while the mapped Bias is
connected directly to that same FX Conv predecessor. Scale and Bias therefore
become siblings in the graph even though their variable IDs were created
sequentially. The deferred-ReLU island accepts the Bias-to-ReLU path and treats
the successorless Scale as a dead sibling.

Consequently the actual main lazy core at ReLU36 is:

```text
Conv29 | ADD32 | Conv33
```

The other three ADD32 terms are prior prefixes followed by the shared Conv33.
The common ADD occurrence and shared outer Conv can be proved; the decisive
failure is the main branch segment before ADD32.

S0-C2-v1 froze the nonidentity segment as one implicit Conv followed by one or
more channel-stationary diagonals. The actual single-Conv segment therefore
returns `nonempty_branch_without_complete_chain`. Identity skips cannot repair
a missing complete branch. Synthetic tests containing
`Conv29 -> diagonal -> ADD32 -> Conv33` remain valid operator capability tests,
but they are not evidence of a real target match.

Changing the cardinality from `diagonal+` to `diagonal*` after observing the
target would expand the selector and reverse a registered closure condition.
It cannot be called a clarification or patch to C2-v1. More importantly, the
empty graph-operator segment is not certified semantic identity here: a
nonidentity Scale exists in the variable producer chain. A future abstract
identity-middle rule needs a graph-event/operator bijection certificate and
must fail closed on this target until the loader graph is corrected.

Likewise, repairing the BatchNorm graph is not C2 plumbing. It restores a
model operation currently absent from this graph path and may change the
2,413-row baseline. Such a loader repair requires an independent campaign and
a complete baseline rebuild; it cannot be mixed into a Neural-HZ gain against
the frozen 1,870.

## Independent whole-state risk inherited by any successor

ADD32 has successors 33 and 40. At the ReLU36 boundary, consuming the Conv33
path cannot release `_sparse_affine_expr_cache[32]`; layer 40 still owns one
future consumer. A path-specific candidate must not write a Conv33 descriptor
back into the ADD32 expression because that would change the layer-40 path.
The original C29 and three prefix roots therefore remain strongly reachable.

A new composed descriptor and strong CSR artifact are additional roots unless
they are request-local and released. Even if released, an otherwise identical
materialized HZ may make the persistent before/after states equal rather than
strictly smaller. Any successor must measure this at the same simulated
consumer-GC boundary and require both resident bytes and entries to decrease.
It may not omit the ADD32 alias, use a global operator-id substitution or count
logical expanded nnz as physical reduction.

## Reusable work and no-claims

The C2 pure lineage planner, hardened descriptor V2, request-local CSR artifact
transaction and whole-state ledger prototype remain reusable isolated
infrastructure. Their unit tests establish mechanics only. They do not create
a C2 target hit, actual V2 emission, production lineage, whole-state reduction,
family replay or score.

Trial 9 continues under its original source freeze and may seal its independent
runtime census after exit. Its result cannot change this graph proof or give
C2-v1 gain credit.
