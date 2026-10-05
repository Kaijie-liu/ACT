# C3 closure and default-off BN loader checkpoint

Date: 2026-09-05. Branch: `redu-hz`. Base commit:
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`.

## Current C3 decision

The read-only negative preflight rebuilt the exact pinned Tiny143 source and
19-edge repaired clone. It expanded every operand occurrence back to the
nearest C3-permitted source (INPUT, INPUT_SPEC, RELU), with no caller-selected
affine cut and no omitted branch. The four complete paths into ReLU36 contain
one legal necessary shape and three nested-ADD shapes. Since the frozen rule
requires every term to match, this target is a structural zero-hit. C3-v1 is
closed with gain 0. There is no jump to ReLU63 or another iid under this rule.

This negative proof does not depend on unfinished runtime migration: even
granting every preceding RELU a fresh exact source leaves nested ADDs. Longer
phase-sliced histories cannot remove these events. An ADD materialization cut
is outside this adapter's frozen source grammar. The proof grants no positive
planning authority and says nothing against a separately registered rule that
supports a nested affine DAG.

The exclusive evidence is `evidence/s0_c3_graph_preflight_20260905_v1.json`,
SHA-256 `7999b02f1af74667a354f59497b9763e2374742bd9d6deead4c53f94c96ae106`.
The negative-screen and concrete-dataflow tests pass 9/9.

## Preserved adapter checkpoint

The prior agents' final isolated adapter, discovered in the current worktree,
passes 264/264 on this date. Source SHA-256:
`c2f33e6df5ba016739d582030266a8a63ea4e6cd07c873dc0aa75dad093d386e`;
test SHA-256:
`e424f82832c63e4e943f24309ef81042ccc7026de5db426d76babc53ea44e31c`.
It now records complete ADD operand multisets, exact SparseHZono schemas,
factor allocation custody and source/operator/payload mutation checks.

The present agent ran the suite and reviewed the relevant entry points; this
is not a new independent-agent review of the entire final file. It remains
non-executable, unintegrated experimental infrastructure. Its own declared
limits include materializer rebinding, request locking/mutation epochs,
post-return publication and snapshot cost. No such claim is promoted by the
264 test count. Further C3-specific migration is stopped after the zero-hit.

## Production loader prerequisite implemented

`TorchToACT(..., repair_batchnorm_producer_graph=False)` now has an explicit
default-off option. The off path never imports or invokes the repair helper.
The enabled path runs while the loader privately owns the graph, accepts only
complete marked sibling SCALE/BIAS pairs, checks finite exact-width vectors,
clones predecessor maps, checks complete variable-producer flow, and rebuilds
successors. Invalid/unrelated inconsistencies raise before a Net is published.
Already-correct and no-BN graphs keep their original edge-map objects.

This is a correctness prerequisite; it does not change HZ arithmetic or earn
representation gain. No worker or default configuration enables the option.
The loader tests pass 13/13. A dyadic residual example checks all affine
coefficients against an independent closed-form map, keeps its continuous
factors, both binary phases, equality/inequality predicates and frame, and
reconstructs concrete-network outputs exactly.

The real Tiny143 flag-on rebuild changes exactly the same 19 edges as the
independent pinned clone; all 81 layer outputs match that clone byte for byte.
The clone also agrees with its variable program at every layer and its final
output differs from PyTorch by at most 1.0658141036401503e-14 at the one fixed
property-box center. This last check is diagnostic, not universal numeric or
family-retention proof. Evidence:
`evidence/bn_loader_opt_in_20260905_v1.json`, SHA-256
`042e0536892cffa1e9cf19dd05942ce65156afb1d9195fd4fbef792ad8f8a276`.

## Baseline provenance recovery

The complete experiment test invocation reports 730 passed and one failure:
the original V1 live-path manifest test detects that the Overleaf table bytes
changed. Read-only Git inspection finds the original exact table at commit
`01923a5bc896667bd0d0a19b310ee52e88bb2b70`. The change removes revision coloring
and changes whitespace; all 13 family numbers remain the same. The other four
authority files retain their exact original hashes.

The recovered isolated table has the original SHA-256
`de13942115a2a1ba79eb3af82a4c435a4919492c044d174cfdef6be1c41429d0`.
A separate process-local input adapter supplies only that verified original
table metadata to the untouched V1 generator. The entire original 543-row
manifest rebuilds BYTE IDENTICALLY, including its original source hashes,
13-family vector and 2,413/1,870 accounting. Evidence:
`evidence/frozen_authority_recovery_20260905_v1.json`, SHA-256
`7c33afc0872aa9e52c8387222e8fa54de75e06c88ec7e58603035350d19493ba`.
The live-path V1 test is not weakened and still rejects the changed live table.
No historical file was restored, overwritten or edited.

## Score and next work

Formal score remains 1870/2413, E0 remains 61/400. Neither this checkpoint nor
Trial9 ran full-family or 2,413-row retention. Trial9's nine-file source freeze
still verifies; the naturally completed UNKNOWN record remains gain 0.

With C3 closed, the next concrete score qualification is the already-existing
TLL signed/dead-ReLU candidate, previously 29/32 against 17/32. It must first
reproduce under the current source before further cross-family and full replay.
This resumes a measured HZ candidate; it neither changes its width-adaptive
rule nor counts its provisional +12 as formal gain. Corrected Tiny/CIFAR work
still requires a newly registered nested-DAG representation before resumption.
