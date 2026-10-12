# Sparse enclosure in the checked template and block source path

This freeze follows `5e3ee0decb3dac9041393499cada99e4318f0692` and the
[checked sparse row primitive](hz_sparse_rows_20261012.md). It changes one
factor: replace local row projection/enclosure/assembly with the complete
sparse-row enclosure at each source step. Retain once-per-expert templates and
direct common/private block support. Do not fall back to per-pair propagation
or materialized joint support to simplify this integration.

## Source and mathematical contract

Use a separate opt-in source package and context identity. Before accepting
each enclosure, independently check the input box, exact affine reference,
ReLU equations or conditional route reference. The affine nominal is actual
SparseHZono propagation with checked exact error compensation. Preserve the
existing ReLU interleaved/blocked row permutation. Then check the new enclosure
and its actual SparseHZono coefficient snapshot. A hash match alone is not a
source inclusion proof.

The common template domain restores the original input expression and retains
the full router. Propagate each expert once on that common domain. Pair guards
add conditional inequalities without treating them as globally true, or
transplanting pair-specific tightened ranges into the common templates.
Guard enclosures must retain exactly the common factor inventory and ownership;
unexpected extra guard factors are refused by this version. Each expert's new
error/activation factors stay private; shared identities and binary offsets
survive every source step and block query.

All unordered tie-legal pairs, properties and gate endpoints remain required.
Generate fresh requests, matrices and candidate evidence in both arms. Use the
unchanged CPU block algorithm, 128 updates and `1/10000000` threshold. New schema
and context hashes reject transplanted old candidates even if coefficients
coincide. Missing evidence stays UNKNOWN. No native optimizer, GPU, real model,
new training, source selection, relation ablation or performance search belongs
to this stage.

## Admission and counting

The new source path accepts only the intersection of existing source/global
limits and the new row limits: the old 128-factor, 128-output and 256-total-
constraint checks remain, and each sparse lift additionally has at most sixteen
outputs, sixteen equalities and sixteen inequalities. All endpoint, expert,
property and package limits remain. Refuse incompatible objects; do not silently
split rows, switch formats or raise limits. The 3072-factor given-row control
does not admit a 3072-output input box.

Read-only inspection of the archived four declarations finds maxima no larger
than four outputs, three equalities and eight inequalities per source step;
they need no row splitting. The new receipt derives row counts from its bound
reference and reports core source/target parses and residual/layout visits.
Exact source-step checks, ownership checks and snapshots still parse source
states separately; do not claim one parse for the whole network proof.

## Fixed execution and controls

Use the four declarations and exact hashes in
`configs/hz_source_representation_20261001.json`, with old block-source and new
sparse block-source arms. Run cases in that file's order, reversing arm order
on odd case positions. Each arm recreates its own source and proofs. This is
eight full calls; required duties remain 3/18/6/6 and endpoints 6/18/6/12 per arm.
Record all outcomes; the unsafe control cannot become positive. Compare every
source target, factor inventory, block domain/objective and final endpoint bound,
not only aggregate positive counts.

Retain the three existing affine/ReLU/guard rounding bridges as old/new
operator differentials (six calls), with the original exact coefficients.
These are necessary because the old four declarations used zero nonzero lift
compensation. Reuse the old dense nine-factor single-row reference only to
check old refusal/new acceptance, without a model or support query. Do not
repeat the 3072-factor width study. Derive one partial package per arm by
removing the last pair's candidates from tied_partial_reuse; no extra proposal.

Freeze twelve control groups: complete roster; full numerical differential;
template/block dispatch preservation; exact source-reference mutations;
sparse enclosure/factor mutations; three operator bridges; dense row and
intersection capacity refusal; partial and stale evidence; deadline faults;
late source/package identity faults; producer independence; complete archive
and cost accounting. Concretely bind each negative query and expected rejection,
so an unrelated exception or unexpected timeout cannot stand in for the intended
check. Include source/affine/ReLU/guard, owner/frame, layer/template/pair omission,
property/gate, factor map, inward compensation, missing rows, old proof reuse,
17-output and 129-factor refusals. Source mutation after construction must
terminate before additional proposals; checker mutation after endpoint checking
must not return success. Expired producer/checker and delayed live snapshot are
separate fault controls with phase evidence.

Each complete arm includes creation, construction, proposal, serialization and
checking within one cooperative 300-second budget; each auxiliary call is capped
at 30 seconds. Imports, archive publication and all auxiliary operations are
separately charged. One suite envelope is 1800 seconds. At most three preserved
attempts may repair implementation/audit errors; no automatic retries, source
changes, wider controls, timing repeats or result-directed tuning.

Save actual inputs, packages, bridges, partials, negative objects and ordered
cost intervals. The independent process reconstructs the frozen inventory and
checks all saved evidence without calling source, enclosure or candidate
producers or an optimizer. Do not inherit stdlib-only/portable or hard-watchdog
claims from a different schema. Current library package imports may load numeric
libraries even on a checking-only path; disclose that boundary.

## Exit decision

On a mismatch, stop and repair within the fixed attempt envelope. Do not tune
coefficients, iterations, gates or tolerance to restore a desired outcome.
On success, this version connects the sparse enclosure to the existing
MoE-specific template/block algorithm. It does not by itself establish real
capacity, tighter bounds, speedup, CUDA benefit or a new real certificate.

The next gate is a bounded end-to-end capacity and compatibility decision for
this combined source/block path, including full row/property inventories,
source storage and operators. Do not automatically add another supervisor,
raise old caps or rerun sealed inputs. Physical device execution remains a
separate algorithm-compatible stage with fresh admission. G1–G6 stay OPEN.
