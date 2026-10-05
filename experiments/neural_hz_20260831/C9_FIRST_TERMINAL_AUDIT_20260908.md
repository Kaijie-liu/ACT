# C9 first end-to-end terminal attempt: UNKNOWN, gain0

Branch redu-hz. Results in results/c9_first_terminal_20260908_v1/ are terminal:
384 tests passed in4.60s; test/worker exits0; source/provenance drift false.
Supervisor wall208.49386645201594s under the frozen240s cap. Full pipeline
traversed the original network through final DENSE79/ASSERT80; no intentional
layer stop, no substituted checkpoint, no input/phase/solver-status repair.

## Passed representation and terminal-entry gates

The SAME frozen c9_live_runtime_v1 constructed its full original-source DAG
and used unchanged native ReLU. Construction89.2298339s, conservative resident
growth955437056 bytes; native ReLU1.7324386s, growth248025088 bytes. Both original
1GiB gates passed. Fresh post-ReLU content matched the fully audited live HZ:
8549d44ed7254710e86d94dd9ce8be3d9bc279246fe78f94f9a56cba25b46fe7.
Ordered phase slots/global widths matched, old slots preserved, no overlap.
Eight focused binding guards plus376 existing tests passed before this run.

At real terminal entry,814 complete numeric roots charge original model/input/
spec, full live TF/cache state, retained definition graph/maps/value views and
actual final affine HZ.480553316 bytes /45519319 entries are strictly below
the unchanged reference lower bound629346312 bytes /52428800 entries. Same
selection rule chose the same distinct-content tied leaves10 and21 as the
fresh live run. Reference construction bounds523595776 and523612160 bytes
passed; every original reference row checked bitwise.

Final HZ native ingestion retained11161930/11161930 nonzeros; all coefficients,
row/column bounds and integrality unchanged. Final Gc has40000 entries, minimum
nonzero absolute coefficient1.0077427424251098e-5. No tolerance/option changes.
Native inspection here concerns the lowered base HZ; no property-clause MILP
was reached in this run, so it is not evidence about such an augmented matrix.
terminal_gate.json SHA:
698d314aa3740c03607aeaf7b4c7c6db459a91184004d0b657441511b765427e.

## Ordinary terminal outcome

The unchanged HZSolver was actually called after all above checks. Output:
UNKNOWN, reason base_unknown, one base-feasibility solve, nodes0. Lowered model:
254965 continuous /1350 binary variables,247240 rows,11161930 predicate nnz.
Only333 unused continuous factors were pruned. No predicate projection, phase
fixing, binary pruning or row coalescing occurred. No concrete witness was
returned, and concrete_validations is empty. No property decision was reached.

Configured terminal budget remained45s. Ordinary evaluate_spec wall time was
61.57687197718769s; the underlying evaluator reports58.64510485716164s after
lowering. The outer240s deadline was not exceeded. Do NOT describe45s as an
observed strict wall duration, silently enlarge it, or infer a specific raw
HiGHS status from elapsed time. The current native wrapper records only
base_unknown here, not the raw solver return message/status. Thus timeout,
numerical rejection and absence of an accepted feasible point are not strictly
distinguished by this log. It does NOT prove mathematical infeasibility.

The shadow worker's solver_s97.989115 includes the preceding diagnostic
terminal gate; it is NOT pure solver speed. This run is no concurrency or
performance-promotion measurement. Diagnostic peak RSS including oracles and
native solving3765388KiB is distinct from the numeric ledger and the per-
construction1GiB cap.

## Immutable artifacts and next action

result.json SHA:
b432becfa9e6dd88149587b9c594e9a3c6e5794515abeb3b898d338c558017d9.
terminal_audit.json SHA:
b33c916db90883a1694985750bebd391e3979d9a2be9b3aebf9607ed4ac04d6b.
Final HZ content SHA:
b2024dfbe2af20f7c9e729bf79c835d5cf7b2050ab57113cc33bebe78f9801c8.
final_hz.pickle138648040 bytes, SHA:
841af01fb74ffa8cfdb7ac434a4f8da632739d866983f2b44ed0f34c353ed0b0.

This terminal attempt is CLOSED with capability gain0, not promoted and not
retried under changed parameters. Its passed live/construction evidence stays
valid; the earlier offline failure also stays failed. The blocker has moved
from inability to construct ReLU78 to ordinary base-HZ feasibility at this
model size. Next safe step is a preregistered read-only structural census of
the saved defining predicates: identify dead value factors with uniquely
defining equalities whose exact elimination can strictly reduce total nnz
without binary elimination, fill growth or loss of prefix reconstruction.
Do not implement LP-status-dependent repair, invoke dual rescue, or extend to
CIFAR/shadows as if a terminal capability success had occurred. Further raw
solver diagnosis, if needed, must be separately preregistered and cannot
change the recorded verdict or count as representation gain.

Formal1870/2413 and separate E061/400 unchanged. No old historical data was
written, no production default enabled, no commit/push performed. Overall goal
remains active and incomplete. All13 preceding archival manifests verified.
