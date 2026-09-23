# Scoped lAs compatibility repair and freeze-only followup

Observation reproduced the old MNIST7 failure. Initial domain keys9 include
5 expert layers with exactly zero coefficients after property pruning. Only
the router dominance property remains (`C=[-1,1,0,...,0]`). The next backward
pass omits all5 expert lAs as None, returns4 router layers and the unchanged
domain store asserts9!=4. Do not delete that assertion or discard properties.
The saved-only [observation archive](metamoe_las_observe_archive_20260923_r1.json)
binds the reproduced error, physical box/specification equality and costs.

The explicit wrapper registers the complete initial node/shape/dtype/property
schema. It restores missing zero entries ONLY if they were zero initially,
are None currently, and the node is not an ancestor of ANY nonzero current
property block in the final axis1 concat. Exact zero test, union across all
properties; no tiny-coefficient threshold. Graph identity, C, feature shapes,
returned keys and dtype must agree. Unsupported/active/mismapped omissions
remain ERROR. Only branching metadata changes; original lAs are reused by
identity, original bound tensors/optimizers/options/acceptance untouched.
Original domain insertion still runs and its original assertion remains.
All restorations and successful insertions are logged. External repositories
are not edited and no dependencies are installed/upgraded.

Controls: mutation/refusal/differential tests, deadline/partial/error/cost
supervision, then exactly two author requests: old MNIST1 (unchanged positive
path) and old MNIST7 (previous error). Same model/physical input/19global
output rows/router/nonzero obligations, CPUfloat64/two threads/300s/8GiB.
Whole request includes wrapper import, checks, logging, loading and solving.
Parent inventory/audit are separate and recorded. No retries, extra radius,
samples, alpha search or bound tuning; ERROR fail-stops the roster.

Gate requires saved audit+replay, MNIST1 numerical positive, and MNIST7
reaching at least one justified restoration AND one original domain insertion
without ERROR. A properly accounted TIMEOUT/UNKNOWN can pass compatibility,
not efficacy; no requirement to manufacture a positive result. Partial logs
cannot establish the gate. Raw results/errors/timeout remain visible.

Only after that gate: freeze (do NOT execute) a new full20-call version using
the SAME10 original clean-correct inputs and rotating order. Keep ACT unchanged
and apply the disclosed wrapper only to the author path. Do not splice old14
normal results into a mixed-version table. This is observed-cohort followup,
not new holdout. R1 stays sealed. No claim of unchanged literal author code
execution after repair: label the compatibility wrapper explicitly. Numerical
filters, HZ-policy acceptance and source-complete proofs remain distinct.
