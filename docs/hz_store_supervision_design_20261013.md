# Checked-store endpoint supervision: finite CPU interface freeze

This stage connects the immutable-source endpoint schema at `1b271e2cf` to
`scoped_proof.device_lifecycle` and its owned-process cleanup. It changes no
mathematics, optimizer, thresholds, endpoint inventory or model. There is no
GPU admission, real request, trained model, width study or performance claim.

## Fixed calls and acceptance

Three normal calls: `half`, `negative`, `partial`. The half source has one input
in [-1,1], constant router (0,1,0), and three identical experts
`(2+ReLU(x), ReLU(x)/32)`. Label 0, zero margin, ANY_LEGAL_TOPK,
selected-softmax top-2. All three pairs remain obligations, even the infeasible
(0,2) pair. Their gate intervals are [0,1/2], [1/2,1/2], [1/2,1]; three properties
need five distinct endpoints. `negative` is exactly the prior negative3 recipe.
`partial` uses half but explicitly omits the final endpoint candidate: execution
can complete, mathematical aggregation must be UNKNOWN. Tile width stays eight.
No missing file is silently replaced by an explicit missing candidate.

Fault calls use half only: delays in numerical import, source construction,
candidate generation, tile serialization, independent checking and reception;
exceptions in producer and checker; invalid candidate identity; missing checker
receipt. These ten fault calls plus three normal calls are the whole roster.
Every invocation has a single 300-second parent clock. Faults additionally cap
the targeted phase at 15 seconds, without changing the overall budget. Fault
reach is required; unrelated import failures do not count as successful controls.

The frozen endpoint algorithm is CPU float64, one thread, 128 projected dual
updates and exact selection against zero duals. Complete source and all endpoint
checks remain enabled. Positive acceptance retains the 1e-7 exact threshold.
The new controls cover both previously unexercised half-gate orientations. They
do not establish relational benefit or generality beyond declared affine/ReLU.

## Binding and cost contract

Bind source declaration and complete classification property before generation,
then bind store, contract and proof manifests in small parent-anchored receipts.
Use the existing five phases: admit, produce, release, check, receive. Only the
producer may import numerical libraries. Check and receive run isolated Python
with `-I -S`, using a fixed, hashed repository stdlib module; the existing relocated
checker remains independently tested. No fresh relative checking budget.

Carry the producer's cumulative exact-work counters into the checker, with the
same 96-million invocation ceiling. Checkpoint hashes are not used. Interrupted
work counts are explicitly incomplete, not zero. Independent offline rechecking
has a separate clock and must never upgrade the online terminal.

Publish tiles atomically and exclusively. A small progress journal records only
completed tile identities, inventory position and counters; it is diagnostic,
not a substitute for the final manifest. No automatic completion of interrupted
proofs. Completed invocations bind all nested evidence files and total bytes;
failed invocations retain a bounded inventory of partial files. No full source
copies per pair and no numerical proof arrays inside the small receipt.

Count imports, source creation, exact source validation, proposals, exact dual
selection, serialization, independent checking, receipt validation, cleanup,
terminal publication and observed API return. Interrupted operations are right
censored. A late API is not accepted. Process-group cleanup and sampled RSS are
not cgroup hard-memory enforcement or a guarantee about escaped daemons/OS I/O.

First run pure controls for wrong source/property/tile binding, absent obligations,
cumulative counter overflow, nested inventory contamination, late observations,
unknown cleanup, API exceptions and cost mutations. Then freeze implementation
hashes and execute the thirteen calls in a new directory. Stop after unconfirmed
cleanup/API interruption and preserve pending slots. No implicit retry or budget
increase. Archive failure before correcting an implementation error.

The stage cap is 64 MiB of regular-file payload, checked before calls and at the
end, not an OS disk quota. This finite protocol does not admit the earlier large
source/given-HZ controls or any sealed input. All G1–G6 remain OPEN.
