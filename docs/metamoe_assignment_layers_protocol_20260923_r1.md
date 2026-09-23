# Same-point layer localization, R1

Follow-up to the audited HZ/source point mismatch. Exactly old MNIST0,
expert1, saved factor assignment; no new optimizer calls, assignments, samples,
range refinement, native time, or property queries. New directory only.

Use the frozen replay worker, validate its entire dependency chain, and
redirect only its artifact destination and validated parent lookup. Observe
the expert's original HybridZ transfers without changing them. At every
visited layer independently evaluate the same ACT IR operation concretely in
torch and evaluate the HZ using the fixed final factor vector. Bind the final
matrix to the saved original model before accepting any comparison. Preserve
partial progress on exception/deadline, never as proof.

Budget: 30 s outer wall, 8 GiB sampled group RSS, same 2 GiB representation
policy, pinned CPU/float64 intake, two threads. All capture, original replay,
array serialization, checks and progress publications run inside the worker.
Outer validation/publication separately recorded. No retry or escalation.

Freeze and commit/push code/config after controls. Run once, then audit the
saved arrays, layer ledger, final model check and outer terminal without a
second propagation. A 1e-9 diagnostic comparison labels point discrepancies;
it is NOT a new solver acceptance margin or all-domain equivalence tolerance.

Interpretation: first HZ-versus-concrete-IR discrepancy localizes a propagation
or factor-interpretation issue at that point. If IR/HZ agree but source differs,
conversion remains suspect; this does not identify one converter bug. Multiple
discrepancies may exist. No claim of global soundness/unsoundness or new SAFE.
Do not repair core code or rerun certification within this diagnostic.
