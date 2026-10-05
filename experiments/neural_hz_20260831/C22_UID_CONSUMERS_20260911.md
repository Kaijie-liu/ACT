# C22 reader audit before any graph-field retirement

Previous goal turn is PROGRESS: C21 completed a fresh generator and all source/
ownership proofs, then correctly failed the full entry gate. C21 remains CLOSED.
This note records read-only current-source inspection, not proof-state deletion.

| Reader | Required data | Consequence |
| --- | --- | --- |
| C17 OwnedIntegrated.fingerprint / C10 portable identity | all node fields, source/operator sharing, HZ and maps | Cannot remove arrays without a new exact content/proof binding |
| C9 independent original-affine audit | support, needed, slots, exponents | Must finish the FULL original proof before any retirement; no bare exact flag |
| C10 fused quotient audit | original graph for its nested C9 proof; tagged maps for quotient | Old proof entry point cannot silently accept an incomplete graph |
| C17 row_uid_tables | node widths, needed and slots | Needs a new exact UID/physical-row index |
| C17 phase_event_audit | row_uid_tables, ownership, maps, complete post oracle | Index replacement plus exact append-event proof required |
| C10 reconstruct_fraction | candidate.validate, dimensions, alias tags/ratios | Does not directly read node arrays, but validation still must bind a genuine completed proof |
| C15 unit-splice reconstruction | final HZ plus compact splice certificate and original prefix | No direct node use; not proof that its native/live integration is done |

The four node fields are therefore NOT simply dead data. Mathematical checking
and state authentication use them. A future closed proof representation needs
an actual independent-checker-produced source/HZ receipt before retiring those
dependencies, and a mutation/reseal-resistant binding afterward. No receipt or
graph retirement is implemented in C22.

Stable original EQ and INEQ UIDs already resolve through existing old prefix
maps; radix UIDs resolve through existing def_rows and the scalar radix base.
Surviving MAIN UIDs follow increasing node/coordinate order, with physical
interruptions at radix rows and UID holes from unused coordinates/erased aliases.
Consecutive UID+physical-row runs are therefore an exact compact index candidate.
A single60-bit word stores20-bit UID start,20-bit physical-row start and20-bit
(length-1). Run starts are monotone in both coordinates, permitting binary
search in either direction with explicit end checks. No hash/digest alone proves
membership: every occupied row AND every unused/erased UID must be checked.

Do not assume the run count or that an additional scan fits140566 work headroom.
C22 first implements/tests a fully priced standalone index and measures its
complete real population from a successful source archive. The prototype keeps
ALL original graph arrays and does not claim HZ/whole-state reduction. If later
fusion can reuse paid row-emission/filtering traversal, it requires a genuinely
different implementation and exact operation accounting, not a lower price for
this standalone scan. C19 also proved the all-row original C10 sort tariff is
6440928, below C21's order-check+normalization6556384 by115456; a future fixed
whole-rule choice may use that evidence, but cannot select per iid or rescue
a failed budget. No such generator has been implemented or run here.
