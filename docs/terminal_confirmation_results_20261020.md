# Terminal confirmation compatibility result

The once-only V3 compatibility batch **failed overall**. All 27 deterministic
controls passed, as did the standard-library child. The native-library child
still lacked accepted terminal evidence when RSS was unavailable. The third
child was not launched. The protocol stopped without retry or limit changes.

Execution HEAD: `7fbc55c8fcbe85314a4892535516e7d30d24446c`.
Freeze SHA-256: `51e83fdc346c721931db795956b1aaccbaf4badf379d691a0b84d81e3c8d2b0b`.
See the [frozen protocol](terminal_confirmation_control_20261020.md),
[lossless saved records](terminal_confirmation_records_20261020.json) and
[offline record audit](terminal_confirmation_saved_audit_20261020.json).

## Observed outcomes

| Item | Outcome | Execution and cleanup cost |
| --- | --- | --- |
| 27 deterministic cases | 27 PASS | Included in batch cost |
| Standard-library child | COMPLETED, exit 0 | 0.1412 s + 0.0337 s |
| Native import and CPU zeros child | ERROR, final exit 0 | 3.9196 s + 0.0481 s |
| Native explicit exit 7 child | Not executed after failure | No execution cost |
| Online saved-record audit | Consistent records, not accepted | Included in batch cost |

The batch reached reception after 4.3007 seconds; the export timestamp was
4.3020 seconds after the first Python monotonic read. Interpreter startup
before that read and the last timestamp's own write/exit tail are not measured
as full cold-start cost. Peak sampled caller-plus-session RSS was 678,932,480
bytes (647.48 MiB). Original result and reception files total 16,152 bytes.
Algorithm Work was zero; all executed control/import/observation work still
consumed the common wall-clock budget.

The deterministic controls covered missing RSS with later identity-bound
readiness, live/unknown rejection, wrong identities and events, permission and
parsing failures, actual post-poll timeout, a live peer, retained partial
evidence and bounded records. These private fixtures are not observations of a
real native shutdown. The real children recorded **zero accepted V3 terminal
confirmations**.

## Failure boundary

The native child's first and last stat observations both reported state `R`,
SID 1966899 and start ticks 1133644606. Its status record was readable
(1,156 bytes) but lacked VmRSS. The optional confirmation path raised
`live process RSS unavailable: terminal unconfirmed`.

This establishes failure to obtain the frozen acceptable terminal evidence,
not a unique kernel/native-library cause. The failure record does not include
the raw poll event list, so it cannot distinguish no readiness from another
rejected event pattern. The stored observations do not justify simply extending
the poll wait or treating absent RSS as zero.

Both launched children were reaped through the existing owned cleanup path,
with no live session members in their cleanup receipts. The native child
eventually had exit code 0; no execution acceptance followed. The batch-level
final FD-balance and direct-child assertions were not reached after its
unexpected outcome. Individual deterministic descriptor-close assertions did
pass. Neither successful cleanup nor the offline audit fills those missing
final batch checks.

The offline audit verifies frozen source hashes, original record hashes,
rosters, costs, failure and missing suffix. It confirms an overall rejected
attempt, not a successfully repaired execution path.

## Research consequence and stop decision

The authorized compatibility attempt is consumed and sealed. V3 stays
default-off. Do not retry this batch, lengthen the wait, rerun M1b/M3 or start
M2, a trained model, external runtime or GPU under this authorization.
Earlier M1b/M3 FAILs remain intact.

There is no new algorithm, full-request certificate, speed or external-win
claim from this stage. G1–G6 remain OPEN. The next research decision should
return to the incremental value of gate–input coupling against the existing
N1 and a semantically matched strong external path; the
[completed G4 semantic audit](genbab_guarded_pair_semantics_20261020.md)
already identifies that comparison. Any external adaptation/execution and
any different native resource-observation contract need their own bounded
approval. This result does not justify an automatic observation-repair chain.
