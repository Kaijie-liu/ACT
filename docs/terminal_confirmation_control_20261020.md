# Identity bound terminal confirmation control

This finite compatibility stage addresses the execution observer that rejected
missing RSS in the M1b and M3 attempts. It changes the optional terminal
classification contract, not mathematical acceptance. It is not research
evidence for gate coupling, speed, full-request certification or external
competitiveness. Both earlier overall FAIL records remain unchanged.

## Authorization and exit condition

Starting branch `feat/moe-route-verification`, clean HEAD
`8d1299234282c80a7a036f645ba7d74753f878ca`.
The user replied “批准” to the proposal to extend only the existing observer's
identity-bound terminal confirmation, freeze and run one compatibility batch
within 300 seconds and existing caps. Missing RSS for a process not confirmed
terminal still rejects. No new supervisor, M1b/M3 retry, M2, model, GenBaB or
GPU execution is authorized. Approval is consumed by this single attempt,
including an unexpected failure; no higher limit or repair retry is implicit.

The exit condition is a bounded control result and saved-record audit. Success
only establishes this compatibility stage. It neither authorizes algorithm
reruns nor restores acceptance to the old failed attempts.

## Optional observation contract

The public `execute(..., terminal_confirmation=False)` retains the old V2
default. Explicit `True` selects `CONFIRMED_IDENTITY_V3`. After a same-SID,
same-start-ticks live observation without RSS, V3:

1. Opens a pidfd and rereads stat to bind that descriptor to the observed
   identity. Disappearance, permission failures or changed identity reject.
2. Polls that FD once for at most 10 ms and the remaining existing deadline.
   Only its POLLIN, optionally accompanied by HUP, is accepted. HUP alone,
   ERR, NVAL, another FD, multiple events or no event reject.
3. Closes the descriptor on success and failure. It sends no signal and does
   not reap the observed process.

The record retains the original missing observation and `rss=null` in
`terminal_without_rss`. Such a process is not renamed a zombie. The aggregate
RSS field sums the observed live processes only; it does not reconstruct a
missing historical sample or bound an unsampled peak. A bounded, deduplicated
ledger retains terminal evidence across later scans and cleanup, including a
partial scan failure. Overflow rejects rather than truncates.

This is a finite explicit OS/pidfd identity and terminal-state assumption,
consistent with the existing pidfd cleanup contract. It is not proof of a
unique cause of the historical failure. EOF, leader exit status, final session
scan, live descendant rejection, cleanup and online reception remain required.
Later successful cleanup cannot override an earlier execution error.

## Frozen controls and costs

The JSON freeze binds the code, this protocol, all 27 deterministic cases and
three public-supervisor calls. Deterministic cases cover normal RSS, the old
default rejection, Z/X/x, missing and disappeared status with later terminal
confirmation, unknown-live and invalid readiness, identity failures, permission
and parsing errors, deadlines, a live peer, partial evidence, first-error
retention and the evidence cap. They use private dependency seams, not global
monkeypatching. Exact missing-event records and descriptor closure are checked.

The only compatibility payload processes are, in order (bounded Git identity
checks also create short metadata subprocesses during admission):

| Call | Payload | Required result |
| --- | --- | --- |
| stdlib zero | Standard-library import and print | COMPLETED, exit 0 |
| native zero | Existing NumPy, SciPy and Torch imports, one-element CPU zeros | COMPLETED, exit 0 |
| native seven | Same fixed imports and CPU allocations, explicit exit 7 | ERROR, exit 7, no observation error |

No model, forward pass, support query, optimizer or solver is invoked. CUDA is
not initialized or used. Native imports can map installed library dependencies;
that is not GPU execution. A successful real child need not trigger the missing
RSS branch: actual confirmation counts and deterministic branch coverage are
reported separately.

The measurement begins at the script's first Python monotonic read, before its
remaining standard-library imports. Interpreter startup before that boundary
is unknown, not claimed as full cold-start time. No preloaded measurement host
is introduced. All subsequent setup, admission, imports, child startup,
observation, control execution, cleanup, serialization and online audit are
charged to the same 300-second alarm. Admission is once, within 60 seconds;
control work stops at 240 seconds, reserving the rest for reception/export.
The last export timestamp precedes its own small write and process exit; that
tail is not claimed as exactly measured.

Caps remain 8M algorithm Work, 128 MiB artifacts, 8 GiB address space per
managed process and a 2 GiB sampled caller-plus-session RSS stop. These controls
consume zero algorithm Work, not zero wall time. RSS sampling is not a hard
aggregate resident-memory guarantee. Log and record caps remain bounded.
Each real child has a 45-second self alarm, at most 50 seconds run allowance
and 55 seconds through cleanup, also bounded by the common deadline.

## Execution and audit

Freeze and commit/push before the single attempt. The runner validates hashes,
the environment, clean authorized branch and fresh available memory/disk. It
claims a new exclusive result directory and never resumes. An unexpected
failure stops all remaining cases and records missing suffixes.

The saved-record checker validates scope, roster, status, evidence form,
descriptor/cost/deadline accounting and final cleanup. It runs online under
the same deadline. A later independent read-only audit can verify the records
but cannot turn a missing or failed online reception into success.

Static AST and separate AI source reviews precede execution. They are code
checks, not independent human review or evidence that real native shutdown
will succeed. Final records, failures and costs are archived without rerunning
the cases.
