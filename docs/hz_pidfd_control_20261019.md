# Local pidfd compatibility — one finite CPU control stage

## Decision and scope

The user approved the concrete pending question on 2026-10-19: implement the
local pidfd bridge and execute one bounded compatibility-control batch. Starting
branch `feat/moe-route-verification`, HEAD
`1f85bc7022d882af2967da761ff1db72e5655580`, clean and synchronized.

This does **not** authorize another endpoint seven-position algorithm batch,
real models, GPU, new quotas, different checks or dependency changes. The V4
pre-admission failure remains intact. The endpoint approval registry stays empty.
The new control has its own source/environment freeze and unique directory;
unexpected failure stops the unexecuted suffix, without a retry.

## Minimal implementation

`scoped_proof/pidfd_compat.py` uses both Python pidfd APIs when present; otherwise
it resolves the libc **named** `pidfd_open` and `pidfd_send_signal` symbols with
explicit ctypes signatures and immediate errno propagation. Only zero flags,
null siginfo, signal 0 and SIGKILL are accepted. No numeric syscall fallback,
bare PID/group signal, global monkeypatch or dependency install is introduced.
An opened descriptor must be non-inheritable and is closed on validation failure.

The existing owned-session runner probes open/signal(0)/close on itself **before
Popen**. Later target/resource failures still fail closed. It retains the direct
leader unreaped while checking the SID and signalling via pidfds, then reaps only
after no live session members remain. The batch observer uses the same probe
before admission. `owned_bounded.py` is unchanged; its old group implementation
is not used as a compatibility fallback.

Libc, ctypes, the linker, Python import machinery and Linux pidfd/proc semantics
remain trusted. A symbol existing does not prove the kernel permits its use.
The current environment lacks both Python wrappers; the finite control must
test the actual libc path. No general daemon, escaped-session, zombie-reaping,
uninterruptible-kernel or GPU cleanup guarantee is claimed.

## Frozen control roster and resources

The eight case groups run in this order, once:

| Group | Required observation |
| --- | --- |
| self_capability | Self open, signal 0, CLOEXEC and close |
| invalid_and_closed_fd | Strict argument rejection and actual EBADF |
| local_abi_errors | Private fake symbols: missing symbol, ESRCH/EPERM/ENOSYS/EMFILE/EINVAL, missing errno, invalid return; explicit ABI and argument order |
| normal_and_error_exit | Owned normal exit and nonzero exit, both cleaned; nonzero never accepted |
| owned_signal | This batch's own direct fixture killed through pidfd and reaped |
| foreign_session_refusal | Another self-created SID is refused and exits naturally; no other user's target |
| deadline_and_descendant_cleanup | Expired deadline creates no child; timed-out same-SID/different-PGID descendant is cleaned |
| descriptor_balance | No leaked handles or unreaped direct fixtures after preceding branches |

The old 22 algorithm controls are **not** this roster and remain unexecuted.
Local ABI fakes do not replace actual self/child operations, nor modify globals.
All fixture children self-expire within six seconds of their own bootstrap.
Cleanup failure retains FAIL and uses bounded observation/wait only, never an
unbound signal. A successful cleanup cannot turn a failed control into PASS.

Unchanged limits: batch 300 seconds, each managed process 8 GiB virtual address
space, 2 GiB sampled session-plus-caller RSS stop, 128 MiB artifact limit, 8M
algorithm Work cap. These syscall controls do not generate mathematical source,
candidate or Work objects; they are not a full-capacity test. Work allowance is
not reset or spent on an algorithm. Reports/logs have a fixed finite roster and
bounded strings/session logs, well below the artifact cap. No raw arrays exist.

The existing **provisioned stdlib measurement-root** boundary is retained:
interpreter cold startup before `PROVISION_ENTRY` is unknown, not zero. Installing
inherited hard limits and stdlib setup are reported as provisioning; all project
imports, file hashes, admission, probes, fixture creation, checks, cleanup,
serialization and reception occur after original t0. No new supervisor layer.
Control work must stop by t0+240; cleanup reserves to t0+260; final acceptance
must occur before t0+300. SIGALRM terminates only the root at the hard deadline;
it does not send a bare PID signal to children. This relies on normal OS signal
delivery, as does the existing supervisor.

Fresh admission needs 4 GiB available host memory and 128 MiB free disk; other
jobs are inspected, never interrupted. The result directory is an exclusive
one-attempt claim. No observations or fixture launches occur from the static
checker. Native execution begins only after implementation, freeze, review,
commit/push and a fresh resource check.

## Acceptance and audit

`scripts/pidfd_finite_control.py --execute-approved-once FREEZE_SHA256` is the
default-off entry, using existing act-py312 with `-I -S -B`. The freeze binds
source and selected environment file hashes, exact roster, caps and directory.
The surrounding invocation retains separate stdout/stderr and process exit
code. Output files must be exclusive, not overwritten.

The saved result/reception are candidate records, not self-authenticating proof
of timely process exit. `scripts/audit_pidfd_finite_control.py` independently
reads the result, stdout export acknowledgement and **observed process status**.
PASS requires all eight groups, correct normal/error/timeout receipts, actual
foreign-sentinel natural exit, descriptor balance, full costs under 300 seconds
and successful exit after output flushing. A truncated/missing receipt or failed
exit is not PASS. The post-stdout clock includes terminal encoding and flush;
the final stderr acknowledgement is followed by a deadline check before exit.
External command wall time also accompanies the archive; provisioning/cold-start
boundaries are not silently conflated. Offline audit cost is separately reported
and cannot repair late online acceptance.

This is a finite observed compatibility check plus saved-record audit, not an
independent proof of OS behavior, every error interleaving, mathematical bounds,
algorithm efficacy, GPU performance or G1–G6 completion. If it passes, the next
scientific action is separately freezing/authorizing the already-defined bounded
algorithm batch, not repeating these small controls or expanding supervision.

## Preparation status

AST/compile preparation passed: eight named groups; endpoint static closure now
77 files, original seven positions/22 names and numeric caps unchanged, zero
endpoint approvals. The ABI/ownership review is read-only. No native control
has run at this preparation stage. Execution result will be archived separately.
