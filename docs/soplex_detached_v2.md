# SoPlex V2: detached launch and interruption accounting

The user authorized continuation after V1's pre-job supervisor loss. This is a
new execution directory/identity, **not** resuming V1, retrying an LP, or changing
the scientific comparison. V1 had zero timed jobs/native solves and remains
preserved. Original four jobs, exact matrices, candidate admission/checker,
218/298/300-second boundaries, memory/output caps and resource gate are unchanged.

V2 calls the frozen V1 `run_jobs`, `supervise`, worker, exact receiver and
independent auditor directly. Its protocol JSON differs from the original
scientific freeze **only in output directory**; startup asserts this equality.
The old readiness fields are historical; the separately bound V2 execution
freeze/readiness records authorize this new process wrapper. No dependency
installation or ACT environment change. Existing `/usr/bin/tmux` is hash-bound.

## Process ownership

A private tmux socket `/data1/Kane/MOE/run/soplex-v2.sock` runs a guardian outside
the transient tool-launch process tree. This does not use or alter other tmux
sessions. User systemd bus was unavailable. A controller executes the fixed
queue; guardian journals process PID **and start ticks**, boot identity and a
one-second heartbeat. It is a Linux child subreaper: if the controller dies,
orphaned descendants are adopted and only owned identities are stopped.

SIGTERM/INT, controller exception/nonzero exit or missing batch terminal lead
to a four-slot interruption ledger, never resume. Completed summaries are
preserved as **requiring audit**, not blindly accepted; worker candidates alone
grant nothing. Missing complete costs are null/right-censored; last resource
poll and phase artifacts remain. No solver deadline or budget is reset.

Private tmux survival is tested by killing the launching process tree while a
synthetic controller runs. It cannot guarantee survival of machine reboot,
account-wide kill or tmux-server termination. After total process loss,
`python -m soplex_detached.run reconcile` checks boot/PID-start identity and
writes a separate post-mortem **only if both guardian/controller are gone**.
An existing live process is not called stalled/dead merely for lack of progress.
Reconciliation never reruns an obligation. Machine-wide loss cannot create an
instantaneous on-machine record; the next inspection provides that record.

## Accounting and audit

Guardian lifetime includes resource waiting and is not LP solve time. Each
timed job retains the frozen V1 complete cost ledger. Guardian cleanup and
post-terminal audit are separately recorded. On normal batch completion the
guardian runs the original `audit_batch` using the new path-bound protocol and
writes `final_review.json` and `completion.json` **inside the raw result tree**.
Thus asynchronous completion does not silently dirty tracked project files.
Audit exceptions are preserved as AUDIT_ERROR and never trigger a rerun.

Preparation is committed/pushed before launch. After completion, manually
review and archive compact results and commit/push; no unattended repository
mutation. All old raw files remain, no destructive cleanup or overwrite.

## Controls

Seven controls: unchanged scientific fields/new path only; launch-tree death with detached completion; controller SIGKILL
and owned orphan cleanup; live/stale PID identity and idempotent reconciliation;
partial costs/truncated final wait line; completed record requiring audit;
duplicate private socket rejection. No real LP or native solver query.
First control attempt (`/data1/Kane/MOE/soplex_life_6qf_2m24/controls.json`)
retained a test-fixture quoting NameError; fixed fixture,
not time limits or solver behavior. Earlier V1 mathematical controls remain
hash-bound, with no changes to the candidate, LP or checker implementation.
