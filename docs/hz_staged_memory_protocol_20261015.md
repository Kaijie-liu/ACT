# Staged memory supervision protocol

2026-10-15. Starting branch `feat/moe-route-verification`, clean HEAD
`0e0f250153ea4c98dbdf478e791e9da3115e82b8`.
**DESIGN_FROZEN_PENDING_LIMITS_AND_IMPLEMENTATION_APPROVAL.** The user approved
freezing this memory-supervision direction. No limit value, runtime change or
control execution is authorized by this freeze. The accompanying
[contract record](hz_staged_memory_protocol_20261015.json) leaves the address-space
value null and execution disabled. It is not a runnable method configuration.

## 1. Scope and unchanged contracts

The proposed opt-in version separates production (including source construction
and checking), portable checking and final reception into fresh processes. It
moves the current unsupervised parent reception into an owned helper. Existing methods remain the
default and historical results remain immutable.

Only the existing `half` and `guarded` standard-library fixtures and fixed
candidate profiles are eligible for a later control freeze. No general callable,
checkpoint, dataset, real model, native proposer, numerical package or GPU is
admitted. This is trusted-code lifecycle isolation, not a sandbox for hostile
Python, native code or another writer with the same UID.

Preserve `hz_request_store.POLICY`, the 128 MiB owned-payload arena, candidate
width 8, scoped check width 2, dense reference width 8, page width 8, 64 KiB
result pages and the existing report/index ceilings. Preserve both complete
mathematical passes, exact residual checks, all tie-legal pair/property duties,
numerical gates, and the cumulative 300-second budget. In particular:

- Splitting processes does not reset the 8,000,000 Work ceiling or reinterpret
  a charged scalar visit as an uncharged lookup.
- Source-defined affine operators still have the registered two-pass legality
  floor of **13,922,736 Work** under the unchanged charging contract. This new
  memory policy cannot admit that full model.
- An address-space cap is an additional refusal mechanism, not an increase to
  the owned arena, archive, Work or sampled-RSS thresholds.

See the [implemented resource account](hz_paged_execution_resources_20261015.md)
and [source-validation decision](hz_domain_guard_decision_20261015.md). Successful
capacity, a complete whole-call memory bound and a completion-time upper remain
unestablished; `full_capacity_admitted` stays false.

## 2. One address-space value and an explicit bootstrap boundary

Use one separately approved integer `as_bytes` for the dedicated watchdog,
controller and every stage. Its value is **not selected here**. It must be
positive, representable by the platform, page-aligned and no larger than any
inherited finite soft or hard AS limit;
otherwise refuse before evidence intake. A caller, package or candidate cannot
supply a substitute value. A later signed-off configuration binds the exact
value, protocol hash, implementation inventory, interpreter identity and fixture
roster. Missing approval or a null value means `ADMISSION_REFUSED`.

Launch a dedicated minimal standard-library bootstrap with `python -I -S -B`.
Before importing project modules or reading request, archive or candidate
payloads, it validates only the fixed-shape launch arguments and lowers its own
`RLIMIT_AS` soft and hard limits to the approved value. It then verifies
`getrlimit`, checks its current virtual footprint is not already above the cap,
and `exec`s the watchdog. Only the exec'd watchdog, after rechecking the limits
and footprint, may publish `LIMIT_ACTIVE` and start the controller. A failed exec
or missing acknowledgement rejects the attempt. Never change the Codex process, interactive shell, shared SSH
service, another user's process or system-wide settings.

The bootstrap interpreter's initial load and its minimal stdlib imports precede
`setrlimit`. They remain a **trusted pre-limit startup boundary**, charged in
time but not proved hard-memory-bounded by this protocol. Lowering AS does not
retroactively remove existing mappings. An externally enforced pre-exec boundary
would require a separate design; do not hide this gap behind a small observed
startup RSS.

After activation, the watchdog launches one dedicated controller, which launches
at most one stage at a time by fresh exec, inheriting the same soft/hard AS pair.
Controller and each stage verify the inherited limits before project imports
and payload parsing. No stage raises either limit.
The uniform value avoids the impossible design in which a controller lowers its
hard cap and a worker later attempts to exceed that inherited cap. Do not use
`preexec_fn`; the dedicated bootstrap performs the setup before its exec.

`RLIMIT_AS` limits per-process virtual address space, not aggregate resident
memory. Allocation failures and stack-growth failures need not raise a recoverable
Python exception. See [Linux resource-limit semantics](https://man7.org/linux/man-pages/man2/getrlimit.2.html)
and [Python subprocess guidance](https://docs.python.org/3.12/library/subprocess.html#popen-constructor).
Once armed, with the trusted no-extra-descendant rule and no stage overlap,
watchdog plus controller plus stage have a conditional virtual-space envelope
of `3 * as_bytes`. This is not a request-RSS, kernel-memory, file-cache or
successful-computation bound.
Keep the existing 2 GiB sampled-RSS refusal separately named and unchanged;
sample watchdog, controller and current stage together, without double counting.
Sampling can miss peaks. Do not substitute `RLIMIT_RSS` or modify shared cgroups.

## 3. Phase machine and check ownership

All stages run pinned trusted code without forks, daemonization, new sessions
inside a stage, or background threads. The watchdog creates one owned controller
session; stages remain in that same session/process group. The controller uses
PID-specific supervision and reaping for its one stage. Do not nest the existing
`run_owned(... start_new_session=True)` for stages: an escaped stage group could
survive controller death. This is a reviewed-code assumption, not an OS-enforced no-fork
guarantee. A detected unexpected child or ownership-inspection failure rejects
the attempt. General native process containment is out of scope.

| Stage | Required work | Nonfinal outcome |
| --- | --- | --- |
| PRODUCTION | Capture the fixed source, construct/publish the shared forest, bind ownership and perform the original complete source check; construct the full property plan; propose fixed candidates and check every endpoint; publish result pages and package. | `PENDING_REPLAY` |
| PORTABLE | Relocate the owned package; in a fresh interpreter reconstruct source and plan, check every original candidate and endpoint, compare every row and summary, and retain inventory and source-seal checks. | `PENDING_RECEPTION` |
| RECEIVE | Perform the existing complete structural reception, exact sign/roster aggregation, implementation and identity checks, storage census and cost reconciliation. Prepare the final receipt. | `PENDING_COMMIT` |
| Controller preparation | After RECEIVE exits and is reaped, validate the bounded handoff and phase chain; prepare the final bounded metadata and exit. | `PENDING_OUTER_COMMIT` |
| Watchdog commit | Reap the controller and establish its group is quiescent; validate the bounded handoff and publish/read back the final terminal before the deadline. | `COMPLETE` only if every gate passes |

Source construction is not a third independent proof. Keeping construction and
its online check in PRODUCTION avoids introducing a separately trusted checked
object across exec. The first complete source/endpoint pass belongs to PRODUCTION;
the fresh second pass belongs to
PORTABLE. Neither phase can accept a prior `CheckedSharedForest`, norm cache,
receipt or Python object as a substitute for its own checks. Reading the
published source/forest wherever the new execution requires it adds I/O and must
be counted; process separation is not assumed free.

Retain the current package check inventory: copied checker-code identities,
source reconstruction, forest identity, regenerated property contract, actual
candidate profile, ordered tile roster, every endpoint, source reseal, every
saved/recomputed exact result row, aggregate equality, and inventory before and
after checking. Missing-candidate markers can yield complete `UNKNOWN` only
after all duties are traversed. An interrupted prefix is incomplete, not that
form of complete UNKNOWN.

RECEIVE is not a third LP proof. It preserves all checks currently made by
`direct_supervised.receive`, including source-derived duties and exact signs.
It must not accept PRODUCTION's report in place of a trusted PORTABLE result.
Neither controller nor watchdog reconstructs rosters, parses pages, loads roots
or rehashes the complete archive after the helper exits; that would recreate
unsupervised reception. The bounded names/lengths closure and fixed suffix checks
specified below are their only post-helper storage checks.

## 4. Bindings, relocation and ownership

The root invocation anchors protocol and approved-resource identity, exact code
inventory, interpreter, case/profile, arm, source identity, absolute deadline,
initial cumulative counters and a fresh attempt nonce. Each stage binds its
role, sequence number, predecessor receipt hash and the same invocation/method.
Package, forest/index, ordered candidates and result references are added as
they become available. The controller uses its observed process identity and
exit status, not a child-selected stage name. A hash binds content; it does not
prove a mathematical check occurred.

Each stage receives bounded file references and counters, never live parent
source/Fraction/checker objects. Intermediate records are fixed-schema and
bounded before allocation. The controller's handoff is at most the existing
64 KiB metadata ceiling, contains only scalar identities, flags, counters and
bounded fixed-role summaries, and contains no per-duty arrays or exception text.
All other phase reports remain helper-read files within the existing archive
and member caps. The later implementation must prove its concrete field/encoding
upper and reserve bytes for metadata and failure records before payload work;
the design cap alone is not that proof.

On one filesystem, relocation transfers the single owned package to a new
request-local path after the previous stage exits. It does not create a second
logical source bank. Copy fallback is forbidden in this profile: cross-device
or conflicting destinations reject. If a future separate control copies a
package for offline checking, its copy and cost are recorded separately. No
hardlink or old writable alias is allowed to stand in for ownership transfer.
File hashes, regular-file/no-symlink checks, bounded lengths and inventory checks
remain necessary. Read-only handles or chmod alone do not protect against a
hostile same-UID writer; private request ownership and trusted stage code remain
assumptions.

PORTABLE uses only relocated package/checker files and the stdlib. No repository,
original path, checkpoint, dataset or solver access is allowed during its proof
reconstruction. Execution approval and root invocation remain external anchors;
an archive cannot approve its own resource policy or checker version.
Before its first package-code import, an independent stdlib bootstrap verifies
the relocated code against the externally approved root implementation map,
including new wrappers/bootstrap modules. Bundle-reported code hashes alone are
insufficient. Do not import a checker and then rely on it to authenticate itself.
The same relative member identities survive relocation; no path-based rebinding.

Only the current stage may write its allowed outputs. Do not serialize traceback
objects or call arbitrary exception `str`/`repr`. The controller holds no pointers
to stage heaps, so process exit—not `del` or `gc.collect`—is the lifetime boundary
even on exception paths. Before the next stage, the prior actor must be reaped
and the controller group contain no unexpected live members. The watchdog keeps
the controller group leader unreaped until signalling its owned group, avoiding
a reused PID. It cleans the group even when the controller exits abnormally or
cannot report the actor PID. No outer success precedes group cleanup. A cleanup
failure stops the phase chain; offline examination cannot repair online acceptance.
The watchdog itself is a trusted live supervisor. If it dies, the caller must
record an incomplete terminal and unknown cleanup, never accept a prepared
controller record or blindly signal a potentially reused PGID. Automatic cleanup
after watchdog death is not proved by this stdlib process-group profile.

## 5. One deadline and complete cost accounting

The absolute monotonic start is recorded before bootstrap launch. For successful
controls, `deadline = start + 300`; only future explicitly frozen failure controls
may use shorter deadlines. Carry the original absolute clock through every exec.
No phase obtains a fresh 300 seconds. Keep the existing final reserve rule
`r = min(1, seconds/5)`: all stage work and reception finish before `deadline-r`,
owned-process cleanup before `deadline-r/2`, and final bounded publication/readback
before `deadline`. A stage receiving no remaining budget is not launched.

All bootstrap/import, AS setup, capture, propagation/construction, parsing,
candidate generation, checking, relocation, hashing, serialization, fsync,
reception, group observation, cleanup and terminal work count. Collect a
nonoverlapping wall-time partition into bootstrap, each stage's full owned
lifetime, controller-only intervals and outer finalization. Watchdog polling
overlaps these intervals and is not added again as wall time. Report actor
execution/cleanup as subcomponents, not extra time added twice. CPU time, sampled RSS, virtual size,
logical bytes, read/write/hash byte totals and exact Work remain distinct metrics.

Every stage resumes the previous verified three-counter Work vector. Every
successful transition verifies monotonicity, phase deltas and the unchanged
cumulative ceiling. Hashing and parsing wall time cannot disappear merely
because the old Work schema does not charge every machine operation. Added
reads/receipts are included, not recast as the former implementation's cost.
Archive/member limits cover all retained stage files, logs, partial prefixes,
reports and terminal reserves together; not one independent allowance per process.

After a kill, last verified counters are lower bounds, not totals. Report
`work_complete=false` and any unknown quantity as null, never zero. Controller
elapsed time includes waiting and cleanup even if actor counters are missing.
If a complete bounded final census cannot finish, mark storage/cost incomplete;
do not run an unbounded census outside supervision to manufacture completeness.
Such a run cannot accept. Later forensic accounting is explicitly offline.

The watchdog commits only after helper output, expected code identity, actor and
controller exit, clean group, role chain, exact resource policy, accumulated costs
and the current clock all pass. The controller can only prepare that decision.

Final storage uses a closed prefix plus a fixed suffix, not the early helper
census alone. RECEIVE seals the payload, previous phase reports and finished logs;
it inventories/hashes that prefix and publishes its ordered path/length digest,
bytes and file count. No stage may subsequently modify that prefix. The only
remaining writers are restricted to these six new relative suffix paths, each
at most 64 KiB:

```text
control/receive_tail.json      control/controller_tail.json
control/terminal.json          logs/receive.log
logs/controller.log            logs/watchdog.log
```

Reserve their aggregate 393,216 bytes and six members from the original limits,
not an additional allowance. Pre-terminal records bind the prefix and allowable
suffix rather than an unfinished suffix hash. Logs close before final hashing.
The watchdog validates actual lengths/hashes of the finished suffix and performs
a bounded streaming directory-name/length check against the sealed prefix digest
and six-name allowlist. It rejects unexpected names, changed lengths, symlinks,
unclosed writers or over-cap traversal; it does not parse or rehash prefix
contents. Same-length malicious rewriting remains excluded by private ownership
and trusted writer code, not claimed detectable by a size scan. All these steps
remain AS-limited and deadline-charged. If even this bounded metadata closure
cannot finish, storage is incomplete and the result cannot accept.

The final terminal includes exact totals using the reserved, bounded terminal
layout and actual sizes; its own hash belongs to the outer handoff, not inside
itself. The concrete encoding must close this suffix bound before implementation
execution, with no circular receipt digest. A post-write clock expiry downgrades the online result even
if a provisional file exists. The API returns an accepting terminal only when
the final record and handoff agree before the deadline.
The caller must also observe normal watchdog exit and receive the bounded final
handoff within that same deadline; a `COMPLETE` file alone is insufficient. This
last call-level record is bounded metadata, not another evidence-parsing phase.
If persisted online, it must occupy an explicitly reserved suffix slot in the
later execution freeze; no unlisted seventh file is implicitly authorized here.

OS-delayed kill, wait or filesystem calls can exceed the clock. This protocol
enforces timely acceptance, not a universal finite return-time guarantee. A
missing/late final record is `INCOMPLETE_TERMINAL` or `TIMEOUT`, never implicit
success. A watchdog/caller outside this controller may retain only its own bounded
status metadata; it must not parse evidence or provide a late accepting fallback.

## 6. Failure outcomes

| Observation | Required result |
| --- | --- |
| No approved AS value, binding mismatch or unsupported inherited limit | `ADMISSION_REFUSED`; no source/worker payload |
| Candidate/source/check failure, malformed or swapped phase receipt | `ERROR`; retain charged prefix |
| Observed budget or existing sampled-RSS threshold exceeded | `TIMEOUT` or `RESOURCE_LIMIT`, with the specific observed gate |
| MemoryError, ENOMEM or abnormal signal without conclusive attribution | Nonaccepting error plus observed code; do not invent proof of AS exhaustion or OOM |
| Helper killed, incomplete roster, missing page or partial metadata | Incomplete nonaccepting outcome; missing cost is not zero |
| Live owned descendant, unreadable ownership state or failed reap | `CLEANUP_INCOMPLETE`; no next stage |
| Every duty checked but some lower bound nonpositive/missing by schema | Complete `UNKNOWN`, not UNSAFE |
| All stages complete and every required bound passes unchanged gates | Accepted result at its existing guarantee level, not a new floating-point proof |

## 7. Finite control plan, not yet executable

After a separate limit/implementation approval, freeze the concrete invocation
schemas, all implementation hashes, metadata bound and exact control list before
execution. Start only with the original two fixtures, two arms and two candidate
profiles; do not add models or capacity/timing sweeps. Keep successful controls
at 300 seconds. No automatic cap tuning after failure.

Required control families:

1. Default-off, missing approval, limit/config/method/role drift and bootstrap
   failure; verify the caller's own limits do not change.
2. Same-candidate staged/old-path exact differential for every obligation,
   positive/zero/nonpositive/missing results and both complete source passes.
3. Fresh relocation on both arms, old-path denial, no live-state inheritance,
   invalid ownership, destination conflict and unsupported cross-device move.
4. Missing/duplicate/reordered duties, role substitution, source/property/candidate
   mismatch and altered lower bound with all archive hashes recomputed. Rejection
   must still reach independent mathematical comparison when hashes are valid.
5. Fixed allocation failure and deadlines in construction, production, replay,
   reception and final publication; errors may not invoke arbitrary formatting.
   Verify OS AS inheritance in a later separately bound allocation control, not
   by assuming an injected MemoryError proves enforcement.
6. Partial output, stage death, unexpected group member, group-observation/reap
   failure and no-next-stage behavior. Synthetic bookkeeping faults must not
   launch uncontrolled children or touch unrelated processes.
7. Preconsumed Work, truncated counter records, all retained-prefix costs, metadata
   reserves, over-cap prechecks and complete/noncomplete terminal audit. Verify
   no stage, transition or receiving helper resets the deadline or budget.

Passing these controls would validate only the new lifecycle interface. It would
not prove sufficient memory for a real model, improve a bound, close G1–G6 or
authorize a later real/GPU experiment.

## 8. Approval gates and stop rule

This commit freezes **design only**. The next decision must separately approve
the concrete AS bytes and implementation/control scope, without changing any
existing quota. If no useful cap is justified, keep admission off; do not silently
borrow the old RSS threshold or restart capacity trials. Model/native/GPU source
admission, any original resource-policy change, and scientific experiments each
remain separate decisions.

The [earlier source/lifetime draft](hz_source_lifetime_protocol_20261014.md)
remains historical. This protocol selects fresh staged processes and supervised
reception for the stdlib-only profile; it does not implement its restricted
checkpoint parser or its native-source proposal. The established source Work and
dense arena refusals survive. No scientific table or sealed experiment changes.

## 9. Static validation

The read-only [design checker](../scripts/check_staged_memory_protocol.py) verifies
12 source/document anchors, all 15 existing policy entries, the unchanged checks
and widths, null limits, nonexecution and the source Work refusal. It does not
instantiate the new supervisor or change process limits:

```text
/data1/Kane/miniconda3/envs/act-py312/bin/python -I -S -B scripts/check_staged_memory_protocol.py
```

The existing nine project-layout regressions also pass. These are design and
navigation checks, not the future deadline/allocation/fault controls. No worker,
model, native solver or GPU request ran in this stage.
Two read-only AI reviews checked lifecycle/check preservation and supervision.
Their corrections covered shared process-group cleanup, pre-import code binding,
post-census suffix accounting and caller-observed watchdog exit. This is not the
independent human technical review required by G6.
