# pidfd compatibility control — completed, not an algorithm result

The one approved finite batch passed **8/8 groups**, with no retry and no skipped
suffix. Execution commit: `92244f51e6dc302a249b30b4b403480a685c46a3`.
Freeze SHA-256: `753be81d3fb727299f2f88d760ca733f8a59117d257be8f54a2a45095f0a2040`.

## Evidence

- [Frozen protocol](hz_pidfd_control_20261019.md) and
  [exact freeze](hz_pidfd_control_freeze_20261019.json).
- [Complete stdout, including all result/session records](hz_pidfd_control_stdout_20261019.json)
  and [post-stdout export acknowledgement](hz_pidfd_control_stderr_20261019.json).
- [Compact result and externally observed exit status](hz_pidfd_control_results_20261019.json).
- [Read-only archive audit](hz_pidfd_control_archive_audit_20261019.json).

The existing act-py312 still has neither Python pidfd wrapper. The new local
`LIBC_PIDFD` path actually executed. No interpreter/dependency replacement,
global monkeypatch, bare PID/group fallback or quota increase occurred.

| Observation | Result |
| --- | --- |
| Self capability, strict inputs, closed FD | PASS |
| Local fake ABI/errno errors | PASS; not a kernel fault injection |
| Normal / nonzero / deadline sessions | `COMPLETED / ERROR / TIMEOUT`, exit `0 / 7 / -9`, all expected |
| Owned signal and different-PGID descendant cleanup | PASS, no live session at reception |
| Foreign-SID sentinel | Refused; natural exit 0, not killed |
| Open FD count | 4 → 4 |
| Batch process exit, observed by command service | 0 |
| Request t0 through stdout encoding/flush observation | 3.096817708 s |
| Stdlib host provisioning, observed prefix | 0.018409785 s; earlier interpreter startup unknown |
| Sampled batch session-plus-caller RSS peak | 43,782,144 bytes (41.754 MiB) |
| Four runtime artifact files | 9,011 bytes |
| Separate offline archive audit | 0.083757365 s; no runtime controls repeated |

The final stderr flush is followed by a check of the same absolute deadline;
exit 0 is required in addition to the earlier candidate acceptance flags. The
API's returned wait duration is **not** total subprocess wall time. Unlike the
protocol's requested companion observation, no independent full external wall
duration was obtained; it is explicitly null in the archive. Do not convert tool
poll waits into a speed claim. The retained original clock, final deadline check
and successful exit support this finite compatibility acceptance; they are not
a cold-start benchmark or an algorithm cost comparison.

An independent read-only agent verified all 16 frozen file hashes, the saved
normal/error/timeout receipts, identities and descriptor counts. Its post-run
`/proc` inspection found none of the seven recorded PIDs or session members
remaining. This is an observation, not a general escaped-session/zombie/kernel
cleanup theorem. Error-path ABI mocks and static pre-spawn ordering are reported
separately from actual native operations. OS, libc, ctypes and the measurement
root remain trusted. Nothing here independently proves neural-network bounds.

## Decision and authorization boundary

The concrete pidfd API compatibility blocker is resolved for this environment
and observed control set. The single finite authorization is consumed; its
exclusive directory remains at
`/data1/Kane/MOE/results/pidfd_compat_20261019_r1` and must not be reused.

The previous V4 early failure is unchanged. Endpoint approvals remain **empty**;
the old seven-position freeze does not match the new source identity and cannot
be resumed. No source factory, candidate generation, mathematical solver, real
model or GPU was run, and no new positive certificate or efficacy result exists.
Full eight-expert resource refusal, numerical gates and G1–G6 remain unchanged.

Next: separately freeze and obtain execution scope for the already-defined
finite endpoint algorithm comparison using the compatible source closure. Do
not add more supervisor layers, repeat these controls to accumulate PASS counts,
or silently treat this approval as authorization for that batch. The scientific
question remains whether the MoE-specific support increment improves complete
obligations under the unchanged registered budget, not whether pidfd works.
