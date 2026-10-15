# Direct lifecycle and hard supervision results

The default-off fixed-candidate connection passes **133 tests, with zero errors
or failures**. The [protocol](hz_direct_lifecycle_protocol_20261015.md) was committed
and pushed at `c4e8c6253` before implementation. The [final report](hz_direct_lifecycle_controls_20261015.json)
binds code hashes, exact differentials, supervised calls and their costs.
No native optimizer, numerical library, real model, checkpoint, dataset or GPU
was executed. Resource ceilings, exact checks and production defaults are unchanged.

## Implemented boundary

- [Lifecycle and leases](../scoped_source/direct_lifecycle.py): one setup and close,
  Driver-issued roots bound to source object, Work, request, contract, task order
  and tile token. Mapping/iterator access revalidates the lease; returned rational
  values are copies. Proposal completion or failure revokes access. Close is
  attempted after setup failure; cleanup failures preserve the primary error and
  record unsuccessful cleanup costs. This is not a hostile-Python memory sandbox.
- [Method registry](../scoped_source/direct_method.py) and
  [portable checker](../scoped_source/direct_lifecycle_package.py): candidate
  width eight, dense-owned or pair-scoped check-two, and the same registered
  consumer in production and replay. Trusted expected method and invocation
  are required. Actual workspace and frame profile are checked, not only the
  manifest labels. Mathematical checking does not independently prove an
  optimizer's generation trajectory; this version admits only fixed stubs.
- [Execution](../scoped_source/direct_lifecycle_execution.py): release proposer,
  leases and production source/arena owners before fresh portable checking.
  Complete source construction/checks, proposed and zero roles, all pair/property
  obligations, publication and online recheck use one cumulative Work/deadline.
  The old V1 functions and their evidence are unchanged.
- [Hard supervisor](../scoped_source/direct_supervised.py): a closed synthetic
  specification launches isolated `act-py312 -I -S -B`; no arbitrary command or
  callback is accepted. The existing owned-group primitive keeps the leader PID
  until group signalling, bounds reaping and requires no live group members.
  Timely exit, cleanup, source/package identity, complete aggregation and final
  reception are required. A worker file marked COMPLETE remains only a proposal.

The trusted stub must not escape its process group. Sampled 2 GiB RSS is an
observation/refusal policy, not an instantaneous memory bound or native allocator
proof. The OS may delay cleanup beyond the deadline; such outcomes are rejected.
The control caller also measures API begin/end independently. There is no claim
that the OS guarantees an absolute wall-time return within 300 seconds.

## Control results

| Control | Result |
| --- | --- |
| Four frozen sources × two scalar profiles × two consumers | 16 configurations; 328 proposed/zero exact bounds match the expanded scalar oracle and old Session; complete outcomes agree |
| Two full supervised sources × two profiles × two consumers | All eight normal requests complete; producer/portable consumer bindings and complete results agree between arms |
| `half` normal source | All three obligations positive in both arms/profiles; synthetic source only |
| `guarded` normal source | All 12 obligations retained; both profiles return UNKNOWN_NONPOSITIVE, including 0/12 or 2/12 positive rows |
| Explicit missing candidates | Execution completes with UNKNOWN_MISSING_EVIDENCE, not fewer required obligations |
| Read-only/foreign/stale roots and method pollution | Rejected; no returned Fraction alias can mutate the checker's value |
| Alternative profile encoding with recomputed hashes | Rejected against trusted frozen method |
| Setup/proposal/close, abort and GC failures | No successful terminal; primary error and cleanup state retained |
| Blocked setup/proposal/close/replay/output | Outer timeout, owned group cleaned; no final acceptance |
| Success-looking candidate written before late exit | Outer timeout; persisted positive-looking output does not rescue the request |
| Late reception and final publication | Timeout; even a saved COMPLETE receipt cannot override the actual late returned call |
| Missing process, unconfirmed cleanup, partial output | Separate failed states, not silently zero-cost or successful |
| Equal-length package/candidate corruption | Full-file hashes and independent structural rereception reject |
| Both portable arms moved to another directory | Same complete result under isolated standard-library checking with original source/repository access denied |

The report contains **31 instrumented supervisor calls**: 10 execution-complete,
10 ERROR, 10 TIMEOUT and one intentionally unconfirmed cleanup. These include
injected/mock failures, repeated reception controls and missing-evidence requests;
they are not a model-coverage or performance table. One additional normal call
outside that instrumented list tests preconsumed Work and audit anchoring. The full
suite took about 31.94 seconds, not an algorithm benchmark.

## Complete accounting and review corrections

The parent clock starts before specification/code binding and subprocess launch.
Worker imports, fixed-source generation, construction, roots, candidates,
serialization, exact checks, online replay, cleanup, parent structural reception,
file hashing and final publication are included. The controls check:

`fixture + inner execution ≤ child execution`,
`child execution + cleanup = child total ≤ parent prepublication ≤ observed return`.

Killed workers have no invented final Work counter: `work=null` accompanies a
bounded progress prefix. All retained source originals, copied source, proof
members, log, candidate report and terminal are charged against the same
128 MiB request limit. Metadata reserves subtract from the worker archive;
member slots reserve future files. The archive's first result is not used as a
free historical input to another arm. Full-file hashes also cover failed prefixes.
The instrumented retained-byte totals range from 5,533 to 429,219 bytes; these
tiny controls do not establish the registered eight-expert capacity.

Two read-only reviews found and prompted corrections before finalization:
complete metadata/file-slot reservation, `..`/symlink path refusal, actual
consumer/profile binding, cleanup-error cost preservation, equal-length evidence
pollution, failure-state auditing and complete cost envelopes. The first
[126-test passing run](hz_direct_lifecycle_controls_20261015_attempt001.json)
is retained. Additional tests cover these review findings and actual late-output
and late-publication cases. No mathematical acceptance gate changed. No test run
failed in this stage; the added coverage is not a new scientific experiment.

## Reproduce and next gate

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -S -m scripts.run_direct_lifecycle_controls
```

This command runs only the frozen standard-library controls. Generated temporary
fixtures are removed by the test harness; the compact bound reports are retained.
The checker rosters exclude the producer; offline relocated/archived checking
has its own cost and cannot upgrade online failure.

The [older implementation cost ledger](hz_direct_join_cost_20261015.md) remains
valid for its bound V1 commit, not automatically for this V2 lifecycle. Next
reconcile the new source-owner release, root-access and parent reception costs
with the complete Work/bytes/lifetime ledger. The fixed-stub matched consumer
path is now implemented; native proposer integration and native matched execution
are not. Full-topology membership/storage/Work barriers, full RSS bounds, real/GPU
admission and all G1–G6 scientific acceptance criteria remain open. Do not infer
native execution permission or higher quotas from this control result.
