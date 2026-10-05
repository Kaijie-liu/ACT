# Trial 9 Tiny143 ReLU36 Exit Audit

Date: 2026-08-31  
Branch: `redu-hz`  
Repository base commit: `f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`

Trial 9 exited naturally.  The PID/start-ticks-bound sealer published its
exclusive sidecar without overwriting the worker result.  The nine frozen
source files still pass `TRIAL9_SOURCE_FREEZE_SHA256SUMS`.  No process was
killed, reprioritized or altered.

```text
2673a77761b67eefa821562453369872e3f804c6758d47a492f8e3aa211a5de1  results/trial9_phase_implicit__tinyimagenet_2024__iid143__relu36_smoke_v1.json
26638eba3e4b3bce0746bd8ac3025d82182b8513b744ff760ff01ef42eb788ca  results/trial9_phase_implicit__tinyimagenet_2024__iid143__relu36_smoke_v1.exit_seal_v1.json
bb9127a7aae29ec47b6bebfbc90da91a42df0f992c57c975380f6747a490f4da  results/trial9_phase_implicit__tinyimagenet_2024__iid143__relu36_smoke_v1.log
```

The sidecar classification is `COMPLETE_JSON_VALIDATED`; every registered
result and provenance field passed.  Its canonical payload digest independently
recomputes to
`03790db0fcf0e05713799349d52b6d5571ed80aefb906b1c47b942c18310f6fa`.
`exit_status=UNRECOVERABLE` means an orphan process's numeric exit code cannot
be recovered after `/proc` disappearance; it is not a result classification.

## Capability result

The worker intentionally stopped at layer 36 after 12,954.997205 seconds of
propagation (12,956.839771 seconds wall time).  It records:

- `census_stop_reached=true`, verdict `UNKNOWN`, zero concrete validations;
- seven implicit Conv occurrences, 15 lazy-affine layers and three lazy
  materializations;
- one exact phase-selective ReLU at layer 36;
- phase N/P/U = 20,648 / 2,268 / 2,172;
- exact nonlinear core `13,610c / 2,101b`, 2,101 equality and 4,202 inequality
  rows, with 5,131,207 stored core entries;
- 2,260 omitted positive rows plus eight deterministic positive probe rows;
- a strict local saving of 8,316 entries after the registered mask/bias charge;
  and
- the resulting five-term layer-36 expression has 84,880,604 logical expanded
  operator entries but only 650,460 resident operator entries and 5,137,492
  resident operator bytes.

The unique-live-cache upper-bound ledger at the stop contains seven HZ objects,
16 expressions and ten operator objects.  It reports 459,290,844 resident
numeric bytes and 38,525,226 resident entries excluding phase bounds.  Python
allocator/object overhead is explicitly absent.  This is a capability census,
not the complete C3 whole-state comparator.

## Structural observations

The frozen source still exhibits the BatchNorm sibling graph:

```text
Conv29 -> Scale30  (no successor)
Conv29 -> Bias31 -> ADD32
ADD32  -> Conv33 -> Scale34 (no successor)
ADD32  -> Conv33 -> Bias35 -> ReLU36
```

ADD24 holds three lazy terms and has successors 25 and 32.  ADD32 holds four
lazy terms and has successors 33 and 40.  The phase-selective layer-36 result
has five terms because the exact nonlinear core is retained beside its lazy
positive expression.  Trial 9 carries no immutable per-term graph lineage, so
it cannot decide whether the three inherited ADD24 terms contain a nested ADD
or a legal exact-source boundary.  It therefore cannot authorize C3.

`consumer_gc_released_sparse_states=0`; no candidate may infer release of
ADD32 while layer40 remains a consumer.  The subsequent lineage-only census
must preserve this second-consumer boundary.

## Accounting conclusion

The result proves that the phase-selective/implicit representation reaches the
registered ReLU36 stop below the 16 GiB process gate and exposes the exact
four-term ADD32/five-term ReLU36 shape.  It does not produce a terminal HZ,
CERT, validated ADV, family replay, concurrency result or strict same-boundary
C3 whole-state reduction.  Trial 9 formal gain is zero; the formal score stays
1,870/2,413 and E0 stays 61/400.
