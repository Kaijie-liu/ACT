# C127 v1 failure audit — test-stage rejection, no source qualification

Disposition: **C127 v1 is CLOSED and FAILED.** Its sole frozen qualification
stopped at the full test gate. The complete-source worker was never launched.
There is no C127 v1 complete kernel/source qualification, measured whole-work
saving, physical reduction, timing gain, network admission or formal gain.
C126 remains the latest qualified ordinary component; C124 remains failed.

This audit reads saved JSON, XML and logs and checks file hashes only. It does
not import experimental modules, rerun tests, load numerical NPZ payloads,
execute a model/solver or alter any frozen v1 source or result. The pre-run
static review missed this executable API error: its earlier no-blocker verdict
is not numerical evidence and is superseded by the failed qualification.

## Authenticated sole-run record

Run directory: `results/c127_systematic_kernel_20260927_v1` (paths here are
relative to `experiments/neural_hz_20260831`). The supervisor process exited1;
saved `exit.json` reports `all_stages_passed=false`, `tests_exit=1`,
`tests_count=3584`, `formal_gain=0`, and failure
`ValueError: complete inherited/new qualification failed`.

| Saved artifact | Bytes | SHA256 |
| --- | ---: | --- |
| `exit.json` | 968 | `eaccb8b1b3848833a7405fd1e9c199f00784ec8dad0b10465184c3fec2235c7c` |
| `preregistered.json` | 304347 | `a615cab70f51568bbdece13ed96f3e9d43963b80ca691788675b13ec72444571` |
| `inventory.json` | 483200 | `dc984a50c6ba2fec899f0af5091de40f23691e2186cfba2aac0f96e6df9878e0` |
| `collection.log` | 458105 | `02723e21cac765ee17e5c9110aff1214ddcda702c6ce9085096725b8dff4b5bc` |
| `tests.log` | 18740 | `0f8094c6057d32ea81f65efe95d2d5aaf899dc14f7f9364739e10f5518bdc1ea` |
| `tests.xml` | 616195 | `6c5d1a6a441265e64db8b5f03bedfda0ee82658d5e3ccac6c180cf4f02868cee` |

All five non-exit hashes agree with `exit.json.artifacts`. These six files are
the complete saved v1 run inventory. There is no `worker.log`, `result.json`,
source-case report, full-point report, complete ledger or numeric proof archive.
The source, input and production-provenance drift flags are all false. The
saved inherited statuses are C124=false, C125=true and C126=true; these concern
their historical ordinary qualification only, not new C127 success.

## Exact test outcome

The saved XML contains3584 test cases,15 failures,0 errors and0 skips:

| Population | Passed | Failed | Total |
| --- | ---: | ---: | ---: |
| Unchanged inherited C126 population | 3556 | 0 | 3556 |
| New C127 v1 population | 13 | 15 | 28 |
| Entire qualification | 3569 | 15 | 3584 |

The supervisor's collection-plus-execution wall was55.95249526388943s, below
the unchanged60s gate. Its total wall was55.95669922605157s. The pytest log says
43.89s and13 warnings; XML records43.875s for its suite. These are test timings,
not worker timings or evidence that the new kernel is faster.

All15 failures terminate at the identical source statement in
`c127_systematic_kernel_proof_v1.py:210`:

```python
minimum = np.where(minimum == 300, 0).astype(np.int32)
```

The exception is `ValueError: either both or neither of x and y should be
given`. This call supplies the condition and replacement value but omits the
third, false-branch argument. It cannot preserve an ordinary nonempty kernel's
minimum exponent; it raises before alignment, source-kernel reconstruction or
the returned eight-array evidence. The intended expression is
`np.where(minimum == 300, 0, minimum).astype(np.int32)`: use0 for the all-zero
kernel sentinel, otherwise preserve the minimum. This audit does NOT patch v1.

The15 failures comprise all11 ordinary source variants in
`test_every_owned_array_matches_C125_and_all36_match_full_Fraction` (signed,
positive, scaled, sparse, zero, mixed_zero, random, noncontiguous, minnormal,
maxnormal, span33), plus the exact binary64-lift, fresh-owned-evidence,
complete-prepayment and full324 source-basis-output tests. They did not reach
their successful-path evidence assertions.

The13 passes are default-off, six unsupported-domain rejection cases, four
header rejection cases and two corrupted-program rejection cases. In
particular, the span34 test accepts any `ValueError`; it encounters this same
missing-argument error before reaching the intended span guard. Its green test
does not prove span34 rejection by the proper guard. The other early guard
passes and the fixed basis work executed before this error do not qualify the
complete kernel program.

## Frozen source custody

All eight new files still match their v1 preregistered hashes:

| Frozen file | SHA256 |
| --- | --- |
| `C127_SOURCE_BIRTH_PREFLIGHT_20260927.md` | `a4d8fafcaf4e6de3841e7275ca342ee4c0fa0df4836d15a31a8d046ad18d7ea7` |
| `C127_SYSTEMATIC_KERNEL_PREREG_20260927.md` | `a365d3625cb827b905caa6fb1206d420e80a9a53e62d2433383257fbcd776907` |
| `c127_systematic_kernel_proof_v1.py` | `620df9ef08bf90a05977b5bfa1bfb06b0a6b1de95f62337e06bb0c7fcd4d165e` |
| `c127_systematic_native_oracle_v1.py` | `af3e006e25884d641a173dc1450a0604fbf5fc9c28e565d9a28f6069715f3d30` |
| `c127_complete_mixed_source_v1.py` | `6de97df939870d74bf3951c785c4647f2c5edc3a9c3c50a9753bbd4be842a710` |
| `c127_mixed_source_worker_v1.py` | `52a162c468f6dda671b3879923211c8203e915f1222f45f584ea8cd132a0a13c` |
| `run_c127_mixed_source_supervisor_v1.py` | `63896acfa31e2238c7565f6afd4b527e054bad3b824a5d10449e319efbc20ca1` |
| `test_c127_systematic_kernel_proof_v1.py` | `5140a9556cb15e3d596c5b57f90fed2e609dd25911d5c2236f6ab749b980ad24` |

The scalar source-birth preflight remains an analysis of unchanged C126
program charges; it is not invalidated or converted into a C127 execution by
this failure. C127's proposed32768+3364KC fee,48515328 ordinary proof bound and
42110976 real-kernel fee saving remain planned scalar quantities, NOT observed
qualified C127 v1 results. No complete worker memory, ledger, cost, physical,
source-equivalence or exact-evidence gates were evaluated for v1.

## Permitted next version and unchanged boundaries

A separately named v2 may correct the missing false branch and redirect its
new wrapper/import chain to that corrected file. It must explicitly authenticate
this v1 failure and preserve the immutable v1 sources/results. This is a changed
executable program, not an unchanged retry or a bookkeeping relabeling of green
results. The28 new test assertions and all3556 inherited qualified tests remain;
full collection/execution and the four complete source fixtures must qualify
anew. No v2 success is asserted here.

No scope narrowing, tariff reduction, cap increase, proof omission, easier
fixture, cache credit or relaxed guard is authorized by this defect. Whole256M,
branch200M, the254M category ceiling, both1GiB gates,64M entries, native-window
and512-bit guards, complete source/owner/inverse/evidence scope remain unchanged.
Formal baseline1870/2413 and all13 family counts are unchanged; independent E0
remains CIFAR25/Tiny36=61/400. No production/default/commit/push action, archived
HZ mutation, network/solver run or formal advancement occurred.
