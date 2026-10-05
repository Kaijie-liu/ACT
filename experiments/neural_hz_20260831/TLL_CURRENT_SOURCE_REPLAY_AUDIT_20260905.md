# Current-source TLL result: formal retention passes, candidate qualification fails

Date: 2026-09-05. `redu-hz`, base
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`.

The complete V2 run reports **11 CERT + 16 validated ADV + 5 UNKNOWN** at the
fixed 45-second solver budget and four-way concurrency. All 17 formal TLL
solved rows remain solved. The previously observed 29/32 candidate is not
reproduced: iid8 and iid26 return UNKNOWN. The pre-registered qualification
therefore fails, and cross-family/default/score promotion does not follow.
The 27/32 result has 10 solves beyond the formal 17, all capability-only until
the outstanding gates and complete 2,413 replay pass. No historical union with
iid8/26 is substituted for this one-source result.

## Preserved attempts

V1 finished all 32 rows with the same loader StopIteration before HZ execution.
The current converted TLL module has zero parameters and floating weights in
buffers (28 buffers on iid0, all floating buffers float64). A versioned worker
fixes only dtype discovery and its provenance list; the original worker and
all V1 records remain intact. V1 summary SHA-256:
`4d1e0ec18aae0ed99ffb21eec6da76eee5dc53c9aeba0c651359f981fc1594c1`.

V2 reran every row uniformly with the same HZ rule, sparse representation,
45-second solver budget, 16 GiB per worker and 240-second supervisor wall cap.
All 32 process exits and result/log files are retained. Model, original spec,
converted spec, source and config hashes show no drift. The conversion's
assertion text is checked against the original for every row. Total batch
elapsed time was 121.863 seconds; this is not a controlled speed comparison.
V2 summary SHA-256:
`fb30cfabcb824b47592bbdc7771c952f075dc42a2713aea2da316333674cdfe0`.

## Two candidate regressions

- iid8: previous CERT in 15.806 solver seconds; current UNKNOWN at 45.005
  seconds. Both records have 282 continuous and 280 binary lowered variables,
  840 rows and 3,646 predicate nnz, with the same 243 eliminated groups.
  Matching counts do not prove coefficient identity or establish whether the
  difference comes from numerics, the dependency stack or concurrency. No
  solver-seed search, selected retry or enlarged budget was used.
- iid26: solver proposed `[-2.0000000000000004, -0.93239752775598]`; the first
  coordinate lies one binary64 step below the exact input lower bound -2.
  The worker's strict concrete check rejected it and returned UNKNOWN. The
  witness was not clamped, repaired, rounded into the box or credited.

## Precise invalid-result accounting

The conservative supervisor labels iid26 ERROR and increments its field
`invalid_adv`, because that field counts any rejected solver proposal. That
label must not be interpreted as an invalid ADV reported by the verifier.
The untouched raw worker record explicitly has verdict/status UNKNOWN and
`reason=invalid_concrete_witness`.

Authoritative distinctions from the raw records are:

- reported invalid ADV: **0**;
- rejected solver witness proposals: **1**;
- concretely validated reported ADV: **16**;
- raw worker counts: 11 CERT / 16 ADV / 5 UNKNOWN;
- stricter supervisor counts: 11 CERT / 16 ADV / 4 UNKNOWN / 1 ERROR;
- lost formal solved: none;
- lost prior candidate solved: iid8 and iid26; and
- qualification/default/score promotion: false.

All 16 reported ADV also pass independent CPU ONNX Runtime evaluation and the
original VNNLIB evaluator at zero tolerance. The replay records the precise
native ONNX dtype input, any cast change from the stored float64 input, the
native output and both input/property checks. It does not process or rescue
the rejected iid26. Evidence:
`evidence/tll_reported_adv_onnx_replay_20260905_v1.json`, SHA-256
`5b468ecbe001a8e0a69d5d369e1e098e4894c94496975e31e19facf69b72a2be`.

## Verification and remaining work

Current isolated suites: **734 passed, 1 failed**. The only failure remains
the old live Overleaf table hash check. The original exact table was recovered
from Git and the complete frozen manifest rebuilt byte-identically in the
separate recorded authority replay. No test, original generator, old manifest
or original authority hash was weakened. New dtype tests pass 4/4; loader
tests 13/13; C3 preflight tests 9/9; runtime adapter tests 264/264.

Next work is to distinguish coefficient/dependency differences from concurrent
solver performance under a controlled, fully registered comparison, while
retaining the strict concrete input check. The generic BN fix remains off by
default. C3 stays closed; future Tiny/CIFAR representation work needs a
separately registered nested residual DAG rule. The full goal is still open:
formal 1870/2413 and external E0 61/400 have not changed.
