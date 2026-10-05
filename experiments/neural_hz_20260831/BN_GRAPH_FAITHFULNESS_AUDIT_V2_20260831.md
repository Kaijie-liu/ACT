# BN Graph Faithfulness Audit V2

Recorded on 2026-08-31 on branch `redu-hz` at commit
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`. This is a read-only rebuild of
the registered TinyImageNet iid143 model/spec followed only by the isolated,
default-off BN graph certificate. It ran no propagation, transfer function,
solver or verifier. It changed no production source, historical result,
`/data1/Kane/HyZor` file or Trial9 file.

The formal baseline remains exactly **1,870/2,413** with all 13 family counts
as hard lower bounds. V2 gain is zero and default enablement remains false.

## Why V1 is superseded

V1 remains preserved at
`evidence/tiny_iid143_bn_graph_faithfulness_audit_v1.json`, SHA-256
`9ace466d6bd2ba27439289aac1cc1752c8d0f8b2cf3c47b2782dda83175705d6`.
It is superseded as current evidence because:

- its graph digest did not bind the numerical `SCALE.a` and `BIAS.c` payload;
- it preceded exact width, one-dimensional real-numeric and finite checks for
  every marked BN payload;
- it preceded strict SSA/single-producer and multi-operand arity checks;
- its clone consumer did not independently rederive the repair plan; and
- it preceded the post-callback source/private-evidence dual CAS and the
  independent adversarial suite.

The first V2 real-graph preflight published no JSON because the newly strict
SSA rule initially rejected layer 80: ACT's final ASSERT intentionally aliases
layer 79's variables with exact `out_vars == in_vars`. The rule was narrowed
before this record: only nonempty exact INPUT_SPEC and ASSERT aliases are
accepted; partial ASSERT aliases remain rejected. The extra regression test
raises the requested 77-test checkpoint to the current authoritative
**78/78 passed**.

## Current isolated implementation evidence

The combined official and adversarial suites pass:

```text
........................................................................ [ 92%]
......                                                                   [100%]
78 passed in 0.70s
```

The current source/test hashes are:

```text
69e669ee5b6c4e51213f0742b7e3ac9cce1ff07ad1378611fcb8c514d4d72a8d  bn_graph_faithfulness_certificate_prototype.py
632e2ce90cb56a8585563154cefa2b75c074f6c184835267c2dbd3bc1cc2af0f  test_bn_graph_faithfulness_certificate_prototype.py
a57a9d4303c51c67a4c6a94bdcc2d2d292eb5bc2c144ee2b9d4a8d2c36e9dd32  test_bn_graph_faithfulness_certificate_adversarial.py
```

The hardening now binds every BN layer's payload key, source dtype, exact
width and canonical finite float64 bytes into the graph digest. It rejects
missing, nonnumeric, Boolean, complex, nonfinite, wrong-rank and wrong-width
payloads. Every public repair plan is independently rederived from the current
source graph before use. A candidate callback is followed by a fresh source
plan/CAS and a fresh private clone/certificate comparison before publication.

## Actual Tiny143 V2 rebuild

The strict parser accepts the real ACT objects without coercion:

- 81 layers;
- 88 predecessor entries;
- 1,475,736 checked input-variable occurrences; and
- 19 complete marked BatchNorm SCALE/BIAS pairs.

The source graph still rejects for exactly the known systematic loader fault:
19 `predecessor_producer_mismatch` plus 19
`bn_scale_bias_graph_event_missing` issues, at BIAS layers
`4,8,12,15,19,23,27,31,35,39,43,47,50,54,58,62,66,70,74`.
There is no malformed-input, extra SSA, operand-arity or edge-asymmetry issue.

Because V2 binds all BN numerical payloads, its graph digests intentionally
differ from V1:

```text
source     1993b610eb5bac28e5246cba3d994f81be995e94ad1cd39afa05b1bc9195cd60
candidate  51c3e6fb0309e9add6063958a6fb891249bf495d7ffacc939346a33bf8952833
```

The one generic all-or-nothing plan contains 19 sibling-to-chain predecessor
replacements. It is independently rederived before clone use. The resulting
hypothetical clone passes the complete certificate with 81 layers, 88 edges,
19 pairs and zero issues. This remains a clone capability result, not a
production patch or baseline result.

## Numerical payload evidence

All 38 marked BN payloads are one-dimensional float64, exact-width and finite:

| event | width | pairs | total entries | relevant nontrivial entries |
|---|---:|---:|---:|---:|
| SCALE | 46,656 | 1 | 46,656 | 46,656 unequal to 1 |
| SCALE | 25,088 | 9 | 225,792 | 225,792 unequal to 1 |
| SCALE | 6,272 | 9 | 56,448 | 56,448 unequal to 1 |
| BIAS | 46,656 | 1 | 46,656 | 46,656 nonzero |
| BIAS | 25,088 | 9 | 225,792 | 225,792 nonzero |
| BIAS | 6,272 | 9 | 56,448 | 56,448 nonzero |

For the fixed ReLU36 path:

| layer | kind | width | payload SHA-256 | min | max |
|---:|---|---:|---|---:|---:|
| 30 | SCALE | 25,088 | `94847bf4...bc150` | -0.04747119835380177 | 2.1341762976063428 |
| 31 | BIAS | 25,088 | `0ba5c2cb...f4aa` | -0.9691451270649756 | 0.8492648103052189 |
| 34 | SCALE | 25,088 | `37aa39c0...a2c7` | 0.00014627055224732068 | 0.02266037406504889 |
| 35 | BIAS | 25,088 | `7d240abf...c476` | -3.2770164686075174 | 0.026329170991795436 |

The exclusive JSON contains every full 64-hex payload digest, not the
abbreviations used in this table.

## ReLU36 lineage diagnostic remains NO-GO

Against the corrected hypothetical graph, the registered path is:

```text
Conv29 -> Scale30 -> Bias31 -> ADD32
      -> Conv33 -> Scale34 -> Bias35 -> ReLU36
```

The content-addressed Trial8 lazy-core interpretation provides Conv29 and
Conv33 observations. The V2 path validator therefore rejects with exactly:

```text
unaccounted_linear_event  Scale30
unaccounted_linear_event  Scale34
```

Trial8 remains SHA-256
`f1f6abcdc87d96e9d8898d0f667d8087b2a72a9898f44d61b4e123426d45b274`,
verdict UNKNOWN. No new cache or verifier execution was used.

The path observations still come from the prior content-addressed
interpretation, not an immutable production runtime-lineage adapter, and they
do not yet bind live operator payload digests. Consequently C3 identity-middle
and any HZ gain remain **NO-GO** even though the graph clone is faithful.

## Reproducible exclusive runner

The new runner is
`run_tiny143_bn_graph_faithfulness_audit_v2.py`, SHA-256
`07cea9d717ae42415a64093d523ea61af7209d55a06c41806fea6e359ada4d68`.
It requires explicit ACT root, benchmark root, iid, converted spec, Trial8 and
output paths. It validates the registered target, rebuilds the graph, runs the
78 frozen tests, performs both certificates, and publishes JSON through an
exclusive same-directory temporary file plus atomic hard-link. A second run
refuses the existing target before model loading.

The exact command used was:

```text
python experiments/neural_hz_20260831/run_tiny143_bn_graph_faithfulness_audit_v2.py \
  --act-root /data1/Kane/FSE/ACT \
  --benchmark-root /data1/Kane/data/vnncomp2025_benchmarks/benchmarks/tinyimagenet_2024 \
  --iid 143 \
  --converted-spec /data1/Kane/FSE/ACT/experiments/neural_hz_20260831/vnnlib_v2_full_v1/tinyimagenet_2024/vnnlib/TinyImageNet_resnet_medium_prop_idx_3553_sidx_3392_eps_0.0039.vnnlib \
  --trial8 /data1/Kane/FSE/ACT/experiments/neural_hz_20260831/results/trial8_phase_selective__tinyimagenet_2024__iid143__relu63_v1.json \
  --output /data1/Kane/FSE/ACT/experiments/neural_hz_20260831/evidence/tiny_iid143_bn_graph_faithfulness_audit_v2.json
```

Published evidence:

```text
31ad2f6b0d5a30cff226340d554ff9fa92883c894a3b9f1c0e02dcfd4f8f2c2a  evidence/tiny_iid143_bn_graph_faithfulness_audit_v2.json
```

## Advancement boundary

The V2 record proves a systematic graph/variable discrepancy, complete
numeric-payload-bound clone repair mechanics, and the continued missing-Scale
lineage diagnosis. It does not prove forward/HZ equivalence, a runtime
lineage adapter, production integration, any family result or any score.

A production loader repair plus runtime lineage adapter must first pass exact
forward/HZ equivalence and then the complete 2,413-row replay. Enablement still
requires retention of all 1,870 solved rows, every one of the 13 family counts
and every validated ADV witness.
