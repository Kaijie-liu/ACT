# BN Graph Faithfulness Audit

Recorded on 2026-08-31 on branch `redu-hz` at commit
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`. This is an isolated,
default-off loader-foundation audit. It modifies no production source, runs no
verifier, writes no historical result and does not touch the running Trial 9.
`/data1/Kane/HyZor` remained read-only.

The formal baseline remains exactly **1,870/2,413**, and every solved count in
all 13 families remains a hard lower bound. This audit has gain zero. A clone
certificate is not a production repair and cannot authorize C2, C3, a family
claim or any score change.

## Result

The TinyImageNet iid143 rebuild contains 81 ACT layers and 19 decomposed
BatchNorm pairs. All 19 have the same graph/variable disagreement:

```text
variable program: upstream -> SCALE -> BIAS
ACT edge graph:   upstream -> SCALE
                         \-> BIAS
```

The source graph certificate checks 1,475,736 input-variable occurrences and
88 predecessor entries. It rejects with exactly 38 issues:

- 19 `predecessor_producer_mismatch` issues, one at each paired BIAS; and
- 19 `bn_scale_bias_graph_event_missing` issues at the same layers.

There are no malformed-layer issues, no predecessor/successor asymmetries and
no other producer mismatch. The affected BIAS layer IDs are
`4,8,12,15,19,23,27,31,35,39,43,47,50,54,58,62,66,70,74`. Thus this is a
systematic conversion fault, not a Tiny iid143/layer-36 selector.

The strict parser accepted the real `Layer.kind`, `in_vars`, `out_vars`,
`preds` and `succs` objects without coercion. Exact nonnegative `int` IDs,
exact `bool` BN markers, contiguous layer IDs and complete edge-map keys are
required. Malformed values fail closed.

The source certificate digest is
`5f5b63d0f0d914d887496ace7d3fdf9059cba03ce1168cc4c3dce72219a1820c`.

## Clone-only repair plan

The prototype recognizes only the complete sibling failure shape. For each
paired BIAS it requires:

1. its `in_vars` to equal the immediately preceding marked SCALE's
   `out_vars`;
2. the SCALE's own producer edge to be faithful;
3. BIAS and SCALE to currently share exactly the same upstream predecessor;
4. the complete source graph to be predecessor/successor symmetric; and
5. the graph obtained by all replacements to pass the full producer
   certificate with no remaining issue.

The real graph produces one all-or-nothing 19-edge plan. It changes each BIAS
predecessor from its upstream layer to its SCALE layer. The plan is bound to
the exact source digest; application to a changed graph fails its CAS check.
It creates private predecessor/successor containers, recomputes inverse edges
and returns an immutable clone. It never mutates a `Net`, `Layer` or
caller-owned dictionary.

The resulting hypothetical clone passes all graph/variable checks with 81
layers, 88 predecessor entries and 19 complete BN pairs. Its digest is
`0681a567dd0a008dbbc8a83f2829652209f0b3dbad66f29640da1bb19995430e`.
This proves that the one generic edge rule is sufficient for this graph. It
does not prove forward/backward/HZ equivalence, retained-set stability or a
safe production patch.

## ReLU36 graph-to-HZ consequence

The corrected hypothetical graph makes the target path explicit:

```text
Conv29 -> Scale30 -> Bias31 -> ADD32
      -> Conv33 -> Scale34 -> Bias35 -> ReLU36
```

The path validator treats Conv/Scale/Dense as multiplicative graph events and
BIAS only as a bias transition. Every multiplicative event must have exactly
one HZ-lineage observation with the same occurrence and semantic kind, in
order. Missing, extra, duplicated, reordered or wrong-kind observations all
reject.

Using the already recorded Trial 8 lazy core observations
`(Conv29, Conv33)` against that corrected path rejects with two
`unaccounted_linear_event` issues: Scale30 and Scale34. This is diagnostic
reuse of content-addressed Trial 8 evidence; no new cache or verifier run was
performed. Both scales are provably nonidentity:

| layer | entries | entries unequal to 1 | min | max |
|---:|---:|---:|---:|---:|
| 30 | 25,088 | 25,088 | -0.04747119835380177 | 2.1341762976063428 |
| 34 | 25,088 | 25,088 | 0.00014627055224732068 | 0.02266037406504889 |

Therefore the present C3 identity-middle target remains **NO-GO**. The pure
path validator exists, but no production runtime-lineage adapter has yet
proved that live HZ terms emit immutable observations for every corrected
graph event. Hand-authored observations cannot discharge that obligation.
The observation currently binds occurrence and semantic kind, not a digest of
the live kernel/scale payload; a production adapter must add and compare that
current-payload snapshot rather than trust a constructor-time label.

## Transaction and adversarial coverage

The isolated test file has 36 passing tests. Coverage includes:

- faithful BN chain, historical sibling graph, missing/ambiguous pair and two
  simultaneous pairs;
- exact marker/type parsing, latest producer after aliasing, symmetric graph
  enforcement and digest CAS;
- multi-operand order, including explicit rejection when one logical operand
  spans multiple producers;
- immutable clone/no caller mutation and no repair menu on an already faithful
  graph;
- Conv/Scale path coverage, BIAS transition, missing/extra/wrong/duplicate/
  reordered operators, skipped graph events and rejection of unregistered
  path kinds; and
- ordinary `Exception` rollback plus `KeyboardInterrupt`/`SystemExit`
  propagation after private staging, without publication.

The prototype imports no production ACT module. An ordinary callback
`Exception` returns a rejected transaction with no clone. A `BaseException`
outside `Exception` discards local staging by stack unwinding and propagates.

## Evidence and hashes

The compact, exclusive audit record is
`evidence/tiny_iid143_bn_graph_faithfulness_audit_v1.json`. It contains the
full affected-layer and repair tuples, the ReLU36 path diagnostic, variable
endpoints, scale statistics and all no-claim flags.

```text
177fd6a8f3ba178431039f5a39250378a8b07d366c2566cf267ce5c17ca9daeb  bn_graph_faithfulness_certificate_prototype.py
6f67393ad8ab19f8c2f4038a60153ada12b627a2692d7f7fe29fb691d8adddd7  test_bn_graph_faithfulness_certificate_prototype.py
9ace466d6bd2ba27439289aac1cc1752c8d0f8b2cf3c47b2782dda83175705d6  evidence/tiny_iid143_bn_graph_faithfulness_audit_v1.json
d6cb15ec6b0dedef0d30bd0441f1dc98e02f6202bcb1e279c337d219c7c33d44  act/pipeline/verification/torch2act.py
8c2ebc9c3cfac78360af36770f54990704c2e031d4c42ddb58532e515c6762a9  act/back_end/hybridz_tf/tf_cnn.py
f1f6abcdc87d96e9d8898d0f667d8087b2a72a9898f44d61b4e123426d45b274  results/trial8_phase_selective__tinyimagenet_2024__iid143__relu63_v1.json
```

## Advancement boundary

A production loader repair is a separate correctness campaign. Before it can
be enabled, it needs forward semantic comparison, graph/HZ runtime-lineage
coverage, affected-family replay and finally the complete 2,413-row replay.
Enablement requires all 1,870 baseline solved rows, every one of the 13 family
counts and all validated ADV witnesses to remain intact. Until that replay,
the formal baseline, E0 evidence and Neural-HZ gain are unchanged.
