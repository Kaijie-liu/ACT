# D002 saved-only structural relevance evidence

Date: 2026-09-28; branch `redu-hz`, commit
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`.
Only existing JSON graph descriptors, reports and source were read. No current
model/HZ was loaded, coefficient tensor scanned, solver called or test run.
This is not a new benchmark verdict or a complete current-source census.

## Tiny graph

Source: [Trial8 saved graph](../results/trial8_phase_selective__tinyimagenet_2024__iid143__relu63_v1.json),
SHA256 `f1f6abcdc87d96e9d8898d0f667d8087b2a72a9898f44d61b4e123426d45b274`.
All 81 descriptors are saved even though numerical propagation stopped early.

| ReLU id | Immediate successor id / kind |
| --- | --- |
| 5 | 6 Conv, 13 Conv |
| 9 | 10 Conv |
| 20 | 21 Conv |
| 28 | 29 Conv |
| 36 | 37 Conv |
| 44 | 45 Conv |
| 55 | 56 Conv |
| 63 | 64 Conv |
| 71 | 72 Conv |
| 78 | 79 Dense |

The later [BN graph audit](../evidence/tiny_iid143_bn_graph_faithfulness_audit_v2.json),
SHA256 `31ad2f6b0d5a30cff226340d554ff9fa92883c894a3b9f1c0e02dcfd4f8f2c2a`,
records an authorized clone-only repair of the known SCALE/BIAS sibling-edge
defect (`production_patch:false`). It does not turn these
ReLU->Conv/Dense boundaries into diagonal-only threshold chains. Do not
interpret the older graph's dead-end SCALE successors as correct semantics.

The corrected [S0 path preflight](../evidence/s0_c3_graph_preflight_20260905_v1.json),
SHA256 `7999b02f1af74667a354f59497b9763e2374742bd9d6deead4c53f94c96ae106`,
records all four source paths into Bias35/ReLU36. The closest path is
ReLU28 -> Conv29 -> Scale30 -> Bias31 -> Add32 -> Conv33 -> Scale34 -> Bias35.
All four contain convolutions. Its old `structurally_impossible` result was
about C3's own shape rule, not D001; we use the recorded paths, not that flag.

## CIFAR graph

Source: [Trial6 saved graph](../results/trial6_cnn_census__cifar100_2024__iid166__v1.json),
SHA256 `03aff8b0cb3aaa80a15ff27774cb74dfef599f180849878a098c807b30f66ce4`.
It records 44 descriptors. ReLU ids 3,5,9,13,18,22,27,31,36 feed Conv;
ReLU41 feeds Dense42. ReLU3 also feeds Add7, whose downstream path encounters
mixing before another ReLU. There is no saved diagonal-only serial motif.

This old census has `missing_hz_state` and `output_hz_exact:false`. Its zero
sharing counters do not establish a mathematical absence of shared rows.
They reflect failure to reach a complete HZ state. The analogous early Tiny
census has the same evidence limitation and a different, 43-descriptor graph;
it must not be conflated with the 81-descriptor explicit-BN graph.

## Scope and other-family context

BN layer type does not guarantee positive scale or negative bias. The source
conversion uses gamma/sqrt(var+eps) and beta-scale*mean. Some saved BN extrema
are favorable, but their intervening Conv remains. A convolution might have
special diagonal action on particular coordinates or constrained subspaces;
no such full-coefficient proof was performed, so this audit neither assumes
nor rules it out.

The earlier TLL census summarized in [README](../README.md) found exact
duplicate/signed groups. Saved iid7/9/15 examples have duplicate versus
positive-proportional group counts 68/68, 144/144, 244/244, and signed-duplicate
versus signed-proportional counts 313/313, 740/740, 1222/1222. These support
old signed-sharing, not a new population of threshold chains. Historical TLL
partial/union capability results are not formal promotion.

Decision: do not implement a scalar-chain-specific domain as the presumed
CIFAR/Tiny breakthrough. Use ordinary mixing Conv/Add/ReLU structures for
the next domain definition/comparator. This is a relevance decision under
limited saved evidence, not a claim that either family has no opportunities.
