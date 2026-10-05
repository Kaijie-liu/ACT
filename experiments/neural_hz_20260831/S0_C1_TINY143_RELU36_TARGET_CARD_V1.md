# S0-C1 Tiny iid143 ReLU36 Target Card V1

This is the first fixed target in the S0-C1 expansion order. It records a
structure and resource preflight, not a verdict or score. The formal baseline
remains 1,870/2,413 and E0 remains 61/400 with no Neural-HZ gain credit.

## Frozen structural evidence

The source record is
`results/trial8_phase_selective__tinyimagenet_2024__iid143__relu63_v1.json`,
SHA-256
`f1f6abcdc87d96e9d8898d0f667d8087b2a72a9898f44d61b4e123426d45b274`.
It reached the layer-63 census stop and records at layer 36:

- output shape `128 x 14 x 14 = 25,088`;
- 20,648 interval-stable-negative, 2,268 stable-positive and 2,172 unstable
  rows;
- the frozen phase-selective probe takes all 2,172 unstable rows plus the first
  8 stable-positive rows, hence exactly 2,180 selected rows;
- four lazy residual terms exist at the preceding ADD layer 32; after the
  layer-36 phase-selective transform, its retained expression has five terms
  because the exact unstable core is added as one same-frame identity term;
  and
- the relevant ordinary convolution geometry is 128 input, middle and output
  channels with two 3x3 convolutions. A later real-graph audit found that the
  emitted Scale30/Scale34 layers are successorless siblings rather than part
  of the Bias/ADD/ReLU graph path; they cannot be assumed to be middle
  diagonals in the real lazy tuple.

The source JSON does not expose every pre-ReLU operator tuple or unique
descriptor count. A subsequent graph-lineage audit nevertheless proves the
strict S0-C1 match count without guessing payloads: Conv29 is before ADD32,
Conv33 is the only Conv after ADD32 and before ReLU36, and no lazy checkpoint
resets the source. Every two-Conv terminal core therefore crosses the ADD32
cut and is forbidden by the frozen v1 rule. Trial 9 remains the authoritative
runtime tuple/profile census, but it cannot change this topological zero-hit.

## Hardened V2 hypothetical diagonal-chain preflight

For the originally assumed diagonal-chain geometry and exactly 2,180 selected
rows, hardened isolated
V2 source SHA-256
`5f7963fdec0924148ee683bf87289f4cd2bb8b5de5c46120f44de5ab3823c646`
computes the following before coefficient compilation:

| Ledger | Exact preflight value |
|---|---:|
| selected / active rows | 2,180 / 2,180 |
| conservative unfused path products | 2,445,737,984 |
| group-valid contraction products | 169,869,312 |
| selected emission contributions | 19,107,328 |
| contraction + emission | 188,976,640 |
| unfused/fused work ratio | 12.942012219076389 |
| coefficient entries | 1,327,104 |
| selected structural logical nnz upper | 5,815,808 |
| full structural logical nnz | 67,108,864 |
| descriptor resident bytes | 10,616,920 |
| selected CSR buffer upper bytes | 93,070,376 |
| immutable input snapshot bytes | 2,359,296 |
| compilation numeric-buffer envelope | 10,753,176 |
| controlled transient numeric-buffer envelope | 103,687,296 |

Thus that hypothetical descriptor is below every frozen local v1
arithmetic/resource gate:
200M contraction, 256M transaction work, 2M coefficients, 64 MiB resident,
1 GiB controlled transient, 64M selected result nnz and the one-quarter work
rule. `full structural logical nnz` is not the selected-result cap and includes
structural zero positions by protocol.

The preflight intentionally supplies no favorable physical comparator, so its
decision is `physical_metric_not_reduced`. That is the correct current status.
The complete four-term transaction must still prove cumulative budgets and a
strict reduction of the registered whole reachable physical state. A local
12.94x work bound cannot substitute for that proof.

## S0-C1/C2 decision and next admissible transition

S0-C1 v1 is a structural miss at this target and may not enter production
integration. Missing production ADD metadata and synthetic `Barrier("ADD")`
objects cannot be used to reinterpret “never crosses ADD.” Trial 9 may finish
and seal its isolated census, with gain zero.

The separately preregistered S0-C2 residual-distributive theorem subsequently
passed isolated lineage tests, but its frozen nonidentity branch requires one
or more middle diagonals. The authoritative real graph instead yields
`Conv29 | ADD32 | Conv33`, so C2-v1 is also closed as a structural zero-hit.
Full evidence is in `S0_C2_REAL_GRAPH_ZERO_HIT_AUDIT_20260831.md`.

The missing diagonal cannot be interpreted as identity: Bias31 consumes the
Scale30 output variable IDs, Scale30 has 25,088 nonidentity values, but the
graph predecessor incorrectly points Bias31 to Conv29. Repairing this
graph-event/operator mismatch is an independent loader campaign that must
rebuild the baseline. Any future identity-middle rule must first prove a
complete producer/graph bijection and therefore fails closed on the current
target. That independent rule and its mandatory certificate are now frozen in
`S0_C3_IDENTITY_MIDDLE_PREREG_V1.md`; it is not authorized to run Tiny143 until
the certificate passes. The favorable arithmetic above remains reusable
capacity evidence, not a real match or score.
