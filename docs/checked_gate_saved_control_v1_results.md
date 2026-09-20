# Checked gate control: narrower range, request still unclosed

Preparation was committed/pushed as `1ba13b8f8` before execution. This new
supplied-evidence control follows `docs/checked_gate_v1.md`; original R1 files,
precision settings and production acceptance rules are unchanged. It does not
reopen the four-LP SoPlex study or the sealed AdvMoE searches.

## Result

The fixed first-family request seed1/index4018 has three declared legal pairs
and27 output obligations. Fresh independent checking reproduces26 positive
obligations and one nonpositive blocker: pair[3,5], property1. The revised
gate uses the same checked router bounds, exact Taylor/interval arithmetic,
and the same old weighted dual multipliers; no optimizer is invoked.

| Quantity | Old proof | New fixed-dual control |
|---|---:|---:|
| Gate lower | 0.5 | 0.5157532095909119 |
| Gate upper | 1 | 0.7666619420051575 |
| Checked blocking lower bound | -0.6040892741185394 | -0.5517098676561972 |
| Complete positive request | No | No |

Exact gate endpoints are `8652903/16777216` and `12862453/16777216`.
The exact bound and hashes are in `checked_gate_saved_control_v1.json`.
The result is **not** evidence that the new LP cannot prove the property:
the multipliers were deliberately not optimized for its new coefficients.
It does show that simply checking those same multipliers with the stronger
gate does not close the full request. The width reduction is not counted as
an additional certificate or a performance win.

The analytic suite passes5 tests, including12 corrupted-proof subcases,
seven high-precision test-oracle endpoints, exact ties, sign crossing,
bounded-input admission and a weighted safety control. Decimal exponentials
appear only in tests; the checker uses exact rational arithmetic and calls
neither the producer nor numerical libraries.

The saved control completes in25.5586s under an external300s cap. It rehashes
parent dependencies, rechecks all27 old obligations, constructs/checks the
new gate and weighted LP, and serializes its three artifacts. Those costs
start from stored HZ evidence; original propagation and proposal costs are
excluded and no production300s speedup is claimed. There are0 model calls,
0 optimizer calls and0 new dual searches. Gate proof size is2,238 bytes;
the weighted export is17,296,182 bytes, and the certificate2,290 bytes.

A separate standard-library process independently rechecks the source router
certificates (not merely the manifest's displayed bounds), the original
obligation inventory, the new sigmoid proof, exact weighted construction,
unchanged duals and the resulting bound. PASS,0issues,26.3351s separately.
The generator consumed hash-bound saved margin metadata; this fresh review
additionally confirms equality to freshly recomputed exact certificate bounds.
No historical result is relabelled and no extra cases or higher precision are
tried after observing this negative result.

## Why the requested stronger claims remain unavailable

- AdvMoE has85.67% clean accuracy but its two global hard-top1 paths are not
  the weighted-top2 objects supported by this evidence interface. The frozen
  strict pilot remains0/2; the100-row numerical study has no numerical
  two-path-positive/observed-route-flip intersection.
- The convolutional family has67.06% clean accuracy; its full comparison and
  new-input evidence study still provide no cross-family route-changing strict
  certificate. Input98 remains a single-pair conditional positive control.
- This new gate control belongs to the first-family48%-accuracy model. It
  cannot provide either missing empirical claim, even if its bound had crossed
  zero. Network-to-HZ, guard lowering and route exclusions also remain trusted.

The request is therefore **not completed**, and stronger paper claims are
**not upgraded**. Research targets are not success conditions that can be
satisfied by changing wording. The next complete-proof change must separately
address evidence strength and source semantics, with a new scoped protocol;
there is no automatic new optimizer search, retraining, larger budget or
unsealed holdout in this negative control's disposition.
