# N015 preregistration: E0 single-path replay of Neural-HZ candidate path v1

Written and frozen before the first full-universe row is run (2026-10-02,
Australia/Sydney). Smoke rows 2, 28 and 76 of CIFAR were run once during
development (`results/n015_smoke_v1.jsonl`); they are not part of this
replay's result file.

## Candidate

Code frozen by `FREEZE_SHA256SUMS` in this directory:
`nhz_engine.py` (VNNLIB parser, ONNX attribute helper), `nhz_sound.py`
(sound engine n004.2), `nhz_terminal.py` (rigorous terminal LP and
level-plan MILP), `run_n015_e0_sound_pipeline.py` (runner). Production ACT is
not imported; default ACT behaviour is unchanged.

One configuration for every row, no per-row menu:

1. sound projection-aligned HZ propagation in float64 with rounding radii and
   300-iteration batched LP tightening (lr 0.05);
2. rigorous batched terminal LP (1000 iterations) for every unsafe disjunct;
3. witness candidates from the box maximiser at the LP multipliers;
4. open disjuncts, worst first: level-plan MILP (gamma = last ReLU layer with
   phases, lambda = all layers) solved by HiGHS 1.14 with objective cutoff,
   single thread, remaining budget as time limit; MILP incumbents are also
   witness candidates;
5. outcome: CERT if every disjunct is excluded with margin 1e-4; ADV if an ONNX
   Runtime replay of a candidate satisfies the original VNNLIB unsafe
   disjunction; TIMEOUT if the 100-second per-row budget is exhausted (a
   CERT/ADV found after the budget is recorded but counted as TIMEOUT);
   UNKNOWN otherwise; ERROR on exceptions.

Witness decoding is a terminal-query primal heuristic (LP rounding / MILP
incumbent) validated by an independent concrete replay. It is reported
separately and is not a representation claim. The user must decide whether it
is admissible under the no-attack rule; until then E0 ADV counts from this run
are provisional.

## Universe and ledger rules (unchanged, from GOAL_CHARTER_E0_AMENDMENT)

All 400 rows of `evidence/{cifar100,tinyimagenet}_2024_evidence_baseline_v2.json`,
model and spec SHA-256 checked per row, timeout 100 s per row. E0 baseline:
0 CERT + 61 validated historical-origin ADV + 339 UNKNOWN.

- Soundness conflict: CERT on any of the 61 ADV rows. Any conflict fails the
  candidate.
- Retention: all 61 ADV rows must be ADV. Any loss fails E0 promotion (the
  float probe N007 already lost 4/25 CIFAR rows, so failure is expected).
- Gain: CERT or ORT-validated ADV on the 339 UNKNOWN rows.
- Invalid ADV: by construction zero (every ADV is an ORT replay of the original
  property at zero tolerance on the float32 model input).

Promotion of the E0 ledger requires retention, zero conflict and at least one
gain. Regardless of outcome, the result is capability evidence for the
Neural-HZ tower; it never changes the formal 1870/2413 score.

## Resources

Shared GPU (other users' jobs present), `torch.cuda.set_per_process_memory_fraction(0.3)`.
CIFAR first, then TinyImageNet. Wall-clock times are measured under shared
load and are not a speed claim.
