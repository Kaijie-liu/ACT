# AdvMoE router bracket and staged-verifier protocol

## Methodological regime

AdvMoE supplies the nonlinear-CNN-router point in a three-level route-analysis
ladder:

1. affine RT-ER router: exact closed-form pixel-space boundary;
2. verification-scale one-hidden-layer router: exact unrelaxed HZ feasibility;
3. AdvMoE CNN router: attack/lower-bound bracket with an explicit undecided
   band.

For `E=2`, clean route `e`, and competitor `1-e`, the property is the scalar
margin `r_e(x) - r_(1-e)(x)`. A concrete PGD input that changes literal router
argmax is a route-instability witness. A backend lower bound above the frozen
positive tolerance is evidence for stability only at that backend's numerical
soundness level.

## Numerical discipline

The installed auto_LiRPA path is not outward-rounded. Therefore this project
does **not** promote its positive IBP/CROWN lower bounds to formal `SAFE` or
formally route-stable. Pilot states are:

- `ATTACK_CONFIRMED_ROUTE_UNSTABLE`: concrete adversarial input replays to the
  other route within the registered box;
- `POSITIVE_NUMERICAL_BOUND_FILTER`: finite lower bound at least `1e-7`, with no
  conflicting witness;
- `UNDECIDED`: neither condition;
- any positive-filter/witness overlap is an audit failure.

Formal route-stability counts remain zero until an outward-rounded or otherwise
validated backend is supplied. Negative relaxation bounds are always UNKNOWN.

## Source-to-CROWN adapter

The literal router is first tested and its current auto_LiRPA rejection is
recorded. A fixed-shape adapter then performs two exact 32x32 specializations:

- channel-padding strided slices become fixed 1x1 stride-2 identity
  convolutions;
- dynamic full-spatial average pooling becomes `AvgPool2d(8)`.

Random, zero, and one inputs must produce bit-identical scores and routes before
any bound is accepted. The adapter rejects use outside `[B,3,32,32]` at its
entry gate.

## Deep-path specialization

For route 0 or 1, every MoE convolution is replaced by a static convolution
containing the selected contiguous weight slice. All 16 replacements must
succeed. Dynamic full-model output and the corresponding specialized path must
agree for concrete inputs within tolerance. The route-specialized model has no
dispatch operator and is the eventual expert/property backend input.

The staged verifier is:

1. determine route stability/uncertainty with the router bracket;
2. for a stable route, verify its one static deep path;
3. otherwise verify both static deep paths on the full input box;
4. if a property backend is inconclusive, attacks may establish only
   full-model replayed UNSAFE witnesses;
5. guarded-cell boxing and eta compilation are retained as one bounded
   ablation, with no expected advantage preclaimed.

## Frozen engineering pilot

While B1 runs, only the first 20 ordered CIFAR-10 test inputs are used, over
`{0.5,1,2,4,8}/255`. PGD uses 20 steps, two restarts, and step size epsilon/4.
The bound worker uses CPU, one thread, and IBP solely to validate the harness;
the paper target remains CROWN after the B1 resource gate. This pilot is not a
full-test census and is not used for prevalence or certification claims.

The full AdvMoE line remains bounded to seed-0 official-code training, init and
final full-test bracketed census, deterministic intermediate telemetry subset,
five-radius staged-verifier table, and one guard ablation. No ratio or expert
count sweep is added here.

## First init-pilot result: weakest bracket only

The accepted run is
`data/moe/results/advmoe_router_bracket_init20_20260830_r3`. Its independent
audit is tracked at
`act/pipeline/moe/results/advmoe_router_bracket_init20_20260830_r3.json` and
reports zero issues. The literal router is rejected by the installed
auto_LiRPA frontend (`AssertionError`), while the fixed-shape adapter is
bit-exact on all 20 registered inputs (`max_abs_error=0`, identical routes).

| epsilon | attack-confirmed unstable | positive numerical filter | undecided |
|---:|---:|---:|---:|
| 0.5/255 | 0/20 | 0/20 | 20/20 |
| 1/255 | 0/20 | 0/20 | 20/20 |
| 2/255 | 0/20 | 0/20 | 20/20 |
| 4/255 | 0/20 | 0/20 | 20/20 |
| 8/255 | 0/20 | 0/20 | 20/20 |

Thus all 100 sample-radius rows remain undecided. PGD not finding a flip is not
route-stability evidence. The one-thread IBP relaxation is unusably loose on
this deep init router: the largest absolute lower and upper bounds are
`2.0738076672e10` and `2.8447031296e10`. These magnitudes diagnose abstraction
explosion; they do not describe concrete router margins. The result validates
the orchestration, identity, adapter, fail-closed aggregation, and independent
audit paths only. It is not a route census or a certificate result.

Two earlier launch directories are permanently excluded and retained. The
first invoked the worker as a module, causing Python 3.11 to import ACT before
the local `typing.override` shim. The second invoked it as a file but lacked
the ACT repository root on `sys.path`. Each contains a `FAILED.json` identity
record and neither reached bound construction. The accepted `_r3` run uses the
same frozen mathematics after a worker-bootstrap-only repair; no failed
directory was reused or overwritten.

This result is an intermediate tool-layer state, not a conclusion that the
deep router is intrinsically difficult to certify. Both sides of the bracket
were deliberately weak: interval propagation is the least precise bound tier,
and the attack used only 20 steps and two restarts. The next frozen engineering
pilot therefore strengthens both sides without changing the 20 inputs or five
radii.

## Frozen three-tier nonlinear-router protocol

The nonlinear CNN router uses a staged backend rather than ACT's exact HZ/MILP
path:

1. `IBP`: free diagnostic only; interval explosion is reported, not tuned;
2. `CROWN/alpha-CROWN`: census tier, with sparse intermediate CROWN, bounded
   CROWN batch width, and one sample per bounded graph;
3. `beta-CROWN + BaB`: final-table closure for a frozen unresolved subset only.

The second pilot config is
`act/pipeline/moe/configs/advmoe_init_router_crown_strong_pilot.json`. Its attack
uses 100 steps, 10 vectorized restarts, and a step size of epsilon/4 halved at
50% and 75% of the trajectory. It records clean/attacked margins, fractional
margin compression, clean input-gradient L1/L2/Linf norms, and every best
endpoint. The CROWN worker runs the margin lower bound only, in eval mode, with
sparse intermediate bounds, `crown_batch_size=128`, `max_crown_size=512`, and
one input per graph. Positive numerical lower bounds remain non-formal because
the installed backend is not outward-rounded.

Resource probes establish why these controls are necessary. Default pure
CROWN exceeded 90 seconds on CPU and exhausted the shared GPU after allocating
more than 62 GiB. CROWN-IBP finishes 20 inputs in 0.26 seconds but only reduces
the 0.5/255 bound scale from about `1.94e9` to `1.01e9`. With official-style
sparse intermediate handling, pure CROWN completes one 0.5/255 input in 2.15
seconds on GPU, peaks at 20.98 GiB, and gives `-3.62e8`. This is a real CROWN
bound and a substantial tightening, but still an undecided numerical result.
The full pilot has a 24-GiB worker peak gate and requires 30 GiB free before it
creates a result directory; it never interrupts the protected B1 job.

At initialization, all 19 BatchNorm layers must be in eval mode. Their running
means, running variances, and batches-tracked state are stored as part of the
deployment identity. Init results validate methods only; certification yield is
evaluated on the trained checkpoint. The project does not use exact HZ/MILP to
propagate this 269K-parameter convolutional router.

## Strong-attack and sparse-CROWN engineering result

The accepted result directory is
`data/moe/results/advmoe_router_bracket_init20_20260830_r7_strong_crown`.
Its independently replayed audit is
`act/pipeline/moe/results/advmoe_router_bracket_init20_20260830_r7_strong_crown.json`
and reports zero issues. All 100 stored attack endpoints replay inside their
registered boxes with the same margins and routes.

| epsilon | route flips | median margin compression | maximum compression | median CROWN lower bound | CROWN seconds | undecided |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5/255 | 0/20 | 0.735% | 0.841% | -374,473,952 | 35.03 | 20/20 |
| 1/255 | 0/20 | 1.469% | 1.678% | -660,480,704 | 36.11 | 20/20 |
| 2/255 | 0/20 | 2.930% | 3.331% | -1,197,689,792 | 36.33 | 20/20 |
| 4/255 | 0/20 | 5.788% | 6.614% | -2,118,697,024 | 36.76 | 20/20 |
| 8/255 | 0/20 | 11.324% | 13.048% | -3,732,893,056 | 35.26 | 20/20 |

The clean route margin has median `0.3087212` and range
`[0.2618431, 0.3610316]`. Median clean input-gradient norms are
`L1=1.1474504`, `L2=0.02989978`, and `Linf=0.0019159`. At 8/255 the attacked
margin remains positive on all 20 inputs, with median `0.2731661` and range
`[0.2319052, 0.3232641]`. These are empirical attack diagnostics, not a
stability proof.

Relative to the earlier IBP pilot, the median lower-bound magnitude is reduced
by `5.20x--5.37x`, but every CROWN lower bound remains negative by hundreds of
millions or more. Thus the stronger two-sided bracket is still 100/100
undecided. This result does not establish that the deep router is intrinsically
difficult to certify or that it is stable; alpha-CROWN and beta-CROWN/BaB remain
unexecuted closure tiers, and trained-checkpoint behavior remains the target.

The accepted worker peak was `22,523,740,672` bytes (20.98 GiB), below the
frozen 24-GiB gate, while the protected B1 process remained alive. Init
BatchNorm identity was 19 eval-mode layers with zero running means, unit
running variances, and zero batches tracked. Failed `_r4`, `_r5`, and `_r6`
directories are permanently excluded and preserved: they exposed, in order,
bounded-graph reuse, cyclic garbage retention, and retained bound-local
references. All three contain the same independently hashed attack endpoints
as the accepted run, and none was overwritten.

## Empirical boundary-estimate diagnostic

The accepted strong-pilot artifacts also support two non-formal, per-input
radius estimates without another model execution:

1. the local first-order estimate `clean_margin / ||grad_x margin||_1`;
2. the 8/255 attack-slope extrapolation `epsilon / fractional_compression`.

The stored L1 quantity is explicitly the gradient of the clean-route margin
`r_clean-r_competitor` with respect to the unit-pixel input, not a single-logit
gradient. On the 20 frozen inputs, the first-order estimate has median
`67.850/255` and range `59.261--75.800/255`; the attack extrapolation has median
`70.644/255` and range `61.313--77.935/255`. Pearson correlation is `0.926`,
Spearman correlation is `0.910`, 16/20 pairs agree within 5%, and 19/20 agree
within 10%.

The exact K=20 RT-ER aggregate pixel-box radius median is `0.5324/255`.
Consequently, the corresponding architecture-regime scale ratios are `127.4x`
and `132.7x`, which the project reports as approximately `130x` rather than a
fixed `137x`. The AdvMoE values are local/extrapolated estimates, not route
boundaries, witnesses, or certificates. The comparison does not isolate weight
sharing, pooling, depth, or initialization as a causal mechanism.

The two estimators are not independent: margin-directed PGD and the first-order
ratio share the same local-gradient geometry. A preregistered large-epsilon
attack therefore tests the extrapolation at 16, 32, 64, and 96/255. Strong PGD
finds 0/20 flips at every radius. Median margin compression is 21.04%, 38.45%,
61.82%, and 76.60%, respectively. Thus the near-70/255 values are validated as
closely agreeing local linear-scale estimates, not observed route boundaries;
all attack-diagnostic intervals remain open beyond 96/255. Attack non-discovery
is not a certified lower bound.

Raw CSV and an SVG with live text are stored under
`data/moe/results/advmoe_init_boundary_estimates_20260830_r1`. Independent
replay reports zero issues at
`act/pipeline/moe/results/advmoe_init_boundary_estimates_20260830_r1.json`.

## Float32 CROWN numerical-reach probe

A frozen first-five-sample probe tested the proposed “maximum positive CROWN
epsilon” construction. It is not a sound-reach experiment: the installed
backend is float32 and is not outward-rounded. The worker therefore records
both requested epsilon and the effective representable input box.

| rank | last positive requested epsilon | first negative requested epsilon | positive effective max width | negative effective max width |
|---:|---:|---:|---:|---:|
| 0 | 3.707e-9 | 3.731e-9 | 3.725e-9 | 7.451e-9 |
| 1 | 2.328e-10 | 2.343e-10 | 2.328e-10 | 4.657e-10 |
| 2 | 1.856e-9 | 1.868e-9 | 1.863e-9 | 3.725e-9 |
| 3 | 7.405e-9 | 7.452e-9 | 0 | 1.490e-8 |
| 4 | 1.856e-9 | 1.868e-9 | 1.863e-9 | 3.725e-9 |

Linear extrapolation from the 0.5/255 CROWN bound predicts a median zero near
`1.609e-12`, but the executed median sign bracket is
`1.856e-9--1.868e-9`. In all 5/5 rows, the sign transition coincides with a
discrete expansion of the float32 box and an increase in changed coordinates.
The requested-epsilon bracket is therefore a frontend/relaxation quantization
diagnostic, not a real-domain CROWN reach. It must not appear as a sound marker
or support an “eleven-order certificate gap” claim.

Independent audit replays all 75 requested boxes and reports zero issues at
`act/pipeline/moe/results/advmoe_crown_numerical_reach_init5_20260830_r1.json`.
No 10,000-input init CROWN census will be run. AdvMoE telemetry uses full-test
first-order estimates, fixed-subset strong PGD, and trained-checkpoint-only
CROWN/alpha/beta closure. The two-path end-to-end table remains independent of
router-bound closure.

### Requested set versus represented set

The float radius requested by an experiment and the tensor box passed to a
backend are different certificate-identity fields. Every future CROWN row must
record:

- `requested_radius`;
- representation dtype and tensor shape;
- SHA-256 identities of the centre, represented lower/upper tensors, per-side
  deltas, and total coordinate widths;
- effective lower/upper L-infinity radii;
- minimum/maximum represented coordinate width;
- zero-width and unchanged-coordinate counts.

`represented_linf_box` materializes this record in
`act/pipeline/moe/certified_artifact_identity.py`. The CROWN reach worker now
uses the same returned lower/upper tensors for both the backend call and the
identity record. A positive requested radius with a zero-width represented box
is a point query. It cannot establish the requested real-ball property. More
generally, this metadata diagnoses representation; it does not prove outward
containment. A formal real-domain label still requires a validated outward-
rounding policy or equivalent set-containment argument.

## Dimensionless relaxation inflation

At the five registered non-microscopic radii, the project reports

`(clean_margin - CROWN_LB) / (clean_margin - strongest_PGD_margin)`.

The median values are `1.664e11`, `1.471e11`, `1.336e11`, `1.186e11`, and
`1.073e11` at 0.5, 1, 2, 4, and 8/255. This recovers the eleven-order empirical
separation without relying on an ULP-scale radius axis. It is a dimensionless
relaxation-versus-observed-attack-drop diagnostic, not a certified
approximation ratio and not a bound on the unknown true reachable margin drop.
All 80 large-epsilon endpoints and all 100 inflation values replay under the
zero-issue audit
`act/pipeline/moe/results/advmoe_large_epsilon_and_inflation_20260830_r1.json`.
