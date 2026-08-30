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

## Init-pilot result

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
