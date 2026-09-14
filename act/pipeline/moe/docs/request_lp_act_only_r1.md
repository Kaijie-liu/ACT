# Fixed ACT-only request proof transfer, R1

Registered before new model queries in response to Advice/bb.md. This is a
post-selected mechanism/evidence experiment, not a certificate-rate estimate,
new holdout, or adjustment of production numerical acceptance.

Fixed cases, in order: seed0/index4029 (historically one pair), seed1/index4018
(three), seed2/index4014 (two). All are ACT-only positives in the complete-cost
external follow-up. Keep the original checkpoint, materialized CPU/float64 input
box, 2/255 radius and nine classification properties. Do not replace a case,
search a gate range, add sigmoid refinements or retry a negative LP at a larger
budget. R1/R2/R3 on index3000 remain untouched.

## Construction and budget

`request_lp_cases.generate` accepts a model, explicit center/lower/upper tensors
and request identity. It recomputes the router HZ and all 28 tie-legal feasibility
queries, then membership-guarded expert HZs and nine rationally checked LP
supports per candidate expert. A pair/property is reused only when both relevant
expert supports are checked positive above the unchanged exact interpretation
of float1e-7. Otherwise propagate a shared-factor expert pair, check both router
order bounds, and use only the resulting dyadic gate enclosure in [0,1].
Check both property-difference supports, then build the direct rational
McCormick LP from the stored **pre-F0 joint HZ**, with exact rational bounds.
There is no call to floating F0 construction. Construction/check failures are
errors; failed LP proposals or nonpositive checked bounds are UNKNOWN.

One CPU thread; 10 seconds per feasibility/LP proposal; 1,800 seconds per entire
generation request, including startup/imports, identity checks, checkpoint and
tensor loading, network-to-HZ propagation, all proposals, inline rational checks
and evidence writes. The supervisor kills only its owned process group on the
hard deadline, retains partial inventories and never promotes a timeout. A
separate read-only request check is timed outside the generation cap and reported
in addition, not hidden. All three terminal outcomes must be retained.

No previous bounds, route census, common facts or proof packages are borrowed.
Only the immutable raw tensor files are reused. Their original preparation cost
was **0.2768000243231654 seconds for all ten input files** in the external run;
report this provenance cost separately, not as a measured per-case cost. New
loads/hashes are charged. Historical model training and the experiment used to
select these cases are outside proof-generation cost and must not be presented
as free end-to-end model creation.

## Evidence and interpretation

For each case publish route coverage, every required obligation's disposition,
reused/residual/unknown counts, relaxed binary counts, checked bounds, complete
generation and independent-check times, and serialized evidence size. A positive
request remains `CHECKED_REQUEST_CONDITIONAL_ON_TRUSTED_LOWERING`: network/input
propagation, guards and route-infeasibility exclusions remain trusted. Checked
LP arithmetic is not a full network or deployed floating-point proof.

Success is executing and independently checking the fixed procedure, not forcing
three positive requests. LP relaxation failure does not refute the original
HZ-policy SAFE and is not evidence of an actual unsafe input. Preserve failure
attempts and do not edit frozen raw result directories.

Run after tests, commit and push:

```sh
python -m act.pipeline.moe.request_lp_cases --config act/pipeline/moe/configs/request_lp_act_only_r1.json --run data/moe/results/request_lp_act_only_20260915_r1
python -S scripts/check_moe_request_lp.py data/moe/results/request_lp_act_only_20260915_r1/seed1_4018
```

The second command needs only Python's standard library and this checkout;
it does not load checkpoints, invoke a solver, or import Torch. Its isolated
launcher bypasses ACT's eager parent-package imports, not the bound checker.
The research selection still references local checkpoint/input locations; this
is **not** yet the complete downloadable model artifact. The generic generation
control in `test_request_lp_cases` builds a three-expert model from code and
checks its proof from a fresh `python -S` process in a different directory.
