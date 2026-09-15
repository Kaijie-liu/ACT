# Full-request supplied-F0-HZ evidence: mixed result

Execution `0ed86005345cc09f1ec746261827dc43c7ee0724`, frozen and remotely
confirmed before creating the one-shot result root. Both original inputs,
checkpoint, materialized boxes, 2/255 and numeric policies retained. No model
selection, range refinement, retry or property-MILP solve. Fresh review:
`act/pipeline/moe/results/conv_request_sign_lp_review_20260915_r1.json`.

## Outcome

| Input / true label / sole feasible pair | Required | Checked LP positive | Checked reuse positive | Nonpositive LP bound | Complete conditional request |
|---|---:|---:|---:|---:|---|
|16 /5 /{0,3}|9|7|0|2|No: UNKNOWN|
|98 /0 /{1,2}|9|8|1|0|Yes; minimum0.17727452738441674|

All18 obligations are generated and checked, not all18 positive. The source
interval fact for input98/competitor8 is independently reconstructed and its
two expert LP proofs checked. Its minimum is the aggregate bottleneck. Eight
residual supplied-HZ LPs are positive; the smallest residual is2.4826297268651.
All six tie-legal candidate pairs per request are accounted for by fresh route
terminals, but the infeasibility bounds excluding five pairs remain trusted.
There are no omitted properties, proposal failures, timeouts or outer kills.

| Competitor | Input16: class5 minus competitor | Input98: class0 minus competitor |
|---|---:|---:|
|0|3.690808268875|—|
|1|1.857577810193|5.712379468352|
|2|1.692663191990|2.482629726865|
|3|0.360946533837|4.376036137481|
|4|4.153275701126|4.320868020311|
|5|—|6.419369860685|
|6|3.626009857502|3.059444367038|
|7|**-0.273751858021**|6.251274977141|
|8|3.126320722189|0.177274527384 (reused)|
|9|**-0.125589947433**|7.416089656567|

Displayed decimals are descriptive conversions of exact rational checked
bounds; the archive retains the full fractions and source/certificate hashes.
Nonpositive lower bounds are NOT UNSAFE, violating inputs, independently
checked upper bounds, or proof that no alternative certificate can succeed.
Input16 does not close despite all its historical diagnostic MILP duals being
positive. Those old coefficients were not saved; the new LP also relaxes all
binary factors. This is not a same-coefficient causal separation of numerical
error, integrality and construction differences. No additional diagnostic is
automatically authorized or queued by this result.

## Cost and independent review

| Input | Fresh capture | LP proposals + local checks | Isolated check | Second isolated check | All stages |
|---|---:|---:|---:|---:|---:|
|16|34.99s|20.90s|21.10s|21.36s|98.35s|
|98|27.67s|11.34s|10.58s|10.54s|60.13s|

Each stage is charged independently under the frozen proof-experiment caps
(300/900/300/300 seconds). These are NOT same-algorithm 300-second comparison
times or an improvement to the frozen three-arm main table. Construction
records384/350 paid native calls for routing/support; zero weighted-property
queries and no production verdict. Journal chains, balanced native returns,
allocation bounds and all six route terminals match. Two separate Python -S
checks plus a fresh archival review independently check downstream arithmetic
without Torch, NumPy, SciPy or the production F0 constructor.19 targeted tests
pass, including coverage/source/certificate mutations, remote-publication
failure, and a mocked full-nine-property construction without property solving.
The pre-freeze test draft's syntax typo was corrected before any real request;
no executed experiment was retried or overwritten.

Raw evidence:69 files,326,533,583 bytes, under
`data/moe/results/conv_request_sign_lp_20260915_r1`; hashes and compact results
are committed, raw model/data/LP arrays are not. Previous full V2 and sign
controls remain byte-identical, including the prior publication-order deviation.
This execution's remote publication order is separately recorded and passes.

## Exactly what is proved, and what is not

Input98 has `CHECKED_REQUEST_CONDITIONAL_ON_TRUSTED_F0_LOWERING`: complete
output-property aggregation from checked supplied-HZ LP bounds and checked
scoped interval facts. The trusted base is still:

- network/input→HZ and interval-source construction;
- membership/pair guard lowering;
- router infeasibility exclusions;
- floating F0 construction and its binding to the original output property.

The independent arithmetic starts AFTER floating F0 construction. It does not
inherit the earlier MLP study's stronger pre-F0 rational McCormick contract.
This is not an unconditional/native floating-point proof, not a full MILP tree
check, and not a new route-changing certificate: both selected inputs have
one feasible pair. The67.06%-accuracy convolutional model is not the high-
accuracy AdvMoE target. Neither the high-accuracy strict-certification goal
nor cross-architecture net competitive advantage is closed by this study.

Both original production TIMEOUTs stay TIMEOUT. The evidence checker was not
installed as an early-termination or SAFE-acceptance gate. One of two selected
controls closing is feasibility evidence, not certified accuracy or a fresh
cohort coverage estimate. The user-requested all-obligation check is complete;
next work requires a separately scoped choice, not automatic bound tightening.
