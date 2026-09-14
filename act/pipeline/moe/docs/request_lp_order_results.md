# Order-only request LP refinement R2: conditional positive closure

Separate frozen execution `db0b4b281`;33.13seconds,normal exit. Exactly five
new LP proposals:two router-order queries and three residual outputs. No
membership/disagreement LP was re-solved; the reconstructed shared expert HZ
matches the frozen parent source hash. All R1 files and its UNKNOWN stay intact.

For pair{2,4}, the independently checked real-model score difference satisfies
approximately -3.42220684 <= r2-r4 <= -3.11462675. The negative upper endpoint
is checked as a positive lower bound for r4-r2. Therefore lambda2<=1/2; the
R2 envelope[0,1/2] follows from order alone, not approximate sigmoid evaluation.

| Pair{2,4},competitor (clean class5) | R1 checked output LB | R2 checked output LB |
|---|---:|---:|
|2|-.03526004|2.60543882|
|3|-.05940228|1.55273374|
|4|-.01215695|4.10959609|

Complete output inventory:18/18 positive obligations,15 via existing scoped
membership facts and3 via new residual LPs. Minimum aggregated checked lower
bound is approximately0.18304675 (an existing reused fact). Separate review
checks all41 supplied LP exports/available duals,including old negative results,
all18 obligations,parent hashes,unchanged route partition and the dyadic order
rule. PASS,zero issues; no additional solve in review.

Strongest status is **CHECKED_REQUEST_CONDITIONAL_ON_TRUSTED_LOWERING**.
This is a positive complete-request downstream evidence result on one observed
verification-scale model, NOT an independently proved network implementation,
new confirmed benchmark gain,high-accuracy certificate or machine-checked
floating-point program. Network/input-to-HZ,guard lowering,router exclusions
and F0 outer-HZ floating construction remain trusted. Exact rational checking
begins at those supplied objects. R1's universal-weight UNKNOWN is not replaced.

Artifacts: `../results/request_lp_order_review_20260914_r2.json`,
`../request_lp_order.py`, `../review_request_lp_order.py`,
and the schema-v2 path in `../check_request_lp.py`. No further gate refinement,
domain splitting,LP budget expansion or production-policy change is planned
as part of this bounded control. External static-pair adaptation is separate.
