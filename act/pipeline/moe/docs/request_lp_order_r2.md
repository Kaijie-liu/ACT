# Separate request LP order-only refinement R2

Authorized follow-on to the closed R1 infrastructure control, not a changed R1
result, performance ablation or new holdout. Freeze seed0/index3000,2/255,
the same model/input/tie semantics and all18 output obligations. Parent R1
manifest SHA256:90aad930c9f8b1f7018c3a7ef30cee80ab4487e5fae16b231bc6432c7b3cf082.
Keep parent files byte-for-byte; materialize hash-checked copies in a new root.

Only enhancement: for each pair with residual properties (currently{2,4}),
export two guarded router support LPs for m=r_a-r_b and -m. Propose duals
with10seconds/LP and independently check exact rational lower bounds L,N.
When both are available require L<=-N. Then use the following fixed rule:

- L>=0 implies lambda_a>=1/2;
- N>=0 implies lambda_a<=1/2;
- neither sign known leaves[0,1]; both zero imply[1/2,1/2].

Proof: lambda_a=exp(r_a)/(exp(r_a)+exp(r_b))=sigmoid(m); monotonicity and
sigmoid(0)=1/2 establish these inclusions including ties. No numerical exp or
sigmoid endpoint is trusted or evaluated to derive the certified envelope.
This is an order fact, not a sharper fitted range or multiplier search.

Reuse ALL existing checked membership and disagreement LPs, without re-solving
them. Reconstruct the needed shared expert HZ and demand byte-identical source
snapshot hash to the parent disagreement export BEFORE reusing its endpoints.
If the gate envelope strictly shrinks, rebuild each of the three residual F0
properties with the same parent checked disagreement endpoints and run one LP
proposal/check. Otherwise retain the old residual without a fresh solve.
Unknown proposal or negative bounds remain UNKNOWN; no replacement, larger
timeout, splitting, sigmoid approximation or further strategy is run.

Expected maximum new queries:2 router LPs+3 output LPs, not a new27-expert-LP
run. Outer watchdog600s,CPU/float64,OMP/BLAS1; preserve partial manifests and
errors. Complete inventory and all proof references bind the parent request.
The v2 checker verifies order support scopes/objectives and rational signs,
then all18 output obligations. Positive aggregation is still only
CHECKED_REQUEST_CONDITIONAL_ON_TRUSTED_LOWERING: network/guard/route-exclusion
and F0 floating outer-HZ construction remain trusted as in R1. No production
acceptance policy changes. LP failure does not mean network UNSAFE.

Commit tests/protocol/code before the five potential new queries. Archive
result even if the order-only refinement produces no improvement. External
static obligations are a different protocol, root and execution stage.
