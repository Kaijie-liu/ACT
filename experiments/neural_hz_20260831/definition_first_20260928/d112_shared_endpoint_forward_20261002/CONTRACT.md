# Shared endpoint forward comparison

This is an opt-in component experiment for the definition-first Neural-HZ goal. It tests whether a shared nonlinear relation can be consumed by an actual existing Softmax-to-value routine. It is not another elimination experiment and it does not establish a new abstract domain, GPU implementation, network admission, or a formal score improvement.

The preceding user-facing status turn was no progress: it checked existing evidence but produced no new research result. This experiment instead fixes two ordinary controls and measures the old implementation, a known incremental reference, and the endpoint candidate on the same source and readout.

## Mathematical carrier and obligations

The intended carrier remains D=(H,R,E,decoder) from the preceding endpoint definition. H retains all original continuous factors, signed binary factors, equalities, inequalities, and shared source identity. R binds actual same-input nonlinear endpoints. E is a finite sound relaxation of R that downstream readouts consume. Empty R embeds H. This module implements only the exact-rational lowering of one such relation block; it is not the full carrier, a native source authenticator, or a proof of novelty relative to HZ, ImageStar, mixed-symbolic, or polynomial domains.

The caller must establish that s and t are finite logits with the same support and temperature, p=softmax(s), q=softmax(t), and both queries use exactly the same value bank V at the same original assignment. The seven frame labels are checked for equality, but labels alone do not prove this obligation. All original predicates and binary column indices remain an immutable prefix. No original bit is relaxed, eliminated, pivoted, or enumerated by the compiler. The LP controls below contain no binary variables; preserving a binary metadata fixture is not native phase qualification.

Let d=s-t, c=logsumexp(s)-logsumexp(t), and lambda_i=L(p_i,q_i), the positive logarithmic mean. Then z_i=p_i-q_i=lambda_i(d_i-c), sum(z)=0. Given certified positive probability boxes, lambda lies between the minimum of the two lower endpoints and the maximum of the two upper endpoints. This is a conservative consequence of the mean lying between its arguments; no transcendental evaluation is required by the compiler.

If d_i lies in [a_i,b_i], then c lies in [min(a),max(b)]. This follows pointwise from exp(c)=sum(q_i exp(d_i)). The affine box calculation preserves all common columns while obtaining sound, possibly loose, intervals. It does not use an LP objective, dual certificate, or solver status.

Each product z=lambda*h is relaxed by four McCormick rows and the valid row 2 lambda<=p+q. The latter follows from logarithmic mean<=arithmetic mean. Bounds for z are the intersection of the independent probability-difference interval with the four corners of the lambda/h box. Those bounds are explicitly installed and then used for value products. In particular, product envelopes must not reuse a wide pre-relation difference box and silently assume its later LP tightening changes the envelope.

Value centering is shared with the reference: Y=V_0+sum(p_i*(V_i-V_0)), and DeltaY=sum(z_i*(V_i-V_0)). These equalities follow from the two simplex identities. A point centered value gives an exact affine product; otherwise a continuous product and four McCormick rows are introduced. All output channels share the same c, lambda, p, q, and old input assignment. No lambda elimination is used.

For every real common-source network state, choose its actual c, lambda and centered products. They satisfy every appended row and preserve its original decoder. This proves sound extension under the stated caller obligations, not equality of the linear relaxation to the nonlinear graph. Exact rational operations are used for emitted rows. Converting them to floating native storage would require a separate outward-rounding and binding proof; this experiment does not provide that admission.

## Fixed ordinary controls

Both controls use x in [-1/2,1/2], epsilon=1/32, t=(x,0), s=(x+epsilon,0). The probability coordinate is logistic(x). All probabilities are strictly between 1/3 and 2/3 because 17/32<log(2). No data label, terminal margin, or solver status selects a case.

The first control uses a second common source w in [-1,1] and V=(1+w/8,w/8). Unlike a constant V, this is non-point and belongs to the fused path's supported structure: tf_mlp first diverts point operands into a constant MatMul. Common value translation cancels exactly in DeltaY. The second uses V=(v,0), v in [1,2], so the shared probability difference must actually be multiplied by a dynamic value.

The tests call the unchanged public sparse_hz_softmax_value_relaxation, including its existing cross-radius LP, rather than a weakened private helper. Scores are direct exact affine readouts of one HZ source, so this is a Softmax-to-PV component comparison, not trained-model execution or complete QK attention admission. Original probability birth, simplex/ratio rows, score/value predicates, and fused output expressions are retained. The existing probability-to-value graph binding and value centering are supplied to every reference arm.

Four systems are compared for the same DeltaY objective:

- A is the full production component output plus the common probability/value bindings.
- D is A plus the endpoint block alone.
- B is A plus the classical two-token incremental bound 0<=z<=epsilon/4 and its shared centered value product.
- C is B plus the endpoint block.

The classical bound follows directly from logistic'(x)=logistic(x)*(1-logistic(x))<=1/4. It is not a discovery. Thus B is a practical stronger reference, not an exact function oracle. The fixed controls are expected to show D improving A, but need not show C improving B. A component pass does not promote a candidate whose apparent innovation gain is absorbed by B.

For the common-translation control, the endpoint rows give z<=epsilon/3=1/96: c in [0,epsilon], z<=U*(epsilon-c), z<=U*c, U<=2/3. For the multiplicative control the shared product gives DeltaY<=2z<=1/48. The stronger classical reference gives 1/128 and 1/64 respectively.

The previous paper audit constructed an old independent-error point from the two rows' actual Taylor residuals at x=-1/2 and x=1/2. Its common-translation difference exceeds 1/96 even with both exact probability boxes, simplex, ratio and linear PV bindings. Numerical execution will check the actual floating implementation, not assume this argument is already an execution receipt. Scalar ReLU readout bounds may be reported using monotonicity of max(0,y-threshold); no new native child phase is thereby certified.

## Costs and qualification limits

For N tokens and C output channels, one relation adds at most 1+N+NC continuous variables, 2+C EQ rows and 13N+8NC LE rows in this literal compiler. Common-centered point products cost fewer columns and rows. All old columns, including signed binary factors and original constraints, are retained. Both base probability/value products and the known reference's products must be counted when comparing the arms; they are not free.

The preflight upper-bounds each additional row's support by the final column population, including its bias, and bounds the retained-entry count before constructing new rows. A final exact count is also checked. This conservative count is not a peak-memory proof: immutable tuple reconstruction, Fraction objects, solver copies, source authentication, terminal conversion, old storage and witness reconstruction are separate costs. Stored rational coefficients are limited to 512 bits. No full physical, 256M whole-work, 200M branch-work, 40M evidence-work, 64M retained-entry, or GPU certification is inferred from this small component run. Those inherited gates remain prerequisites for later native stages; their scope is not waived or raised here.

This scalar Fraction prototype is not GPU-ready. Any GPU lowering must retain shared column identity and valid outward rounding, pay conversion/guard/query costs, and show complete same-path benefit. It is premature to duplicate this candidate across ViT models merely because a small CPU comparison is positive.

## Prior art and interpretation

The elementary logistic bound above is proved directly, so the comparison does not depend on an unavailable oracle. Softmax bounds using exponential-reciprocal and log-sum-exp decompositions are established literature, including [Wei et al. 2023](https://proceedings.mlr.press/v206/wei23c.html). Finite interpolation conditions for smooth convex functions are also established, including [Taylor, Hendrickx and Glineur](https://arxiv.org/abs/1502.05666). Neither source is claimed to provide this exact implementation or a new Neural-HZ domain; they motivate keeping known relational baselines in the comparison.

The definition contribution still requires a general joint semantics and unified forward transformers with independently demonstrated precision or full-cost advantages. If D beats A but C does not beat B, record a source-consumption mechanism result and a negative innovation control. Do not scale this particular justification into a claimed domain breakthrough.

## Provenance

Date 2026-10-02 Australia/Sydney, branch redu-hz, commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac. Tracked binary diff SHA256 is 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5. All new files are isolated here. Production code, frozen experiments and /data1/Kane/HyZor remain unchanged.

Formal baseline remains 1870/2413 (1063 CERT and 807 validated ADV). Independent E0 remains CIFAR100 25 plus TinyImageNet 36, 61/400. These controls cannot change either score, establish family zero regression, or enable a default. The complete Goal remains active.
