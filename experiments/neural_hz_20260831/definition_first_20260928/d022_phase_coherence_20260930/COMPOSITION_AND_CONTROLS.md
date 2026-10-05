# Affine composition and shared source controls

The shared-phase block can be composed across ordinary mixed-sign convolutions without replacing its differences by independent intervals. That statement is an exact relational construction, not proof of a new domain or faster verification. A separate two-layer control shows that source-conditioned local improvement can reach a downstream ReLU, but the simplest such construction is already a known strong single-node formulation.

## Exact affine difference interfaces

For y=Wx+b and fixed linear observations B_in x and B_out y, a fixed T,c satisfies

```text
B_out(Wx+b) = T B_in x+c for every x
```

if and only if ker(B_in) is contained in ker(B_out W), equivalently row(B_out W) is contained in row(B_in). Necessity follows by testing x=0 and kernel vectors; sufficiency follows by expressing each output row in the input row span. Necessarily c=B_out b.

For incidence B_in, its kernel consists of independent constant offsets on connected components. The condition therefore precisely tests whether the next affine difference ignores those offsets. Absolute anchors can restore missing directions, but a full-rank anchored representation can require dense reconstruction; existence of T does not prove cheap T. These are standard linear algebra facts used as design tests.

## Ordinary CNN propagation and padding

For convolution with stride s, dilation d and padding extension x_tilde,

```text
f[o,q] = b[o] + sum_(c,k) w[o,c,k] x_tilde[c,s*q+d*k-p].
```

For output displacement t, subtraction gives

```text
Delta_t f[o,q]
 = sum_(c,k) w[o,c,k] Delta_(s*t) x_tilde[c,s*q+d*k-p].
```

Same-channel bias cancels; weights may have either sign. Dilation changes sampling locations and stride changes the required difference displacement. In valid/interior regions this is a direct convolution of the corresponding input difference field. Representing larger displacements using fixed nearest-neighbor paths is exact but its path length and nnz must be counted.

For zero padding, a pair with one endpoint outside the image is an absolute input-to-zero readout, not an ordinary neighbor difference. If both are outside it is zero. No imaginary original ReLU bit is assigned to padding. For example y1=x1, y2=x1+x2 can arise at a padded edge; y2-y1=x2 cannot be recovered from x2-x1 alone. Inputs (0,0) and (1,1) have the same input neighbor difference but different output differences. This is a boundary semantic requirement, not a special numerical workaround.

A concrete two-block valid CNN is

```text
f_i=x_i-(1/2)x_(i+1)+b1,   r_i=ReLU(f_i),
g_i=r_i-(1/3)r_(i+1)+b2,   z_i=ReLU(g_i).
```

With delta_i=f_(i+1)-f_i, d_i=r_(i+1)-r_i, gamma_i=g_(i+1)-g_i and e_i=z_(i+1)-z_i,

```text
delta_i=Delta x_i-(1/2)Delta x_(i+1),
gamma_i=d_i-(1/3)d_(i+1).
```

Attach the retained-phase factor to both layers, keeping these equalities, all original gates and the same source. The integer relation is unchanged and the LP is contained in the old LP. In particular, gamma is not formed from independent copies of the d intervals. This proves a legal relational composition, not strict final-output improvement for every such CNN.

Differences cannot become a stand-alone exact ReLU function: preactivation pairs (-1/4,3/4) and (-3/4,1/4) have the same difference and phases but different output differences. The original amplitude/source graph remains necessary.

Add and residual composition obey B(a+b)=Ba+Bb on a shared source; channel Concat uses block-diagonal spatial incidence. Cross-channel pairs and spatial concatenation seams require their own readouts or anchors. The CNN identity must not be generalized to arbitrary affine row pairs without the kernel condition above.

## GPU forward and transpose operator

For oriented incidence B with each row tail minus head, let S_t,S_h select those endpoints, L and U denote diagonal edge lower/upper bounds, and f,r,beta be retained coordinates. Let one denote the all-ones edge vector. The four edge blocks are

```text
B r - U S_t beta <= 0,
-B r + L S_h beta <= 0,
-B(r-f) + U S_h beta <= U one,
B(r-f) - L S_t beta <= -L one.
```

For block multipliers lambda1 through lambda4, the transpose contributions are

```text
r:    B^T(lambda1-lambda2-lambda3+lambda4),
f:    B^T(lambda3-lambda4),
beta: -S_t^T U lambda1 + S_h^T L lambda2
      +S_h^T U lambda3 - S_t^T L lambda4.
```

If f=Wq+b is substituted, the source contribution includes W^T B^T(lambda3-lambda4) and the RHS includes the corresponding bias terms. If f is retained, W instead belongs to the original affine equality module. Neither may be omitted or counted twice.

This suggests gather, scatter, diagonal operations and convolution primitives. It still requires all original gate/source work, four multipliers and residual entries per edge, metadata, temporary buffers, any bound rows, terminal integration and certified evidence handling. For already computed f, gathering Bf may be cheaper than separately convolving every directional difference field. A matrix-free formula does not prove a vendor LP interface accepts it, and scatter rounding must be handled by the existing numerical qualification rules.

No GPU operation or timing was executed. No retry of the failed initialization-trace route, cap change, security change or solver replacement is proposed here. Transposes inside a fixed terminal LP are ordinary solver arithmetic, not network backward/dual rescue.

## A known source conditioned two layer positive control

Take 0<=a<=A, 0<=b<=B, max(A,B)<theta<A+B, and r=ReLU(a+b-theta) with original bit beta. Besides the ordinary four rows, add

```text
r <= a+(B-theta)beta,
r <= b+(A-theta)beta.
```

Together with the box and beta in [0,1], these describe the full two-coordinate labeled gate hull. They are precisely a two-dimensional instance of [Anderson et al., section 5.2, Proposition 12](https://www.columbia.edu/~wm2428/papers/mip_neural_networks.pdf), not a new result. With existing a,b columns they add two rows and six nnz, no bits or continuous variables. Affine readouts instead require their actual support or added defining columns to be charged.

A constructive check is useful. Set T=r+theta*beta. For 0<beta<1, choose u+v=T with

```text
max(0,a-A(1-beta)) <= u <= min(a,A*beta),
max(0,b-B(1-beta)) <= v <= min(b,B*beta).
```

The four combinations of upper bounds follow from the four upper gate/hull rows. The sum of lower endpoints is at most T: zero or one positive endpoint uses theta>max(A,B) and r>=0; two positive endpoints use r>=a+b-theta and theta<A+B. Thus such u,v exist. The active source is (u,v)/beta; the inactive source is (a-u,b-v)/(1-beta). Both belong to the box, have the appropriate signs and mix to the original tuple. Endpoint beta values follow directly. This proof does not instruct a runtime to enumerate phases or copy all source coordinates.

For a residual q=alpha*b-lambda*r+c with alpha>=lambda>0 and c>0, the new row gives

```text
q >= c+(alpha-lambda)b+lambda(theta-A)beta >= c.
```

Consequently a second original gate t=ReLU(q) has t>=c even in its ordinary LP, since t>=q. Choose A=B=1, theta=6/5, alpha=6/5, lambda=1, c=1/100. Use the same certified interval bounds [-79/100,121/100] for q in both comparisons, obtained from 0<=b<=1 and 0<=r<=4/5. The old first-gate relaxation allows a=9/10,b=2/5,beta=13/20,r=13/25. Its q=-3/100 permits t=0 and second bit zero under those gate rows. The source-conditioned rows instead prove t>=1/100, sufficient for the positive-margin property t>=1/200. No stronger second-gate bound is silently given only to the new formulation.

This works on an open coefficient region, not only an equality of weights. However, HZ with the same two rows obtains the identical benefit. If a,b are readouts from a larger constrained source, validity remains but local ideality does not imply full-source ideality. General mixed downstream coefficients need not satisfy the displayed cone condition. This is a required comparator and constructive evidence that source conditions can survive a second ReLU, not the requested Neural-HZ innovation.

## Status

All formulas are paper derivations independently reviewed, with the same provenance as PHASE_COHERENCE.md. No executable tests, model census, GPU qualification, shadow or full replay were run. The remaining contribution must concern useful compositional relation generation or complete cost beyond these known formulations, without weakening any existing promotion gate.
