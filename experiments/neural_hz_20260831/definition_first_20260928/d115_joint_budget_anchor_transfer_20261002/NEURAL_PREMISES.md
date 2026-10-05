# Joint budget premises in an ordinary biased residual block

The joint-budget transfer theorem has nontrivial premises on an ordinary, biased, nonparallel ReLU bank and a genuinely changing downstream phase. This paper example proves those premises without a feasibility solver or phase split. It does not establish a neural-query separation or native applicability to the archived CIFAR100 and TinyImageNet networks.

## Uniform budget construction

For x,y in [-1,1], retain the three original gates and all their original bits:

```text
g1=x+y/4+1/5,       q1=ReLU(g1)
g2=-x+y/5+3/10,     q2=ReLU(g2)
g3=-y+x/10+1/4,     q3=ReLU(g3).
```

Their exact scalar affine-box intervals are [-21/20,29/20], [-9/10,3/2], and [-17/20,27/20]. All are unstable, biases are nonzero, and normals are pairwise nonparallel. Applying the same ordinary secant rule U*(g-L)/(U-L) gives

```text
q1 <= 29*x/50 + 29*y/200 + 29/40
q2 <= -5*x/8 + y/8 + 3/4
q3 <= 27*x/440 - 27*y/44 + 27/40.
```

Merge the common source coefficients before taking their box support:

```text
q1+q2+q3 <= 43/20 + 9*x/550 - 189*y/550
         <= 251/100.
```

The last bound is strictly smaller than the sum of the individual maxima, 43/10. No optimized multipliers or LP states are used: each local secant follows the same endpoint rule and the aggregate support is a fixed absolute-value sum. This budget construction itself is known affine-bound propagation, not a new Neural-HZ theorem.

The normalized bundle v_i=100*q_i/251 is nonnegative with sum(v)<=1. Normalization is an observation view, not a change to the concrete model. The real source is generally a strict subset of this simplex and all its original predicates remain. A future implementation must certify the endpoints and rounding; these rational paper formulas do not certify a floating model with interval BN parameters.

## Genuine old and new phases

Let alpha be q1's original active-bit view. Add the actual mixed residual successor with its own original bit beta:

```text
h=q1-3*q2/5+2*q3/5+x/10-1/5
p=ReLU(h).
```

All four old/new bit combinations occur at strict interior source points with nonzero original preactivations:

| Source x,y | g1,g2,g3 | h | alpha,beta |
| --- | --- | --- | --- |
| -1/2,0 | -3/10,4/5,1/5 | -13/20 | 0,0 |
| -1/10,-9/10 | -1/8,11/50,57/50 | 57/500 | 0,1 |
| 0,0 | 1/5,3/10,1/4 | -2/25 | 1,0 |
| 1/2,0 | 7/10,-1/5,3/10 | 67/100 | 1,1 |

Thus changing alpha to beta is not silently a relabeling of identical phases or a stable-gate substitution. These signs, nonparallel normals and the strict improvement of the aggregate budget persist under sufficiently small perturbations, but no perturbation radius has been numerically certified.

In beta's active readout of h, the residual x/10 must also be conditioned. The q bundle does not supply beta*x for free.

There is also a decisive own-gate identity: since alpha is q1's phase, alpha*q1=q1, including q1=0 with either original label. Therefore t1=alpha*v1=v1. The local separating point in THEORY.md has t1=1/5 and v1=1/4 and is incompatible with this identity. It definitely does not extend to this block's faithful old-observation interface. Do not drop the identity or select a different anchor merely to salvage that point.

## What was established and what remains

This control establishes a sound way to obtain a useful shared budget and shows a real nonnested anchor change in a normal mixed residual structure. It also proves the nonembedding of the particular local moment control. A later output-query gain and favorable cost remain unproved. The next comparison must bind every old/new observation and residual to one original source and keep the same complete original gates, own-output identities and phase labels in both arms; any separating point must arise from that faithful comparison.

No code, model, LP, GPU or numerical test was run. All values above are paper arithmetic independently reviewed, not execution results. This file does not authorize a numerical run without the unchanged candidate freeze and preregistration.
