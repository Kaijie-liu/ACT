# A2 saved-only diagnosis: exact supports, misplaced improvement

The fixed-tilt A2 mechanism remains stopped. Analysis of its original saved
coefficients now separates two previously unresolved issues: the four distinct
global support bounds are already exact optima, and the synthetic weighted
property has an explicitly feasible negative point. Neither stronger candidate
duals for those same queries nor more accurate scalar arithmetic can turn this
property into a positive certificate. This is not a general negative result
about margin-conditioned support or a counterexample to a trained MoE.

Starting revision: `e29013e4c0dbcc9a09de7273dfb9f656612a34ae`, clean
`feat/moe-route-verification`. This follow-up performs no new support query,
solver/model execution, candidate search, source selection or policy change.
It reads the [immutable A2 archive](margin_a2_records_20261021.json), whose
SHA256 is `c80863cec99bb522c6d5461833abea1b79268f6ed22423bcceced4bfeb5ea355`.
The [original experiment](margin_a2_results_20261021.md) and its online outcomes
remain unchanged. This dated addendum supersedes only its *unknown optimality*
statement for the four distinct global objectives, not the remaining LP queries.

## Exact reduction of the original source

Use the archived factor ordering: continuous factors `z0…z3074`, followed by
four relaxed binary factors `z3075…z3078`. The original four equalities imply

    z0 = z1 = 0,    z3072 = z3076/4,    z3074 = z3078/4.

Of the four inequalities, only `z2 <= 3/4` remains nonredundant after these
substitutions and the factor boxes. In particular, the other left sides become
`z3075/4`, `z3076/4` and `z3078/4`, whose maxima are 1/4, below their respective
right sides 1/2, 3/4 and 3/4. This is an exact reduction for this source, not
an optimization procedure applied to another LP.

Let `x=z2`, `beta=z3075`, `u=z3076`, `v=z3078`. Projecting all ten original
expert output rows with the saved property, including its constant, gives

    b = -11/32 + 9*x/512
        + sum_{j=3}^9 [(-1)^j*(j+1)^2/512]*zj
        - 55*beta/1024 - 165*v/1024,
    d = 5/128 + 165*(u+v)/1024,
    m = 1/2 + 3*x/8 + 3*beta/16.

Here `x in [-1,3/4]`; `beta,u,v` and the other remaining factors range
independently over `[-1,1]` in continuous P. No output class was discarded.
For any anchor `t in [0,1]`, the private contribution to `b+t*d` is
`165*(t*u+(t-1)*v)/1024`. It is minimized by the same `u=-1,v=1`, with value
`-165/1024`; the other nuisance factors contribute `-371/512`. Thus

    psi(mu,t) = -1259/1024 + 5*t/128
                + min(9*x/512 - 55*beta/1024 | m=mu).

Eliminating `beta=(16*mu-8-6*x)/3` leaves
`x/8 - 55*mu/192 + 55/384`. The smallest feasible x is
`max(-1,(16*mu-11)/6)`. It follows that the exact reachable margin interval
of this continuous P is `[-1/16,31/32]`, not the looser registered factor-box
interval `[-5/16,21/16]`, and the exact conditional value function is

    psi(mu,t) = -333/256 + 5*t/128 +
        (55/192)*(5/16-mu),    -1/16 <= mu <= 5/16;
        (3/64)*(mu-5/16),       5/16 <= mu <= 31/32.

This formula is attained on both entire intervals: use `x=-1` and
`beta=(16*mu-2)/3` on the left; use `beta=1` and `x=(16*mu-11)/6` on the right,
with the same minimizing private/nuisance factors. These paths are affine,
so checking all their linear constraints at the interval endpoints checks
the whole paths. The algebra above supplies the matching global lower bound;
checking endpoints alone would not prove optimality.

Because this psi is affine in t, exact anchor interpolation has no concavity
gap on its feasible fibers. This conclusion is source-specific; it does not
remove the finite-anchor gap demonstrated by the earlier Diamond example.

## A feasible negative point, not an inferred counterexample

For `f=b+sigma(m)*d`, the left derivative of `psi(mu,sigma(mu))` is at most
`-55/192+5/512=-425/1536<0`, using `0<sigma'<=1/4`. The right derivative is
at least `3/64>0`. Therefore

    inf_P f = -333/256 + (5/128)*sigma(5/16).

An attaining factor assignment has the following nonzero entries; all other
continuous factors are zero:

    z2=-1, z3=1, z4=-1, z5=1, z6=-1, z7=1, z8=-1, z9=1,
    z3072=-1/4, z3074=1/4,
    z3075=1, z3076=-1, z3077=1, z3078=1.

All four binary factors are actual ±1 here, not fractional relaxations.
The saved-only script checks all 3,079 boxes and all four equalities/four
inequalities of the augmented HZ. It also maps the assignment back into all
six original stored HZs and checks their 15 equalities/15 inequalities, including
duplicate lifted rows; these are not 15 distinct independent constraints.
Original expert outputs agree with the augmented outputs. Original router
scores are `(1/4,-1/8,-1/16)`, so the synthetic pair `{0,2}` is legal at this
point. This is one point, not a route-change witness.

The point has `(b,d,m)=(-333/256,5/128,5/16)`. Even the coarse fact
`0<sigma<1` gives a strictly negative upper enclosure `f<-323/256<0`.
This time the upper enclosure comes from an explicitly checked feasible point,
not from misusing a certified lower bound. The sign/quarter-Lipschitz rule also
gives the more informative closed enclosure

    -41/32 <= inf_P f <= -10471/8192 < 0.

This concerns the **given synthetic HZ property**. Its property vector is not
being reinterpreted as a classification margin. No checkpoint, training input,
network-to-HZ lowering, native full-model replay or real UNSAFE is established.
The continuous conditional paths need not preserve binary integrality; only
the displayed negative witness explicitly does. Nothing changes the original
online statuses or adds a certificate to a research main table.

## Why the four global support queries are not the bottleneck

For `nu in [-3/64,55/192]`, the two slopes of
`psi(mu,t)+nu*mu` bracket zero at `mu*=5/16`, so

    Phi(t,nu) = -333/256 + 5*t/128 + 5*nu/16.

Both frozen tilts, 0 and -1/128, lie in that interval. The script independently
reconstructs each saved dual bound from the original CSR constraints, including
signs and exact box-residual compensation, and evaluates the same objective
at the feasible witness. All four distinct values match:

| Anchor t | Tilt nu | Exact support value |
| --- | --- | ---: |
| 27/64 | 0 | -10521/8192 |
| 27/64 | -1/128 | -10541/8192 |
| 53/64 | 0 | -10391/8192 |
| 53/64 | -1/128 | -10411/8192 |

These account for six executions in the original, separately audited event
ledger: two in zero and four in finite. The new script checks six saved support
entries; their execution count is inherited from that earlier ledger audit.
Primal/dual equality now proves exact optimality for these objectives. It does
not establish optimality for all 78 LP calls or for the product relaxations.

For either anchor the finite-tilt envelope adds the same hinge to zero tilt:

    S_t(mu) = -333/256 + 5*t/128 + H(mu),
    H(mu) = max(0,(mu-5/16)/128).

The resulting per-cell gains are `0,0,3/2048,19/4096`. The first cell remains
the aggregate bottleneck. It uses an unconditional minimum attained at
`mu=5/16`, outside that cell `[-5/16,3/32]`. The frozen negative tilt cannot
correct this mismatch: it is weaker there, and the zero line wins.

This pinpoints a limitation of the frozen conditional approximation/tilt
selection. It does not justify searching for another tilt on the same source,
or asserting that any different margin representation must fail. Nor does the
negative property invalidate the original bound-improvement gate: that gate
did not require the synthetic property to become positive. Its registered
zero-gain result still stands.

## Pointwise sigmoid aggregation is not a rescue here

The original operator contract already proposes keeping affine support lines
until pointwise gate interpolation. Implementing that ideal is not a newly
discovered research mechanism. For the saved lines its exact envelope is

    G(mu) = -333/256 + (5/128)*sigma(mu) + H(mu).

It is increasing on the registered interval, so its minimum is at `-5/16`,
where `H=0`. Zero and finite tilt therefore still have identical minima.
Both anchor envelopes are nondecreasing; both attain every cell minimum at
its left endpoint. There is no loss here from combining incompatible anchor
minimizers, so delaying that part of scalarization alone gives no improvement
under the same gate enclosure.

Even replacing the quarter-Lipschitz enclosure with exact sigmoid at the
registered left endpoint can improve the aggregate by at most

    (5/128)*(5/16)^3/48 = 625/25165824 < 1/1024.

For completeness, `sigma(-a)=1/2-tanh(a/2)/2` and
`0<=u-tanh(u)<=u^3/3` for `u>=0`. The latter follows by differentiating
`tanh(u)-u+u^3/3`, whose derivative is `u^2-tanh(u)^2>=0`.
This proves the stated bound relative to `1/2-a/4`. It benefits zero and finite
equally, not the tilt mechanism. It is an upper bound on a **surrogate-envelope
improvement**, not a feasible network objective. It holds only with the
unchanged registered interval and saved support lines; it is not a cap after
substituting the newly derived reachable margin interval. The separate feasible-point
argument above establishes the negative synthetic property.

## Reproduction, cost and next decision

Run from this repository using the existing environment:

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -I -S -B scripts/analyze_margin_a2_saved_20261021.py
```

The source-specific standard-library script binds the immutable archive and
source/property hashes, reconstructs the ten-class projection, checks the
reduction/path identities and original feasible witness, rechecks the six dual
records, and reproduces the scalar gains. It does not call the producer or an
optimizer. Its 60s alarm and 8GiB address-space bound limit this offline read;
they are not a new production policy. [Saved output](margin_a2_saved_diagnosis_20261021.json)
records final PASS and approximately 0.0731s inside the script, excluding
interpreter startup/exit. An earlier pass took 0.0775s; after review, an explicit
input/entry-output equality check was added and the saved-only audit repeated.
Neither invocation called a solver. This is offline diagnostic cost, not an online speedup or cost
replacement for A2. The analytic derivative/minimization proof above remains
an inspectable mathematical argument, not a machine-checked theorem prover.

Separate read-only AI reviews independently derived the value function,
checked the aggregation math, and reviewed novelty against the existing
contract. They are not external human technical review. No new model,
support query or candidate was generated; real-development usage stays
0 stages/0 calls/0 seconds. G1–G6 remain OPEN.

**Keep `STOP_CURRENT_FIXED_TILT_MECHANISM`.** Do not run A3/A5, add tilts/cells,
improve sigmoid on this source, revive the observation chain, or rename the
already-proposed ideal aggregation to bypass the stop. The saved-only
diagnostic gap is now closed. No distinct, justified next mechanism was
identified in this bounded review. Further algorithm execution needs a new
scientific direction with a concrete nonredundant representation and strong
comparator separation, not another run of this candidate. Independent G3/G6
work is a separate agenda and must not be described as algorithm improvement.
