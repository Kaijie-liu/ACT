# Checked input / affine / ReLU / shared-frame source prefix V1

This separate optional proof representation addresses the upstream gaps from
`upstream_source_v1_results.md`. It changes neither production HZ/HiGHS
acceptance nor old source matrices/certificates. The bounded real control uses
the same pinned convolutional input98 and experts1/2. The endpoint is **both
experts after their first Conv2d and ReLU**, not classification or complete MoE
safety. No additional sample, optimizer, forward, checkpoint load, old output
certificate reuse or whole-network propagation is part of this stage.

## Proof representation and independence

A sparse rational state consists of centers, continuous/binary generators,
equalities and inequalities, plus globally distinct named factor IDs. Every
serialized coefficient is interpreted exactly. The producer is untrusted;
`source_enclosure/check.py` checks its construction without importing it.
Only serialization/identity helpers and the previously controlled mathematical
Conv operator parser are shared. No numerical zero tolerance is an acceptance
rule. `exact=False` avoids confusing this outer enclosure with an exact image.

The portable composer binds the original request/state inventory and input
bytes, both expert parameter identities, all required states and their source
hashes. All seven steps must pass: input, guard, two affine steps, two ReLUs,
and the joint factor map. Checking a partial prefix cannot produce the terminal
prefix certificate, and no prefix status denotes complete network SAFE.

## Input enclosure

The producer chooses a binary64 center and a radius at least
`max(center-lower, upper-center)`, increasing a rounded-down radius with
`nextafter` when needed. It never drops a radius based on a1e-12 cutoff.
The independent checker uses exact rational endpoint comparisons and verifies
one distinct coordinate factor per input component. A production rounding
recipe or `nextafter` call is not trusted in place of the containment check.

For every input point in the represented box, its coordinate factors lie in
`[-1,1]`. Zero-radius coordinates need no division. This is containment of the
pinned represented box, not a proof of preprocessing or an exact epsilon-ball.

## Guard attachment

The real control imports the four saved pair rows only after proving each
redundant on the **whole** factor box. They cannot delete any assignment even
after input-radius enlargement. The checker compares their actual coefficients,
RHS and factor positions, and requires exact source preservation.

This special-case argument is not a general lowering proof for nonredundant
pair guards or membership big-M guards. For general guards, source semantics
and any rounding error would need their own checked implication certificate.

## Affine error compensation

Let the exact real affine image of a supplied source state be `Wz+b`. A
floating stored matrix from the preceding audit is used **only as a nominal
proposal** `p` on the same old factor positions. Its inherited constraints are
not authoritative: the new state inherits the checked source's constraints.
The checker recomputes exact residual coefficients

`delta = Wz+b-p = delta_c + delta_Gc xi + delta_Gb beta`.

For row i it verifies

`rho_i = |delta_c_i| + sum |delta_Gc_ij| + sum |delta_Gb_ij|`.

The output contains an additional **fresh continuous** factor with coefficient
`rho_i`, independently for each nonzero row. Given any old feasible assignment,
choose that factor as `delta_i/rho_i`; its magnitude is at most one, it changes
no old constraint, and the output equals the exact affine image. Zero error
requires no fresh factor. The checker validates every position, ID, coefficient
and inherited constraint. This is an outer-enclosure argument, not a claim that
the old approximate image was exact. New matrices have new identities; old LP
duals are not used against them.

## ReLU graph and checked ranges

Current ranges use exact generator-box sums, retaining all inherited constraints
but not optimizing them. They may be loose; no uncertified numerical support
bound is used. The checker independently proves each proposed range contains
that generator box. Positive/negative stable cases copy the source/return zero;
zero is handled without a strict tie exclusion.

For an unstable preactivation `a in [l,u]`, `l<0<u`, introduce continuous
`t_minus,t_plus in [-1,1]` and a fresh binary `z in {-1,1}`. Define

`y = u/2 * (1-t_plus)`

and require

`a = l/2*(t_minus+z) + u/2*(1-t_plus)`,
`-t_minus-z <= 0`, `-t_plus+z <= 0`.

If z=+1, the box forces t_plus=1, so y=0 and a<=0. If z=-1, it forces
t_minus=1, so a=y>=0. Conversely each a in[l,u] has such an extension:
for a<=0 choose z=+1,t_plus=1,t_minus=2a/l-1; otherwise choose
z=-1,t_minus=1,t_plus=1-2a/u. Thus the constructed graph represents ReLU on
the supplied enclosure. All projections, coefficients, RHS subtraction and
factor allocations are exact and independently checked. This does not justify
the old floating ReLU states or old support-tightening facts.

## Shared/private factor composition

Both expert prefixes start from the identical checked guarded input. All input
factor IDs and base constraints are preserved. Each expert's error and ReLU
factors are private; their names must be disjoint even across factor kinds.
The checker reconstructs the maps into one shared-input frame and checks every
output coefficient, equality and inequality after remapping. Base constraints
occur once, each expert's residual constraints remain present.

For a common input assignment, both prefixes have satisfying private extensions;
disjointness lets these extensions coexist. This proves containment of the
joint expert-prefix relation, not a Cartesian product of independent inputs.
The check is not merely comparing an integer `frame_id` flag.

## Controls and bounded real execution

Eleven new tests include exact affine/ReLU witness extensions, positive/negative/
zero activations, subnormal and tiny input intervals, continuous/binary error
terms, changed source/constraints, missing compensation, wrong sign/range,
private aliasing and wrong maps. A moved bundle works after its synthetic source
directory is removed; six hash-rebound semantic mutations reject. Two unchanged
lifecycle regressions cover exceptions, expired/active deadlines and partial
evidence. No real prefix is generated before preparation commit/push.

Real root: `data/moe/results/source_enclosure_conv98_20260920_v1`.
Single300s budget: build at most90s, check at most180s, each capped by total
remaining time. The existing owned-process deadline mechanism is reused, not a
new arithmetic supervisor. Complete construction, serialization, child startup,
checking and publication are charged. Per-state serialization is a nested cost;
all remaining serialization is included in whole-build time. Independent
relocation review and four semantic mutations are separately timed. No retry or
automatic tuning on a LIMIT/ERROR/UNKNOWN outcome.

After commit/push, act-py312 `python -m source_enclosure.run`, then
`python -m source_enclosure.review <new root>`; archive either outcome.

Even success proves only the declared real prefix. Later expert layers, complete
weighted output properties, and the old reused-fact source are not covered.
Graph correspondence to the intended program and checker/interpreter execution
remain trusted; deployed floating execution is not verified. A new full proof
would have to use the new source identities and discharge every remaining
obligation. No empirical-table or high-accuracy/cross-family claim is upgraded.
