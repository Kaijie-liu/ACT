# D015 supporting paper note — one sign-dual four-row normalizer

2026-09-28; redu-hz, f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac.
Independent algebraic review only; NOT selected/implemented in frozen D015
v1 or v2. No additional test/model/solver run or score gain follows here.
This records a bounded mathematical investigation performed while the source
pilot was running, not a replacement for that real-source experiment.

## Exact simultaneous rule

Original gate: r=ReLU(f), with its original beta and SAME valid l<0<u.
Let p,n be nonnegative linear forms of existing nonnegative amplitude columns.
On the complete original integer SOURCE relation, require

    p>0 => f-p>=0,       n>0 => f+n<=0.

Nonnegativity p,n>=0 must also hold in the retained source LP relaxation.
Use ONE set of four rows:

    r>=p,   r>=f+n,   r<=u*beta,   r<=f-l*(1-beta).

There is no alternate terminal path or instance dispatch. Both certificates
must concern the SAME original f, never a successively altered approximation.
All source predicates, input identities, original bits and raw-f consumers
remain. p,n are expressions, not new columns or phases.

Proof: new rows imply old rows since p,n>=0. Conversely, for an old integer
tuple with beta=0, r=0 and f<=0. The positive premise forces p=0; if n>0 the
negative premise ensures f+n<=0, and if n=0 this is already true. For beta=1,
r=f>=0. The negative premise forces n=0; the positive premise implies p<=f
when p>0, and p=0 is harmless. Thus every old integer tuple remains. At f=0,
both premises force p=n=0, retaining both original zero-phase choices.
For fractional source/phase points, new rows directly imply the old rows, so
the new LP is a subset even though the premises need only be integer-valid.

This includes D014 negative-only shielding by p=0. The positive-only analogue
by n=0 strengthens the original nonnegative-output lower row, without adding
an output readout or variable. It is NOT the stronger exact r=P elimination.

## Uniform source certificate and limitation

Write f=s+sum_P a_i e_i-sum_N d_j e_j, with a_i,d_j>0, 0<=e_i<=u_i,
Slo<=s<=Sbar, and a certified amplitude-conflict graph E. Set

    D_i=sum_{j in N, {i,j} not in E} d_j*u_j,
    I={i in P: Slo-D_i>=0},       p=sum_{i in I} a_i*e_i;
    C_j=sum_{i in P, {i,j} not in E} a_i*u_i,
    J={j in N: Sbar+C_j<=0},      n=sum_{j in J} d_j*e_j.

If p>0, choose an active selected i. Its conflicting negative amplitudes
vanish. After removing ALL selected positive terms, f-p is still bounded
below by Slo-D_i, because the unselected positive terms are nonnegative.
The negative proof is the D014 argument with signs reversed. No input/phase
split, solver query, LP status, public label or attack is involved.

Because D_i,C_j>=0, the unconditional certificate can have BOTH sides nonempty
only if Slo=Sbar=0. The symmetric form covers either baseline sign, not generic
two-sided extraction on a varying biased baseline. Failed sufficient tests
do not prove the corresponding term essential.

## Full local cost and a strict paper control

For distinct canonical amplitude columns separate from baseline columns,
the positive lower row adds|I| nonzeros; the negative lower row saves|J|.
Net native row-nnz change is|I|-|J|; variables/bits/rows are unchanged. Aliases,
coefficient merging, duplicate rows, source discovery and evidence must be
counted explicitly. Positive strengthening is NOT free compression.

Reflect the D014 biased control:

    f=1/2-e1-(1/4)e2+e3+e4,
    e_i=sigma_i*t_i, t_i in[0,1], sigma_i binary, sigma1+sigma3<=1.

Conflict13 and caps1 give I={3}, p=e3, n=0; old tight bounds are[-3/4,5/2].
The source-inclusive formulation has9continuous/5bits/21rows48nnz before
scalar bounds/RHS. This positive rule produces49nnz, not47 or a saving.
At t=(1,1,1,0), sigma=e=(1/2,1,1/2,0), beta=1,r=1/4, all old rows hold with
f=1/4, while the new r>=e3=1/2 rejects the fractional-source tuple. These
are hand-derived facts, not executed LP results or concrete network inputs.

## Research classification

This follows from ReLU sign duality and valid lower-row strengthening. The
ordinary HZ encoding supplied the same valid rows has identical semantics;
the scalar algebra and generic constraint normalization are not claimed new.
It does not solve the outstanding compositional/domain-novelty problem.
Its possible value is broader source applicability at a measurable predicate
cost. Do not add it to a runtime menu or silently change the frozen D015
negative-only experiment. Any selected successor needs fresh preregistration,
complete costs, full inherited tests and real source evidence under the goal.

## Interpreting source hits: stable tightening is not nonconvex shielding

For the negative-only rule, define B0=Sbar+sum_P a_i*u_i and Bj=Sbar+Cj.
If an accepted j has no positive-mass certified conflict neighbor, Bj=B0<=0,
so f<=B0<=0 on the whole source. This proves identically zero ReLU OUTPUT,
not a fixed original phase: original f=0 still permits both beta choices.
An old independent full_bounds interval crossing zero does not refute this;
it only demonstrates that the old bound was too loose.

Three evidence levels must therefore remain distinct:

1. B0<=0: unconditional zero-output certificate, even when irrelevant
   conflict edges happen to exist. Potential source-bound improvement, not
   evidence that nonconvex dependency reasoning was necessary.
2. B0>0 and Bj<=0: a conflict is necessary for THIS fixed sufficient test.
   This still does not prove a positive target value or positive extracted
   amplitude actually reachable under all original source constraints.
3. Separate original-source reachability proofs for f>0 and selected e_j>0:
   the latter plus d_j>0 and shielding implies f<0. This establishes a genuine
   cross-phase local instance, still not a property solve or native net gain.

Count distinct target rows separately from extracted terms. Preserve B0,Bj,
removed positive mass, responsible edges, actual denominators and unmeasured
reachability/LP/native cost flags. Window-level tested_pairs and shielded_terms
alone are insufficient to classify individual hits. D015's already frozen
workers are NOT modified to add this analysis or extra source evaluations.
Only sufficiently complete saved evidence can support a later saved-only
classification. Missing evidence is UNKNOWN, not an inferred zero or success.
