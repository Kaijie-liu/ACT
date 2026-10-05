# D005 — phase-compatible contextual quotient for Neural-HZ

2026-09-28; `redu-hz`, commit `f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`.
Paper-only definition/proof draft, independently reviewed ordinary example.
No code implementation, numerical qualification, model census or formal gain.
After the user's alignment reminder, this is **supporting research, not the
selected main domain contribution or next engineering task**. See
`../ALIGNMENT_D005_20260928.md`; the potential source preflight below is retained
as a conditional follow-up, not a decision to expand this local rewrite.
D001 remains the phase-dependent generator language; D003 remains its frozen
unnormalized reference. This changes the candidate **equivalence relation**,
not the old results, native resource caps or promotion standard.

## 1. A consumer-sensitive equivalence, not equality of every real node

Let S be a certified common retained prefix domain containing original inputs,
all original/free prefix bits, ALL original prefix ReLU guards, shared frame
identity and original predicates. Expressions f,h and their certificates may
not depend on the new parent bit or descendant guards; that would be circular.
For a real preactivation expression f define its full guarded-ReLU relation

    R_f(s) = { (delta,v): delta in {0,1},
                         (2*delta-1)*f(s)>=0, v=delta*f(s) }.

Define `f ~[S,ReLU-phase] h` iff R_f(s)=R_h(s) for EVERY SAME s in S.
Equivalently, at each s either both f,h are strictly negative, or f=h>=0.
Proof: a negative input permits only (delta,v)=(0,0); zero permits both
(0,0),(1,0); positive input permits only (1,f). These three cases distinguish
sign/zero and positive magnitude exactly. No runtime case search is required.

This relation is weaker than real-value equality but stronger than equality of
rectified values alone. It preserves every original parent phase witness,
including both choices at zero. It is an equivalence on a fixed S, and a
congruence only for this typed consumer—not an ideal/module identity usable
under arbitrary affine contexts. For example -1 and -2 are equivalent here;
after adding3/2 they are not. A shared raw-value consumer therefore forbids
global replacement unless it has its own exact certificate.

A candidate Neural-HZ element can retain D001's original continuous symbols,
binary identities and phase-dependent affine forms, but distinguish raw-value
ports from guarded-ReLU ports. A certified representative may replace f by h
only at the latter. All original EQ/LE consumers remain raw unless separately
proved invariant. The concretization is still the complete Gamma=(original
input, ALL original bits, declared outputs), subject to all predicates.

The exact replacement theorem follows immediately from equality of R_f and
R_h, then compositional equality of every unchanged downstream operator on
the same frame. Keep original source expressions and a reconstruction recipe
for hidden values; do not claim a constant-affine decoder for retired values.
Their source/certificate/reconstruction storage must be included in the cost.
This is not permission to erase a bit, predicate, source branch or raw consumer.

## 2. One uniform forward certificate on mixed fan-in

Let r_i=ReLU(f_i), and let parent preactivation g contain selected terms
a_i*r_i with a_i>0. Other terms may have mixed signs or be residual inputs.
For each selected i define the exact remainder q_i=g-a_i*r_i. Suppose a
non-circular certificate on the SAME retained prefix S proves

    q_i <= -epsilon_i < 0.

Set ghat=g-sum_selected a_i*(r_i-f_i). This is the result of replacing the
selected rectified inputs by their original preactivations in this consumer.
Then `g ~[S,ReLU-phase] ghat`.

Proof: Delta=sum a_i*(ReLU(f_i)-f_i)>=0, hence ghat=g-Delta<=g. If any selected
f_i<=0, then r_i=0 and g=q_i<0, so also ghat<0. Otherwise all selected r_i=f_i
and Delta=0. Thus negative values may change, but signs/zero sets and all
nonnegative values agree. At a selected child's zero the parent is strictly
inactive. At the parent's zero every selected child is strictly positive,
and both parent phase choices remain legal. This proves simultaneous rewriting;
same-parent sequential rewrites also preserve the remainder upper bounds because
earlier nonnegative Delta terms only decrease the remaining remainders.

With ordinary affine row g=b+sum_j a_j r_j+d*u, a sufficient certificate is
the forward interval upper bound on the row with term i omitted. It can be
computed with exact/outward-safe coefficient arithmetic from already certified
bounds; it neither fixes a phase at runtime nor invokes a solver, LP status,
dual certificate, backward search, attack or split. A positive-edge-only scan
can share row-sum accounting, but must account for exact arithmetic, bounds,
all consumers and rewritten coefficient fill. No discovery/runtime cost has
yet been measured. The certificate cannot depend on the parent guard being
replaced, a dropped predicate, a target identity or a favorable final margin.

Strict negativity matters for FULL phases. For g=ReLU(x1)+ReLU(x2)-1,
replacing the first child using only q1<=0 fails at x1=-1,x2=1: g=0 admits
parentdelta=1, whereas ghat=-1 forbids it. This is an ordinary proof boundary;
the first candidate rejects zero remainder bounds rather than building a
special-case repair path.

## 3. Ordinary mixed positive example and full terminal scope

On x in [-1,1]^2 let

    f1=x1+x2/2,       f2=x1/2+x2,
    r1=ReLU(f1),      r2=ReLU(f2),
    g=r1+r2-2,        y=x1+ReLU(g).

The two distinct child hyperplanes cross the box; each child ranges [0,3/2].
Both omitted-term remainders are <=-1/2, so

    ghat=3*x1/2+3*x2/2-2,   ReLU(g)=ReLU(ghat)

with the SAME parent sign/zero relation and original child bits. The parent
is active at (1,1), inactive at (0,0), and zero at (2/3,2/3), where both child
values are positive. These are paper witnesses, not sampled evidence or ADVs.
This is mixed two-input fan-in plus residual, not a scalar threshold chain,
duplicate neuron, dead parent or extreme-number example.

For this isolated interface only, r1/r2 have no other raw consumer/predicate.
Retire their native values, retain each child's two exact sign rows and its
bit, and use the parent's four rows with new sound ghat bounds[-5,1]. The
integer-terminal bill changes from5 to3 continuous variables and12 to8 gate
rows; all3 original bits remain. Gate-row nnz changes from30 to22 for the
displayed coefficients, counting nonnegativity rows and all guard rows. Add
the 12 versus8 RHS entries, all variable bounds, original-source metadata,
certificate and reconstruction costs; these are symbolic counts, not a paid
native-HZ component qualification or measured speedup. An equality-only HZ
backend's slack/encoding costs require a separate full comparison.

In a network, a child native value is retired only if ALL its consumers allow
it, including every raw preactivation, original EQ/LE predicate, residual,
Concat and output. Keeping an unrewritten consumer means keeping that value;
local edge rewrites alone must not be counted as a global variable saving.
Arbitrary predicates on original g likewise prevent deleting its original
construction. The transformation does not automatically preserve D003's
all-hidden-node constant-affine readout contract; full hidden reconstruction
uses the original shared source expressions and unchanged input/phase values.

## 4. Integer equivalence does not preserve the LP relaxation

The original parent bound is[-2,1]; the rewritten preactivation bound is[-5,1].
At x=(0,0), take childbits=(1,1), relaxed parentdelta=1/4 and parent value1/4.
The new child guards and parent rows admit this, even after adding valid
implications parentdelta<=childbit_i. The old child rows force r1=r2=0 and its
parent upper row requires value<=-1/2. Thus the new rows admit fractional tuples
the old system excludes; phase-implication cuts alone do not fix this.

Conversely x=(1/2,1/2), childbits=(3/4,3/4), r1=r2=9/8, parentdelta=1 and
parentvalue=1/4 satisfy the original LP rows: g=1/4. The new ghat=-1/2 forbids
that same parentdelta. Without added cuts the two retained-phase relaxations
are therefore INCOMPARABLE, not a proved one-way inclusion. Both paper witnesses
are fractional diagnostics only. They do not invalidate the integral theorem,
but they forbid claiming old CERT preservation or an LP-neutral simplification.

Preserving the old projected LP as well would require its additional projected
constraints, with their full derivation/row/storage cost, or retaining the old
lift. No such cost-free strengthening or fallback menu is assumed. Any actual
candidate remains opt-in and must pass same-structure, old-solve/family and full
replay requirements; exact integer equivalence alone cannot promote it.

## 5. Prior art and falsifiable research claim

Mixed Boolean/continuous functional sets already support variable-dependent
generators; D001/D005 cannot claim that language alone is new.
[Combastel](https://arxiv.org/abs/2009.07387).

Botoeva et al.'s consecutive-layer dependency analysis already uses an
omitted-contribution upper-bound condition (Lemma3) to infer inactivity
dependencies and uses them in MILP search. Our strict sufficient bound is
related prior art, NOT a new discovery. We do not import its splitting or
runtime branch-dependent procedures. The paper does not by itself establish
this draft's all-phase contextual rewrite, full consumer retirement or new
domain novelty. [AAAI2020, Section3](https://ojs.aaai.org/index.php/AAAI/article/view/5729/5585).

Zhang/Bölcskei give a compositional calculus for functional equivalence and
deep ReLU rewrites. Therefore mixed-layer value simplification alone is also
insufficient novelty. Our stronger same-original-phase/zero contract must be
compared explicitly, not presumed novel because the vocabulary differs.
[Complete Identification](https://arxiv.org/html/2602.00266v2).

Ordinary HZ/shared-circuit processing given the SAME certificate can perform
the SAME rewrite and keep the same bits. A useful new abstract-domain claim
would need a substantive compositional invariant, efficient proof/normalization
theorem and actual common-structure advantage beyond that comparator. None is
established here. The current progress is identifying a strictly more permissive
semantic quotient than D004 exact-preactivation equality, proving a nontrivial
mixed positive control, and exposing its LP/consumer limitations.

## 6. Conditional supporting follow-up, not the current main action

The existing Tiny/CIFAR descriptors contain mixed Conv/Add/ReLU structures,
but no per-row full certified bound/certificate population for this rule.
They do not establish even one removable real child value. Next conduct a
fresh scoped source-evidence preflight: faithful graph, original bound origins,
uniform positive-edge omitted-term certificates, ALL consumer coverage,
coefficient fill and old/new LP comparison requirements. If only stable/dead
neurons or isolated unsupported motifs qualify, stop this main route and keep
the theorem as supporting work. Do not build another test/accounting framework
or assert large-CNN applicability before that evidence.

A read-only source check found that the historical corrected Tiny census is
also insufficient as proof authority: `run_corrected_phase_support_census_v1.py`
stores hashes and aggregate support counts, not the full per-neuron bound
arrays, stops at layer36, and labels its own scope as existing IntervalTF
rather than independent outward-rounding authority. The saved record has
`hz_verifier_run:false` and `positive_authorization:false`. Neither fixed-center
containment nor a bound hash proves the required universal negative remainder.
The old source and result remain unchanged. Source SHA256
`df04ee8b2b3f924a5c4a0d2d1f6a1e7baf0009d27cc9fa12c2826705fe47a63f`;
`evidence/corrected_phase_support_census_20260905_v1.json` SHA256
`425e3997e6c68b388145b4b7e4e86b211d115880fb7c17871704f0594f0b30ff`.
Any real bound/certificate census therefore needs a fresh sound source protocol;
copying historical support totals into the new theorem is not a safe shortcut.

The actual corpus scan, imports, bounds computation and implementation each
need a fresh frozen scoped plan and all applicable existing caps. No relaxation
of those caps, LP/capability gates, original2413/400 populations, or historical
custody is made. Formal1870/2413 and separateE0=61/400 remain unchanged.
