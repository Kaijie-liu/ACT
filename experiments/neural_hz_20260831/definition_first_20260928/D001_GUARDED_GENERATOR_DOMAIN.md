# D001 — Guard-aware Boolean-affine generators: first definition exploration

Date: 2026-09-28. Branch `redu-hz`, commit
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`.
Status: mathematical candidate and source/literature review only. No numerical
qualification, implementation, network run, score gain or novelty claim.
Governed by [the revised objective](../GOAL_DEFINITION_FIRST_AMENDMENT_20260928.md).

## 1. Change to the definition, rather than to matrix storage

An ordinary HZ has a constant affine value map

    x = c + Gc xi + Gb s,
    xi in [-1,1]^p, s in {-1,1}^q,
    Ac xi + Ab s = b,

with the project's additional inequality predicates retained. The candidate
changes this constant-generator language: generator coefficients themselves
may depend on the discrete phase factors, while all dependence on continuous
input factors remains affine.

Let Rq = R[beta_1,...,beta_q] / <beta_i^2-beta_i>. A value expression belongs
to the free Rq-module M = Rq^(p+1), interpreted as a(beta)+g(beta)xi. Store
coefficient expressions as shared arithmetic circuits, not expanded monomials
or a list of phase assignments. Exact rational coefficients are the intended
first implementation sublanguage; the mathematics is over real coefficients.

A candidate element D comprises the shared factor environment, value forms
v, equality forms E, inequality forms A, and a finite ledger of certificates J:

    gamma(D) = { v(beta,xi) : xi in [-1,1]^p, beta in {0,1}^q,
                 E(beta,xi)=0, A(beta,xi)<=0 }.

J certifies additional forms that vanish on the feasible latent assignments;
it does not relax or replace E/A. Factor identities, tensor shapes, and the
original-input map are part of the element. A Boolean variable is genuinely
integral, never an interval-valued approximation.

Semantic order is inclusion of gamma. This is a candidate concrete-set domain
with exact transformers on the stated language. We have NOT constructed a
computable best abstraction, canonical quotient normal form, general join or
widening; no claim of a complete abstract-interpretation implementation follows.

## 2. HZ embedding and explicit nonconvexity

Set s=2 beta-1. Embed an ordinary HZ with

    a(beta) = c + Gb(2 beta-1), g(beta)=Gc,
    E(beta,xi) = Ac xi + Ab(2 beta-1)-b,

and the corresponding unchanged inequalities. This introduces no extra factor
and preserves every original binary identity (with its explicit recoding).

The element v=(t, beta*t), t in [-1,1], (2 beta-1)t>=0 is the graph of ReLU.
It contains (-1,0) and (1,1), but excludes their midpoint (0,1/2), so this
language is not a single CZ or zonotope relaxation.

For each fixed Boolean valuation the feasible slice and its image are
polytopes. Thus its bounded denotation is still a finite union of polytopes,
already representable by HZ. This is a set-theoretic argument only: enumerating
valuations is NOT an authorized algorithm. Greater set expressiveness is not
the proposed contribution; useful neural algebra is the research question.

## 3. Exact neural transformers and the continuous-rank invariant

Affine/Conv: v' = Wv+d, preserving the environment and E/A. Tensor-native W
may be a convolution but its complete reachable storage and consumers count.

ReLU of f=a+g xi: append a fresh binary delta and its guard

    r = delta*f,
    (2 delta-1)*f >= 0.

If delta=0 the guard gives f<=0 and r=0; if delta=1 it gives f>=0 and r=f.
At f=0 both phases are feasible and both outputs are zero. This proves exact
closure, including the boundary, without a phase-search/splitting procedure.
The two logical cases here are a proof, not a runtime enumeration proposal.

Residual/Add: combine forms over the SAME shared latent environment and
conjoin both branches' predicates; shared ancestor predicates are identified
by their original identity. Independently copying each branch's input factors
and taking a Minkowski sum would be wrong. Concat stacks the same-frame forms.
Intersection with a linear output property appends the composed inequality.

Induction: affine maps preserve continuous-affine degree; multiplication by a
new Boolean and each guard do too. Consequently propagation through affine,
ReLU and residual layers keeps exactly the original p continuous factors.
Every original/new binary and all predicates remain. This is a front-end rank
invariant, NOT a proof of fewer variables in the final decision procedure.
Smooth activations/attention products do not fall under this proof.

For exact propagation from an embedded input HZ, a feasible assignment directly
recovers the original input and its output. A general imported HZ must keep its
existing input/witness map as well. Concrete network/property replay and all
predicate checks remain mandatory; a symbolic model is not a validated ADV.

## 4. Candidate distinguishing operation: guard-aware quotient reduction

Boolean idempotence alone is existing algebra. The potential NN-specific
operation is to reduce forms modulo a CERTIFIED submodule J of forms zero on
the actual feasible set. Equality rows are members. Rq-linear combinations
remain zero. Only local proof-carrying rules are proposed; no free general
ideal membership, nonlinear SMT, complete Boolean equivalence or Groebner
basis computation is assumed.

Derived identities (paper proofs, not novel claims by themselves):

* Same preactivation f with retained phase bits beta,delta:
  `(beta-delta)*f=0`. Away from zero the guards agree; at zero the product is
  zero even if the bits disagree. Do not replace this by beta=delta.
* Opposite preactivations f,-f with retained bits beta,eta:
  `(beta+eta-1)*f=0`. Away from zero exactly one is active; at zero either bit
  may be chosen. Do not impose beta+eta=1 at the boundary.
* Hence ReLU(f)-ReLU(-f)=f. Positive/negative scaled copies reduce to affine
  combinations of f and one retained ReLU value, with all other phase guards
  still represented. The project's old signed-sharing work is prior art here,
  not a new result to claim again.
* A certified nonnegative form h satisfies `(delta-1)*h=0` under its ReLU
  guard. ReLU outputs and nonnegative constant combinations supply such
  certificates. This includes absorption; ordinary stable-ReLU treatment is
  again a comparator, not sufficient novelty.

A more informative rule to investigate is a guarded threshold chain. Let

    r=beta*f, (2 beta-1)f>=0,
    h=delta*(r-theta), (2 delta-1)(r-theta)>=0, theta>0.

If delta=1, then r>=theta>0, forcing beta=1. Thus

    delta*(1-beta)=0, and h=delta*f-delta*theta.

This is stronger than a Boolean-cube identity: it uses the numeric sign guard.
It removes the mixed product delta*beta from the value expression, without
dropping either bit or either guard. The strict positive threshold is the
rule's premise; it cannot be silently generalized to all thresholds. Detection
requires an exact same-frame expression and a certified positive constant,
not approximate weight matching or an iid-specific choice.

Formalization detail: certify the complete vanishing form
`delta*(1-beta)*f` directly from the phase implication. Multiplying a scalar
identity placed in M's constant coordinate by a continuous-affine f does NOT
follow merely from Rq-linear module closure. A dedicated local certificate
must justify the whole form (or an explicitly defined Boolean-ideal action).

This illustrates the desired object-level algebra. Whether this or a more
general reconvergent/threshold relation occurs sufficiently often in the
target CNNs has NOT been established. A rare scalar chain is not grounds for
a long implementation campaign. Need a reusable relation on ordinary blocks.

Guard expressions may be rewritten only with exact equivalence certificates;
their constraints cannot be discarded because an output no longer references
a bit. Binary elimination/interface existential projection is OUT OF SCOPE
under the current no-binary-deletion constraint, even if a literature idea
suggests it. All certificate/guard storage must be included in any gain claim.

## 5. The terminal-cost objection is part of the research

The dangerous shortcut is to count only the fixed p front-end factors. A
linear integer terminal query must lower beta*f or equivalent products. For
an unstable ReLU with certified l<=f<=u, l<0<u, a standard exact lowering is

    r>=0, r>=f, r<=u*beta, r<=f-l*(1-beta), beta in {0,1}.

It introduces a continuous value variable and constraints. Applying this to
every original gate may reconstruct essentially the old network/HZ system.
Even when a value product disappears, its guard may still require the same
auxiliary. No solver speedup or total-variable saving follows automatically.
The coefficients of the full bounded DAG admit finite bounds, but obtaining
useful certified bounds and representing their arithmetic are paid work.

Expanded Boolean coefficients can be exponential. Shared circuits prevent
literal expansion but may merely preserve the original network graph. J can
also grow; keeping original guards and all phase factors limits savings.
Any uncertainty falls back to the unchanged rule/UNKNOWN, not convexification.

Next substantive question: find a guard-aware reduction that reduces BOTH
the propagated object and complete terminal lowering on repeated ordinary
structures, while retaining the bits and reconstructable shared input. Compare
against ordinary HZ, its already implemented exact simplifiers, and an equally
shared unnormalized coefficient circuit. If the only improvement is front-end
notation, reject that hypothesis rather than calling it a new useful domain.

## 6. Primary-source positioning (bounded initial review, not novelty proof)

- [Ortiz et al., HZ exactly represents ReLU networks, 2023](https://arxiv.org/abs/2304.02755):
  exact ReLU graph representation and linear binary growth already exist.
- [Tran et al., ImageStar, 2020](https://arxiv.org/abs/2004.05511):
  image-shaped generators support tensor-native affine propagation; local
  `/data1/Kane/HyZor/star.pdf` is this paper. Its linear-predicate single sets
  do not themselves supply the proposed compact nonconvex algebra.
- [Combastel, mixed polynotopes, 2022](https://arxiv.org/abs/2009.07387):
  closest overlap: mixed interval/Boolean symbols, polynomial dependencies,
  shared symbolic identity and eager/lazy composition. Boolean-dependent
  generators and beta-squared reduction alone cannot establish novelty.
  Its broader neutral/inclusion-preserving rewriting framework must also be
  distinguished; a generic claim of "rewriting a symbolic domain" is not new.
- [Kochdumper and Althoff, CPZ](https://arxiv.org/abs/2005.08849):
  polynomial value maps with polynomial predicates overlap semantically.
  Our inference: Boolean polynomial constraints and bounded slack lifting can
  encode this candidate in CPZ; that says nothing about efficient lowering.
  CPZ is a NONCONVEX polynomial family, not the prohibited convex CZ.
- [Xie et al., Hybrid Polynomial Zonotopes](https://arxiv.org/html/2506.13567v1):
  nearby definition/naming; its displayed form uses continuous polynomial
  monomials and additive binary generators, not literally these coefficient
  circuits. Do not claim either exact identity or established novelty.
- [Scaled HZ, 2025](https://arxiv.org/abs/2501.13023) and
  [HZ RNN reachability, 2026](https://arxiv.org/abs/2603.11547):
  screened adjacent work; scaling/training and selective phase relaxation
  are not adopted as this definition-first innovation.

Two independent read-only reviews agree on the main caution: definition-only
Boolean-affine circuits are insufficient novelty, and terminal costs cannot
be deferred out of the accounting boundary. Paper identities above have not
received a machine-checked proof or a numerical qualification run.

## 7. Next work and disposition

Keep this first note as a dated hypothesis, not a frozen runtime admission.
Next derive a bounded guard-rewrite calculus and a complete comparative
lowering example; check its overlap with existing symbolic/NN simplifications
and its prevalence in ordinary structures before building a large prototype.
Do not restart C131 budget optimization as a substitute.

No test, numerical import, payload restoration, model/solver execution,
production change or benchmark replay occurred in D001. Formal1870/2413,
every-old-solve/all13-family requirements and independent E061/400 are unchanged.
