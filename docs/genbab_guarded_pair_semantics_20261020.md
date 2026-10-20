# Guarded weighted pair semantics for a strong external comparison

2026-10-20, ACT `36e8936f5a61ccce559de2fb9a9edea657241852`, clean at entry.
This static audit closes a specification-design gap, not a runtime gate.
**A top-2 guard can be expressed using appended router outputs and the existing
external unsafe-conjunction interface.** General input-polytope support is not
required for that encoding. The current ACT adapter does not implement it.

The research consequence is to retain a guarded nonlinear baseline in G4.
The new ACT gate–margin rows repair a concrete missing connection, but belong
to a standard sigmoid envelope family; they do not alone establish G1 novelty.
No external import, adapter implementation, solver, model, data or GPU run
occurred. M1b/M3 execution failures and exhausted attempts remain unchanged.

## The same guarded obligation without a TopK operator

Assume the registered real-valued output-layer selected-softmax top-2 object,
no shared expert, common output space, unit temperature, and finite router
scores. For an unordered canonical pair S={a,b}, a<b denotes an expert index
order, NOT an ordering of router scores. Define

```
m_S(x) = r_a(x) - r_b(x)
F_S(x) = E_b(x) + sigmoid(m_S(x)) * (E_a(x) - E_b(x))
G_S(x) = AND_{i in S, j not in S} [r_j(x) - r_i(x) <= 0]
p_S,t(x) = q_t^T F_S(x) + c_t.
```

All subgraphs consume the same input x and the same model state. G_S is
equivalent to S being tie-legal top-2: every selected score is at least every
unselected score. Do not require r_a>=r_b. Equality is included, so multiple
pairs can be legal at the same x.

For the strict margin target p_S,t>0, the exact bad set is

```
OR_{S,t} {x in X : G_S(x) AND p_S,t(x) <= 0}.
```

Thus every guarded obligation can be expressed by a static graph with outputs
Z_S=[F_S,r] and one unsafe matrix C_S,t Z_S <= d_S,t. It contains
2(E-2) guard rows and one property row; the last right-hand side is -c_t.
For a frozen nonzero margin threshold tau, use tau-c_t instead of silently
changing the target. For general q,c, an alternative graph appends the affine
property values p_S,t and uses scalar output comparisons to zero.

**Rows within a matrix are AND; pair/property matrices are OR.** To rule out
one bad conjunction on a domain it suffices to prove one atom impossible.
Different subdomains may exclude different atoms. To prove the complete request,
all bad conjunctions must be excluded. Proving each guard false, or treating
guard rows as separate output-safety obligations, is not the same problem.

If the actual property is non-strict p>=0, excluding p<=0 is a stronger
sufficient test, not an exact complement: equality alone is not UNSAFE.
An apparent bad point must satisfy the original input, route and property
contract and the registered full-model replay rule. A tie-legal branch witness
is not automatically a witness for a deployed implementation's deterministic
tie selection. Never change weak/strict signs or add a tolerance to remove ties.

For intuition only, writing g_S=max(r_j-r_i) gives the exact strict statement
max(g_S,p_S,t)>0. Replacing it with max(g_S,p_S,t)>=0 is unsound for a weak
property implication at g_S=0,p_S,t<0. Use the direct conjunction encoding;
this audit does not propose another max/eta adapter.

## Coverage and cost variants must remain separate

| Variant | Obligations and source of coverage | Current state |
| --- | --- | --- |
| Existing ACT front end plus plain CROWN | ACT enumerates feasible pairs; each pair is verified on the whole box | Implemented; no pair guard enters the CROWN call |
| Guarded static pair | Same input box, actual variable gate, router outputs and bad conjunctions | Specification-level route established; adapter/runtime unvalidated |
| Complete all-pair aggregation | Every unordered pair and every property, each with its guard | Mathematically covers all tie-legal routes without ACT route exclusions; runtime unvalidated |

The last variant may run one static pair at a time or use a shared graph of all
pairs, but these are different cost configurations and must be frozen separately.
One original request has one 300-second budget, not 300 seconds per pair.
If ACT route pruning is used instead, that analysis is charged and its exclusions
remain part of the trusted/proved coverage boundary. A pair UNKNOWN cannot be
silently removed.

For E=8 and nine classification properties, all-pair aggregation has 28 pairs,
252 bad conjunctions and 13 atoms per conjunction, or 3,276 logical rows.
This is a combinatorial inventory, not a memory/work admission bound or runtime
prediction. It does not relax the separate rejected ACT full-source capacity.

The static algebra is equivalent in the declared real semantics. Replacing
selected softmax with sigmoid of a difference, adding property projection, or
exporting a shared graph does not prove identical native floating-point
execution. Actual lowering, input containment, weight identity and evidence
grade still require controls; finite probes alone are not a global proof.

## Pinned source findings

Read-only local sources were clean:

- alpha-beta-CROWN `e5c7e17bf0488843acb77b7519f59876717a49f4`,
  `/data1/Kane/MOE/baselines/alpha-beta-CROWN`.
- auto_LiRPA `5a098e8f9fb5786a428a024981d833d303921f2d`,
  its existing `auto_LiRPA` submodule.

These are compatibility-audit revisions, not a claim to be the newest release
or an executed GenBaB configuration.

| Source anchor | Static finding |
| --- | --- |
| ACT `act/back_end/moe/static_pair.py:12` | Same-input variable selected-softmax pair graph exists; rejects a shared expert |
| ACT `act/pipeline/moe/external_pair_worker.py:63` | Route completeness is checked before external execution |
| Same worker `:87`, `:105`, `:124` | Nine ten-class properties, whole-box pair bounds, `method="CROWN"`; no guard/specification adapter or nonlinear BaB |
| External `complete_verifier/read_vnnlib.py:111`, `:273`, `:309` | Input box restriction is distinct from output comparisons; Y-to-Y guard comparisons stay in a common matrix, and DNF AND groups stay together |
| External `complete_verifier/specifications.py:326` | Records the number of conjuncts per disjunct instead of flattening their semantics |
| External `complete_verifier/incomplete_verifier_func.py:193` | A single unsafe conjunction uses the any-atom exclusion stopping rule |
| External `complete_verifier/activation_split/bab_bootstrap.py:97` | Activation BaB uses the corresponding conjunction exclusion rule |
| External `complete_verifier/api.py:3141`, `abcrown.py:205` | `auto` can choose input BaB or `bab-refine`, not the intended nonlinear activation-split configuration |

The legacy [input-frontend compatibility result](../act/pipeline/moe/docs/external_compatibility_results.md)
remains valid: its relational INPUT expression was rejected. That observation
does not reject the appended-OUTPUT route established above. Clip-and-Verify
can be a separate domain-tightening capability; the Boolean encoding does not
depend on enabling it.

The current worker also hardcodes ten classes and nine classification rows.
The static graph's output flexibility is not evidence that this complete-request
worker already accepts arbitrary properties or output widths.

Four entrypoint details affect whether a comparison is actually the same task:

1. `VerificationSpec.build_from_input_bounds` (`api.py:1238`) accepts explicit
   bounds and raw unsafe `(C,rhs)` clauses. In contrast, the expression API
   (`:1273`) negates its supplied output property. Passing an unsafe AND to
   that safe-property DSL reverses the problem. Its safe expression would be
   the OR of a positive property margin and a strictly false guard atom.
2. `build_from_center` (`:1223`) first casts center/epsilon to float32; expression
   output matrices (`:973`) also use float32. The explicit-bounds/raw-clause
   constructors use the configured default dtype (`:1043`, `:1121`). A future
   binary64 comparison should use the latter and verify the final endpoints,
   coefficients and dtype, not reconstruct the box from center/epsilon.
3. The simple VNNLIB text parser accepts scalar output-to-output/constant
   comparisons, not arbitrary q expressions. Emit explicit property outputs
   or use raw matrix clauses; do not assume the text parser is a general SMT
   linear-expression parser.
4. With INVPROP output-constraint propagation enabled, the current incomplete
   verifier rejects multiple ORs each containing multiple ANDs
   (`incomplete_verifier_func.py:245`). One pair/property conjunction per call
   with charged, complete outer aggregation avoids that specific restriction;
   it is not evidence of runtime compatibility or a free extra budget.

A future explicitly authorized control must bind the actual selected verifier,
nonlinear branching configuration, dtype and result semantics. A successful
small `auto` API run is not evidence that GenBaB executed. In this pinned source,
the default low-dimensional path can select input splitting with plain CROWN;
the non-convolutional larger path can select `bab-refine` and disable clipping.
No configuration is changed or activated by this audit.

The official [tool description](https://github.com/Verified-Intelligence/alpha-beta-CROWN)
advertises nonlinear operators, output specifications and GenBaB; the
[GenBaB paper](https://arxiv.org/html/2405.21063v2) describes nonlinear branching
and interval-dependent relaxations. Those capabilities motivate this comparison,
not a measured result on ACT's objects.

## What the M3 rows do and do not add

Write T_t(m)=sigma(t)+sigma'(t)(m-t). The M3 relation is T_0(m)=1/2+m/4,
plus a half-range bound. On a positive interval [l,u], sigmoid is concave:
the chord is at least 1/2 and the tangent T_l is no greater than T_0 for m>=l.
On a negative interval, the chord is at most 1/2 and T_u is no less than
T_0 for m<=u. Hence a conventional endpoint-tangent/chord family can imply
both M3 rows; no new nonlinear operator is needed.

This is a statement about mathematical expressivity, NOT every computed
generic bound. For example on [0,1], the lower chord and upper tangent at 1
admit m=0,lambda=51/100: that upper intercept is sigma(1)^2>25/49>51/100.
M3 excludes that point. Optimized choices, intervals, integer versus relaxed
ReLU factors and finite budgets can change the actual ordering.

The pinned `auto_LiRPA/operators/s_shaped.py:279` implements chord/tangent
families; `:327` clips optimized tangency parameters to the interval and
`:1196` supplies sigmoid via the S-shaped implementation. This supports the
family comparison, not a claim that an optimizer chooses the desired endpoint
or that one complete verifier dominates another.

The saved M3 separation therefore remains useful evidence about a missing
ACT relationship, but not an irreproducible advantage over generic shared-graph
verification. Do not add more slopes or synthetic examples simply to strengthen
a novelty claim. The consequential open test is complete-request precision/cost
against a correctly expressed strong baseline.

## Decision and remaining gate

Separate read-only AI reviews checked the Boolean/mathematical contract and
the pinned source paths; neither review executed a backend or replaces human review.

Retain three distinct comparisons: whole-box sufficient pair, guarded pair and
complete route aggregation. Prioritize validating the guarded specification
route over another ACT-only envelope control. This advances G4 experimental
design; it neither closes G1 nor establishes external performance.

This audit does not approve deployment, external adaptation/execution, another
M3 run, M2, GPU or observer repair. The repeated RSS/reception failure still
requires a separately scoped decision before further native execution.
Any subsequent adapter control must explicitly check guard orientation, ties,
property offsets, AND/OR grouping, incomplete coverage and actual engine
selection, under an approved finite scope. Existing failures, sources and
resource limits are retained.
