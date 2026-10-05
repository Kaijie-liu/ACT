# Shared-source attention: bounded directional query

This document separates paper theorems from an unexecuted implementation and
from source/model qualification. Novelty is not established. The native object
retains the original source, phase identities, EQ/LE, frames and decoder;
temporary two-dimensional convex geometry is a query image, not a replacement
of Neural-HZ by a zonotope/CZ. No source partition, phase enumeration, solver,
backward rescue or concrete-network claim is introduced.

## One token and the positive-tail theorem

For a compact rational polygon P, define

    F(t) = max_(s,v in P) exp(s-shift) (v-t).

Because the objective increases in v, only the upper boundary U(s) matters.
Let m=max U, L=min s and R=max s. The upper boundary is concave piecewise affine.
When t<m the maximum is positive. On the set U(s)>t,

    log(exp(s-shift)(U(s)-t)) = s-shift + log(U(s)-t)

is concave. More constructively, a segment of slope b has derivative sign
D=U(s)-t+b. Let k be the rightmost maximum of U. To its left the upper boundary
is nondecreasing, so any positive objective is increasing. On its right U and
the segment slopes are nonincreasing; D decreases along negative-slope edges
and jumps downward at vertices. The maximum is therefore a vertex, the final
endpoint, or the unique relevant root D=0 inside one edge.

After the upper hull and slopes are prepared, binary search on its right-hand
vertex derivatives identifies that edge/kink using O(log number_of_vertices)
rational comparisons. A stationary point on the edge beginning (s0,v0) is

    s* = s0 - (v0-t+b)/b,     v* = t-b,     b<0.

One exponential at this rational point evaluates the positive support. Point,
vertical-line and horizontal-edge degeneracies are included. Preparation sorts
all supplied points and constructs their complete upper hull, not a sampled
or selected subset. The score/value lower bounds retain every original point.

## The negative branch is not unimodal

At t=0 the concave upper chain

    (0,-1), (1,-1/3), (2,-1/7), (3,-1/27), (4,-1/64)

has two strict local maxima of exp(s)U(s), at scores 1 and 3: the objective's
incoming derivative is positive and outgoing derivative negative at both.
Reflecting this chain about (2,-2) supplies a lower boundary, yielding a
centrally symmetric rational polygon, hence a genuine two-dimensional zonotope.
This refutes using the positive-tail argument on all-negative token supports;
it is not an information-theoretic lower bound on every possible exp algorithm.

The new query deliberately uses

    H(t) = F(t)                    if t < m,
           exp(L-shift)(m-t)       if t >= m.

If B is the support of the independent score/value rectangle from the SAME
polygon information, then

    F(t) <= H(t) <= B(t)

for all t. The negative branch is generally strictly weaker than F. For the
segment (0,-1),(1,-1/10), t=0, shift=0, F=-e/10 but H=-1/10. A point/vertical
token has constant score and is exact in both branches; at t=m both are zero.
The error is bounded by

    0 <= H(t)-F(t) <= (exp(R-shift)-exp(L-shift)) (t-m)_+.

Each F is a maximum of affine functions in t. For delta>0,

    -exp(R-shift)delta <= F(t+delta)-F(t)
                         <= -exp(L-shift)delta.

Since F(m)=0 and the negative branch has slope -exp(L-shift), the same bounds
hold for H across the switch. H is continuous, strictly decreasing (indeed
convex). It is an APPROXIMATE SUPPORT QUERY, not the D127 exact polygon oracle.

## Product queries, roots and multiple directions

All token terms use one fixed shift. Their sum has a unique zero, bracketed by
the minimum and maximum of all token values. In the independent-token exact
source setting, the zero of sum F is the true attention maximum, so the zero
of sum H is a safe upper bound lying no higher than the rectangle root. With
source overlap, original predicates, or coefficient-error rectangles, there
is an additional explicit outer approximation. A positive lower bound on H
does not show a realizable positive network value.

The returned lo/hi bracket is for the ROOT OF sum H, NOT a two-sided bound on
the native attention's range. Actual lower bounds require querying the opposite
output direction. Across heads, sums of upper bounds are sound but generally
not jointly attainable; no witnesses can be concatenated. Source/model binding
belongs to the separately authenticated caller, not to this scalar oracle.

The implementation makes at most twelve bisection steps. It lowers hi only
when the sum's interval upper endpoint is nonpositive, and raises lo only when
the lower endpoint is positive. An exact zero interval identifies the root.
An unresolved sign leaves a safe bracket and reports uncertified precision;
it must not claim unconditional initial_width/4096. The matched rectangle root
is independently bracketed using cached endpoint exponentials. Intersecting
the H upper bound with that rectangle upper bound preserves soundness even
when finite-precision bisection paths differ.

The D124/D127 threshold-1/2 and threshold-5673/10000 controls have a constant
zero token as their only negative branch. H equals F on every token there.
Their strict comparisons therefore survive the branch change in real
arithmetic; the new finite-precision tests must still certify the signs. The
strong reference remains the SAME true sigmoid probability graph with the
specified scalar McCormick/simplex/energy constraints checked in inherited
D127 tests. This does not claim dominance over full RLT, source Taylor, old
production HZ, or every competing attention method.

## Certified sixteen-term exponential, not changed historical precision

D127 remains read-only. The candidate's exp16_bounds uses the same 512-bit
Fraction gate, argument range [-64,64], 72-bit outward core arithmetic and
168-bit outward reciprocal. After reducing |x| by powers of two to r<=1/2,
sum terms 0 through 15. A lower sum is safe because all omitted terms are
positive. Bound the first omitted term r^16/16! outward and its following
geometric tail using ratio at most ru/17, where ru is the rounded-up r. Repeated
outward squaring reverses range reduction; negative arguments invert and swap
positive endpoints, then round outward. Every operation preserves enclosure.

This variant may be wider. Neither equivalence to the old 64-term intervals
nor unconditional nesting is assumed; tests require both certified intervals
to overlap and establish independent rational brackets/tail checks. Resource,
type, range and intermediate-width failures raise rather than substitute float
arithmetic, an uncharged oracle, a narrower source population or a rescue path.

## Full cost and authenticated caching

The caller supplies one unchanged D127 Budget across source construction,
polygon preparation, all thresholds/directions/models and evidence. This module
creates no budgets. Sorting comparisons, hull arithmetic, checks/container
visits, preparation/caching, query arithmetic, bisection and rectangle comparison
are charged; upstream D127 polygon construction is additionally charged by its
caller. Counters are scalar-work units, not bit complexity or wall-clock proof.

Prepared handles are frozen records with immutable geometry. A module-private
weak-identity registry records the actual issued object and original field
identities plus its issuing Budget. Public reconstruction/dataclass replacement
or altered metadata is rejected without an O(d) structural hash per query. The
registry does not hold its key alive. Each handle retains only one immutable
endpoint-exponential pair for one shift; changing shift replaces it. Checks,
creation and use are charged. Returned entries and conservative entry_upper
cover retained and construction/cache symbolic entries; physical memory and
full aggregate/source ledgers remain the worker's responsibility.

For 192 signed directions per model and three heads, the two models require
1152 root queries, with respectively 17 and 5 tokens per root. Twelve threshold
rounds give at most 152064 positive-branch exponential calls, plus at most
25344 endpoint-cache calls before reuse. Geometric construction still processes
all original generator columns for every direction; only subsequent threshold
search is logarithmic. These counts are not a successful whole-work or timing
qualification. The unchanged worker gates and actual complete population must
decide that. Formal 1870/2413 and the separate 61/400 ledger gain remain zero.

## Status

Paper derivations reviewed independently; this text describes their intended
implementation. At file creation the candidate has not been imported, parsed,
collected or executed. Only the parent-controlled frozen unique run may change
that status. No novelty or model/source/physical/GPU qualification is implied.
