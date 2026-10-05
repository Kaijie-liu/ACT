# C114 exact sink projection and commuting-footprint lemmas

For normalized continuous parents z and continuous sinks m_j, let each unique
definition be p_j*m_j+a_j*z=b_j with p_j a positive power of two and
abs(b_j)+sum(abs(a_j))<=p_j. The triangular definitions supply a unique extension
m_j=(b_j-a_j*z)/p_j in[-1,1]. Thus eliminating these factors from every consumer
and retaining all other constraints gives exactly the same projection onto
the remaining coordinates. The extension formula supplies the inverse. The
claim is conditional on complete consumer incidence; omitting one predicate
would invalidate it. Binary factors are neither pivots nor deletion targets.

For independent sinks S and a complete consumer c*m+d*z=e, substitution gives
(d-sum_j c_j*a_j/p_j)*z=e-sum_j c_j*b_j/p_j. C114 computes the actual rational
coefficients and RHS, bounds rational width at512bits, and independently reads
back the exact native binary64 row after its positive power-of-two gauge. The
inherited complete coefficient window remains[2^-20,2^40]. No epsilon or relaxed
rounding is used. Expected guard failures cannot become a positive result.

In each expanded consumer, a column contributed by exactly one nonzero term
cannot cancel. Counting such columns gives a necessary lower bound on the new
nnz without full rational expansion. If the sum of these lower bounds is at
least the original defining-plus-consumer nnz, strict local nnz reduction is
impossible. Failure of this necessary test is not a statement about different
larger elimination blocks or a different exact source representation.

The primitive's ordinary shared-parent test has two four-parent sinks and two
consumers. Each isolated no-collision elimination costs+1nnz, while joint exact
substitution changes16nnz to6. An independently recursive rational projection
of all original equations equals the projected native equations, including an
offset variant. Disjoint-parent, nonredundant-box, nontriangular, window,
incorrect-alias, complete-consumer and zero-budget cases exercise rejection.
These are reusable algebra checks, NOT a claim that real filters are identical.

## Separate whole-transaction observation

C114 rejects locally neutral groups. A later atomic candidate would be a
different version and must meet the charter's strict TOTAL nnz and whole-state
physical gates. After a fixed exact root quotient, two sink eliminations
commute if their defining rows and consumer rows are disjoint and no selected
sink is a parent of another selected sink. The last condition follows here
from complete output-only sink incidence. Then the union changes each affected
row once, and its nnz delta equals the sum of the individual deltas.

The separate footprint check verified this for ALL128 neutral real-source
groups:256 defining factors and256 distinct original consumer rows. It checked
that no defining factor overlaps a removed root alias. Consequently their
combined delta AFTER the137-root quotient is0, not a positive nnz gain. Adding
the root quotient's already known-411 gives strict-411 for the full hypothetical
atomic transaction. The additional M benefit is256 fewer factors, not additional
nnz reduction. Conditional packet formulas predict-22528B beyond root reuse.

No new packet, source object, owner delta or inverse allocation was emitted by
the footprint check. The archived individual native checks plus disjointness
prove conditional arithmetic, not an implemented or admitted candidate. A
fresh source producer must still prove all original HZ rows, EQ/INEQ/binaries,
owner/UID/frame and concrete-input inverse identities, global storage and work.
