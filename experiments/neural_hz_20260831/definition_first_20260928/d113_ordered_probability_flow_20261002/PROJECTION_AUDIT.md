# Probability flow substitution and the innovation boundary

This paper audit explains the representational content of the ordered-flow component after its frozen execution. It is not an implementation request to eliminate continuous factors and does not weaken the original nonconvex HZ requirements. An independent review checked the substitution argument. It is a mathematical argument, not a machine-checked theorem or an additional executed benchmark.

## Exact projection with the same relation information

Fix one source-wide certified permutation pi and the exact rational probability and product bounds used during compilation. Let x denote all original coordinates, including original signed binary phases, and T the newly allocated product coordinates. The N-1 conservation equalities uniquely determine F_k=A_k(x)=sum_{r<=k}(q_pi(r)-p_pi(r)); F_0=0 and the two simplex equalities give F_N=0.

Let R(x,F,T) contain every remaining appended condition: flow bounds, operand bounds, McCormick rows, probability bounds and output equalities. Then the following two predicates have exactly the same projection onto (x,T):

    exists F: P_H(x) and F=A(x) and R(x,F,T)
    P_H(x) and R(x,A(x),T)

The forward direction substitutes the unique conserved flow. The reverse direction chooses that same A(x) as its flow witness. Every original coordinate, phase label and predicate is unchanged. This proof does not depend on whether x is continuous or mixed discrete/continuous.

The expanded reference is the known prefix relation together with products of the same aggregate prefixes and adjacent value differences, using the identical precompiled bounds. Merely adding prefix inequalities to old independent per-token products is not this reference. Recomputing looser intervals after substitution would also change the comparison and invalidate the claimed equivalence.

Point products cause no exception: retain the compiler's exact affine substitution and its operand bounds. Ties cause no exception because the certified order is fixed before substitution. Multiple channels use the same A_k(x), so the proof covers shared flows across all those channels when every occurrence is replaced consistently. No original binary factor is pivoted, deleted, relaxed or enumerated.

## What can and cannot follow

The flow coordinates provide a sparse extended formulation: adjacent conservation has linear token support, whereas writing every prefix into every row can increase support quadratically. This can be a useful implementation property, but it adds no precision over the matched aggregate-product reference. There is no measured GPU or full-query cost advantage in the present prototype; its small strongest-reference fixture is larger than the equally precise grouped formulation.

This is not a claim that any representation compilable to HZ lacks innovation. Most useful abstractions admit equivalent encodings. A genuine contribution could still be a new reusable structural abstraction, a provably advantageous composition or a full-cost improvement over a stated practical reference. The present theorem only prevents attributing known prefix/aggregate-product precision to the addition of flow variables itself. It does not prove that all future cross-query domains are redundant.

The current candidate's mathematical premise is also selective: whole-source score-change order must be certified. The actual ViT graph audit did not establish that premise on the target models. Increasing the number of synthetic tokens or channels cannot by itself establish native applicability or a definition breakthrough. A new hypothesis should be evaluated against these conclusions before more code or qualification infrastructure is added.
