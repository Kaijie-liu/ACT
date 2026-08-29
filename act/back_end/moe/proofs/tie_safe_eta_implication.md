# Tie-safe eta implication proposition

For hard top-1 branch `i`, define

\[
g_i(x)=\max_{j\ne i}(r_j(x)-r_i(x)),
\qquad
s_i(x)=\min_k(C_kE_i(x)+d_k).
\]

Under `ANY_LEGAL_TOPK`, branch `i` is legal exactly when `g_i(x) <= 0`;
equality is a legal tie. The zero-margin reduction
`max(g_i,s_i) >= 0` is not sound: `g_i=0` and `s_i<0` satisfies the reduced
property while leaving a legal unsafe branch unchecked.

## Proposition

Fix `eta > 0`. If

\[
\max(g_i(x)-\eta,s_i(x))\ge0
\]

for every input in the original box, then the expert safety property holds
whenever branch `i` is a legal hard-top1 route.

### Proof

At every legal branch point, `g_i(x) <= 0`, hence
`g_i(x)-eta <= -eta < 0`. The maximum of this strictly negative value and
`s_i(x)` can be nonnegative only if `s_i(x) >= 0`. Thus every legal execution
of branch `i` satisfies every safety row. This includes `g_i(x)=0`, so all
tie-legal routes remain obligations. □

## Exact incompleteness domain

The exact implication imposes no condition on `s_i` when `g_i > 0`. The eta
reduction is also automatically satisfied when `g_i >= eta`. Therefore the
additional obligations introduced only by eta are exactly

\[
0 < g_i(x) < \eta.
\]

The endpoint `g_i=0` is not incompleteness: it is a legal tie and must be
checked. ACT reports the strict mathematical band separately from the frozen
numerical boundary-tolerance band. A negative relaxation candidate remains
`UNKNOWN`; it is never a full-model counterexample.
