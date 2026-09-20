# Auxiliary feasible-point diagnostics: closed, not a MoE result

These studies investigate why existing checked LP lower bounds fail to close.
They do not change the original verification verdicts, sample sets, budgets or
positive-margin acceptance rules. They are supporting diagnostics, not a new
exact linear-algebra contribution of the MoE method.

For a minimization relaxation, a valid checked lower bound L > 0 suffices to
prove that obligation, conditional on its lowering. A checked feasible point
with objective U <= 0 would instead show that this relaxation cannot have a
strictly positive minimum. It would **not** be a counterexample to the original
MoE: the outer relaxation may contain unrealizable factor/gate assignments.
L <= 0 alone proves neither that obstruction nor unsafety. Without accepted U,
weak lower-bound proposals and relaxation imprecision remain unseparated.

The final finite mature-solver comparison used the original four LPs, one
SoPlex call each and unchanged exact-input/bit/budget contracts. All four input
readbacks passed and all native calls returned reported optimality. All four
candidate files failed the frozen 4096-bit rational-token admission; **zero
original-LP feasible-point checks were entered**. A saved-output lexical scan
found 92, 157, 641 and 362 oversized serialized coordinates, respectively. It did
not parse/reduce the large fractions or prove that every possible feasible
witness needs that many bits. No native objective was promoted to a checked U.

Complete diagnostic request costs were 10.781, 10.481, 44.813 and 17.812 seconds.
They start at supplied LPs, not at the MoE input; they are not end-to-end
verification costs. Input/terminal/cost audit PASS does not certify feasible
points. Limits, prior attempts and missing evidence remain archived in
[the finite result](../../docs/soplex_finite_real_v2_results.md), its summary
and full review. No additional arithmetic search is required to state the
paper's current result: this diagnostic did not resolve the LP obstruction.

Further work requires a separate decision tied to complete output obligations.
Neither a larger bit cap nor a faster internal elimination counter alone
establishes a new complete MoE proof. The main paper instead reports the
confirmed first-family benefit, direct relation ablation, cheaper external
path, convolutional transfer failures and explicit conditional proof contract.
