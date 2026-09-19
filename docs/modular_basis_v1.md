# Bounded multimodular basis candidates — V1 control protocol

Scope: separate arithmetic research after the four frozen primitive diagnostics.
No real LP reconstruction, new native query, increased 4096-bit cap, production
hook, or original-checker change is authorized in this stage. Analytic controls
test whether modular arithmetic avoids *some* large cross products; they cannot
establish efficacy on the four real LPs.

## Fixed method and resource contract

Keep original-coordinate basis assembly and all original rows/residuals. Reduce
rational coefficients individually modulo successive primes below 2^30; no row
denominator LCM. Skip a prime dividing a denominator or singularizing the chosen
basis; neither event is a rational infeasibility verdict. Sparse field elimination
uses the declared incidence pivot heuristic, independently for each field.
Each field product is below 2^60. CRT stores bounded residues; symmetric rational
reconstruction proposes values which must satisfy the original assembled rational
equations exactly. A modular solution or reconstruction alone is not accepted.
The unchanged standalone original-LP checker subsequently checks every original
constraint, bounds, objective and statement identity.

One fixed basis, at most 128 primes, enumerated deterministically descending from
2^30-1 (at most 4096 odd candidates, primality trial division charged). All rounds
share one <=300-second deadline and 20,000,000 operation counter. Keep original
16,384 dimension, 1,000,000 input/fill, 2,000,000 live coefficient and 200,000 heap
caps. Count retained exact source plus current modular rows/pivots as live; no
cross-round retained matrices. CRT, reconstruction and exact rational values keep
the 4096-bit cap. Whole-prime skips and rejected early reconstructions are logged.
No fallback basis or arithmetic retry after any limit. Exhausted reconstruction
is UNKNOWN, not an infeasible LP or unsafe model.

This bounds explicit arithmetic values, not every temporary in Python Fraction,
integer division, gcd or interpreter memory. Exact assembly/residual/objective
arithmetic can still fail. At most ~3840 modulus bits and symmetric reconstruction
are deliberately incomplete even for solutions whose numerator and denominator
separately fit 4096 bits. Candidate-generation work is not independent checking.

## Controls before any real diagnostic

- Huge cross-product / small exact solution, compared with frozen primitive code;
  original LP checked independently. Also denominator-LCM adverse control.
- Fractional random differential, sparse synthetic scale, empty/redundant basis.
- Bad denominator primes, singular primes, genuinely singular systems, premature
  modular aliases and exhausted schedule; none may certify infeasibility.
- Intrinsically large answer, deadline, operations, sparsity/fill and bit caps.
- Identity mismatch, missing rows, nonzero equality residual and invalid original
  candidate: standalone original-LP checker remains decisive.
- Relocate checks under python -I -S; re-read saved sources and original equations
  in a fresh no-elimination review. Preserve all failed control attempts.

## Attribution

Multimodular exact solving and rational reconstruction are established techniques,
not a new MoE theorem. [FLINT rational matrix documentation](https://flintlib.org/doc/fmpq_mat.html)
documents multimodular/Dixon solvers with reconstruction; our bounded Python
prototype is neither FLINT nor a claim of equal completeness/performance. Unlike
the documented denominator-clearing pipeline, this prototype maps each rational
coefficient directly into a finite field.
