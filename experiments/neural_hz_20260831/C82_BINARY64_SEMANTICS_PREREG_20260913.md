# C82 v2: original affine binary64 semantics, not unrounded-real equality

V1's complete1946-test suite passes. Its actual preparation rejects the FIRST
center equation after complete root/source/native authentication,27.229s,
137810649work; both1GiB gates pass. The failure is preserved in
results/c82_native_terminal_20260913_v1 (result SHA237780b7071bfc250ab400508e94dd0b60d3150e4599e3e1e4e9ea98a4c22972).
No solver call, final proof or score gain. Supervisor56125 terminal exit1,
source/provenance unchanged. This is an overly strong NEW checker, not an
observed defect in the old affine implementation or a failed unit inverse.

Local implementation authority: solver_hz.sparse_hz_linear computes CSR W@c,
then adds bias, and computes CSR W@Gc/Gb. These are binary64 operations; its
existing C34 final proof compares all original realized bits, not an unrounded
Fraction sum. V1 wrongly required the latter and additionally started center
accumulation at bias rather than adding bias after the matvec. Ordinary
multi-term rows need not have exactly representable rational sums.

V2 checks the SAME original operation order independently: exact Fraction
multiply -> binary64 round -> exact Fraction add -> binary64 round per ordered
nonzero contribution; bias is added only after the center dot. Generator sums
use each original CSR contribution in order. Compare exact resulting values,
with no epsilon/tolerance and no changed output coefficient. No general
unrounded-real-exactness claim is made. The exact C81 unit/local inverse and
independent original equations remain unchanged. An ordinary non-dyadic-weight
test must expose the old unrounded mismatch and pass the binary64 realization.

This corrects the reference semantics to the already frozen native path; it
does not relax HZ projection exactness, witness validity or a baseline gate.
No new rounded HZ, convex domain, extra solver path or property is introduced.
Increase the NEW independent rounded-arithmetic prepayment from64 to128 per
term/comparison to cover both exact operations and explicit conversions.
The original256M/60s/both1GiB/64M/source/owner/input/phase gates remain.

Retain all1946 v1+prior tests and add24 v2 tests (1970 total/88files). Freeze
v2 worker/supervisor before one actual preparation; v1 files/results immutable.
The only worker changes are the checker import and new exclusive output path.
The successful C81 full-restore prerequisite is still required unchanged.
