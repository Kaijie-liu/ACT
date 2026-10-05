# Exact integer F4 numerator program

C119's denominator-bearing source-equivalence theorem is unchanged. This note
proves the new coefficient program used to pay its complete ordinary boundary.

A finite normal binary32 weight is an exact signed integer mantissa times2^p,
with absolute mantissa<2^24. Check every original binary64 value, if supplied,
equals its binary32 round-trip before using the decoded words. No tolerance.
Zero coefficients contribute integer0 with no alignment requirement.

For each3x3 kernel choose the minimum nonzero exponent and shift each mantissa
by its nonnegative exponent difference. With differences<=33, |aligned|<2^57.
Apply the six exact integer forms

`a, -a-b-c, -a+b-c, a+2b+4c, a-2b+4c, c`

first to columns and then rows. Their coefficient matrix is exactly GNUM.
Each form has L1 norm<=7. Every first-stage intermediate is bounded by7*2^57;
every second-stage intermediate by49*2^57<2^63. The concrete evaluation order
uses no larger partial sum. Thus signed-int64 operations cannot overflow.
All36 outputs equal GNUM*aligned*GNUM^T exactly. Multiplication by the original
common2^p scale yields precisely the original dyadic numerators.

Let transformed values be y. Since y0=a, y5=c and y2=-a+b-c, recover b by
y2+y0+y5. Applying this inverse on both axes recovers every original source
word. First-axis inverse intermediate bound35*2^57<2^63; the second operates
on the exact recovered first-stage forms and has the same safe bound. This
full inverse guards decoding/axis errors, but by itself does not prove that
unselected output components are correct.

Accordingly C120 retains the entire unchanged C119 independent oracle: all36
kernel components are independently reconstructed in exact Fraction arithmetic,
the72 basis coefficients are checked, and EVERY actual V/M/output native row,
support, positive pivot, gauge, semantic scale and redundant box is checked.
Ordinary tests also compare complete emitted arrays with the direct C119
Fraction constructor. No inverse checksum or hash replaces exact comparison.

The forward schedule uses99 integer operations per kernel, inverse18. Its
prepaid2768-unit program covers decode/check/align/forward/recovery/conversion
and exact numerator materialization. Original row and independent proof costs
are unchanged. The complete measured batch245413428 fits the256M gate, including
all sources/owners/inverses/evidence and the actual combined held-root ledger.

This coefficient theorem neither proves arbitrary source weights lie in its
domain nor allows reusing a prepared kernel without source identity/custody.
Every actual operator must be freshly checked. It supplies no network verdict,
whole-source runtime payment, benchmark retention replay or score promotion.
