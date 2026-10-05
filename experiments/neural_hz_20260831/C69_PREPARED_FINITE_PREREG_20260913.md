# C69: remove one implied finite-output scan in complete prepared emission

Registered before tests or target. Same redu-hz/f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac,
production15198f4ddc40dfa1c37456737b0f2080ddee2c653e2b9d0010cf245b6c5fec75.
C68 was progress: exact source consumer and actual unchanged-scan rejection.
All original source/history/default/score and goal restrictions remain fixed.

The intended structural-index investigation found a more direct repeated
construction operation before adding an index. C29 _prepare first calls
exponent_data on ALL original continuous/binary operands, proving they are
finite and nonzero and deriving their full shifted native window. It then calls
scaled_exact, retaining forward ldexp, inverse ldexp and exact array equality.
That helper additionally scans np.isfinite(out) on every coefficient.

For finite original x and bounded integral k, if ldexp(ldexp(x,k),-k)==x
elementwise, the intermediate output cannot be NaN or infinity: ldexp of a
nonfinite value remains nonfinite and cannot equal finite x. Forward arithmetic
also retains overflow/invalid exceptions. Thus output finiteness follows from
the retained checks; none of the accepted coefficient rows or rejected inputs
changes. Underflow/reversibility, original finite/nonzero/power/coordinate
validation, normal window proof, original RHS scaled_exact, head metadata,
one-use store, global UID/radix events and final full source proof stay.

Implement a new private coefficient scaler called ONLY inside the new complete
_prepare after exponent_data. No generic scaled_exact replacement, user-supplied
normality flag, old receipt, monkeypatched global helper or proof bypass.
The new encoder and bound normal reader/direct CSR producer types are separate
versions; the entire100965 quotient, products, row gauges and source maps stay.

Payment is exactly ONE absent isfinite vector operation per ORIGINAL logical
coefficient. Use the same source-derived original coefficient inventory as C31;
subtract no RHS, head/control, extra radix coefficient, native consumer or old
index work. Every physical omission is counted at the real one-use store;
independently verify it equals all emitted continuous+binary coefficients and
is at least the original logical inventory. Keep the ORIGINAL conservative
branch encoding price unchanged. Add128 fixed new producer/report binding work.
All original workgroup prices/caps remain. A different implementation removes
an operation; this is not a lower tariff on a retained check. Reject any input
for which the proof's original preparation preconditions are unestablished.

Qualification: compare complete prepared/packed rows, all scalar finite/zero/
power/reversibility failures and ordinary binary/EQ/INEQ/radix cases with C31;
prove the omitted helper performs no output-finite scan while the original
finite-input proof and inverse equality still execute. Fresh chain/shared/
pointwise-Conv outputs/maps/owners must equal C67, except the explicit new report.
Retain all1586 C68 v2 checks and add focused source/full-proof/budget tests.

Then complete source-bound preflight from ORIGINAL C9/C31 inputs; actual fresh
original-expression generation only if fitting. Run the existing full C65
original source/box/UID/owner/local-inverse proof, exact omission-counter proof,
complete same-C31 physical ledger, both1GiB gates and fresh restore. Full
source/hash diagnostics retain all inputs and their existing separate bounded
offline phase; no hidden source/oracle deletion or claim of source-first LIVE.

CPU1/GPU0,AS16GiB,test60s/worker240s/restore60s,whole256M/branch200M,64M entries,
both1GiB transients, radix16384/131072/16M unchanged. Freeze all new source/config/
provenance before execution, exclusive results/c69_prepared_finite_20260913_v1
with automatic logs/result/exit retention. Failed versions remain immutable.

This is construction payment for the SAME exact Neural-HZ structure, not a
score gain. It does not yet admit C68 native lineage or pay an unmeasured full
native transaction. After success derive complete NEW native costs and LIVE
proof; never import the old C34 increment as actual new work. Formal1870/all13,
E0 CIFAR25/Tiny36, concrete witness/shadow/family/full2413/separate400 gates stay.
