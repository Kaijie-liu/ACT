# C81: exact exponent-binned inverse, v1

Default-off isolated continuation of C80 on redu-hz at
f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac. No production edit, promotion,
historical write, compiler change or solver execution is authorized here.
The preceding instruction-maintenance turn was no research progress.

## Repeated blocker and identity

The full authenticated C74 checkpoint now restores its original owners and
storage, but two independent systems of long ordinary affine rows exceed the
256M whole-restore work cap. The original reconstruction consumes surviving
consumer prefixes; the independent check consumes every original producer
coefficient. Neither system, any coefficient, local equation, coordinate or
box check may be removed. The fixed zero qualification input is not feasibility.

Decode each binary64 coefficient exactly as signed integer m and exponent e.
For dyadic x = n * 2**q, accumulate m*n into the integer bin indexed by e+q.
Align all bins once, and construct one Fraction per row. This eliminates
per-term Fraction allocation, gcd normalization and exponent alignment; it does
not lower the tariff of the unchanged C57/C68/C70 algorithms. Every term,
including zero terms, is read. No float multiplication is used. An input with
an odd denominator uses the original exact Fraction row arithmetic instead.

Input bounds price at least a general 53-bit mantissa at exponent -52, even
for zero. Wider or smaller exact dyadics are conservatively bounded, not
rounded. Bounds propagate through unit inverses in order. The full source
producer rows must not reference removed local children: their local equations
are restored subsequently. This is checked on actual row indices, not assumed.
Local equations and their separate independent checks keep the old arithmetic.

## Work accounting, frozen before target observation

Use the same WorkPool, including all unchanged decode/root work and separately
reported source/native authentication. Compile both complete row systems and
derive their full remaining upper bound before executing any inverse dot.
If the bound does not fit the remaining 256M, reject before arithmetic; the
preparation already performed remains charged to that same pool.

- Full-frame Fraction conversion/box checks: unchanged 16 per coordinate.
- Cached numerator/exponent and conservative bound preparation: 8 per coordinate.
- Coefficient bit decode, finite/index guards and two owned arrays: 16 per
  coefficient + 128 per row. Arrays are int64 mantissa/int16 exponent, 10 bytes
  per coefficient; original indices and values are borrowed and never changed.
- Vectorized bound/index analysis: 4 per coefficient + 128 per row.
- Integer-bin core: 8 per coefficient plus twice the sum of input numerator
  64-bit words and running conservative bin 64-bit words. The latter uses the
  maximum input mantissa width up to that term plus coefficient width 53 and
  ceil(log2(row length)); thus it covers every possible bin population.
- Bin merge upper bound: (16 + 4*alignment_words) per possible exponent bin.
  Final Fraction construction: 128 + 64*alignment_words per row.
- Non-dyadic fallback: 64 per coefficient; row preparation is still charged.
- Unit inverse arithmetic: unchanged 128 per unit; independent producer header
  unchanged 64 per row. Point-table updates add 8 per changed coordinate.
- Both full local-tag traversals and both exact local-equation traversals retain
  their original 8/header and 128/local charges. No local check is cached away.
- Removed-local mask: 4 per coordinate; used solely for source-row applicability.

These are algorithmic logical-work units, not instructions/cycles or a runtime
speed assertion. The whole upper bound includes both systems and all new
preparation/storage/update work. Source authentication remains separately
reported exactly as C80; no full-CPU-256M claim is made.

## Qualification and unchanged gates

First compare to an independent plain Fraction oracle on nonzero mixed-sign,
cancellation, repeated-column, shared-parent and multiple-exponent rows, with
ordinary dyadic and odd-denominator inputs. Compare full reconstructed vectors
with C68 on EQ/INEQ, signs, phases and local-child compositions; preserve all
1895 existing tests, exact collection and JUnit equality. Record development
failures honestly before freezing target sources; no old tests are modified.

Only after qualification: one complete C77 checkpoint restore using unchanged
C79 decoder and C78 complete root collector, authenticated C74 source/native
bindings and the original whole input held strongly. CPU1/GPU0, AS16GiB,
60s whole restore, both 1GiB transient tests, 64M entries, original restored
entry envelope and complete 256M pool remain. Fatal-only stack reporting.
No target cap tuning, identity menu, omitted roots or zero-point shortcut.

Exclusive supervisor outputs preserve source/branch/config hashes, exact test
inventory, logs, result and exit even on failure. A successful inverse report
must cover all 199 unit and 100965 local equations and 11708 original coordinates.
It is not a feasible witness or a score gain. After that complete boundary,
return to the new NativeState terminal affine/property binding and exact
witness recovery, then original target/shadow/family/full2413 and separate400.
Formal1870 and E0 CIFAR25/Tiny36 remain unchanged; all charter prohibitions apply.
