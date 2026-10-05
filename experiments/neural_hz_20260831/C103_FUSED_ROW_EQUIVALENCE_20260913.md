# Exact implementation argument for C103

This describes the frozen C103 implementation; it changes no gate or tariff.
Its private C entrypoint is reachable from the encoder only AFTER the original
_bounded_powers has validated each original integer dtype/range and produced
bounded signed int64 arrays. It is not an independent public proof issuer.
The original coordinate/order/shape/finite RHS guards remain in encode, and
generic emit still constructs and validates its own complete powers. Source
and circuit code use strict new concrete encoder types, not widened admission.

For each nonzero finite coefficient write abs(v_i)=m_i*2^e_i by frexp, with
1/2<=m_i<1. The original implicit exponent gives E_i=e_i+p_i. The first fused
scan checks ALL original v_i, retains min(E_i), max(E_i), the largest mantissa
among equal maximal exponents, and the original first continuous coefficient.
This is exactly the information computed by C97's full NumPy intermediates.

The lower shift is -19-min(E_i). The upper shift is41-max(E_i) when the top
mantissa equals1/2, otherwise40-max(E_i). A nonempty admissible interval gives
the identical shift clamp(0,lower,upper). Lower proves every magnitude>=2^-20;
upper proves every magnitude<=2^40. A lower-exponent coefficient cannot exceed
the maximal-exponent group's top. An empty interval returns None before any
payload publication, entering precisely the unchanged radix encoder. The empty
coefficient row retains shift0 and the original generic RHS check.

The second fused scan writes ldexp(v_i,p_i+shift), immediately recomputes its
inverse, and requires exact equality to EACH original v_i. No fast-math/FMA,
tolerance, guessed shift, missing coefficient, binary deletion or RHS rewrite
is introduced. The old scaled_exact performs the complete original RHS finite
and inverse checks. Head codes use the identical first-mantissa/exponent/sign.
The new private payload has the unchanged nonportable one-use contract and is
consumed by the unchanged coordinate-copy/store/entry/UID/ownership sequence.

All original logical prices remain. The native program has two linear passes;
the removed complete values/powers/mantissa/exponent intermediates alone total
32N bytes. New final output arrays occupy8N bytes, already required by C97;
they are ordinary owning NumPy arrays, not hidden native storage. Six borrowed
Py_buffer descriptors and scalar loop/window data are fixed-size stack state;
there is no native heap or retained module cache. Strided input is read with
memcpy, so negative strides and alignment do not require extra copies. The
generic Python dtype/power constructors are unchanged in scope. This is a
logical-work/storage argument, not a claim that generation prices count every
CPU instruction or all external payload/hash traffic.

The new full-source fingerprint check includes all canonical HZ matrices,
maps, report values, source/operator sharing, owner words and circuit inverse
records. Equality on fixtures is not actual-network admission. Actual runtime
must independently reconstruct its source and match the complete C100 proof
identity; every original native/input/final proof and complete LIVE check is
still executed. Old schema/report labels denote inherited mathematical
semantics; the actual source and native binary are identified by the new
1769-entry execution manifest, never by an old class name or a Git commit alone.
