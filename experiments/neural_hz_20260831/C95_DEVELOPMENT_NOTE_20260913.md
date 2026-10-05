# C95 pre-freeze fixture correction

First focused execution:38 passed,1 failed in1.25s. The new test
test_normal_source_requires_exact_transformed_significand incorrectly expected
the all-one kernel with w[0]=float32(1+2^-23),w[4]=float32(2^-31) to be
nonrepresentable. Both original C85 and new C95 correctly returned exact words:
the smallest term still has a power-of-two mantissa, so alignment shift alone
does not imply excessive significant bits.

Correct the fixture, not the guard: w[4]=float32((1+2^-23)*2^-31) has an odd
significand at the smaller scale. Retain the requirement that old and new both
reject with the identical complete failing-coefficient count. No transform,
threshold, domain, tariff or target selection change. This was a focused
pre-freeze test, not a registered target attempt. No actual target started.
