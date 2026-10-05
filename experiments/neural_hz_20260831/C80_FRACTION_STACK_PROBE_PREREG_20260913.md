# C80 diagnostic: isolate fatal periodic stack sampling during Fraction work

C79 passes all1895 tests, exact original storage envelope and complete actual
source/native hashes, then exits with signal11 during inverse arithmetic. The
periodic stack log stops while printing fractions._from_coprime_ints at30s.
This is not yet a proved cause and does not qualify complete restoration.

First a bounded standard-library-only probe: same Python3.13, CPU1/GPU0,
tracemalloc active and repeated Fraction arithmetic, with periodic stack dumps
every0.1s for at most2s (parent5s). No ACT/C78/native helper/Torch/NumPy imported.
Its only purpose is to test whether this failure occurs independently of the
new native collector and tensor decoder. Log stdout/stderr and signal exit,
freeze this file/probe plus Python executable and fractions.py. No checkpoint
is loaded, no numerical result is promoted, no original test/gate changes.

If this reproduces, use fatal-only stack reporting for one new complete C79
restore version. Remove only the optional periodic diagnostic thread, retain
every proof/resource/source test and the full1895-test qualification. If it
does not reproduce, do not assume diagnostic sampling was the cause; use fatal
stack reporting and investigate the actual implementation before promotion.
