# C52 math-worker v2: correct a measurement-log key collision

The first frozen math worker is CLOSED with exit1. Its16 focused tests passed
and all three complete source/Fraction/physical/toy-point checks ran and were
saved, including the138785-byte source/state pickle. At the measurement callback
it raised TypeError: dict() got multiple values for keyword argument elapsed_s.
The callback helper and the measurement dictionary both provided elapsed_s.
Consequently no complete measurement record or transient-gate success is claimed
for v1. Its output, source and failure records are immutable.

This NEW worker changes only the logging prefix to worker_elapsed_s, preserving
the measurement's own elapsed_s and all other fields. It fixes an observed
reporting error; it is not a timing retry, arithmetic change or special-case
test campaign. The signed-congruence writer, independent audit, source fixtures,
exact16 tests, three128-copy cohorts, owner boundaries, tariffs, serializer and
all60s/240s/1GiB/16GiB/64M/256M/200M limits stay unchanged. No larger cap or
test omission. The supervisor inherits and checks the complete v1 source freeze
before adding the new worker/supervisor and this correction note.

Use a new exclusive directory results/c52_signed_first_write_math_20260911_v2.
Run the normal focused16-case pytest once, then the same complete synthetic
transaction once with the corrected measurement callback. Save success/failure
and hashes automatically. Any new failure closes v2; never overwrite v1.

Still math/prototype stage only: no actual Tiny/CIFAR/source census, complete
old qualification suite, original network binding, native solver, benchmark
witness, family replay or promotion. Formal1870 and E0 CIFAR25/Tiny36 unchanged.
The user-directed research focus remains common Neural-HZ structure; the one
logging correction must not grow into another test-framework project.
