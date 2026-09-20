# Modular supervision V1 — control and independent review results

Completed 2026-09-20 from clean branch `feat/moe-route-verification`, starting
HEAD `83a9bbdb91de90b74f12e465bbb9687ac45a60d1`. This is a **separate
supplied-LP supervision interface**, not a changed production MoE verifier.
Frozen modular arithmetic, native fidelity import/options and standalone
original-LP checker remain unchanged. No real LP reconstruction occurred.

## Passing evidence

`docs/modular_supervised_controls_attempt001.json`: **107/107 PASS**, no skipped
tests: 36 supervision/integration controls +7 native-fidelity regressions +64
modular/original-LP arithmetic regressions. Raw controls remain outside Git with
the receipt's hash inventory. Failed/censored synthetic executions are retained.

`docs/modular_supervised_v1_review.json`: fresh process **PASS, zero issues**,
2,624 artifacts, 27 terminal records and reconstructed complete/partial costs.
Six portable original-LP packages checked again with `python -I -S`: five
exact-feasible points and one expected equality-violation rejection. No new
native solve or basis reconstruction in this fresh review. A completed checker
does not imply its point was feasible; a feasible nonpositive LP objective
does not imply a concrete network counterexample.

## What was checked

- One absolute original start, cutoffs218/298/300; load/capture/map/construct,
  package/check, terminal and late-publication behavior. No clock reset.
- Owned process kills at native, prime schedule, finite field, CRT,
  reconstruction, exact residual, result serialization and checker boundaries.
  Partial native readback and malformed final writes remain visible; no proof
  package or checker success is fabricated from partial evidence.
- Arithmetic error/LIMIT and modular-exhaustion terminals, no retries. ERROR
  stops a roster; later unstarted rows keep null costs. TIMEOUT/LIMIT remain
  unresolved. Zero native count is not invented where a return was unobserved.
- Cyclic arithmetic phase ordering, immutable contiguous segment prefixes,
  bounded snapshots, wrong request/hint bindings, missing entries, corrupted
  clocks/round metadata and invalid checker flags.
- Instrumented vs frozen uninstrumented constructor: bundle, system hash,
  row residuals, status, statistics, operations and bit metadata agree exactly
  on the standard control and a three-prime full-native control. The latter
  reconstructs1073741790, rejects premature aliases, checks U=-1073741790, and
  records three prime-schedule/CRT windows within the original clock.
- A synthetic128-cycle journal test exercises >256 saved phase events below
  the new bounded2048-event cap. This is a journal test, not128 new LP solves.
  Arithmetic still permits at most128 primes, with all old operation/bit caps.
- A4096-variable/8192-equality synthetic native pipeline preserves every
  original row and independently checks U=-4096/3. This is not real-LP efficacy.

The journal validates metadata and clocks, not a mathematical elimination
transcript. Event serialization, source hashing, constructor output writing and
all other overhead stay inside the appropriate enclosing clock. It cannot
observe the final instruction of a killed process. Its observed prefix is
explicitly `PARTIAL_DIAGNOSTIC_NOT_PROOF`, not a resumable search or proof.

## Cost interpretation

The107 tests took46.556s; receipt clock49.137s includes subsequent terminal/cost
reconstruction and sealed-source checks, but excludes its initial pre-timer
old-freeze verification. Fresh review4.102s is separate post-control work.

| Synthetic complete path | Whole supplied-LP publication clock | Nested constructor |
| --- | ---: | ---: |
| standard analytic LP |0.471695s|0.005202s|
| three-prime reconstruction |0.472643s|0.012436s|
|4096-variable/8192-equality LP|2.897338s|0.175710s|

These are descriptive shared-machine controls, not matched speedups. Fault
controls that advance the original start report large simulated elapsed clocks;
those values are not real wall time spent solving. Whole request = disjoint
phase windows + residual. Constructor subphases, journal and serialization are
nested inside construct, not extra terms to add again. Missing stages/partial
writes use null and observed windows, never guessed zero durations.

Historical network propagation, range proofs and F0 lowering are deliberately
excluded: this interface begins with a supplied LP. No end-to-end MoE latency
or new SAFE/UNSAFE claim follows from these measurements.

## Next gate

Integration controls are complete. A separate `modular_diagnostic/` protocol
must bind the same four original LPs, ordered terminals, source/runtime/semantic
identity and fresh results directory. Its batch controls and read-only selection
review precede a frozen launch. This document is not authorization to execute
the four real obligations, enlarge caps, change bases or introduce a portfolio.
