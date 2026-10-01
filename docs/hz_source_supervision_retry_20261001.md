# One unchanged source supervision follow up

The R3 batch stopped after 11 of 17 registered calls, before any test methods
ran. Its serialization-delay call never reached the injected fault: the
numerical-import operation was still open at the phase deadline, and leader
reaping was not confirmed within the original cleanup allowance. The other
six slots remain unstarted. The [failure record](hz_source_supervision_20261001_r3.json)
retains the terminal and summary identities. This is not a passed fault control.

A subsequent read-only process observation found no row for the recorded leader
or its original process group, PID/PGID 380848. This permits further owned work;
it does not retrospectively confirm cleanup within the R3 deadline. No process
was interrupted by that observation, and no cause of the import delay is asserted.

Allow exactly one new, complete R4 control batch in
`baseline_runs/hz_source_supervision_20261001_r4`. It uses the same 17 calls,
13 tests, source declarations, implementation bytes, 128 proposal iterations,
thresholds, resource limits and individual budgets as R3. No warmup, rerun of
only the missing slots, budget extension or result splicing is allowed. If R4
again misses an intended fault or cannot confirm cleanup, preserve its failure
and stop this execution-validation attempt rather than loop until PASS.

R1 and R2 remain the pre-hardening batches, not final acceptance of the current
implementation. This follow-up is finite synthetic CPU control only; it admits
no real/full-size source, native solve, physical GPU or sealed input.
