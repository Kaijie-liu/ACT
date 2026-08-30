# ICML 2025 RT-ER B1 unattended landing

## Purpose

The B1 training supervisor can finish while no interactive Codex turn is open.
This landing hook converts a successful supervisor exit into a complete,
audited endpoint rather than merely noticing that a process stopped. It never
signals, restarts, or changes the protected training process.

The frozen protocol is
`act/pipeline/moe/configs/icml2025_b1_landing_protocol_r1.json`. The hook polls
only the run-root JSON state. It runs in `act-py312` until endpoint evaluation,
then invokes the already frozen Blackwell reproduction interpreter at
`/data1/Kane/MOE/envs/rt-er-blackwell/bin/python`.

## Epoch-50 rehearsal

When epoch 50 appears exactly once in `progress.json`, the hook validates:

1. the checkpoint, metrics, and telemetry files exist;
2. every SHA-256 equals the immutable supervisor record;
3. the serialized checkpoint reports public epoch 50;
4. the telemetry reports epoch 50 and the same checkpoint identity.

The rehearsal deliberately excludes ordered full-test evaluation, threshold
interpretation, commit, and push. It writes only
`landing/rehearsal_epoch050/B1_LANDING_REHEARSAL.json` under the external run
root. This is an unattended-chain rehearsal, not an experimental endpoint.

## Final landing

The final path starts only when the supervisor status is `PASSED` and its
completed schedule is exactly epochs 10 through 130 in steps of ten. It then:

1. repeats the identity checks for epoch 130;
2. evaluates standard accuracy on all 10,000 ordered CIFAR-10 test inputs;
3. runs the official PGD implementation for 50 steps at 8/255 and stores every
   endpoint, prediction, route, and pixel-space norm;
4. reloads the checkpoint and independently replays all stored endpoints
   through the full routed model;
5. applies the previously frozen inclusive SA and PGD-50 interpretation
   intervals without changing them;
6. writes the raw `B1_LANDED_summary.json` under the run root;
7. only from a clean, feature-branch worktree exactly synchronized to the
   remote, writes the two tracked landing records, commits, and pushes them.

Any failed identity, replay, branch, cleanliness, synchronization, or numerical
check stops the sequence and preserves a failure JSON. Partial landing output
is never committed. Push is retried three times; a persistent remote failure is
recorded and requires manual recovery without force-push.

## Reporting behavior

The desktop task cannot promise an interactive notification after it has been
closed. The hook instead makes completion durable: the raw and tracked landing
summaries contain the checkpoint hash, ordered SA, ordered PGD-50 accuracy,
frozen interpretation branches, replay count, and audit issue count. A
successful unattended landing is committed and pushed to
`feat/moe-route-verification`, so reopening the project reveals a complete
report without rerunning the endpoint.

## Post-B1 resource order

After B1 lands, the GPU starts the isolated AdvMoE environment build and
official seed-0 training. In parallel, CPU-only resources run the paired
exact-big-M/lazy-enumeration/MIP-start timing study and the monolithic MILP
baseline. Worker counts and process priority remain bounded so CPU timing does
not contend with the data loader. B3 CROWN evaluation starts only after both
lanes finish and an appropriately quiet GPU window is available.
