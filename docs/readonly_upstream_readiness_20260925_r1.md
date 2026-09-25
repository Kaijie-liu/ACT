# Full upstream integration: controlled and frozen, execution not admitted

## Completed

- Implementation:`f916ac82f`. Fresh source generation, input/router and expert
  propagation, join/guard/projection/weighted LP construction, unchanged R1
  serialization, all source checks, bounded receiver and terminal accounting
  are connected under the same300s request clock.
- Controls:`readonly_upstream_controls_20260925_r1.json`:264PASS
  (24new integration+240prior),101.810256s,619source bindings intact.
  Tiny controls cover bytewise old/new construction differential, both modes,
  multiple pairs/ties/dimensions, propagation/construction/publication/check
  cutoff, partial evidence, exceptions, poisoned bindings, late terminal,
  complete cost and relocated solver-free original-checker replay.
- Frozen implementation launch protocol:`4b369cae5`, config SHA256
  `325f544fb9cfe3212c4c014d4dcc02a230cf09697df5d6ea8fa229037658d169`.
  Exactly4 calls:two previously fixed synthetic recipes once per direct and
  readonly arm, opposite order. No new sizes/repeats or microbenchmark expansion.

## Not completed: resource admission, not a scientific outcome

The formal4-call destination does not exist; **0/4 calls launched**. There is
no measured full-upstream comparison table, no terminal failure from a launched
request and no new output certificate. Do not report this as4timeouts,0% SAFE,
or as successful completion of the comparison. The successful tiny controls
are not the frozen experiment.

First full-regression admission was refused at approximately6.2GiB available
memory, before creating its control directory. It later passed the unchanged
gate at approximately110GiB and the264controls completed. After the experiment
freeze was committed/pushed, execution admission was refused because load1
was approximately26 on20logical CPUs (registered limit10). Subsequent read-only
checks observed approximately33,then28–29; the experiment directory still did
not exist. Memory was then ample. No threshold was changed, no external job
was interrupted and no background watcher or experiment was left running.

These are prelaunch resource observations, not rejected scientific outcomes
to replace or concatenate. The frozen run remains available unchanged.

## Exact resumption

No further method design or freeze is needed. On a clean synchronized research
branch, confirm resource admission through the existing runner; do not bypass
it. Follow `readonly_upstream_freeze_20260925_r1.md`:

```sh
PYTHONDONTWRITEBYTECODE=1 /data1/Kane/miniconda3/envs/act-py312/bin/python -m readonly_upstream.run verify
PYTHONDONTWRITEBYTECODE=1 /data1/Kane/miniconda3/envs/act-py312/bin/python -m readonly_upstream.run execute
```

Only after execution creates complete batch terminal records, run the frozen
saved-only audit and fresh replay commands from that document. Archive every
result and all phase/whole costs. Do not rerun slow/error calls, enlarge budgets
or pool earlier parser measurements. No saved source may bypass upstream work.

Even successful comparison calls check source/LP CONSTRUCTIONS only. Native
output lower bounds and full MoE SAFE are not produced by this integration.
Defaults remain OFF, sealed real requests stay sealed, historical23 source
gaps and ISSTA readiness claims are not upgraded.
