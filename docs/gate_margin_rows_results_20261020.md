# Gate margin native representation control results

2026-10-20. **Overall FAIL: online reception did not run.**
The worker completed the fixed constructions and faults, and its saved matrices
support a conditional representation separation. These are two different
conclusions; offline checks do not recover execution acceptance.

Execution commit `9d28bbc8eb0169210cfbfaaeae207e45ec301dae`,
freeze `0deb8518b3344dfd978d4b9e075a53ef71d5fab3d50bbe7292b8e6714d5abf69`.
The once-only directory is
`/data1/Kane/MOE/results/gate_margin_rows_20261020_r1`.
The [protocol](gate_margin_rows_control_20261020.md), [lossless saved archive](gate_margin_rows_records_20261020.json)
and [offline recheck](gate_margin_rows_saved_audit_20261020.json) are retained.
No trained model, data, support query, optimizer or GPU was executed.

## Scientific result on the actual stored matrices

The off/on inputs, property, slices and supplied ranges were held fixed.
Actual SparseHZ/F0 matrices, not just ideal formulas, were serialized.

| Literal case | Old relaxed feasible point value | New checked lower bound |
| --- | ---: | ---: |
| P0 positive first half | −1/16 | 1/16 |
| P1 positive second half | −1/16 | 1/16 |
| N negative mirror | −3/16 | 1/16 |

For each enhanced matrix, two fixed negative inequality multipliers produce
the bound with exactly zero residual. The old points satisfy every old
constraint and violate the added relation. They are relaxation witnesses,
not UNSAFE witnesses of the true nonlinear model.

This demonstrates that the added same-factor gate–margin rows can exclude
points admitted by the existing independent-gate F0 representation, at fixed
slice count and fixed ranges. It is a given-HZ/LP mechanism result, not a new
full MoE certificate, complete-source proof or novelty claim over generic
nonlinear envelopes.

The stronger fixed old-method reference also succeeds: all four quarter-slice
F0 matrices have checked minimum `1/4096`, established by exact dual bounds
and attaining points. Thus this example does **not** establish domination of
N1 with more slices. The supplied dyadic ranges and separately proved
tight-sigmoid analytical comparison must not be conflated with a measured
tight-range solver comparison.

Z/E retain their output bounds `1/2` and `1/8`. Q retains the shared/private
continuous and binary mapping, including private rows preceding slice rows;
it is not a positivity claim. X returns the unchanged F0 for unsupported sign.
R returns the unchanged F0 for an inward native center/radius embedding.
Six enhanced constructions add twelve rows; two fall back.

## Worker and outer execution have different outcomes

The worker recorded twenty attempted and twenty returned native constructions,
twelve saved case records, and all twenty-eight specified fault refusals.
Expected error messages, case identities and the complete Work chain match.
Its saved status is PASS.

The public supervisor recorded `ERROR`, with
`execution_error.message = "live process RSS unavailable"`.
The sampled process identity remained present, but RSS was unavailable.
This is the observed failure; the record does not establish a unique OS-level
cause. Return code 0 and `LEADER_REAPED_NO_LIVE_SESSION` cleanup do not override it.
The controller consequently recorded FAIL and never entered online reception.

There is no online reception ledger, final-FD check or successful final acceptance.
No global no-zombie, escaped-session, continuous RSS or complete cold-start
claim is made. No observation policy was relaxed and no retry was performed.

## Recorded costs

| Quantity | Recorded value |
| --- | ---: |
| Worker cumulative Work | 92,845 |
| Worker time before export | 4.845891 s |
| Controller time before terminal export | 5.943015 s |
| Observed through terminal-file export | 5.944693 s |
| Sampled worker RSS | 610,148,352 B |
| Maximum sampled session plus controller RSS | 742,191,104 B |
| Runtime files | 55 |
| Runtime bytes including terminal | 2,681,290 B |

The eight paired off constructions total about 30.24 ms; their on
constructions/checks total about 208.72 ms. Every on construction was slower
in this single run. Binding preparation, saved-record checks and serialization
are separately included in the worker time; preparation was shared once per
paired control. These figures are not independently timed full requests,
proof of stable overhead, or a speedup result. No solver savings were measured.

The 300-second/8M Work/128 MiB artifacts/8 GiB AS/2 GiB sampled RSS policy
was unchanged. Sampled values cannot prove the maximum during unobserved
intervals, and the missing observation is precisely why acceptance failed.

## Recheck and archive

The compact archive preserves canonical original worker/terminal bytes and
contains the identity/size inventory for all 55 runtime files. All original
prefixes remain in the runtime directory. No frozen source or failed output
was overwritten.

The stdlib command:

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -I -S -B \
  scripts/audit_gate_margin_rows_saved.py \
  docs/gate_margin_rows_records_20261020.json \
  210b6ac86080c9afc67637ca3036b9f80d6a3860ce0013cbf8327c86484c00cf
```

returned `OFFLINE_SAVED_CHECKS_PASS`: twelve records, 23,790 audit Work,
about 0.1101 s before stdout. This used a separately labelled offline ledger,
not the expired online deadline. Fault outcomes were audited from bound records,
not re-executed.

A separate read-only AI mathematical review used `Fraction` directly on the
stored matrices without importing ACT, the producer, checker, solver or model.
It verified the P/N duals and attaining points, all four reference minima, and
the Q objective, equalities, McCormick and added rows. Another read-only AI
review checked the freeze, terminal, hashes, 41 prefixes and costs.
These are not independent human review or full-network reproof.

## Continue and stop decision

Retain the concrete representation-separation evidence, its stronger-baseline
limitation and its measured construction overhead. G1–G6 remain OPEN.
The once-only control attempt is consumed; its overall acceptance gate failed.
Do not start trained-model development, M2, GPU or an automatic M3 retry.
Do not repair the observer within this result stage or alter old FAIL records.

Any subsequent native execution needs a separately scoped decision addressing
the repeated observation/reception failure. Mathematical/static research
remains possible, but more wrappers or repeated small constructions cannot
substitute for complete-request evidence or establish innovation.
