# Two separate deliverables — freeze/readiness report

Implementation commit: `cff54e000`. Preparation started at `caed0354a` on a
clean `feat/moe-route-verification` checkout. No dependencies were changed,
other users' processes touched, checkpoints trained, or real requests solved.

## Execution study: ready for separately authorized execution

The [protocol](metamoe_receipt_reserve_protocol_20260923_r1.md),
[55-test gate](metamoe_receipt_reserve_controls_20260923_r1.json), and
[saved-only freeze review](metamoe_receipt_reserve_freeze_review_20260923_r1.json)
bind the new config `configs/backend_controls/metamoe_receipt_reserve_r1.json`.
Four observed prefix inputs, two ACT variants, eight calls, rotating order.
Only expanded/contracted native soft-cap fraction changes (1.0 vs0.8). The
300-second whole-request and30-second expert caps, property order/count,
equal-share deadlines and acceptance policy remain unchanged.

The new instrumentation is charged to both arms. The native worker is explicitly
rebound; old configs/results remain intact and must be read at their frozen
source identities, not silently revalidated as if the live worker were old.
Regression uses the old composition controls, not its historical live-source
freeze test. Tiny native differential and separate original acceptance audit
passed. Fault controls cover timeouts, exceptions, partial/misbound/late results,
fail-stop denominator and complete accounting. No real speed or coverage effect
has yet been measured. A reduced native cap can lose results.

The real output root remains absent. Execution, when authorized, is:

```sh
PYTHONDONTWRITEBYTECODE=1 /data1/Kane/miniconda3/envs/act-py312/bin/python scripts/metamoe_receipt_reserve.py --config configs/backend_controls/metamoe_receipt_reserve_r1.json
```

After execution, use `audit_metamoe_receipt_reserve.py --replay-only` for a
separate full-model witness record (including a zero-witness ledger), then its
`--replay <record>` audit. Keep the entire eight-call denominator and all partial
files, report gained/lost policy positives and solved, query-return/runtime
censoring, worker starts and ALL-request cost. No post-hoc retries, changed
fraction, extended time or automatic next batch. Copy compact final evidence
to a new archive and commit/push only after the independent review is complete.

## Source/output line: protocol only, adapter/control work remains

The separate [scope](source_output_closure_scope_20260923_r1.md) binds
seed0/rank0, index4088 (not input98). Saved checkpoint/input bytes are hashed;
the model was not loaded or propagated. The exact corrected input enclosure,
full router and experts, checked guard construction and all28x9 output
obligations must share one new source/matrix identity. Old positive bounds and
old pair exclusions are prohibited. Universal gate range[0,1], no range tuning
or sample substitution, whole future attempt300seconds.

The standard-library reviewer independently checks all252 planned identities
and the fixed choice. It does **not** claim the generic graph/guard adapter has
been implemented, those obligations solved, or a source-complete certificate
exists. Implement its finite operator/factor controls before considering a
later execution freeze. Stop and report if the bounded attempt does not close.
Graph/program correspondence, rational abstraction and deployed float semantics
remain separate claims; no historical table or guarantee has been upgraded.
