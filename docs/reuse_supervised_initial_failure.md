# Initial integration control harness failure

First `python -m reuse_supervised.controls` invocation on 2026-09-16 failed
because the mechanically derived test module renamed its fixed tail import to
nonexistent `reuse_on_portable.execution`. The correct unchanged tail is
`single_check_portable.execution`. No real verification request was launched.
The control runner subsequently tried to import that same failed module to
read observations, so it also failed before writing its numbered receipt.
This failure is retained here; the runner now collects observations from loaded
modules and can record unittest import failures. No frozen source, math,
budget, selection or acceptance policy was changed for this harness correction.
