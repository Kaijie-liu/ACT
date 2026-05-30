# NeuralSAT STRICT — no source patches required

Unlike CORA TRUESTRICT and NNV STRICT, NeuralSAT's "no helper" mode is enabled by a **single
upstream CLI flag**: `--disable_attack`. No source modification was needed for the strict
helper-free run. This directory exists for symmetry with the other archives and to document
what the flag actually does.

## The single switch: `--disable_attack`

Wiring (audited 2026-05-27):

1. `src/main.py:44-45` — argparse declares
   ```python
   parser.add_argument('--disable_attack', action='store_false', help="disable attack.")
   ```
   `store_false` means the underlying namespace attribute `args.disable_attack` defaults to
   `True` and becomes `False` when the flag is passed.

2. `src/setting.py:60-61` — `Settings.setup(args)` does
   ```python
   if hasattr(args, 'disable_attack'):
       self.use_attack = args.disable_attack
   ```
   So passing `--disable_attack` sets `Settings.use_attack = False`.

3. `src/verifier/verifier.py` gates three call-sites on `Settings.use_attack`:
   - line 88:  `is_attacked, self.adv = self._pre_attack(...)` (PGD pre-attack)
   - line 102: `is_attacked, self.adv = self._mip_attack(...)` (MIP-based attack)
   - line 433: `self.adv = self._attack(pick_ret, ...)` (in-BaB attack)

   All three are skipped when `Settings.use_attack == False`.

A reaudit (`grep -n 'use_attack' src/`) confirms there are no other code paths that perform
PGD / random sampling / falsification. NeuralSAT in `--disable_attack` mode produces SAT
verdicts **only** through the completeness of bound-propagation + branch-and-bound, which
witnesses a concrete counterexample at a leaf when the BaB tree is fully refined. This is a
sound result, not a falsification helper.

## Driver-level patch (not NeuralSAT source)

One bug fix was applied to our own bash driver `run_neuralsat_strict_parallel.sh` mid-run:

**Symptom.** Lanes were silently exiting after processing a few benchmarks (mysteriously
disappearing from the dual-lane orchestration with no error log).

**Root cause.** The function `run_benchmark` re-enabled `set -e` after each per-instance
`timeout` call. Because the script's top-level `set -uo pipefail` does **not** include
`-e`, this turned on `-e` for the subsequent benchmark setup. The next call to
`gpu_wait()` would then execute the line:
```bash
[[ -z "$free" ]] && free=0
```
which returns exit code 1 whenever `$free` is non-empty (the common case), causing the
lane subshell to abort.

**Fix.** Remove the spurious `set +e` / `set -e` toggle around the inner `timeout`
invocation; let the per-instance return code propagate as a normal exit code without
triggering `errexit`. After this fix, both lanes ran to completion without further early
exits.

The fix is in `scripts/run_neuralsat_strict_parallel.sh` in this archive — it is the
post-fix version that produced the final clean dual-lane sweep.
