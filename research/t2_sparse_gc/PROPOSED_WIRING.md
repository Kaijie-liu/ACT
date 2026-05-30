# T2 sparse-Gc wiring proposal

**Status**: prototype validated, awaiting end-to-end RSS comparison
before patch application.

## Files to change

1. `act/back_end/hybridz_tf/representations.py` — add helpers + knobs
2. `act/back_end/hybridz_tf/hz_routing.py` — call helpers at HZono exit
   of `hz_conv2d` and `hz_dense` (only when env knob enabled)
3. `tests/test_hz_representations.py` — soundness regression for new
   operators (5 tests; already passing in research/test_soundness.py)

## Soundness invariant

For any HZ `Z` and any transformed `Z' = prune(Z, eps)` or
`Z' = dense_to_sparse(Z)`:

> `Image(Z') ⊇ Image(Z)`

Proven by row-slack construction (see prototype.py docstring) and
empirically validated:

- 5 random-seed soundness tests pass on small dim
- Full cifar100 first-conv synthetic: looseness = 0.0000 (i.e. bounds
  preserved EXACTLY because dense Gc had ~0% pruning candidates and
  sparse conversion is exact when zero_eps = 0)
- 98.3% storage saving on cifar100 first-conv (1536 MiB → 26 MiB)

## Env knobs (default OFF for safe rollout)

| Name | Default | Effect |
|---|---|---|
| `ACT_HZ_PRUNE_GC` | `0` | Enable threshold-prune of dense Gc cols |
| `ACT_HZ_PRUNE_GC_THRESH` | `1e-9` | Drop cols with `\|\|col\|\|_∞ ≤ threshold` |
| `ACT_HZ_DENSE_TO_SPARSE` | `0` | Convert low-density HZono to SparseGcZ |
| `ACT_HZ_SPARSE_GC_DENSITY` | `0.05` | Density threshold for conversion |
| `ACT_HZ_PRUNE_GC_INSTRUMENT` | `0` | Print per-conv ng/density/RSS |

## Wiring location

```python
# hz_routing.py:hz_conv2d, AFTER current HZono dense path returns out
# (line ~353 in current source). Add:

from act.back_end.hybridz_tf.representations import (
    _act_prune_gc_enabled, _act_prune_gc_threshold,
    _act_dense_to_sparse_enabled, _act_dense_to_sparse_density,
    _act_hz_prune_gc_dense, _act_hz_dense_to_sparse,
)

# After dense conv exit:
if _act_prune_gc_enabled() and isinstance(out, HZono):
    out = _act_hz_prune_gc_dense(out, _act_prune_gc_threshold())
if _act_dense_to_sparse_enabled() and isinstance(out, HZono):
    out = _act_hz_dense_to_sparse(out, _act_dense_to_sparse_density())
return _propagate_base_any(hz, out)
```

Same pattern in `hz_dense` for the Linear layer exit.

## Rollback path

Both env knobs default `0`, so the patch is a no-op until the user
sets them. If a regression surfaces:

```bash
unset ACT_HZ_PRUNE_GC ACT_HZ_DENSE_TO_SPARSE
```

reverts behaviour exactly to pre-patch.

## Test plan

1. `pytest tests/test_hz_representations.py` — soundness
2. Run `regression_pack.sh` with env knobs OFF — must match canonical
   (8 PASS, 0 FAIL) since patch is no-op by default
3. Run `regression_pack.sh` with `ACT_HZ_PRUNE_GC=1
   ACT_HZ_DENSE_TO_SPARSE=1` — must still PASS (sound, ≤ same number
   of CERTs, possibly some FAL/UNK reclassifications if widening
   matters)
4. Single-instance cifar100 iid 0 under 24 GiB cap — peak RSS should
   drop substantially vs baseline

Only after #1-#4 pass: propose full benchmark sweep.
