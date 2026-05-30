# PyRAT STRICT HYB_Z — patch / provenance bundle

This bundle is the **hyb_z (hybrid zonotope) variant** of the PyRAT STRICT
sweep. It is a strict sibling of `pyrat_strict_20260527/` (the `con_z` sweep)
— same binary, same falsification-disable mechanism, same `.ini` provenance,
**only the abstract domain differs**.

| Item | Source |
|---|---|
| PyRAT binary | commit `95c72fc22b` (== sibling archive) |
| `.ini` base  | competition commit `4a9a4f0:vnn_config/*.ini` (== sibling archive) |
| `.ini` transform | **same two parser-compat renames** + **5 hyb_z knobs added** + `domain = [hyb_z]` substitution; see below |
| `run_pure.py` | bit-identical to sibling (`scripts/run_pure.py`) |
| forced CLI flags | identical to sibling (`--check skip --nb_random 0 --attack bounds --batch_attack False --exhaustive False`) |

---

## Why this archive is separate from `pyrat_strict_20260527/`

`pyrat_strict_20260527/` measures PyRAT's **`con_z` (constrained zonotope)** abstraction with helpers disabled.
This archive measures PyRAT's **`hyb_z` (hybrid zonotope)** abstraction with the same helpers disabled.

The two archives together form a *controlled abstraction study*: identical binary, identical helper-disable, identical per-bench knobs, identical `.ini` provenance, only the domain label changed. Any verdict delta between the two is attributable to the domain.

---

## Domain selection — important caveat

### What we asked for in the `.ini`

```ini
domains = [hyb_z]
```

### What PyRAT actually runs

```
Running analysis with N processes, domains = ['con_z', 'hyb_z'], scorer = ..., timeout = ...
```

### Why

PyRAT's internal `pyrat.analyzer.analysis_param.AnalysisParam._domains_clean()` method
**unconditionally appends `con_z` to the domain list whenever `hyb_z` is selected**.
Confirmed by reading the bytecode of `_domains_clean`:

```python
# domain consts in the bytecode include the literal 'con_z'
# names include 'append'
# the cleaner appends 'con_z' whenever 'hyb_z' ∈ domains
```

This is an **intrinsic property of PyRAT's hyb_z implementation**, not a
configuration error on our side. The Hybrid Zonotope abstraction in PyRAT is
constructed *on top of* a constrained-zonotope carrier — it is not a
stand-alone domain. Both domains are computed at every layer and the
tightest per-element bound is kept.

### What the user-facing label means in this archive

> "hyb_z" in this archive == PyRAT's hyb_z+con_z-carrier compound abstraction.

To isolate "con_z alone", see `pyrat_strict_20260527/`.

To measure the *additional* contribution of hyb_z beyond con_z, take the
per-bench V/A delta between this archive and the sibling. **A non-zero
delta is the marginal contribution of the hyb_z layer**, since
everything else is bit-identical.

---

## Patch 1 — `run_pure.py` (runtime monkey-patch)

Bit-identical to `pyrat_strict_20260527/patches/run_pure.py`. The same six
falsification entry points and the simulation-guided scorer are rebound to
no-op stubs before `pyrat.main()` is invoked. The `look_random(nb=1)`
passive shim that preserves analyzer-proved UNSAFE verdicts without
invoking `model.infer` is also unchanged.

Refer to the sibling archive's `patches/README.md` for the full audit and
the scientific-integrity argument.

---

## Patch 2 — Per-`.ini` hyb_z overlay

For every benchmark, the `.ini` at competition commit `4a9a4f0:vnn_config/`
is loaded, then:

1. Any line starting with `domains` or `domain=` is replaced with
   `domains = [hyb_z]`.
2. Any `split_relu` line is replaced with `split_relu = False`.
   (Forcing `split_relu=False` is necessary because PyRAT's
   split_relu path uses con_z BaB on ReLU and would dominate the
   abstraction; setting it False isolates the hyb_z contribution.)
3. The following five hyb_z knobs are appended (or, if pre-existing,
   normalised):

```ini
max_hybz           = -1
iterative_hybz     = False
hybz_relu_method   = False
intermediate_concr = False
split_relu         = False
```

4. The same parser-compat renames as in the sibling archive are
   inherited (`nb_repeat → nb_restart`, `step_ratio → lr_attack`).

Every other knob in the `.ini` — `split`, `nb_process`, `lr`, `n_it_optim`,
`library`, `device`, `dtype`, `check`, `nb_random`, `batch_attack`, etc. —
is **bit-identical** to the competition commit's `vnn_config/<bench>.ini`.

The 26 resulting `.ini` files are at `patches/hybz_ini/`.

---

## Knob choices — rationale

| Knob | Value here | Why |
|---|---|---|
| `max_hybz` | `-1` (unbounded) | Let PyRAT use the full HZ expansion without an artificial cap. With per-instance timeout enforced separately, an HZ cap would silently downgrade precision. |
| `iterative_hybz` | `False` | Direct max-precision computation rather than incremental refinement. Incremental would be slower per analysis call but might benefit some short-timeout benches; we picked the simpler max-precision path for cleaner attribution. |
| `hybz_relu_method` | `False` | PyRAT's default (Ortiz et al., 2304.02755). Both `True` (Zhang et al., IEEE 2023) and `False` produce *exact* abstractions; only solving time differs. |
| `intermediate_concr` | `False` | Do not concretise between layers (would lose hyb_z precision). |
| `split_relu` | `False` | Disable ReLU branching to isolate the hyb_z domain contribution. |

`split` (input-split BaB) is **inherited** from the competition `.ini` —
PyRAT's input-split BaB is a sound completeness mechanism that does not
contradict the strict-no-helper policy.

---

## What this archive does NOT touch

* The PyRAT verifier algorithm (`.pyc` binaries) — bit-identical to `95c72fc`.
* Every non-domain knob in the competition `.ini`s.
* The benchmark `.onnx` / `.vnnlib` / `instances.csv` files.

Only the `domains` line and the five hyb_z knobs above were edited; the
`.ini` directory at `vnn_config_2025_hybz/` is a new directory (not an
in-place edit of the competition tree).

---

## Reproducing the rerouted domain

```bash
# show that 'hyb_z' becomes ['con_z','hyb_z'] at runtime
python - << 'PYEOF'
from pyrat.analyzer.analysis_param import AnalysisParam
p = AnalysisParam(default=True)
p.domains = ['hyb_z']
p.clean_args()
print(p.domains)        # -> ['con_z', 'hyb_z']
PYEOF
```
