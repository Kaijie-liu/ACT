# PyRAT STRICT — patch / provenance bundle

PyRAT is distributed by CEA as a **compiled-only** `.pyc` archive (no source
release). The STRICT mode is therefore created by a **runtime monkey-patch
shim** (`run_pure.py`) plus a small set of CLI flag overrides — no patches
to the verifier's source code are possible because no source code is
distributed.

This directory contains:

| File | Purpose |
|---|---|
| `run_pure.py` | The runtime-patch launcher (semantically equivalent to a source patch under `falsification_method='none'` — see §"Patch 1" below) |
| `competition_ini_rename.patch` | Unified diff of the trivial parser-compat renames on the competition `.ini` configs |
| `competition_ini/` | Bit-for-bit copy of the `.ini` files actually used at runtime (the renamed versions) |

---

## Provenance

| Item | Value |
|---|---|
| PyRAT binary commit | `95c72fc22bf084cde033fe5a65e1232e7af752f5` (HEAD of `pub/pyrat` `main`, "Merge branch 'fix-public-version' into 'main'") |
| Competition commit | `4a9a4f065a623be395ac4b3385a47ea81638dc48` ("Final version for VNN2025", referenced by arXiv:2512.19007 §PyRAT) |
| Source path of `.ini` | competition commit `4a9a4f0:vnn_config/<bench>.ini`, extracted via `git show 4a9a4f0:vnn_config/<bench>.ini` |
| Renames applied | `nb_repeat → nb_restart`, `step_ratio → lr_attack` (the HEAD pyrat parser no longer recognises the older names) |
| Why HEAD binary + competition ini | HEAD's `.pyc` is the only public binary that loads under our Python 3.10 env. The competition's `.pyc` is not separately published. We pin every other knob (`.ini`) to the competition commit and document the rename. |

---

## Patch 1 — `run_pure.py` (runtime monkey-patch)

**Why**: PyRAT's `.pyc` cannot be edited at source level. Helper-free operation
must be enforced before any analysis call. `run_pure.py` rebinds every
falsification entry point to a no-op stub via `setattr` on the imported
module, **before** PyRAT's `main()` is invoked.

**Semantics**: All six in-tool falsification functions and the
simulation-guided BaB scorer are replaced by no-op stubs that **never call
`model.infer`**. The only "passive" stub is `look_random`, which returns
`True` when called with `nb=1` (the witness-extraction branch in
`check_prop`'s UNSAFE arm) — but it does so **without executing any forward
pass**. See §"What `look_random(nb=1)=True` is" below for the semantic
argument that this preserves analyzer-soundness without invoking any
concrete-point evaluation.

**Patched symbols** (verbatim from `run_pure.py`):

```python
# Gradient-based attacks → (None, None)
pyrat.attacks.attacks.{pgd_attack, pgd_attack_batched, deepfool_attack}

# Falsification wrappers → False
pyrat.attacks.utils_attacks.{counter_adv, look_for_counter,
                             look_for_counter_adv, infer_counter}

# Random-sample falsifier → passive shim
pyrat.attacks.utils_attacks.look_random  ←  True if nb==1 else False
                                            (NO model.infer call in either branch)

# Imported references in analyzer modules' globals (bytecode binds here)
pyrat.analyzer.analyzer.{counter_adv, infer_counter, look_random}
pyrat.analyzer.analyzer_single.{look_random, look_for_counter}

# Simulation-guided BaB scorer
pyrat.partitioning.scorers.output_influence_scorer
  .OutputInfluenceScorer.score  →  abstract-width fallback (no model.infer)
```

**Forced CLI flags** (override anything in the `.ini`):

| Flag | Forced value | Why |
|---|---|---|
| `--check` | `skip` | Disables both the pre-analysis `counter_adv + look_random` block in `analyze()` and the post-analysis `look_for_counter` block in `check_prop()` |
| `--nb_random` | `0` | Defang any remaining random-sample falsifier call (belt-and-suspenders) |
| `--attack` | `bounds` | Reduces the `attack` list to a no-op value |
| `--batch_attack` | `False` | Cosmetic (consumer is `counter_adv`, already patched) |
| `--exhaustive` | `False` | Disables the brute-force integer-enumeration mode in `analyze_exhaustive_integer` |

**Auditing**: every patched-out function was independently re-checked at
runtime (see `_run.meta.json` field `flags.audited`) and a 3453-instance
sweep of all `.out` logs greps **0** matches for `pgd|deepfool|[CEX]|counter[_ ]ex|infer_counter`.

---

## What `look_random(nb=1)=True` is and why it is *not* a helper

In an unmodified PyRAT, `check_prop`'s UNSAFE arm calls
`look_random(box, ..., nb=1)` **after** the abstract analyzer has already
proved the property violated (`single_res.evaluate(to_verify)` returned
`UNSAFE` from over-approximative bounds alone). The purpose of that single
sample is to extract a *concrete* counter-example for VNN-COMP's
`.counterexample.gz` output — not to find one.

The over-approximation in PyRAT's `con_z` / `poly` domains is sound on the
UNSAFE side: if the output box is fully contained in the violation region,
then every input maps to a violating output, and the witness extraction
will succeed deterministically. We replace this single inference with a
passive `return True`, which preserves the analyzer's UNSAFE verdict but
**does not produce a witness or invoke any forward pass**.

Concretely: with `nb_random=0` forced and `--check skip` forced, the only
remaining call site of `look_random` is this UNSAFE-witness path with
`nb=1`. The passive shim eliminates the single forward pass that would
otherwise occur. No falsification-search path can fire.

This treatment is symmetric to how the other strict reproductions in this
paper handle their tools' "1-point deterministic evaluators":
- CORA: `falsification_method='none'` (we patched this in CORA's source)
- abcrown: `--pgd_order=skip` (which also disables `adv_warmup`, `check_adv`)
- NeuralSAT: `--no-pgd-attack` + early-stop disabled
- PyRAT: the `look_random` shim documented here.

The A column in `_summary_overall.csv` therefore reflects only those
instances whose **abstract analysis itself** concluded UNSAFE — there is
no center-of-box, no PGD, no random-sample contribution to A.

---

## Patch 2 — `competition_ini_rename.patch`

Two parameter renames between competition commit `4a9a4f0:vnn_config/`
and the HEAD pyrat parser:

| Old name | New name | What it controls |
|---|---|---|
| `nb_repeat` | `nb_restart` | Number of restarts for PGD/DeepFool (helper-side, fully disabled here) |
| `step_ratio` | `lr_attack` | PGD step-size scaling (helper-side, fully disabled here) |

These are semantic-identity renames in the parser: HEAD pyrat only
recognises the new names, the old ones would be ignored. We apply the
rename to keep the `.ini` files loadable while leaving every other knob
(`domains`, `device`, `check`, `nb_process`, `split_relu`, `lr`,
`n_it_optim`, etc.) bit-identical to the competition commit.

13 of the 26 `.ini` files contained one of these renamed knobs and are
included in `competition_ini_rename.patch`; the other 13 are byte-identical
to the competition source and have no patch entry.

---

## What this archive does **NOT** touch

* The PyRAT verifier algorithm (`.pyc` binaries under `pyrat/` in the
  upstream repo): bit-identical to commit `95c72fc`.
* The competition `.ini` knobs apart from the two parser renames.
* The benchmark `.onnx` / `.vnnlib` / `instances.csv` files
  (`/data1/Kane/data/vnncomp2025_benchmarks/`).

The only artifacts touched are:
1. The launcher `run_pure.py` (this archive's copy is in `patches/`).
2. The `.ini` files at `benchmarks/vnn_files/vnn_config_2025_competition/`
   (which is a *new* directory, not an in-place edit of the original).
