# abcrown VNN-COMP 2025 sweep — reproducibility bundle

**Date.** 2026-05-25 to 2026-05-26
**Tool.** alpha-beta-CROWN (αβ-CROWN), two installations:
- GenBaB clone (older base) at `/data1/Kane/GenBaB/alpha-beta-CROWN`, commit `ff2649c9c` (master HEAD at clone time)
- vnncomp2025 fork at `/data1/Kane/alpha-beta-CROWN_vnncomp2025`, commit `61b5ff8` (with patches, see below)

**Protocol.** Strict "no-PGD" — `--pgd_order=skip` (which transitively auto-disables `adv_warmup`, `input_split.check_adv`, `bab.attack`, `mip_attack`). The CLI shorthand is `--NOPGD` (in vnncomp_main.py wrapper).

**Result.** See `RESULTS_TABLE.tex` next to this file.

---

## Directory layout

```
audit_results/abcrown_nopgd_20260525/
├── README_REPRODUCIBILITY.md           ← this file
├── RESULTS_TABLE.tex                   ← single-page beamer table of all benchmark results
├── _run.log                            ← driver log (interleaved across lanes A/C/F2/F3/H/...)
├── _run.meta.json                      ← provenance (host, GPU, torch, commits, etc.)
├── _run.meta.lane_a_genbab.json        ← snapshot taken before Lane C launched
├── _paused_manual.flag                 ← if pause/resume happened, this is the timestamp
├── _nohup*.out                         ← raw stdout/stderr for each lane / retry
├── <benchmark>/                        ← per-benchmark results (one dir per benchmark)
│   ├── _summary.csv                    ← per-benchmark CSV (idx, onnx, vnnlib, csv_timeout, used_timeout, wall_sec, verdict, exit_code, paths)
│   ├── NNNN__<onnx>__<vnnlib>.result   ← VNN-COMP standard result file (verdict line + adv example)
│   ├── NNNN__<onnx>__<vnnlib>.log      ← abcrown's full stdout/stderr
│   ├── NNNN__<onnx>__<vnnlib>.json     ← per-instance metadata (verdict, exit_code, wall_sec, pgd_disabled, tool=GenBaB|fork)
│   └── _err_attempt*/                  ← errored entries quarantined before retry (preserved for forensics)
└── audit_results/abcrown25_supplemental_20260526/   ← parallel dir for vnncomp25-new benchmarks (cersyve/malbeware/relusplitter/sat_relu) run with the fork
```

Tool tagging: each `.json` has a `"tool"` field, either `"abcrown_GenBaB"` (implied if absent in the main dir) or `"abcrown_vnncomp2025_fork"`. Use this when merging results.

---

## How to reproduce from scratch

### Prerequisites

| Item | Value used |
|---|---|
| Linux | Ubuntu 24.04, kernel 6.14 |
| GPU | NVIDIA RTX PRO 6000 Blackwell, 96 GB VRAM |
| CUDA driver | matching CUDA 12.8 (Blackwell needs cu128) |
| Python | 3.11 (in conda) |
| PyTorch | 2.9.1+cu128 (Blackwell-compatible) |
| Gurobi | 11.0.x (academic WLS license valid through 2026-05) |
| Disk | ~50 GB free for installer + benchmarks + results |

### Step 0: benchmark data

```bash
# Benchmarks come pre-bundled in
ls /data1/Kane/data/vnncomp2025_benchmarks/benchmarks/   # 26 benchmarks

# Initial setup we did (one-time):
cd /data1/Kane/data/vnncomp2025_benchmarks
gunzip -rk benchmarks/                                    # decompress all .gz
# Then specific symlinks for the "broken" benchmarks per upstream setup.sh:
cd benchmarks
mkdir -p nn4sys/onnx vggnet16_2022/onnx
# (note: most of these symlinks point to nn4sys_2023/vggnet16_2023 dirs that
#  don't exist in the 2025 distribution. The real files are in nn4sys/onnx/*.gz
#  and we down-loaded vgg16-7.onnx separately.)
```

### Step 1: download missing models (one-time)

```bash
# vgg16-7.onnx — 528 MB, from ONNX Model Zoo
mkdir -p /data1/Kane/data/vnncomp2025_benchmarks/benchmarks/vggnet16_2023/onnx
curl -fsSL -o /data1/Kane/data/vnncomp2025_benchmarks/benchmarks/vggnet16_2023/onnx/vgg16-7.onnx \
  "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification/vgg/model/vgg16-7.onnx"
# Symlink:
ln -sf ../../vggnet16_2023/onnx/vgg16-7.onnx \
  /data1/Kane/data/vnncomp2025_benchmarks/benchmarks/vggnet16_2022/onnx/vgg16-7.onnx

# mscn_2048d.onnx + mscn_2048d_dual.onnx + cGAN_imgSz32_nCh_3_small_transformer.onnx
# All from VNN-COMP large_models.zip (sciebo)
mkdir -p /data1/Kane/data/_large_models_download && cd $_
wget "https://rwth-aachen.sciebo.de/s/RapAoed1dxG1PMs/download" -O large_models.zip
unzip -j large_models.zip \
  "vnncomp2024/nn4sys_2023/seed_896832480/onnx/mscn_2048d.onnx.gz" \
  "vnncomp2024/nn4sys_2023/seed_896832480/onnx/mscn_2048d_dual.onnx.gz" \
  "vnncomp2024/cgan_2023/seed_896832480/onnx/cGAN_imgSz32_nCh_3_small_transformer.onnx.gz" \
  -d /tmp/extract
gunzip -c /tmp/extract/mscn_2048d.onnx.gz       > /data1/Kane/data/vnncomp2025_benchmarks/benchmarks/nn4sys/onnx/mscn_2048d.onnx
gunzip -c /tmp/extract/mscn_2048d_dual.onnx.gz  > /data1/Kane/data/vnncomp2025_benchmarks/benchmarks/nn4sys/onnx/mscn_2048d_dual.onnx
gunzip -c /tmp/extract/cGAN_imgSz32_nCh_3_small_transformer.onnx.gz \
  > /data1/Kane/data/vnncomp2025_benchmarks/benchmarks/cgan_2023/onnx/cGAN_imgSz32_nCh_3_small_transformer.onnx

# Trap: vnncomp2025 ships a bogus `mscn_2048d_dual.onnx.gz` containing a wget log
# (134-byte HTML-like text). Replace with the real one from large_models.zip.
```

### Step 2: install both abcrown forks

```bash
# Conda env for abcrown
conda create -n abcrown25 python=3.11 -y
/data1/Kane/miniconda3/envs/abcrown25/bin/pip install --no-cache-dir \
    torch==2.9.1+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
/data1/Kane/miniconda3/envs/abcrown25/bin/pip install --no-cache-dir \
    onnx==1.17.0 onnxruntime onnxoptimizer onnxsim skl2onnx \
    appdirs coloredlogs graphviz humanfriendly protobuf pyyaml psutil \
    sortedcontainers tqdm packaging rich pandas scikit-learn scipy gurobipy
# Verified-Intelligence's onnx2pytorch fork (vnncomp25 abcrown expects this)
/data1/Kane/miniconda3/envs/abcrown25/bin/pip install --no-cache-dir --no-deps \
  "git+https://github.com/Verified-Intelligence/onnx2pytorch@8447c42c3192dad383e5598edc74dddac5706ee2#egg=onnx2pytorch"

# GenBaB clone (older base; used as Lane A in our sweep)
cd /data1/Kane && git clone --recursive https://github.com/Verified-Intelligence/GenBaB.git GenBaB
# (the GenBaB clone has alpha-beta-CROWN as a subdir; the env "GenBaB" was
#  set up earlier with similar torch/onnx versions)

# vnncomp2025 fork
cd /data1/Kane && git clone --recursive \
  https://github.com/Verified-Intelligence/alpha-beta-CROWN_vnncomp2025.git \
  alpha-beta-CROWN_vnncomp2025
cd alpha-beta-CROWN_vnncomp2025 && git checkout 61b5ff8
```

### Step 3: apply the patches

**Patch A** — vnncomp25 fork's `auto_LiRPA/parse_graph.py` line 218: torch 2.7+ removed
`torch.onnx._globals`. Wrap the import in try/except:

```python
# at line 218 (approx) of /data1/Kane/alpha-beta-CROWN_vnncomp2025/auto_LiRPA/parse_graph.py:
if version.parse(torch.__version__) >= version.parse("2.1.0"):
    # Needed for BoundConcatGrad with torch 2.1.x — 2.6.x.
    # torch 2.7+ removed the internal torch.onnx._globals module;
    # autograd inlining is handled differently there, so we no-op.
    try:
        from torch.onnx._globals import GLOBALS
        GLOBALS.autograd_inlining = False
    except ImportError:
        pass
```

**Patch B** — GenBaB clone's missing nn4sys yaml symlink (vnncomp_main.py looks
for `nn4sys_2023.yaml` but file is named `nn4sys.yaml`):

```bash
cd /data1/Kane/GenBaB/alpha-beta-CROWN/complete_verifier/exp_configs/vnncomp23
ln -sf nn4sys.yaml nn4sys_2023.yaml
```

(This is necessary or **all 194 nn4sys instances** error with FileNotFoundError.)

**Patch C** (attempted, not effective for nn4sys/lsnc_relu/collins_aerospace) —
GenBaB clone's `auto_LiRPA/operators/constant.py` for the BoundPrimConstant
AttributeError on YOLO/transformer models. Not bundled here because the deeper
auto_LiRPA gaps (onnx::If for nn4sys mscn, aten::ATen sum for lsnc_relu) are
not fixable from outside — they require auto_LiRPA op support.

### Step 4: gurobi license

```bash
export GRB_LICENSE_FILE=/data1/Kane/ACT/modules/gurobi/gurobi.lic
# License must be valid (the one we used expired on 2026-05-25 and was extended)
/data1/Kane/miniconda3/envs/abcrown25/bin/python -c \
  "import gurobipy as gp; m=gp.Model(); m.setParam('OutputFlag',0); m.addVar(); m.optimize(); print('Gurobi OK')"
```

### Step 5: run the sweep

The driver scripts (committed in this bundle and at `/data1/Kane/ACT/scripts/`):

| Script | Purpose |
|---|---|
| `scripts/run_abcrown_nopgd_vnncomp2025.sh` | Main GenBaB sweep with `--NOPGD` |
| `scripts/run_abcrown25_supplemental.sh` | Fork-based supplemental for vnncomp25-new benchmarks |
| `scripts/run_abcrown25_vit_retry_parallel.sh` | 2-chunk parallel retry of vit_2023 (CPU-only chunking) |

Three lanes were used in the final sweep:

```bash
# Lane A — GenBaB clone, all benchmarks per BENCH_ORDER
nohup /data1/Kane/ACT/scripts/run_abcrown_nopgd_vnncomp2025.sh \
  > .../audit_results/abcrown_nopgd_20260525/_nohup.out 2>&1 &

# Lane B — fork, vnncomp25-new benchmarks (cersyve / malbeware / relusplitter
# / sat_relu / soundnessbench), separate results dir
nohup /data1/Kane/ACT/scripts/run_abcrown25_supplemental.sh \
  > .../audit_results/abcrown25_supplemental_20260526/_nohup.out 2>&1 &

# Lane C — fork, "recovery" for benchmarks that Lane A errored on.
# Writes into LANE A's results dir so its resume logic finds the .result and skips on its turn.
RESULTS_ROOT=.../audit_results/abcrown_nopgd_20260525 \
  nohup /data1/Kane/ACT/scripts/run_abcrown25_supplemental.sh \
    "cgan_2023 traffic_signs_recognition_2023 dist_shift_2023 ml4acopf_2024 vggnet16_2022 collins_aerospace_benchmark" \
    > .../audit_results/abcrown25_lane_c_recovery_20260526/_nohup.out 2>&1 &
```

The script supports resume by default: it checks for existing `.result` files
and skips instances where one already exists with non-empty content. So you can
interrupt and resume freely.

### Step 6: targeted retries (post-Lane-A)

For benchmarks where Lane A's GenBaB clone hit specific upstream bugs that the
fork fixes, we re-ran via the fork into the SAME results dir (resume picks up):

```bash
# nn4sys — GenBaB hits FileNotFoundError + onnx::If unsupported on mscn models.
# Fork's auto_LiRPA supports onnx::If for pensieve (but not mscn_2048d_dual).
# (Real mscn files must have been downloaded — see Step 1.)
RESULTS_ROOT=.../audit_results/abcrown_nopgd_20260525 \
  /data1/Kane/ACT/scripts/run_abcrown25_supplemental.sh "nn4sys"

# Quarantine errored entries before re-running so resume logic re-attempts them
# (script writes .json with verdict=missing_result but no .result file; resume
# only checks .result existence). We moved {*.json,*.log} with err verdicts
# into <benchmark>/_err_attempt1/ then re-launched the runner.
```

For vit_2023 we used a small parallel chunking script
(`run_abcrown25_vit_retry_parallel.sh 2`) to do 2 chunks concurrently — vit's
per-instance peak is ~50–80 GB on Blackwell so 2 is the safe parallel max.

### Step 7: get the final tally

```bash
# Per-benchmark JSON-based count (the .json files have the canonical verdict;
# .result files have the verdict's first line + adv example payload)
for d in audit_results/abcrown_nopgd_20260525 audit_results/abcrown25_supplemental_20260526; do
  for bd in "$d"/*/; do
    b=$(basename "$bd"); [[ "$b" =~ ^_ ]] && continue
    n=$(ls "$bd"/*.json 2>/dev/null | wc -l)
    [[ $n -gt 0 ]] || continue
    sat=$(grep -lE '"verdict":"sat"[,}]' "$bd"/*.json 2>/dev/null | wc -l)
    unsat=$(grep -lE '"verdict":"unsat"[,}]' "$bd"/*.json 2>/dev/null | wc -l)
    to=$(grep -lE '"verdict":"timeout' "$bd"/*.json 2>/dev/null | wc -l)
    unk=$(grep -lE '"verdict":"unknown"[,}]' "$bd"/*.json 2>/dev/null | wc -l)
    err=$((n - sat - unsat - to - unk))
    printf "%-32s n=%-4d V=%-3d A=%-3d U=%-3d E=%-3d\n" "$b" "$n" "$unsat" "$sat" "$((to+unk))" "$err"
  done
done | sort
```

---

## Caveats / known-unsupported (and why)

| Benchmark | Verdict | Root cause | Recoverable? |
|---|---|---|---|
| `lsnc_relu` | 80/80 error → mark **unsupported_no_pgd_path** | `aten::ATen sum` op not in auto_LiRPA (both clone+fork). Official VNN-COMP yaml relies on `pgd_order=before` to find a PGD witness BEFORE BoundedModule construction (which would fail). With `--NOPGD`, BoundedModule is forced and it raises NotImplementedError. | No (requires auto_LiRPA op support OR PGD path) |
| `soundnessbench` | 50/50 error → **unsupported_no_pgd_path** | First layer is `[12288 × 128]`. The intermediate `torch.eye(12288).expand(...)` in `backward_bound.py:get_sparse_C` needs ~72 GB single allocation. On 96 GB GPU when share with other workload → OOM. Official yaml uses PGD-find-witness to bypass bound prop entirely. | No (would need ≥80 GB GPU dedicated, or sparse-C rewrite) |
| `collins_aerospace_benchmark` | 6/6 error → **unsupported_oom** | YOLO 640×640 input with float64 bound prop. CROWN-LP fallback path needs ~5–6 GB single allocation on top of model load (>50 GB resident). Fork+TRY_CROWN both OOM in <10 s. | Only with smaller-input variants or aggressive batch-size yaml tuning |
| `nn4sys` mscn_* | 125/194 error | `onnx::If` control-flow op not supported by either clone or fork's auto_LiRPA. Pensieve sub-models DO work (69 verified). | Would need auto_LiRPA `If` op implementation |
| `vit_2023` non-pgd | 77/200 error | `IndexError: shape of mask [1] does not match indexed tensor [4,1,0,3,17,17]` in fork's auto_LiRPA when handling certain vit blocks. Same on GenBaB clone. | Likely a specific vit attention-layer pattern; requires patching auto_LiRPA |
| `relusplitter` | 60/220 error | Fork's vnncomp25 yaml for relusplitter requires **CPLEX** cutting planes (not Gurobi). Without CPLEX, those 60 instances throw `Exception: CPLEX cutting planes are needed.` | IBM Academic CPLEX license (free, separate registration) |
| `cgan_2023` | 3/21 error | (1) `transposedConvPadding_1` instance hits a JIT/CUDA driver error in TorchScript ClampedMultiplication forward kernel; (2,3) older error fixed once we sourced real `small_transformer.onnx` from large_models.zip. Then those re-tried and 2 became timeouts, 1 still CUDA driver error. | Possibly with `TORCH_DISABLE_JIT_OPTIMIZATIONS=1` or downgrading torch |

---

## Protocol clarification (important for paper writing)

`--pgd_order=skip` disables the FOLLOWING attack/falsification paths (verified from source — see `arguments.py:867` and `lp_mip_solver.py:1240`):

1. **PGD attack** — neither "before" nor "after" `self.attack()` calls in `abcrown.py` execute when `pgd_order == 'skip'`.
2. **MIP adv_warmup** — `lp_mip_solver.py` has `if adv_warmup and pgd_order != 'skip'`. Automatically skipped.
3. **input-BaB `check_adv`** — default is `'auto'` which "disables check_adv when pgd_order is skip" (`arguments.py:962` help).
4. **BaB-Attack beam search** (`bab.attack.enabled`) — default `False`, not enabled by us.
5. **MIP-based attack** (`attack.enable_mip_attack`) — default `False`, not enabled by us.
6. **input split RHS-update via attack** (`bab.branching.input_split.update_rhs_with_attack`) — default `False`.

What REMAINS:
- **Branch-and-bound completeness**: the abcrown BaB algorithm itself iteratively
  refines the bound and may discover that a region of input space is entirely
  unsafe — in which case it emits `sat` with the leaf's center as witness. **This
  is not "attack"; it is sound, complete enumeration based on LP/CROWN bounds.**
- All `sat` verdicts in our results come from this path, NOT from gradient-based
  attack or random sampling.

We document this distinction with the cctsdb_yolo_2023 (28 `sat`) and
safenlp_2024 (647 `sat`) results: these are complete BaB findings, not attack
results. See the example `.log` files for any of these — search for "Result: sat"
without preceding `attack(...)` invocation.

---

## Files in this bundle

Top-level scripts referenced by this README are at `/data1/Kane/ACT/scripts/`:

```
scripts/run_abcrown_nopgd_vnncomp2025.sh
scripts/run_abcrown25_supplemental.sh
scripts/run_abcrown25_vit_retry_parallel.sh
```

All three are bash; they wrap calls to `complete_verifier/vnncomp_main.py` and
add per-instance `.result` / `.log` / `.json` writing + a benchmark-level
`_summary.csv`. They are CPU-cheap and Lane A + Lane C + Lane F2/F3/F4 ran
concurrently with no contention beyond GPU.

`RESULTS_TABLE.tex` is a single-page beamer slide of the per-benchmark tally.
Compile with `pdflatex RESULTS_TABLE.tex`.

---

## What we'd do differently next time

1. **Use the vnncomp25 fork from day 1** — the GenBaB clone is older and missed
   features (yaml configs for vnncomp25-new benchmarks, onnx::If support in
   auto_LiRPA, etc.). Lane A spent time on benchmarks that Lane B/C re-ran
   anyway.
2. **Pre-flight the missing data** — `setup.sh`'s symlinks for nn4sys mscn /
   vggnet16 / cgan small_transformer ALL point at directories that don't exist
   in the 2025 distribution. Should download `large_models.zip` (2.6 GB) before
   starting the sweep, not after the first 8 hours.
3. **Run `gunzip -rk` once, then verify file integrity** — at least one `.gz`
   in the distribution (`mscn_2048d_dual.onnx.gz`) was actually a wget log file,
   not a real gzipped onnx. Verify by `file <name>.onnx` after gunzip.
4. **CRLF line endings** — `instances.csv` files have CRLF; bash arithmetic
   chokes on the trailing `\r`. Strip with `${var//[[:space:]]/}` before parsing.
5. **Decimal timeouts** — `metaroom_2023` and `traffic_signs_recognition_2023`
   have decimal timeouts (`480.0`). Bash arithmetic needs integers — use
   `${var%.*}` to strip the fractional part.
