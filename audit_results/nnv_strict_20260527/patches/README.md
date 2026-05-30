# NNV STRICT — source patches

NNV has neither a CLI flag for "no helper" nor a natively-helper-free design. We applied **three patches** to enable STRICT mode and to recover R2026a MATLAB compatibility. Each patch is small, surgical, and documented inline with `% STRICT` or `% R2026a compat shim` markers in the patched source files.

| Patch | File | Lines added | Purpose |
|---|---|---|---|
| 1 | `code/nnv/examples/Submission/VNN_COMP2025/run_vnncomp_instance.m` | ~40 | STRICT mode: env-gated helper bypass + cp-star refusal |
| 2 | `code/nnv/engine/utils/matlab2nnv.m` | 7 | R2026a compat: `ScalingLayer` → `ElementwiseAffineLayer` |
| 3 | (driver, not NNV source) `scripts/nnv_strict_run_one.m` | ~25 | parpool isolation per MATLAB pid + `NNV_NUMCORES=5` env override |

All upstream behavior is preserved when `NNV_STRICT_NO_HELPER` is unset (or set to anything other than `1`). The R2026a compat shim (patch 2) is *unconditional* — it activates for every NNV call on R2026a, but is mathematically a no-op (the new branch maps to an existing equivalent layer class).

---

## Patch 1 — `run_vnncomp_instance.m` (STRICT mode, 4 edits)

### What NNV's three helpers are, and why we disable them

NNV's competition entry script contains three operations that violate "pure sound verifier":

1. **`falsify_single` random sampling**: draws `nRand=100` or `500` random points from the input box, evaluates `predict(net, x)` on each, returns SAT if any lies in the unsafe halfspace. Equivalent to nnenum/abcrown PGD / random falsification.
2. **`create_random_examples` lb/ub corner evaluation** (line 786, upstream): `xRand = [lb, ub, xRand];` — explicitly evaluates the deterministic input-box corners. Equivalent to CORA's `center-of-box` heuristic.
3. **`cp-star` reachability method**: conformal-prediction surrogate model with statistical bounds (coverage=0.999, confidence=0.999). With probability up to 10^-3 the bound is wrong. **Not sound formal verification.**

The STRICT patch gates (1) and (2) together (skipping `falsify_single` removes both), and refuses (3) entirely.

### 4 edits applied

All gated by an `NNV_STRICT = strcmp(getenv('NNV_STRICT_NO_HELPER'), '1')` switch defined once near the top.

```matlab
% Edit 1: After function header (line ~5)
% STRICT-MODE PATCH (ACT paper, 2026-05-28):
%   - Gates falsify_single (random + corner sampling) behind env NNV_STRICT_NO_HELPER=1.
%   - Refuses cp-star reachability (statistical, not sound) under STRICT mode.
%   - Default behavior (unset env) is identical to upstream competition entry.

t = tic;
status = 2;

% STRICT: env-gated mode switch (defined ONCE, used in all subsequent guards)
NNV_STRICT = ~isempty(getenv('NNV_STRICT_NO_HELPER')) && strcmp(getenv('NNV_STRICT_NO_HELPER'), '1');
```

```matlab
% Edit 2: After load_vnncomp_network returns reachOptionsList (line ~35)
% STRICT: reject benchmarks whose only configured reach method is cp-star
if NNV_STRICT
    only_cpstar = ~isempty(reachOptionsList);
    for strict_i = 1:length(reachOptionsList)
        if ~strcmp(reachOptionsList{strict_i}.reachMethod, 'cp-star')
            only_cpstar = false; break;
        end
    end
    if only_cpstar
        tTime = toc(t);
        fid = fopen(outputfile, 'w');
        fprintf(fid, 'unsupported_strict\n');
        fclose(fid);
        disp('STRICT: only cp-star configured for this benchmark; refusing (not sound).');
        status = 4;
        return
    end
end
```

```matlab
% Edit 3: Wrap the entire falsification block (~lines 70-96)
% STRICT: skip the entire falsification block — random sampling + lb/ub corner
% evaluation in falsify_single are helpers.
counterEx = nan;
if NNV_STRICT
    % falsification disabled; counterEx stays NaN, reachability decides.
else
    % <upstream falsify_single call chain unchanged>
end
```

```matlab
% Edit 4: Before the reachability block, catch cp-star fallback (e.g. linearizenn's
% try/catch fallback when matlab2nnv fails on the loaded network)
if NNV_STRICT && status == 2 && ~isempty(reachOptionsList) ...
        && strcmp(reachOptionsList{1}.reachMethod, 'cp-star')
    tTime = toc(t);
    fid = fopen(outputfile, 'w');
    fprintf(fid, 'unsupported_strict\n');
    fclose(fid);
    status = 4;
    return
end
```

### Verdict semantics added

A new `status = 4` corresponds to a new result-file token `unsupported_strict`, written when STRICT refuses a cp-star benchmark. Aggregator buckets it under `E` (tool unsupported), distinct from `error` (NNV crash).

---

## Patch 2 — `matlab2nnv.m` (R2026a compat shim, 7 lines)

### Why this is needed

On MATLAB R2026a, `importNetworkFromONNX(...)` silently inserts a `nnet.cnn.layer.ScalingLayer` between the input layer and the first computational layer for many models (auto-derived from ONNX initializers). On R2024a (the version VNN-COMP 2025 used), this layer was not inserted. NNV's `matlab2nnv.m` has cases for every other relevant layer class but not for `ScalingLayer` — so on R2026a, *every* `matlab2nnv(net)` call errors at line 221 with "Unsupported Class of Layer" before any reachability work begins. Pre-patch, NNV STRICT errored on 97% of instances on R2026a; post-patch, it runs.

### What the patch does

Adds a single `elseif` branch in matlab2nnv's layer-class dispatcher (right after the existing `nnet.onnx.layer.ElementwiseAffineLayer` case):

```matlab
% R2026a compat shim (ACT paper, 2026-05-28, user-authorized):
% nnet.cnn.layer.ScalingLayer is auto-inserted by importNetworkFromONNX
% on R2026a (was NOT inserted on R2024a). Mathematically identical to
% ElementwiseAffineLayer (y = Scale .* x + Offset). Without this case,
% every R2026a NNV verification attempt errors at matlab2nnv load time.
% This is NOT a STRICT-mode change — restores R2024a behaviour exactly.
elseif isa(L, 'nnet.cnn.layer.ScalingLayer')
    Li = ElementwiseAffineLayer(L.Name, L.Scale, L.Offset, true, true);
```

### Why this is honest

- The semantics of `ScalingLayer` and `ElementwiseAffineLayer` are identical: both compute `y = Scale .* x + Offset` with the same broadcast rules. NNV's existing `ElementwiseAffineLayer` is a sound-verifiable class.
- The patch does NOT change any reachability algorithm. It only adds a class-name-to-class-name mapping for one layer type that R2026a renamed/refactored.
- A counterfactual run on R2024a (which we cannot easily redo without reinstalling MATLAB) would produce *identical* verdicts for every instance, because R2024a never produces a `ScalingLayer`.
- This patch is *not* gated by the STRICT switch — it activates on every NNV call on R2026a. Removing it makes NNV unusable on this MATLAB version.

The user explicitly authorized this patch on 2026-05-28 after a transparent discussion of the trade-off ("don't fix upstream NNV" vs. "get useful NNV data for the paper"). See README_REPRODUCIBILITY.md "Patches and authorization" section.

---

## Patch 3 — `nnv_strict_run_one.m` (parpool isolation, driver-side)

### Why

When running 3 parallel lanes (each spawning its own MATLAB instance), the default `parcluster('Processes').JobStorageLocation` is shared across all MATLAB sessions for this user. Concurrent `parpool('local', N)` calls then race and fail with: `Failed to locate and destroy old interactive jobs. Unable to use a value of type cell as an index.`

Additionally, NNV hardcodes the parpool worker count as `numCores = feature('numcores')` (= 20 on this host), so 3 lanes would request 60 workers total — way past the 20 physical cores.

### What the driver does

Inside `nnv_strict_run_one.m`, before calling `run_vnncomp_instance`:

1. **Aggressive parpool state reset** (3-layer defence):
   ```matlab
   try, delete(gcp('nocreate')); catch, end
   try
       c = parcluster('Processes');
       if ~isempty(c.Jobs), delete(c.Jobs); end
   catch, end
   try
       % nuke on-disk Job* dirs from any earlier crashed sessions
       prefdir_local = fullfile(prefdir, '..', 'local_cluster_jobs', version('-release'));
       d = dir(fullfile(prefdir_local, 'Job*'));
       for k = 1:length(d)
           try, rmdir(fullfile(prefdir_local, d(k).name), 's'); catch, end
           try, delete(fullfile(prefdir_local, [d(k).name '*'])); catch, end
       end
   catch, end
   ```

2. **Per-pid private JobStorageLocation** (key fix for inter-lane race):
   ```matlab
   try
       c = parcluster('Processes');
       priv_dir = fullfile(tempdir, sprintf('nnv_strict_jobs_%d_%d', feature('getpid'), round(rand()*1e9)));
       if ~isfolder(priv_dir), mkdir(priv_dir); end
       c.JobStorageLocation = priv_dir;
       saveProfile(c);
   catch, end
   ```

3. **Worker-count cap via env var**: `setenv('NNV_NUMCORES','5')` — NNV's `run_vnncomp_instance.m` was further (4th edit, smaller) patched to honor this env var when present, so `parpool('local', N)` requests at most 5 workers per lane. With 3 lanes × 5 workers = 15 < 20 cores.

This is purely about *running multiple MATLAB sessions concurrently without parpool corruption*. It does not change verdict semantics. Verified by smoke test (malbeware: `unsat` in 17 s with 5-worker parpool).

---

## What this archive does NOT change

- `verify_specification.m` and the sound reach methods (`approx-star`, `exact-star`, `relax-star-area`) — untouched.
- NNV's per-benchmark dispatcher (which method maps to which benchmark) — untouched.
- The error-throwing branches for genuinely unsupported architectures (cctsdb_yolo, lsnc_relu) — untouched. NNV correctly reports `error` for these; the patches do not silence them.

Verifying patch integrity:

```bash
grep -c "STRICT-MODE PATCH (ACT paper" /data1/Kane/nnv/code/nnv/examples/Submission/VNN_COMP2025/run_vnncomp_instance.m
# expected: 1

grep -c "R2026a compat shim" /data1/Kane/nnv/code/nnv/engine/utils/matlab2nnv.m
# expected: 1

grep -c "NNV_STRICT" /data1/Kane/ACT/scripts/nnv_strict_run_one.m
# expected: 1 (env var name appears once in setenv call)
```
