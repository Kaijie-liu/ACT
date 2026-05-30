# CORA TRUESTRICT source patches

The TRUESTRICT mode adds a new `falsification_method='none'` option to CORA that does not exist in the upstream VNN-COMP 2025 release. Three files in `/data1/Kane/cora-vnncomp2025/` are modified. The unified diff for `prepare_instance.m` is captured in `prepare_instance.m.patch` (a backup `.orig` was created before edit). The other two files were edited in place; their patched regions are reproduced verbatim below for forensic reproducibility.

All three edits are gated by the value of `options.nn.falsification_method`. When set to `'none'` (the TRUESTRICT default we enabled in `prepare_instance.m`), CORA runs pure over-approximative reachability with no falsification heuristic of any kind. When set to any other value (`'center'`, `'fgsm'`, `'zonotack'`), upstream behavior is preserved bit-for-bit.

---

## Patch 1 — `prepare_instance.m`

Sets the per-instance options to TRUESTRICT defaults. Backup exists at `prepare_instance.m.orig`.

See `prepare_instance.m.patch` for the unified diff.

---

## Patch 2 — `code/cora/nn/+nnHelper/validateNNoptions.m`

Adds `'none'` to the admissible-values list for `falsification_method`. Without this edit, CORA throws `CORA:wrongFieldValue` before reaching `verify.m`.

**Before** (line 318-319 in the upstream file, single contiguous block):

```matlab
aux_checkFieldStr(options.nn,'falsification_method', ...
    {'center','fgsm','zonotack'},structName);
```

**After** (TRUESTRICT, what is on disk at the time this archive was created):

```matlab
% TRUESTRICT patch: 'none' added so verify.m can skip falsification block.
aux_checkFieldStr(options.nn,'falsification_method', ...
    {'none','center','fgsm','zonotack'},structName);
```

---

## Patch 3 — `code/cora/nn/@neuralNetwork/verify.m`

Bypasses the entire falsification block in CORA's `verify` function when `falsification_method='none'`. Upstream computes adversarial candidates via FGSM / center-of-box / zonotack and then evaluates the candidate points to look for a counter-example. Under TRUESTRICT we skip all of it.

**Before** (lines ~380-465 in upstream, schematic; the original block runs FGSM/center/zonotack and `aux_checkPoints` unconditionally):

```matlab
% 2. Falsification ----------------------------------------------------

% 2.1. Compute adversarial examples.
switch options.nn.falsification_method
    case 'fgsm'
        ...
    case 'center'
        xi_ = xi;
    case 'zonotack'
        ...
    otherwise
        throw(CORAerror('CORA:wrongFieldValue', ...));
end

% 2.2. Check the specification for adversarial examples.
[~,critVal,falsified,x_,y_] = ...
    aux_checkPoints(nn,options,idxLayer,A,b,safeSet,xi_);

if any(falsified)
    res.str = 'COUNTEREXAMPLE';
    break;
end

% Check if the batch was extended with multiple candidates.
if size(critVal,2) > cbSz
    critVal_ = reshape(critVal,1,cbSz,[]);
    critVal = min(critVal_,[],3);
end
```

**After** (TRUESTRICT, lines 380-467 on disk):

```matlab
% 2. Falsification ----------------------------------------------------

% TRUESTRICT patch (for arXiv reproducibility study, see archived README):
% when falsification_method = 'none', skip the entire falsification block.
% This makes CORA strictly reachability-only — no point eval, no gradient,
% no sampling. Without this skip, even 'center' is a 1-point falsification
% helper since CORA's reachability is over-approximative (cannot prove A
% alone). Result: V verdicts come from reachability, A verdicts impossible.
if strcmp(options.nn.falsification_method, 'none')
    falsified = false(1, cbSz);
    critVal   = zeros(1, cbSz);
    x_        = [];
    y_        = [];
else
% 2.1. Compute adversarial examples.
switch options.nn.falsification_method
    case 'fgsm'
        ...
    case 'center'
        xi_ = xi;
    case 'zonotack'
        ...
    otherwise
        throw(CORAerror('CORA:wrongFieldValue', ...));
end

% 2.2. Check the specification for adversarial examples.
[~,critVal,falsified,x_,y_] = ...
    aux_checkPoints(nn,options,idxLayer,A,b,safeSet,xi_);

if any(falsified)
    res.str = 'COUNTEREXAMPLE';
    break;
end

if size(critVal,2) > cbSz
    critVal_ = reshape(critVal,1,cbSz,[]);
    critVal = min(critVal_,[],3);
end
end  % TRUESTRICT patch: close `if strcmp(falsification_method,'none')` outer
```

The wrapper introduces exactly one new conditional branch (`'none'` short-circuits with the stub values that the downstream code expects) and one new closing `end` to balance the outer `if`. No other lines in `verify.m` are touched.

---

## Why these three patches together, and not a single option toggle?

CORA's upstream design assumes that *some* falsification method is always selected, because over-approximative reachability alone can never produce a sound SAT verdict. The three patches lift this assumption:

1. `validateNNoptions.m` would otherwise reject `'none'` at startup.
2. `verify.m` would otherwise still execute `aux_checkPoints` with `xi_ = xi` if it received `'center'` — that 1-point evaluation is itself a (deterministic) falsification helper and must be skipped.
3. `prepare_instance.m` sets the option for every benchmark in the VNN-COMP 2025 suite, so a single sweep can be run with one launcher.

Skipping any one of the three is sufficient to break TRUESTRICT semantics — verify this before relying on any future modification.
