"""GlobalTriangleLP: small-dense ReLU verifier via global triangle LP with
LP-tight intermediate bounds. HiGHS only, forward + LP only, no split / BaB /
backward / Gurobi.

Use case: small dense ReLU networks (e.g. acasxu_2023). Achieves 61/186 on
acasxu under strict no-split principles vs. K-sweep ceiling 5/186 and prior
sweep_C+input-split canonical 42/186 (see paper §6.7).

Public API:
    is_small_dense(onnx_path, in_dim_max=32, total_relu_max=500) -> bool
    verify(onnx_path, vnnlib_path, time_limit_per_lp=5)
        -> ('verified'|'unknown'|'fail(...)', elapsed_s)

Soundness: standard 3-inequality ReLU triangle relaxation. LP relaxation
contains every reachable trajectory; LP-infeasible disjunct ⇒ unsafe set
empty for that disjunct; verified iff all disjuncts infeasible.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

# Lazy imports — only loaded when needed
_onnx = None
_numpy_helper = None
_hp = None


def _lazy_imports():
    global _onnx, _numpy_helper, _hp
    if _onnx is None:
        import onnx as _onnx_mod
        from onnx import numpy_helper as _nh
        import highspy as _hp_mod
        _onnx = _onnx_mod
        _numpy_helper = _nh
        _hp = _hp_mod


# ─── ONNX extraction (Sub-optional, MatMul+Add OR Gemm) ─────────────────────
def extract_layers(onnx_path: Path):
    """Return (sub_const, layers, output_layer):
      sub_const: (n_in,) input shift, OR None if model has no input Sub
      layers:    list of (W, b) for each hidden layer (with ReLU after)
      output_layer: (W, b) for final layer (no ReLU)
                    OR (W, b, skip_W, skip_slice) for linearizenn-style
                    models with an input-slice skip into the output.

    Handles:
      * optional Sub at input
      * optional Flatten
      * MatMul+Add chains and Gemm nodes (both as affine layers)
      * ReLU markers between layers
      * linearizenn-style tail: MatMul-(Slice+MatMul)-Concat-MatMul that
        produces output = main_path + skip_W @ x[slice_range]
    """
    _lazy_imports()
    m = _onnx.load(str(onnx_path))
    inits = {init.name: _numpy_helper.to_array(init) for init in m.graph.initializer}
    nodes = list(m.graph.node)

    sub_const = None
    matmul_b_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    relu_after: List[bool] = []

    # Track per-tensor symbolic linear ops for the post-Gemm-chain tail.
    # Map name -> ('main', last_matmul_b_index)  OR  ('input_slice', start, stop, step)
    #          OR ('linear', (A, b))  meaning value = A @ x_in + b
    # We compose linear maps symbolically through MatMul / Slice / Concat / MatMul.
    pending_W = None
    consumed_outputs = set()
    constants = {}  # name -> np.ndarray for Constant nodes
    main_chain_output = None  # the name of the final Add/Gemm output before tail

    # First pass: build hidden + initial output layer via the Gemm/MatMul+Add pattern
    for n in nodes:
        if n.op_type == "Sub":
            const_name = [i for i in n.input if i in inits][0]
            sub_const = inits[const_name].astype(np.float64).reshape(-1)
        elif n.op_type == "Flatten":
            pass
        elif n.op_type == "MatMul":
            const_inputs = [i for i in n.input if i in inits]
            if const_inputs:
                # Always set pending_W on MatMul. If a subsequent Add+init follows
                # (main-chain MatMul+Add pattern), it gets consumed. If not (tail
                # MatMul before Slice/Concat), pending_W is just left uncommitted.
                pending_W = inits[const_inputs[0]].astype(np.float64)
        elif n.op_type == "Add":
            const_inputs = [i for i in n.input if i in inits]
            if const_inputs and pending_W is not None:
                b = inits[const_inputs[0]].astype(np.float64).reshape(-1)
                matmul_b_pairs.append((pending_W, b))
                relu_after.append(False)
                pending_W = None
                main_chain_output = n.output[0]
        elif n.op_type == "Gemm":
            # Gemm has 2 or 3 inputs: A, B (weight), optional C (bias).
            # When B is an initializer, treat as standard affine: y = alpha*A@B + beta*C
            # ONNX default alpha=1, beta=1, transA=0, transB=0.
            B_name = n.input[1]
            C_name = n.input[2] if len(n.input) >= 3 else None
            if B_name in inits:
                W = inits[B_name].astype(np.float64)
                # Honor transB attribute
                transB = 0
                transA = 0
                alpha = 1.0
                beta = 1.0
                for attr in n.attribute:
                    if attr.name == 'transB': transB = attr.i
                    elif attr.name == 'transA': transA = attr.i
                    elif attr.name == 'alpha': alpha = attr.f
                    elif attr.name == 'beta': beta = attr.f
                if transB:
                    W = W.T
                W = alpha * W
                if C_name and C_name in inits:
                    b = beta * inits[C_name].astype(np.float64).reshape(-1)
                else:
                    b = np.zeros(W.shape[-1], dtype=np.float64)
                matmul_b_pairs.append((W, b))
                relu_after.append(False)
                main_chain_output = n.output[0]
        elif n.op_type == "Relu":
            if relu_after:
                relu_after[-1] = True
        elif n.op_type == "Constant":
            for attr in n.attribute:
                if attr.name == 'value':
                    constants[n.output[0]] = _numpy_helper.to_array(attr.t)
        # Slice / Concat / MatMul on non-init handled in tail pass below

    # Compose consecutive non-ReLU affine layers. The composition rule is:
    #   (x @ W1 + b1) @ W2 + b2 = x @ (W1 @ W2) + (b1 @ W2 + b2)
    # The first ReLU after a composed block terminates that hidden layer;
    # any trailing composed affines (no ReLU after) become the output layer.
    composed_layers = []          # list of (W, b)
    composed_has_relu = []         # parallel: True if this layer is followed by ReLU
    cur_W, cur_b = None, None
    for (W, b), has_relu in zip(matmul_b_pairs, relu_after):
        if cur_W is None:
            cur_W, cur_b = W, b
        else:
            # Compose: x @ cur_W + cur_b → (...) @ W + b
            cur_W = cur_W @ W
            cur_b = cur_b @ W + b
        if has_relu:
            composed_layers.append((cur_W, cur_b))
            composed_has_relu.append(True)
            cur_W, cur_b = None, None
    # Trailing affine(s) without ReLU = the output layer
    if cur_W is not None:
        composed_layers.append((cur_W, cur_b))
        composed_has_relu.append(False)

    hidden = [layer for layer, hr in zip(composed_layers, composed_has_relu) if hr]
    out_layers = [layer for layer, hr in zip(composed_layers, composed_has_relu) if not hr]
    if len(out_layers) != 1:
        raise ValueError(
            f"GlobalTriangleLP: expected exactly 1 output (no-ReLU) layer "
            f"after affine composition, got {len(out_layers)}; "
            f"model not supported."
        )
    output_layer = out_layers[0]

    # ─── Tail pass: detect linearizenn-style skip pattern ───────────────────
    # Look for nodes AFTER main_chain_output (= last Gemm/Add output):
    #   MatMul(main_chain_output, W_a)  -> u
    #   Slice(input, ...)               -> x_slice
    #   MatMul(x_slice, W_b)            -> v
    #   Concat([u, v])                  -> c
    #   MatMul(c, W_c)                  -> output
    # If detected, fold all tail linears into:
    #   output_layer = (W_out_main, b_out_main) such that
    #     output = post_L @ W_out_main + b_out_main + x_in[slice] @ W_out_skip
    if main_chain_output is None:
        return sub_const, hidden, output_layer

    # Trace tail: detect and fold linearizenn-style Slice+Concat+MatMul.
    # See _maybe_fold_linearizenn_tail for the symbolic-linear composition.
    return _maybe_fold_linearizenn_tail(sub_const, hidden, output_layer,
                                          nodes, inits, constants, main_chain_output,
                                          m.graph.input[0].name)


def _slice_size(slice_tuple, n_in):
    """Resolve slice (start, end, step) against n_in to a concrete count."""
    if slice_tuple is None: return 0
    start, end, step = slice_tuple
    if end is None or end > n_in: end = n_in
    if start < 0: start = max(0, n_in + start)
    if step <= 0: step = 1
    return max(0, (end - start + step - 1) // step)


def _maybe_fold_linearizenn_tail(sub_const, hidden, output_layer,
                                   nodes, inits, constants, main_chain_output,
                                   input_name):
    """If we find a Slice+Concat+MatMul tail, fold it into a richer output_layer
    that also includes a skip term from input. Returns:
        (sub_const, hidden, (W_main, b_main, W_skip, slice_idx_range))
    where slice_idx_range is a tuple (start, stop, step).
    If no tail, returns (sub_const, hidden, output_layer) as before.
    """
    # Collect tail nodes (those that consume main_chain_output transitively
    # or take input as a side branch via Slice).
    # Symbolic tensors: dict name -> {kind, ...}
    # kind: 'main' — value derived from post_L via affine (A_post, b_post)
    #       'input' — equals the model input tensor x_in
    #       'slice' — slice of input: (start, stop, step)
    #       'linear' — A_post @ post_L + A_skip @ x_in[slice] + b
    # We propagate symbolic values through MatMul/Slice/Concat.

    # The output_layer's affine is: last_pre = post_L @ output_layer[0] + output_layer[1]
    # (where post_L is the previous ReLU's output). But linearizenn has NO ReLU
    # after the final Gemm — so main_chain_output IS this value, treat it as the
    # "starting symbolic" linear value:
    #   main_chain_output = post_L @ W + b
    # For tail tracing, we let post_L be a placeholder; A_post starts as W and
    # b starts as b for main_chain_output.

    W_init, b_init = output_layer  # last Gemm/MatMul+Add weights
    # Determine model input dim for slice resolution
    n_in_full = W_init.shape[0] if sub_const is None else sub_const.shape[0]
    if hidden:
        # First hidden layer's W has shape (n_in, n_h0); first dim is n_in
        n_in_full = hidden[0][0].shape[0]
    # Tail symbolic state per tensor name:
    syms = {}
    # main_chain_output represents post_L @ W_init + b_init
    syms[main_chain_output] = {
        'A_post': W_init.copy(),     # shape (n_L, n_dim)
        'A_skip': None,              # set when skip term enters
        'b': b_init.copy(),          # shape (n_dim,)
        'slice': None,
    }
    syms[input_name] = {
        'A_post': None,
        'A_skip': None,
        'b': None,
        'slice': (0, None, 1),       # full input
        '_is_input': True,
    }

    has_tail = False
    for n in nodes:
        if all((inp in inits) or (inp not in syms and inp not in constants) for inp in n.input):
            continue
        if n.op_type == "Slice":
            # Slice(input, starts, ends, axes, steps) — assume axis=-1 if axes present
            in_name = n.input[0]
            if in_name not in syms:
                continue
            # Read starts/ends/steps from constants
            starts = constants.get(n.input[1], None)
            ends = constants.get(n.input[2], None)
            # axes optional (n.input[3]), steps optional (n.input[4])
            steps = constants.get(n.input[4], None) if len(n.input) >= 5 else None
            if starts is None or ends is None:
                continue
            start = int(starts.flatten()[0])
            end = int(ends.flatten()[0])
            step = int(steps.flatten()[0]) if steps is not None else 1
            # Clamp end to input dim (ONNX uses INT64_MAX for "to end")
            if end > n_in_full or end < 0: end = n_in_full
            if start < 0: start = max(0, n_in_full + start)
            out_name = n.output[0]
            syms[out_name] = {
                'A_post': None,
                'A_skip': None,
                'b': None,
                'slice': (start, end, step),
                '_is_input_slice': True,
            }
            has_tail = True
        elif n.op_type == "MatMul":
            # tail MatMul: tensor @ W
            in0, in1 = n.input[0], n.input[1]
            W = inits[in1].astype(np.float64) if in1 in inits else None
            if W is None or in0 not in syms:
                continue
            sym0 = syms[in0]
            out_name = n.output[0]
            new_sym = {'A_post': None, 'A_skip': None, 'b': None, 'slice': None}
            if sym0.get('_is_input_slice'):
                # value = x_in[slice] @ W
                new_sym['A_skip'] = W.copy()
                new_sym['slice'] = sym0['slice']
                new_sym['b'] = np.zeros(W.shape[-1], dtype=np.float64)
            elif sym0.get('_is_input'):
                new_sym['A_skip'] = W.copy()
                new_sym['slice'] = (0, None, 1)
                new_sym['b'] = np.zeros(W.shape[-1], dtype=np.float64)
            else:
                # sym0 = A_post @ post_L + (A_skip @ x_slice) + b
                # value = sym0 @ W
                if sym0['A_post'] is not None:
                    new_sym['A_post'] = sym0['A_post'] @ W
                if sym0['A_skip'] is not None:
                    new_sym['A_skip'] = sym0['A_skip'] @ W
                    new_sym['slice'] = sym0['slice']
                new_sym['b'] = (sym0['b'] @ W) if sym0['b'] is not None else np.zeros(W.shape[-1], dtype=np.float64)
            syms[out_name] = new_sym
            has_tail = True
        elif n.op_type == "Concat":
            # Concat along last axis. We concatenate (A_post @ post_L + A_skip @ x_slice + b) blocks.
            # All inputs must be symbolic; concat along axis -1.
            in_syms = [syms.get(i) for i in n.input]
            if any(s is None for s in in_syms):
                continue
            # Pad each block's A_post/A_skip/b to vectors; concat along last dim
            blocks_post = []; blocks_skip = []; blocks_b = []
            # Resolve a common slice. For simplicity, require all skip slices match
            # or be None.
            slices = [s['slice'] for s in in_syms if s.get('A_skip') is not None]
            common_slice = slices[0] if slices else None
            sk_n_common = _slice_size(common_slice, n_in_full) if common_slice else 0
            for s in in_syms:
                A_p = s.get('A_post')
                A_s = s.get('A_skip')
                bb = s.get('b')
                dim_out = (A_p.shape[1] if A_p is not None
                            else (A_s.shape[1] if A_s is not None
                                    else (bb.shape[0] if bb is not None else 0)))
                if A_p is None:
                    A_p = np.zeros((W_init.shape[0], dim_out), dtype=np.float64)
                if A_s is None and sk_n_common > 0:
                    A_s = np.zeros((sk_n_common, dim_out), dtype=np.float64)
                if bb is None:
                    bb = np.zeros(dim_out, dtype=np.float64)
                blocks_post.append(A_p)
                if A_s is not None: blocks_skip.append(A_s)
                blocks_b.append(bb)
            new_sym = {
                'A_post': np.concatenate(blocks_post, axis=-1) if blocks_post else None,
                'A_skip': (np.concatenate(blocks_skip, axis=-1) if blocks_skip else None),
                'b': np.concatenate(blocks_b, axis=-1) if blocks_b else None,
                'slice': common_slice,
            }
            syms[n.output[0]] = new_sym
            has_tail = True
        # Other ops (Constant) are tracked in `constants` dict (already populated)

    if not has_tail:
        return sub_const, hidden, output_layer

    # Find the model graph's final output name to get the folded result
    # (the last symbolic tensor whose name matches model output).
    # We assume `output` is the model's output name.
    final_name = 'output'
    if final_name not in syms:
        # fallback: pick last sym
        candidate_outputs = [k for k in syms if k not in ('input', input_name)]
        if not candidate_outputs:
            return sub_const, hidden, output_layer
        final_name = candidate_outputs[-1]
    final_sym = syms[final_name]
    W_main = final_sym['A_post']
    b_main = final_sym['b']
    W_skip = final_sym['A_skip']
    slice_range = final_sym['slice']
    if W_main is None:
        return sub_const, hidden, output_layer
    if W_skip is None:
        return sub_const, hidden, (W_main, b_main)
    return sub_const, hidden, (W_main, b_main, W_skip, slice_range)


def is_small_dense(onnx_path: Path,
                   in_dim_max: int = 32,
                   total_relu_max: int = 500) -> bool:
    """Heuristic: dispatch only on small dense ReLU networks.

    Conditions:
      - No Conv ops in the graph
      - Input dim <= in_dim_max (default 32)
      - Total ReLU count <= total_relu_max (default 500)
    """
    _lazy_imports()
    try:
        m = _onnx.load(str(onnx_path))
    except Exception:
        return False
    has_conv = False
    n_relu = 0
    in_dim = 0
    for n in m.graph.node:
        if n.op_type.startswith("Conv"):
            has_conv = True
        elif n.op_type == "Relu":
            n_relu += 1
    if has_conv:
        return False
    if n_relu == 0 or n_relu > total_relu_max:
        return False
    # Input dim from graph input shape (last dim)
    if m.graph.input:
        try:
            shape = m.graph.input[0].type.tensor_type.shape.dim
            dims = [d.dim_value if d.dim_value > 0 else 1 for d in shape]
            in_dim = int(np.prod(dims))
        except Exception:
            return False
    if in_dim == 0 or in_dim > in_dim_max:
        return False
    return True


# ─── vnnlib parser ──────────────────────────────────────────────────────────
def _split_top_level_groups(s: str, opener: str = '(', closer: str = ')'):
    out = []
    depth = 0; start = -1
    for i, ch in enumerate(s):
        if ch == opener:
            if depth == 0: start = i
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0 and start >= 0:
                out.append(s[start:i+1])
                start = -1
    return out


def parse_vnnlib(vnn_path: Path, n_in: int, n_out: int):
    """Returns disjuncts: list of (lb_x_disj, ub_x_disj, unsafe_rows).
    Handles top-level X bounds, top-level Y constraints, and arbitrary mix
    of (assert (or (and ...) (and ...))) blocks; final disjuncts are the
    cartesian product over all OR-blocks.
    """
    import re
    raw = open(vnn_path).read()
    lines = []
    for ln in raw.split('\n'):
        i = ln.find(';')
        if i >= 0: ln = ln[:i]
        if ln.strip(): lines.append(ln)
    txt = '\n'.join(lines)

    lb_x_global = np.full(n_in, -np.inf)
    ub_x_global = np.full(n_in,  np.inf)
    for m in re.finditer(r'\(assert \(>= X_(\d+) ([\-\d.eE]+)\)\)', txt):
        lb_x_global[int(m.group(1))] = float(m.group(2))
    for m in re.finditer(r'\(assert \(<= X_(\d+) ([\-\d.eE]+)\)\)', txt):
        ub_x_global[int(m.group(1))] = float(m.group(2))

    def parse_y_in(blk):
        disj = []
        for m in re.finditer(r'\(>= Y_(\d+) ([\-\d.eE]+)\)', blk):
            c = np.zeros(n_out); c[int(m.group(1))] = -1.0
            disj.append((c, -float(m.group(2))))
        for m in re.finditer(r'\(<= Y_(\d+) ([\-\d.eE]+)\)', blk):
            c = np.zeros(n_out); c[int(m.group(1))] = 1.0
            disj.append((c, float(m.group(2))))
        for m in re.finditer(r'\(>= \(\- Y_(\d+) Y_(\d+)\) ([\-\d.eE]+)\)', blk):
            c = np.zeros(n_out); c[int(m.group(1))] = -1.0; c[int(m.group(2))] = 1.0
            disj.append((c, -float(m.group(3))))
        for m in re.finditer(r'\(<= \(\- Y_(\d+) Y_(\d+)\) ([\-\d.eE]+)\)', blk):
            c = np.zeros(n_out); c[int(m.group(1))] = 1.0; c[int(m.group(2))] = -1.0
            disj.append((c, float(m.group(3))))
        for m in re.finditer(r'\(<= Y_(\d+) Y_(\d+)\)', blk):
            c = np.zeros(n_out); c[int(m.group(1))] = 1.0; c[int(m.group(2))] = -1.0
            disj.append((c, 0.0))
        for m in re.finditer(r'\(>= Y_(\d+) Y_(\d+)\)', blk):
            c = np.zeros(n_out); c[int(m.group(2))] = 1.0; c[int(m.group(1))] = -1.0
            disj.append((c, 0.0))
        return disj

    def parse_x_in(blk):
        lb = lb_x_global.copy(); ub = ub_x_global.copy()
        for m in re.finditer(r'\(>= X_(\d+) ([\-\d.eE]+)\)', blk):
            lb[int(m.group(1))] = float(m.group(2))
        for m in re.finditer(r'\(<= X_(\d+) ([\-\d.eE]+)\)', blk):
            ub[int(m.group(1))] = float(m.group(2))
        return lb, ub

    # Extract OR blocks first (we need their spans to compute top_level_y on
    # the text WITHOUT OR contents — otherwise per-disjunct Y rows get
    # duplicated as "top-level" too).
    or_block_alternatives = []
    or_block_spans = []  # (start, end) spans to mask out for top-level parse
    for m in re.finditer(r'\(assert \(or\b', txt):
        start = m.start()
        depth = 0; end = -1
        for i in range(start, len(txt)):
            if txt[i] == '(': depth += 1
            elif txt[i] == ')':
                depth -= 1
                if depth == 0:
                    end = i + 1; break
        if end < 0: continue
        assert_block = txt[start:end]
        body_match = re.match(r'\(assert \(or\s+(.+)\)\s*\)', assert_block, re.DOTALL)
        if not body_match: continue
        body = body_match.group(1)
        ands = _split_top_level_groups(body)
        if ands:
            or_block_alternatives.append(ands)
            or_block_spans.append((start, end))

    # Build a text without OR-block contents for top-level Y extraction
    if or_block_spans:
        pieces = []
        cursor = 0
        for s, e in sorted(or_block_spans):
            pieces.append(txt[cursor:s])
            cursor = e
        pieces.append(txt[cursor:])
        txt_without_or = ''.join(pieces)
    else:
        txt_without_or = txt
    top_level_y = parse_y_in(txt_without_or)

    disjuncts = []
    if not or_block_alternatives:
        if top_level_y:
            disjuncts.append((lb_x_global, ub_x_global, top_level_y))
    else:
        from itertools import product
        for combo in product(*or_block_alternatives):
            lb_d = lb_x_global.copy(); ub_d = ub_x_global.copy()
            y_rows = list(top_level_y)
            for and_block in combo:
                lb_blk, ub_blk = parse_x_in(and_block)
                lb_d = np.maximum(lb_d, lb_blk)
                ub_d = np.minimum(ub_d, ub_blk)
                y_rows.extend(parse_y_in(and_block))
            if y_rows:
                disjuncts.append((lb_d, ub_d, y_rows))
    return disjuncts


# ─── LP build + solve ───────────────────────────────────────────────────────
def _solve_one_obj(h, nvars, obj_coefs, sense='min', time_limit=10):
    """Set objective and solve. obj_coefs: dict col→coef."""
    obj = [0.0] * nvars
    for c, v in obj_coefs.items():
        obj[c] = float(v) if sense == 'min' else -float(v)
    h.changeColsCost(nvars, np.arange(nvars, dtype=np.int32), obj)
    h.run()
    sm = h.getModelStatus()
    if sm == _hp.HighsModelStatus.kOptimal:
        val = h.getObjectiveValue()
        return ('ok', val if sense == 'min' else -val)
    if sm == _hp.HighsModelStatus.kInfeasible:
        return ('infeasible', None)
    return ('fail', None)


def output_affine_rows(output_layer, n_li, n_in, n_out,
                         s_post: int, s_xin: int, s_y: int):
    """Build the LP rows for output y = post_L @ W_main + b_main [+ x_in[slc] @ W_skip].

    Handles both 2-tuple `(W_main, b_main)` and 4-tuple
    `(W_main, b_main, W_skip, slice_range)` output_layer formats.

    Each row encodes: y[j] - Σ wjk·post_L[k] - Σ wjm·x_in[m] = b_main[j]
    """
    if len(output_layer) == 2:
        W_main, b_main = output_layer
        W_skip = None; slc = None
    else:
        W_main, b_main, W_skip, slc = output_layer
    in_to_out_main = (W_main.shape[0] == n_li)
    rows = []
    # Resolve skip indices
    skip_idx = None
    if W_skip is not None and slc is not None:
        s, e, step = slc
        if e is None or e > n_in: e = n_in
        skip_idx = list(range(s, e, step))
        in_to_out_skip = (W_skip.shape[0] == len(skip_idx))
    for j in range(n_out):
        coefs = {s_y + j: 1.0}
        for k in range(n_li):
            wkj = W_main[k, j] if in_to_out_main else W_main[j, k]
            if wkj != 0:
                coefs[s_post + k] = -wkj
        if W_skip is not None:
            for pos, i_in in enumerate(skip_idx):
                wkj = W_skip[pos, j] if in_to_out_skip else W_skip[j, pos]
                if wkj != 0:
                    coefs[s_xin + i_in] = coefs.get(s_xin + i_in, 0.0) - wkj
        rows.append(('eq', float(b_main[j]), list(coefs.items())))
    return rows


def _add_rows_to_lp(h, rows_data):
    """rows_data: list of (sense, rhs, [(col, val), ...])"""
    for sense, rhs, entries in rows_data:
        cols = np.array([e[0] for e in entries], dtype=np.int32)
        vals = np.array([e[1] for e in entries], dtype=np.float64)
        if sense == 'le':
            h.addRow(-_hp.kHighsInf, rhs, len(cols), cols, vals)
        elif sense == 'ge':
            h.addRow(rhs, _hp.kHighsInf, len(cols), cols, vals)
        else:
            h.addRow(rhs, rhs, len(cols), cols, vals)


def _verify_one_disjunct(sub_const, layers, output_layer,
                          lb_x, ub_x, unsafe_rows, time_limit_per_lp=5):
    n_in = lb_x.shape[0]
    n_out = output_layer[1].shape[0]
    n_l = [b.shape[0] for W, b in layers]
    var_offsets = {}
    cur = 0
    var_offsets['x_in'] = (cur, cur + n_in); cur += n_in
    for li in range(len(layers)):
        var_offsets[f'pre_{li}'] = (cur, cur + n_l[li]); cur += n_l[li]
        var_offsets[f'post_{li}'] = (cur, cur + n_l[li]); cur += n_l[li]
    var_offsets['y'] = (cur, cur + n_out); cur += n_out
    nvars = cur

    lb_arr = np.full(nvars, -_hp.kHighsInf)
    ub_arr = np.full(nvars,  _hp.kHighsInf)
    s, e = var_offsets['x_in']
    lb_arr[s:e] = lb_x; ub_arr[s:e] = ub_x

    h = _hp.Highs()
    h.silent()
    h.setOptionValue("time_limit", float(time_limit_per_lp))
    h.setOptionValue("presolve", "off")
    h.setOptionValue("solver", "simplex")
    lp = _hp.HighsLp()
    lp.num_col_ = nvars
    lp.num_row_ = 0
    lp.col_cost_ = [0.0] * nvars
    lp.col_lower_ = lb_arr.tolist()
    lp.col_upper_ = ub_arr.tolist()
    lp.row_lower_ = []
    lp.row_upper_ = []
    lp.a_matrix_.format_ = _hp.MatrixFormat.kColwise
    lp.a_matrix_.start_ = [0] * (nvars + 1)
    lp.a_matrix_.index_ = []
    lp.a_matrix_.value_ = []
    h.passModel(lp)

    # Layer 0 affine: pre_0 = W_0 @ (x_in - sub_const) + b_0
    W0, b0 = layers[0]
    in_to_out = (W0.shape[0] == n_in)
    s_pre0, _ = var_offsets['pre_0']
    s_xin, _ = var_offsets['x_in']
    if sub_const is not None:
        b_eff0 = b0 - (sub_const @ W0 if in_to_out else W0 @ sub_const)
    else:
        b_eff0 = b0
    rows = []
    for j in range(n_l[0]):
        coefs = {s_pre0 + j: 1.0}
        for k in range(n_in):
            wkj = W0[k, j] if in_to_out else W0[j, k]
            if wkj != 0:
                coefs[s_xin + k] = -wkj
        rows.append(('eq', float(b_eff0[j]),
                     [(c, v) for c, v in coefs.items()]))
    _add_rows_to_lp(h, rows)

    layer_pre_bounds = []
    for li in range(len(layers)):
        n_li = n_l[li]
        s_pre, _ = var_offsets[f'pre_{li}']
        lb_pre = np.zeros(n_li); ub_pre = np.zeros(n_li)
        for j in range(n_li):
            st_lb, lb_j = _solve_one_obj(h, nvars, {s_pre + j: 1.0}, sense='min',
                                          time_limit=time_limit_per_lp)
            st_ub, ub_j = _solve_one_obj(h, nvars, {s_pre + j: 1.0}, sense='max',
                                          time_limit=time_limit_per_lp)
            if st_lb != 'ok' or st_ub != 'ok':
                return f'fail(lp_bound_l{li}_n{j}_{st_lb}/{st_ub})'
            lb_pre[j] = lb_j; ub_pre[j] = ub_j
        layer_pre_bounds.append((lb_pre, ub_pre))

        s_post, _ = var_offsets[f'post_{li}']
        new_lb = np.maximum(0, lb_pre)
        new_ub = np.maximum(0, ub_pre)
        for j in range(n_li):
            h.changeColBounds(s_post + j, float(new_lb[j]), float(new_ub[j]))
        rows = []
        for j in range(n_li):
            l, u = float(lb_pre[j]), float(ub_pre[j])
            if l >= 0:
                rows.append(('eq', 0.0, [(s_post + j, 1.0), (s_pre + j, -1.0)]))
            elif u <= 0:
                rows.append(('eq', 0.0, [(s_post + j, 1.0)]))
            else:
                rows.append(('le', 0.0, [(s_pre + j, 1.0), (s_post + j, -1.0)]))
                lam = u / (u - l)
                rhs = -lam * l
                rows.append(('le', float(rhs),
                             [(s_post + j, 1.0), (s_pre + j, -lam)]))
        _add_rows_to_lp(h, rows)

        if li + 1 < len(layers):
            W_next, b_next = layers[li + 1]
            n_next = n_l[li + 1]
            in_to_out_n = (W_next.shape[0] == n_li)
            s_pre_next, _ = var_offsets[f'pre_{li + 1}']
            rows = []
            for j in range(n_next):
                coefs = {s_pre_next + j: 1.0}
                for k in range(n_li):
                    wkj = W_next[k, j] if in_to_out_n else W_next[j, k]
                    if wkj != 0:
                        coefs[s_post + k] = -wkj
                rows.append(('eq', float(b_next[j]),
                             [(c, v) for c, v in coefs.items()]))
            _add_rows_to_lp(h, rows)
        else:
            s_y, _ = var_offsets['y']
            s_xin_eff, _ = var_offsets['x_in']
            rows = output_affine_rows(output_layer, n_li, n_in, n_out,
                                        s_post=s_post, s_xin=s_xin_eff, s_y=s_y)
            _add_rows_to_lp(h, rows)

    s_y, _ = var_offsets['y']
    for c_vec, d in unsafe_rows:
        entries = [(s_y + k, float(c_vec[k])) for k in range(n_out) if c_vec[k] != 0]
        cols = np.array([e[0] for e in entries], dtype=np.int32)
        vals = np.array([e[1] for e in entries], dtype=np.float64)
        h.addRow(-_hp.kHighsInf, float(d), len(cols), cols, vals)
    h.changeColsCost(nvars, np.arange(nvars, dtype=np.int32), [0.0] * nvars)
    h.run()
    sm = h.getModelStatus()
    if sm == _hp.HighsModelStatus.kOptimal:
        return 'unknown'
    if sm == _hp.HighsModelStatus.kInfeasible:
        return 'verified'
    return f'fail(spec_lp_{str(sm)})'


# ─── Public API ─────────────────────────────────────────────────────────────
def verify(onnx_path, vnnlib_path, time_limit_per_lp: float = 5.0):
    """Verify a single (onnx, vnnlib) instance via global triangle LP.

    Returns ('verified'|'unknown'|'fail(...)', elapsed_s).

    Sound: LP relaxation contains the integer-feasible set; LP-infeasible
    disjunct ⇒ unsafe set empty for that disjunct; verified iff all disjuncts
    infeasible. Cannot output 'falsified' (no witness extraction in this
    version; a feasible LP does not prove a counterexample exists).
    """
    _lazy_imports()
    t0 = time.time()
    onnx_path = Path(onnx_path); vnnlib_path = Path(vnnlib_path)
    try:
        sub_const, layers, output_layer = extract_layers(onnx_path)
    except Exception as e:
        return (f'fail(extract:{type(e).__name__})', time.time() - t0)
    n_in = layers[0][0].shape[0] if (sub_const is None) else sub_const.shape[0]
    if sub_const is None:
        # Infer n_in from layer-0 weight orientation
        W0 = layers[0][0]; b0_dim = layers[0][1].shape[0]
        n_in = W0.shape[0] if W0.shape[1] == b0_dim else W0.shape[1]
    n_out = output_layer[1].shape[0]
    disjuncts = parse_vnnlib(vnnlib_path, n_in, n_out)
    if not disjuncts:
        return ('fail(no_disjuncts)', time.time() - t0)
    for lb_d, ub_d, _ in disjuncts:
        if not (np.isfinite(lb_d).all() and np.isfinite(ub_d).all()):
            return ('fail(unbounded_input)', time.time() - t0)
    for lb_d, ub_d, unsafe_rows in disjuncts:
        st = _verify_one_disjunct(sub_const, layers, output_layer,
                                    lb_d, ub_d, unsafe_rows,
                                    time_limit_per_lp=time_limit_per_lp)
        if st == 'unknown':
            return ('unknown', time.time() - t0)
        if st.startswith('fail'):
            return (st, time.time() - t0)
    return ('verified', time.time() - t0)
