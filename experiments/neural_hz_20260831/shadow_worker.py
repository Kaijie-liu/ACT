#!/usr/bin/env python3
"""Run one isolated dense-HZ baseline/Neural-HZ structural shadow pair arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import resource
import subprocess
import sys
import time
from pathlib import Path


def _status_name(status) -> str:
    value = getattr(status, "value", None)
    return str(value if value is not None else status).upper()


def _provenance(act_root: Path) -> dict[str, str]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=act_root, text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=act_root, text=True
    ).strip()
    digest = hashlib.sha256()
    for relative in (
        "act/back_end/solver/neural_hz.py",
        "act/back_end/solver/solver_hz.py",
        "act/back_end/hybridz_tf/hybridz_tf.py",
        "act/back_end/hybridz_tf/tf_mlp.py",
        "act/back_end/hybridz_tf/tf_cnn.py",
        "act/back_end/hybridz_tf/exact_linear_op.py",
        "act/back_end/verifier.py",
        "act/config/config.py",
        "experiments/neural_hz_20260831/shadow_worker.py",
    ):
        path = act_root / relative
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return {
        "branch": branch,
        "commit": commit,
        "candidate_sha256": digest.hexdigest(),
    }


def _operator_storage(operator) -> tuple[int, int, int, bool]:
    """Return logical nnz, resident entries/bytes, and implicit-Conv tag."""
    logical = getattr(operator, "logical_expanded_nnz", None)
    if logical is None:
        logical = operator.nnz
    resident_entries = getattr(operator, "resident_entries", None)
    if resident_entries is None:
        resident_entries = operator.nnz
    resident_bytes = getattr(operator, "resident_bytes", None)
    if resident_bytes is None:
        resident_bytes = (
            operator.data.nbytes
            + operator.indices.nbytes
            + operator.indptr.nbytes
        )
    return (
        int(logical),
        int(resident_entries),
        int(resident_bytes),
        type(operator).__name__ == "ImplicitConv2DOp",
    )


def _matrix_storage(matrix) -> tuple[int, int]:
    return (
        int(matrix.nnz),
        int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes),
    )


def _live_cache_upper_bound_ledger(tf) -> dict[str, object]:
    """Charge unique objects retained by all live HZ/lazy/precomputed caches.

    This is intentionally a conservative live-cache upper bound until
    consumer-aware cache GC is proven. It is not an allocator-RSS estimate.
    """
    expressions = {}
    hz_objects = {}
    phase_bounds = {}
    for expression in tf._sparse_affine_expr_cache.values():
        expressions[id(expression)] = expression
    for precomputed in tf._sparse_precomputed_relu.values():
        if precomputed and hasattr(precomputed[0], "Gc"):
            hz_objects[id(precomputed[0])] = precomputed[0]
        if len(precomputed) >= 4 and precomputed[3] is not None:
            expressions[id(precomputed[3])] = precomputed[3]
        if len(precomputed) >= 5 and precomputed[4] is not None:
            phase_bounds[id(precomputed[4])] = precomputed[4]
    for hz in tf._sparse_hz_cache.values():
        hz_objects[id(hz)] = hz

    operators = {}
    biases = {}
    for expression in expressions.values():
        biases[id(expression.bias)] = expression.bias
        for term in expression.terms:
            hz_objects[id(term.source)] = term.source
            for operator in term.operators:
                operators[id(operator)] = operator

    operator_logical = 0
    operator_resident_entries = 0
    operator_resident_bytes = 0
    implicit_ops = 0
    for operator in operators.values():
        logical, entries, nbytes, implicit = _operator_storage(operator)
        operator_logical += logical
        operator_resident_entries += entries
        operator_resident_bytes += nbytes
        implicit_ops += int(implicit)

    bias_entries = sum(int(value.size) for value in biases.values())
    bias_bytes = sum(int(value.nbytes) for value in biases.values())
    value_entries = 0
    value_bytes = 0
    predicate_entries = 0
    predicate_bytes = 0
    for hz in hz_objects.values():
        gc_entries, gc_bytes = _matrix_storage(hz.Gc)
        gb_entries, gb_bytes = _matrix_storage(hz.Gb)
        value_entries += int(hz.c.size) + gc_entries + gb_entries
        value_bytes += int(hz.c.nbytes) + gc_bytes + gb_bytes
        for matrix in (hz.Ac, hz.Ab, hz.Auc, hz.Aub):
            entries, nbytes = _matrix_storage(matrix)
            predicate_entries += entries
            predicate_bytes += nbytes
        predicate_entries += int(hz.b.size + hz.ub.size)
        predicate_bytes += int(hz.b.nbytes + hz.ub.nbytes)

    phase_bound_bytes = 0
    for bounds in phase_bounds.values():
        phase_bound_bytes += int(
            bounds.lb.numel() * bounds.lb.element_size()
            + bounds.ub.numel() * bounds.ub.element_size()
        )
    resident_entries = (
        operator_resident_entries
        + bias_entries
        + value_entries
        + predicate_entries
    )
    resident_bytes = (
        operator_resident_bytes
        + bias_bytes
        + value_bytes
        + predicate_bytes
        + phase_bound_bytes
    )
    return {
        "scope": "unique_live_cache_upper_bound",
        "allocator_overhead_included": False,
        "expression_objects": int(len(expressions)),
        "operator_objects": int(len(operators)),
        "implicit_conv_operator_objects": int(implicit_ops),
        "hz_objects": int(len(hz_objects)),
        "phase_bound_objects": int(len(phase_bounds)),
        "operator_logical_expanded_nnz": int(operator_logical),
        "operator_resident_entries": int(operator_resident_entries),
        "operator_resident_bytes": int(operator_resident_bytes),
        "expression_bias_entries": int(bias_entries),
        "expression_bias_bytes": int(bias_bytes),
        "hz_value_entries": int(value_entries),
        "hz_value_bytes": int(value_bytes),
        "hz_predicate_entries": int(predicate_entries),
        "hz_predicate_bytes": int(predicate_bytes),
        "phase_bound_bytes": int(phase_bound_bytes),
        "resident_entries_excluding_phase_bounds": int(resident_entries),
        "resident_bytes": int(resident_bytes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bench")
    parser.add_argument("iid", type=int)
    parser.add_argument(
        "--arm",
        choices=(
            "baseline",
            "projection",
            "phase",
            "neural",
            "compact",
            "fill_aware",
            "duplicate_census",
            "proportional_census",
            "signed_proportional_census",
            "share_relu",
            "share_signed_relu",
            "share_signed_compact_relu",
            "signed_cancellation_census",
            "signed_cancellation",
            "signed_cancellation_mixed",
            "signed_cancellation_pairs",
            "signed_cancellation_two_pairs",
            "signed_cancellation_width_adaptive",
            "sparse_affine_nnz_census",
            "sparse_relu_nnz_census",
            "sparse_conv_csr_census",
            "sparse_compact_csr_census",
            "sparse_deferred_relu_census",
            "sparse_lazy_dag_census",
            "sparse_frontier_rebase_census",
            "sparse_phase_separated_relu_census",
            "sparse_phase_selective_relu_census",
            "sparse_phase_implicit_relu_census",
        ),
        required=True,
    )
    parser.add_argument("--act-root", type=Path, required=True)
    parser.add_argument("--bench-root", type=Path, required=True)
    parser.add_argument("--vnnlib-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-hz-checkpoint", type=Path, default=None)
    parser.add_argument("--solver-timeout", type=float, default=15.0)
    parser.add_argument("--memory-gb", type=float, default=16.0)
    parser.add_argument("--sparse-entry-limit", type=int, default=64_000_000)
    parser.add_argument("--debug-structure", action="store_true")
    parser.add_argument(
        "--stop-after-layer",
        type=int,
        default=None,
        help="structural census only: stop after this exact ACT layer id",
    )
    parser.add_argument(
        "--representation",
        choices=("dense", "sparse"),
        default="dense",
    )
    args = parser.parse_args()

    if args.sparse_entry_limit <= 0:
        raise ValueError("sparse entry limit must be positive")

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite shadow result: {args.output}")
    if "/data1/Kane/HyZor" in str(args.output.resolve()):
        raise ValueError("historical /data1/Kane/HyZor results are read-only")
    if args.save_hz_checkpoint is not None:
        if args.save_hz_checkpoint.exists():
            raise FileExistsError(
                f"refusing to overwrite HZ checkpoint: {args.save_hz_checkpoint}"
            )
        if "/data1/Kane/HyZor" in str(args.save_hz_checkpoint.resolve()):
            raise ValueError("historical /data1/Kane/HyZor results are read-only")
        args.save_hz_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    cap = int(args.memory_gb * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    sys.path.insert(0, str(args.act_root))

    record: dict[str, object] = {
        "schema": "neural_hz_shadow_v1",
        "formal_baseline_solved": 1870,
        "formal_baseline_total": 2413,
        "bench": args.bench,
        "iid": args.iid,
        "arm": args.arm,
        "verdict": "ERROR",
        "solver_timeout_s": args.solver_timeout,
        "memory_gb": args.memory_gb,
        "sparse_entry_limit": args.sparse_entry_limit,
        "representation": args.representation,
        "stop_after_layer": args.stop_after_layer,
        "hz_checkpoint": (
            None
            if args.save_hz_checkpoint is None
            else str(args.save_hz_checkpoint)
        ),
        "provenance": _provenance(args.act_root),
    }
    started = time.monotonic()
    try:
        import torch
        import act.back_end.hybridz_tf.tf_mlp as hz_mlp
        from act.back_end.core import Bounds as ABounds
        from act.back_end.solver.solver_hz import HZSolver, sparse_hz_fast_bounds
        from act.back_end.transfer_functions import (
            get_transfer_function,
            set_solver_mode,
            set_transfer_function_mode,
        )
        from act.front_end.spec_creator_base import LabeledInputTensor
        from act.front_end.verifiable_model import (
            InputLayer,
            InputSpecLayer,
            OutputSpecLayer,
            VerifiableModel,
        )
        from act.front_end.vnnlib_loader.onnx_converter import (
            convert_onnx_to_pytorch,
            get_onnx_input_shape,
        )
        from act.front_end.vnnlib_loader.vnnlib_parser import parse_vnnlib_queries
        from act.pipeline.verification.torch2act import TorchToACT

        root = args.bench_root / args.bench
        rows = [
            line.split(",")
            for line in (root / "instances.csv").read_text().splitlines()
            if line.strip()
        ]
        model_path = root / rows[args.iid][0].replace("./", "")
        spec_path = args.vnnlib_root / args.bench / rows[args.iid][1].replace("./", "")
        input_shape = tuple(get_onnx_input_shape(model_path))
        model = convert_onnx_to_pytorch(model_path).eval()
        model_dtype = next(model.parameters()).dtype
        labeled = LabeledInputTensor(
            tensor=torch.zeros(input_shape, dtype=model_dtype),
            label=torch.tensor([0]),
        )
        queries = parse_vnnlib_queries(spec_path, labeled_tensor=labeled)
        if args.save_hz_checkpoint is not None and len(queries) != 1:
            raise ValueError("HZ checkpointing requires exactly one query")
        for input_spec, _ in queries:
            input_spec.lb = input_spec.lb.to(dtype=model_dtype)
            input_spec.ub = input_spec.ub.to(dtype=model_dtype)

        # The structural trial is layered on the already-audited projected
        # exact-ReLU representation; only the final predicate projection varies.
        hz_mlp.hz_apply_relu._rq4_original = False
        set_transfer_function_mode("hybridz")
        set_solver_mode("hybridz")
        tf = get_transfer_function()
        tf._SPARSE_MAX_AFFINE_CELLS = int(args.sparse_entry_limit)
        tf._rq4_original_relu = False
        tf._neural_hz_compact_relu = (
            "fill_aware"
            if args.arm == "fill_aware"
            else args.arm in ("compact", "sparse_compact_csr_census")
        )
        tf._neural_hz_duplicate_census = args.arm == "duplicate_census"
        tf._neural_hz_proportional_census = args.arm == "proportional_census"
        tf._neural_hz_signed_proportional_census = (
            args.arm == "signed_proportional_census"
        )
        tf._neural_hz_share_relu = args.arm == "share_relu"
        tf._neural_hz_share_signed_relu = args.arm == "share_signed_relu"
        tf._neural_hz_share_signed_compact_relu = (
            args.arm in (
                "share_signed_compact_relu",
                "signed_cancellation_census",
                "signed_cancellation",
                "signed_cancellation_mixed",
                "signed_cancellation_pairs",
                "signed_cancellation_two_pairs",
                "signed_cancellation_width_adaptive",
            )
        )
        tf._neural_hz_signed_cancellation_census = (
            args.arm == "signed_cancellation_census"
        )
        tf._neural_hz_signed_cancellation = args.arm in (
            "signed_cancellation",
            "signed_cancellation_mixed",
            "signed_cancellation_pairs",
            "signed_cancellation_two_pairs",
            "signed_cancellation_width_adaptive",
        )
        tf._neural_hz_signed_cancellation_mixed_only = (
            args.arm == "signed_cancellation_mixed"
        )
        tf._neural_hz_signed_cancellation_pairs_only = (
            args.arm == "signed_cancellation_pairs"
        )
        tf._neural_hz_signed_cancellation_elimination_max_cardinality = (
            4 if args.arm == "signed_cancellation_two_pairs" else 0
        )
        tf._neural_hz_signed_cancellation_two_pair_min_outputs = (
            8192 if args.arm == "signed_cancellation_width_adaptive" else 0
        )
        tf._neural_hz_sparse_affine_nnz_guard = (
            args.arm in (
                "sparse_affine_nnz_census",
                "sparse_relu_nnz_census",
                "sparse_conv_csr_census",
                "sparse_compact_csr_census",
                "sparse_deferred_relu_census",
                "sparse_lazy_dag_census",
                "sparse_frontier_rebase_census",
                "sparse_phase_separated_relu_census",
                "sparse_phase_selective_relu_census",
                "sparse_phase_implicit_relu_census",
            )
        )
        tf._neural_hz_sparse_relu_nnz_guard = (
            args.arm in (
                "sparse_relu_nnz_census",
                "sparse_conv_csr_census",
                "sparse_compact_csr_census",
                "sparse_deferred_relu_census",
                "sparse_lazy_dag_census",
                "sparse_frontier_rebase_census",
                "sparse_phase_separated_relu_census",
                "sparse_phase_selective_relu_census",
                "sparse_phase_implicit_relu_census",
            )
        )
        tf._neural_hz_sparse_conv_csr_builder = (
            args.arm in (
                "sparse_conv_csr_census",
                "sparse_compact_csr_census",
                "sparse_deferred_relu_census",
                "sparse_lazy_dag_census",
                "sparse_frontier_rebase_census",
                "sparse_phase_separated_relu_census",
                "sparse_phase_selective_relu_census",
                "sparse_phase_implicit_relu_census",
            )
        )
        tf._neural_hz_sparse_deferred_relu_materialization = (
            args.arm in (
                "sparse_deferred_relu_census",
                "sparse_lazy_dag_census",
                "sparse_frontier_rebase_census",
                "sparse_phase_separated_relu_census",
                "sparse_phase_selective_relu_census",
                "sparse_phase_implicit_relu_census",
            )
        )
        tf._neural_hz_sparse_lazy_affine_dag = (
            args.arm in (
                "sparse_lazy_dag_census",
                "sparse_frontier_rebase_census",
                "sparse_phase_separated_relu_census",
                "sparse_phase_selective_relu_census",
                "sparse_phase_implicit_relu_census",
            )
        )
        tf._neural_hz_sparse_frontier_image_rebase = (
            args.arm in (
                "sparse_frontier_rebase_census",
                "sparse_phase_separated_relu_census",
                "sparse_phase_selective_relu_census",
                "sparse_phase_implicit_relu_census",
            )
        )
        tf._neural_hz_sparse_phase_separated_relu = (
            args.arm in (
                "sparse_phase_separated_relu_census",
                "sparse_phase_selective_relu_census",
                "sparse_phase_implicit_relu_census",
            )
        )
        tf._neural_hz_sparse_phase_selective_materialization = (
            args.arm in (
                "sparse_phase_selective_relu_census",
                "sparse_phase_implicit_relu_census",
            )
        )
        tf._neural_hz_sparse_implicit_conv_dag = (
            args.arm == "sparse_phase_implicit_relu_census"
        )
        if args.representation == "dense":
            tf.get_sparse_hz = lambda _layer_id: None

        statuses: list[str] = []
        metadata: list[dict[str, object]] = []
        propagation_s = 0.0
        solver_s = 0.0
        final_hz = None
        concrete_validations: list[dict[str, object]] = []
        for input_spec, output_spec in queries:
            net = TorchToACT(
                VerifiableModel(
                    input_layer=InputLayer(
                        labeled_input=labeled,
                        shape=input_shape,
                        dtype=model_dtype,
                    ),
                    input_spec=InputSpecLayer(input_spec),
                    model=model,
                    output_spec=OutputSpecLayer(output_spec),
                )
            ).run()
            bounds = ABounds(input_spec.lb.detach().clone(), input_spec.ub.detach().clone())
            tf._neural_hz_compact_factors = 0
            tf._hz_cache.clear()
            tf._sparse_hz_cache.clear()
            tf._cache_net_id = None
            tf._neural_hz_sparse_compact_factors = 0
            tf._neural_hz_duplicate_rows = 0
            tf._neural_hz_duplicate_groups = 0
            tf._neural_hz_duplicate_max_group = 1
            tf._neural_hz_proportional_rows = 0
            tf._neural_hz_proportional_groups = 0
            tf._neural_hz_proportional_max_group = 1
            tf._neural_hz_signed_proportional_rows = 0
            tf._neural_hz_signed_proportional_groups = 0
            tf._neural_hz_signed_proportional_max_group = 1
            tf._neural_hz_signed_duplicate_rows = 0
            tf._neural_hz_signed_duplicate_groups = 0
            tf._neural_hz_signed_duplicate_max_group = 1
            tf._neural_hz_shared_relu_rows = 0
            tf._neural_hz_signed_shared_relu_rows = 0
            tf._neural_hz_signed_compact_layers = 0
            tf._neural_hz_cancellation_rows = 0
            tf._neural_hz_cancellation_groups = 0
            tf._neural_hz_cancellation_max_group = 1
            tf._neural_hz_eliminated_cancellation_groups = 0
            tf._neural_hz_cancellation_profile = []
            tf._neural_hz_deferred_relu_layers = 0
            tf._neural_hz_deferred_zero_rows = 0
            tf._neural_hz_lazy_affine_layers = 0
            tf._neural_hz_lazy_affine_materializations = 0
            tf._neural_hz_lazy_checkpoints = 0
            tf._neural_hz_frontier_rebases = 0
            tf._neural_hz_frontier_rebase_profile = []
            tf._neural_hz_phase_separated_relus = 0
            tf._neural_hz_phase_separated_profile = []
            tf._neural_hz_phase_selective_relus = 0
            tf._neural_hz_phase_selective_profile = []
            tf._neural_hz_linear_op_arena.clear()
            tf._neural_hz_implicit_conv_ops = 0
            tf._neural_hz_implicit_conv_profile = []
            after = {}
            census_stop_reached = False
            tick = time.monotonic()
            for layer in net.layers:
                predecessors = net.preds.get(layer.id, [])
                incoming = bounds if layer.id == 0 or not predecessors else after[predecessors[0]].bounds
                after[layer.id] = tf.apply(layer, incoming, net, {}, after)
                if args.representation == "dense":
                    tf._sparse_hz_cache.clear()
                if (
                    args.stop_after_layer is not None
                    and int(layer.id) == int(args.stop_after_layer)
                ):
                    census_stop_reached = True
                    break
            propagation_s += time.monotonic() - tick
            record["census_stop_reached"] = census_stop_reached

            assert_layer = net.layers[-1]
            output_layer_id = net.preds[assert_layer.id][0]
            final_hz = (
                tf.get_hz(output_layer_id)
                if args.representation == "dense"
                else tf.get_sparse_hz(output_layer_id)
            )
            input_hz = None
            if args.representation == "sparse" and final_hz is not None:
                for layer in net.layers:
                    candidate = tf.get_sparse_hz(layer.id)
                    if (
                        candidate is not None
                        and candidate.frame_id == final_hz.frame_id
                        and candidate.n_out == int(bounds.lb.numel())
                    ):
                        input_hz = candidate
                        break
            if args.debug_structure:
                def debug_layer(layer):
                    sparse = tf.get_sparse_hz(layer.id)
                    lazy = tf._sparse_affine_expr_cache.get(int(layer.id))
                    lazy_operators = {}
                    if lazy is not None:
                        for term in lazy.terms:
                            for operator in term.operators:
                                lazy_operators[id(operator)] = operator
                    lazy_storage = [
                        _operator_storage(operator)
                        for operator in lazy_operators.values()
                    ]
                    data = {
                        "id": int(layer.id),
                        "type": type(layer).__name__,
                        "kind": str(layer.kind),
                        "successors": [
                            int(value) for value in net.succs.get(layer.id, [])
                        ],
                        "weight_shape": (
                            None
                            if not isinstance(layer.params.get("weight"), torch.Tensor)
                            else list(layer.params["weight"].shape)
                        ),
                        "sparse_cached": sparse is not None,
                        "sparse_drop_reason": tf._sparse_drop_reasons.get(layer.id),
                        "lazy_affine_cached": lazy is not None,
                        "lazy_terms": (
                            None if lazy is None else int(len(lazy.terms))
                        ),
                        "lazy_operator_entries": (
                            None
                            if lazy is None
                            else int(
                                lazy.bias.size
                                + sum(item[0] for item in lazy_storage)
                            )
                        ),
                        "lazy_operator_resident_entries": (
                            None
                            if lazy is None
                            else int(
                                lazy.bias.size
                                + sum(item[1] for item in lazy_storage)
                            )
                        ),
                        "lazy_operator_resident_bytes": (
                            None
                            if lazy is None
                            else int(
                                lazy.bias.nbytes
                                + sum(item[2] for item in lazy_storage)
                            )
                        ),
                        "lazy_implicit_conv_operators": (
                            None
                            if lazy is None
                            else int(sum(item[3] for item in lazy_storage))
                        ),
                        "n_out": (
                            None
                            if sparse is None
                            else int(sparse.n_out)
                        ),
                        "frame_id": (
                            None
                            if sparse is None
                            else sparse.frame_id
                        ),
                        "n_cont": (
                            None if sparse is None else int(sparse.n_cont)
                        ),
                        "n_bin": (
                            None if sparse is None else int(sparse.n_bin)
                        ),
                        "n_eq": (
                            None if sparse is None else int(sparse.n_eq)
                        ),
                        "n_ineq": (
                            None if sparse is None else int(sparse.n_ineq)
                        ),
                        "storage_entries": (
                            None
                            if sparse is None
                            else int(tf._sparse_storage_entries(sparse))
                        ),
                        "gc_nnz": (
                            None if sparse is None else int(sparse.Gc.nnz)
                        ),
                        "gb_nnz": (
                            None if sparse is None else int(sparse.Gb.nnz)
                        ),
                        "ac_nnz": (
                            None if sparse is None else int(sparse.Ac.nnz)
                        ),
                        "ab_nnz": (
                            None if sparse is None else int(sparse.Ab.nnz)
                        ),
                        "auc_nnz": (
                            None if sparse is None else int(sparse.Auc.nnz)
                        ),
                        "aub_nnz": (
                            None if sparse is None else int(sparse.Aub.nnz)
                        ),
                    }
                    if str(layer.kind) == "RELU":
                        predecessors = net.preds.get(layer.id, [])
                        interval_bounds = (
                            None
                            if (
                                not predecessors
                                or predecessors[0] not in after
                            )
                            else after[predecessors[0]].bounds
                        )
                        if interval_bounds is not None:
                            interval_lower = interval_bounds.lb.reshape(-1)
                            interval_upper = interval_bounds.ub.reshape(-1)
                            data.update(
                                {
                                    "interval_stable_positive": int(
                                        (interval_lower >= 0).sum()
                                    ),
                                    "interval_stable_negative": int(
                                        (interval_upper <= 0).sum()
                                    ),
                                    "interval_unstable": int(
                                        (
                                            (interval_lower < 0)
                                            & (interval_upper > 0)
                                        ).sum()
                                    ),
                                }
                            )
                        source = (
                            None
                            if not predecessors
                            else tf.get_sparse_hz(predecessors[0])
                        )
                        if source is not None:
                            hz_bounds = sparse_hz_fast_bounds(source)
                            lower = torch.maximum(
                                interval_bounds.lb.reshape(-1),
                                hz_bounds.lb.reshape(-1),
                            )
                            upper = torch.minimum(
                                interval_bounds.ub.reshape(-1),
                                hz_bounds.ub.reshape(-1),
                            )
                            unstable = (lower < 0) & (upper > 0)
                            unstable_rows = (
                                unstable.nonzero(as_tuple=False)
                                .reshape(-1)
                                .cpu()
                                .numpy()
                            )
                            data.update(
                                {
                                    "relu_source_n_out": int(source.n_out),
                                    "relu_stable_positive": int((lower >= 0).sum()),
                                    "relu_stable_negative": int((upper <= 0).sum()),
                                    "relu_unstable": int(unstable.sum()),
                                    "relu_source_gc_nnz": int(source.Gc.nnz),
                                    "relu_source_gb_nnz": int(source.Gb.nnz),
                                    "relu_unstable_gc_nnz": int(
                                        source.Gc[unstable_rows].nnz
                                    ),
                                    "relu_unstable_gb_nnz": int(
                                        source.Gb[unstable_rows].nnz
                                    ),
                                }
                            )
                    return data

                record["layer_hz_structure"] = [
                    debug_layer(layer) for layer in net.layers
                ]
            record.setdefault("live_cache_ledgers", []).append(
                _live_cache_upper_bound_ledger(tf)
            )
            record["output_hz_exact"] = bool(
                getattr(final_hz, "exact", False)
            )
            record["input_hz_available"] = input_hz is not None
            record["output_frame_id"] = getattr(final_hz, "frame_id", None)
            record["input_frame_id"] = getattr(input_hz, "frame_id", None)
            if (
                args.save_hz_checkpoint is not None
                and final_hz is not None
                and bool(getattr(final_hz, "exact", False))
            ):
                temporary_checkpoint = args.save_hz_checkpoint.with_name(
                    args.save_hz_checkpoint.name + f".tmp.{os.getpid()}"
                )
                checkpoint = {
                    "schema": "neural_hz_checkpoint_v1",
                    "bench": args.bench,
                    "iid": args.iid,
                    "provenance": record["provenance"],
                    "input_shape": tuple(int(value) for value in bounds.lb.shape),
                    "input_hz": input_hz,
                    "final_hz": final_hz,
                }
                with temporary_checkpoint.open("xb") as handle:
                    pickle.dump(checkpoint, handle, protocol=5)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_checkpoint, args.save_hz_checkpoint)
                record["hz_checkpoint_written"] = True
            solver = HZSolver(
                time_limit=args.solver_timeout,
                tolerance=1e-7,
                neural_hz_projection=args.arm in ("projection", "neural"),
                neural_hz_phase_fixing=args.arm in ("phase", "neural"),
            )
            tick = time.monotonic()
            verdicts = solver.evaluate_spec(
                final_hz,
                output_spec,
                batch_size=1,
                n_out=len(assert_layer.in_vars),
                input_hz=input_hz,
                input_shape=tuple(bounds.lb.shape),
                timelimit=args.solver_timeout,
            )
            solver_s += time.monotonic() - tick
            record["compact_relu_factors"] = int(
                record.get("compact_relu_factors", 0)
            ) + int(
                getattr(tf, "_neural_hz_compact_factors", 0)
                if args.representation == "dense"
                else getattr(tf, "_neural_hz_sparse_compact_factors", 0)
            )
            record["duplicate_relu_rows"] = int(
                record.get("duplicate_relu_rows", 0)
            ) + int(getattr(tf, "_neural_hz_duplicate_rows", 0))
            record["duplicate_relu_groups"] = int(
                record.get("duplicate_relu_groups", 0)
            ) + int(getattr(tf, "_neural_hz_duplicate_groups", 0))
            record["duplicate_relu_max_group"] = max(
                int(record.get("duplicate_relu_max_group", 1)),
                int(getattr(tf, "_neural_hz_duplicate_max_group", 1)),
            )
            record["shared_relu_rows"] = int(
                record.get("shared_relu_rows", 0)
            ) + int(getattr(tf, "_neural_hz_shared_relu_rows", 0))
            record["signed_shared_relu_rows"] = int(
                record.get("signed_shared_relu_rows", 0)
            ) + int(getattr(tf, "_neural_hz_signed_shared_relu_rows", 0))
            record["signed_compact_layers"] = int(
                record.get("signed_compact_layers", 0)
            ) + int(getattr(tf, "_neural_hz_signed_compact_layers", 0))
            record["cancellation_relu_rows"] = int(
                record.get("cancellation_relu_rows", 0)
            ) + int(getattr(tf, "_neural_hz_cancellation_rows", 0))
            record["cancellation_relu_groups"] = int(
                record.get("cancellation_relu_groups", 0)
            ) + int(getattr(tf, "_neural_hz_cancellation_groups", 0))
            record["cancellation_relu_max_group"] = max(
                int(record.get("cancellation_relu_max_group", 1)),
                int(getattr(tf, "_neural_hz_cancellation_max_group", 1)),
            )
            record["eliminated_cancellation_groups"] = int(
                record.get("eliminated_cancellation_groups", 0)
            ) + int(
                getattr(tf, "_neural_hz_eliminated_cancellation_groups", 0)
            )
            record.setdefault("cancellation_profile", []).extend(
                getattr(tf, "_neural_hz_cancellation_profile", [])
            )
            record["deferred_relu_layers"] = int(
                record.get("deferred_relu_layers", 0)
            ) + int(getattr(tf, "_neural_hz_deferred_relu_layers", 0))
            record["deferred_zero_rows"] = int(
                record.get("deferred_zero_rows", 0)
            ) + int(getattr(tf, "_neural_hz_deferred_zero_rows", 0))
            record["lazy_affine_layers"] = int(
                record.get("lazy_affine_layers", 0)
            ) + int(getattr(tf, "_neural_hz_lazy_affine_layers", 0))
            record["lazy_affine_materializations"] = int(
                record.get("lazy_affine_materializations", 0)
            ) + int(
                getattr(tf, "_neural_hz_lazy_affine_materializations", 0)
            )
            record["lazy_checkpoints"] = int(
                record.get("lazy_checkpoints", 0)
            ) + int(getattr(tf, "_neural_hz_lazy_checkpoints", 0))
            record["frontier_rebases"] = int(
                record.get("frontier_rebases", 0)
            ) + int(getattr(tf, "_neural_hz_frontier_rebases", 0))
            record.setdefault("frontier_rebase_profile", []).extend(
                getattr(tf, "_neural_hz_frontier_rebase_profile", [])
            )
            record["phase_separated_relus"] = int(
                record.get("phase_separated_relus", 0)
            ) + int(getattr(tf, "_neural_hz_phase_separated_relus", 0))
            record.setdefault("phase_separated_profile", []).extend(
                getattr(tf, "_neural_hz_phase_separated_profile", [])
            )
            record["phase_selective_relus"] = int(
                record.get("phase_selective_relus", 0)
            ) + int(getattr(tf, "_neural_hz_phase_selective_relus", 0))
            record.setdefault("phase_selective_profile", []).extend(
                getattr(tf, "_neural_hz_phase_selective_profile", [])
            )
            record["implicit_conv_ops"] = int(
                record.get("implicit_conv_ops", 0)
            ) + int(getattr(tf, "_neural_hz_implicit_conv_ops", 0))
            record.setdefault("implicit_conv_profile", []).extend(
                getattr(tf, "_neural_hz_implicit_conv_profile", [])
            )
            record["consumer_gc_released_sparse_states"] = int(
                record.get("consumer_gc_released_sparse_states", 0)
            ) + int(
                getattr(tf, "_neural_hz_released_sparse_states", 0)
            )
            record["proportional_relu_rows"] = int(
                record.get("proportional_relu_rows", 0)
            ) + int(getattr(tf, "_neural_hz_proportional_rows", 0))
            record["proportional_relu_groups"] = int(
                record.get("proportional_relu_groups", 0)
            ) + int(getattr(tf, "_neural_hz_proportional_groups", 0))
            record["proportional_relu_max_group"] = max(
                int(record.get("proportional_relu_max_group", 1)),
                int(getattr(tf, "_neural_hz_proportional_max_group", 1)),
            )
            record["signed_proportional_relu_rows"] = int(
                record.get("signed_proportional_relu_rows", 0)
            ) + int(getattr(tf, "_neural_hz_signed_proportional_rows", 0))
            record["signed_proportional_relu_groups"] = int(
                record.get("signed_proportional_relu_groups", 0)
            ) + int(getattr(tf, "_neural_hz_signed_proportional_groups", 0))
            record["signed_proportional_relu_max_group"] = max(
                int(record.get("signed_proportional_relu_max_group", 1)),
                int(
                    getattr(tf, "_neural_hz_signed_proportional_max_group", 1)
                ),
            )
            record["signed_duplicate_relu_rows"] = int(
                record.get("signed_duplicate_relu_rows", 0)
            ) + int(getattr(tf, "_neural_hz_signed_duplicate_rows", 0))
            record["signed_duplicate_relu_groups"] = int(
                record.get("signed_duplicate_relu_groups", 0)
            ) + int(getattr(tf, "_neural_hz_signed_duplicate_groups", 0))
            record["signed_duplicate_relu_max_group"] = max(
                int(record.get("signed_duplicate_relu_max_group", 1)),
                int(getattr(tf, "_neural_hz_signed_duplicate_max_group", 1)),
            )
            for verdict in verdicts:
                status = _status_name(verdict.status)
                validation: dict[str, object] | None = None
                if status == "FALSIFIED":
                    validation = {
                        "solver_status": status,
                        "valid": False,
                        "reason": "missing_counterexample",
                    }
                    counterexample = verdict.counterexample
                    if counterexample is not None:
                        candidate = counterexample.detach().to(dtype=model_dtype)
                        if candidate.shape != bounds.lb.shape:
                            candidate = candidate.reshape(bounds.lb.shape)
                        input_ok = bool(
                            torch.all(candidate >= input_spec.lb).item()
                            and torch.all(candidate <= input_spec.ub).item()
                        )
                        with torch.no_grad():
                            concrete_output = model(candidate).reshape(1, -1)
                        encoded = output_spec.encode_linear(
                            B=1,
                            n_out=concrete_output.shape[1],
                            device=torch.device("cpu"),
                            dtype=torch.float64,
                        )
                        coefficients = encoded["C"].detach().cpu().double()
                        thresholds = encoded["thresholds"].detach().cpu().double()
                        margins = (
                            coefficients @ concrete_output.detach().cpu().double().reshape(-1)
                        ).reshape(1, int(encoded["M"]))
                        kind_name = getattr(encoded["kind"], "name", str(encoded["kind"]))
                        if "UNSAFE_LINEAR" in kind_name.upper():
                            violates = bool(torch.all(margins <= thresholds).item())
                        else:
                            violates = bool(torch.any(margins >= thresholds).item())
                        valid = input_ok and violates
                        validation = {
                            "solver_status": status,
                            "valid": valid,
                            "reason": "concrete_network_violation" if valid else "invalid_witness",
                            "input_ok": input_ok,
                            "violates": violates,
                            "counterexample": candidate.detach().cpu().reshape(-1).tolist(),
                            "output": concrete_output.detach().cpu().reshape(-1).tolist(),
                            "margins": margins.reshape(-1).tolist(),
                            "thresholds": thresholds.reshape(-1).tolist(),
                        }
                    if not bool(validation["valid"]):
                        status = "UNKNOWN"
                        verdict.metadata["reason"] = "invalid_concrete_witness"
                if validation is not None:
                    concrete_validations.append(validation)
                statuses.append(status)
                metadata.append(dict(verdict.metadata))

        record.update(
            {
                "verdict": (
                    "CERT"
                    if statuses and all(s == "CERTIFIED" for s in statuses)
                    else "ADV"
                    if any(s == "FALSIFIED" for s in statuses)
                    else "UNKNOWN"
                ),
                "statuses": statuses,
                "metadata": metadata,
                "n_queries": len(queries),
                "n_g": None if final_hz is None else int(final_hz.Gc.shape[1]),
                "n_b": None if final_hz is None else int(final_hz.Gb.shape[1]),
                "n_rows": None if final_hz is None else int(final_hz.Ac.shape[0]),
                "propagation_s": round(propagation_s, 6),
                "solver_s": round(solver_s, 6),
                "verify_s": round(propagation_s + solver_s, 6),
                "concrete_validations": concrete_validations,
            }
        )
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["worker_wall_s"] = round(time.monotonic() - started, 6)
    payload = json.dumps(record, sort_keys=True) + "\n"
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
