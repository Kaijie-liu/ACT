#!/usr/bin/env python3
"""
ACT Pipeline Command-Line Interface.

Provides fuzzing capabilities for neural network verification with support for:
- VNNLib verification benchmarks (default)
- TorchVision datasets (alternative)

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import logging
from pathlib import Path
from typing import Any, List, Optional
import sys
import torch

from act.util.cli_utils import add_device_args, initialize_from_args
from act.back_end.config import VALID_SOLVER_TIERS

logger = logging.getLogger(__name__)
from act.front_end.specs import OutputSpec
from act.front_end.spec_creator_base import LabeledInputTensor
from act.front_end.vnnlib_loader.create_specs import VNNLibSpecCreator
from act.front_end.vnnlib_loader import data_model_loader as vnnlib_loader
from act.front_end.vnnlib_loader import category_mapping as vnnlib_mapping
from act.front_end.torchvision_loader.create_specs import TorchVisionSpecCreator
from act.front_end.torchvision_loader import data_model_loader as tv_loader
from act.front_end.torchvision_loader import data_model_mapping as tv_mapping
from act.front_end.model_synthesis import synthesize_models_from_specs
from act.pipeline.fuzzing.actfuzzer import ACTFuzzer, FuzzingConfig, FuzzingReport
from act.pipeline.verification.per_neuron_bounds import PerNeuronCheckConfig


# -----------------------------------------------------------------------------
# Per-neuron bounds validation settings (Level 2)
#
# Zero-tolerance check: any concrete activation outside [lb, ub] is reported as
# unsoundness. 
# -----------------------------------------------------------------------------


def print_header():
    """Print simple header."""
    print(f"\n{'=' * 80}")
    print(f"ACT: Abstract Constraint Transformer")
    print(f"Inference-based whitebox fuzzing for neural network verification")
    print(f"{'=' * 80}\n")


# ============================================================================
# Data-Model Pair Management Commands
# ============================================================================


def cmd_list_available(creator: str):
    """List available datasets/categories."""
    print(f"\n{'=' * 80}")
    print(f"AVAILABLE DATA-MODEL PAIRS ({creator.upper()})")
    print(f"{'=' * 80}\n")

    if creator == "vnnlib":
        categories = vnnlib_mapping.list_categories()
        print(f"VNNLIB Categories ({len(categories)}):")
        print("-" * 80)
        for cat_name in sorted(categories):
            info = vnnlib_mapping.get_category_info(cat_name)
            print(f"  {cat_name:30s} ({info['type']}) - {info['description']}")
            print(f"    └─ Models: {info['models']}, Properties: {info['properties']}")

    elif creator == "torchvision":
        datasets = sorted(tv_mapping.DATASET_MODEL_MAPPING.keys())
        print(f"TorchVision Datasets ({len(datasets)}):")
        print("-" * 80)
        for ds_name in datasets:
            info = tv_mapping.DATASET_MODEL_MAPPING[ds_name]
            models = info.get("models", [])
            print(f"  {ds_name:30s} [{info.get('category', 'N/A')}]")
            if models:
                print(
                    f"    └─ Models: {', '.join(models[:5])}{'...' if len(models) > 5 else ''}"
                )

    print(f"\n{'=' * 80}\n")


def cmd_search(query: str, creator: str):
    """Search for datasets/categories."""
    print(f"\n{'=' * 80}")
    print(f"SEARCH RESULTS: '{query}' ({creator.upper()})")
    print(f"{'=' * 80}\n")

    if creator == "vnnlib":
        matches = vnnlib_mapping.search_categories(query)
        if matches:
            print(f"Found {len(matches)} VNNLIB categories:")
            print("-" * 80)
            for cat_name in sorted(matches):
                info = vnnlib_mapping.get_category_info(cat_name)
                print(f"  {cat_name:30s} ({info['type']}) - {info['description']}")
        else:
            print(f"No VNNLIB categories found for '{query}'")

    elif creator == "torchvision":
        matches = tv_mapping.search_datasets(query)
        if matches:
            print(f"Found {len(matches)} TorchVision datasets:")
            print("-" * 80)
            for ds_name in sorted(matches):
                info = tv_mapping.DATASET_MODEL_MAPPING[ds_name]
                print(f"  {ds_name:30s} [{info.get('category', 'N/A')}]")
        else:
            print(f"No TorchVision datasets found for '{query}'")

    print(f"\n{'=' * 80}\n")


def cmd_info(name: str, creator: str):
    """Show detailed information about dataset/category."""
    print(f"\n{'=' * 80}")
    print(f"INFO: {name} ({creator.upper()})")
    print(f"{'=' * 80}\n")

    if creator == "vnnlib":
        try:
            info = vnnlib_mapping.get_category_info(name)
            print(f"Category: {name}")
            print(f"Type: {info['type']}")
            print(f"Year: {info['year']}")
            print(f"Description: {info['description']}")
            print(f"\nModel Information:")
            print(f"  • Models: {info['models']}")
            print(f"  • Properties: {info['properties']}")
            print(f"  • Input Dim: {info['input_dim']}")
            print(f"  • Output Dim: {info['output_dim']}")

            # Check if downloaded
            downloaded = vnnlib_loader.list_downloaded_pairs()
            matching = [p for p in downloaded if p["category"] == name]
            if matching:
                print(f"\n✓ Downloaded: {len(matching)} instances")
            else:
                print(f"\n⚠ Not downloaded (use --download {name})")
        except ValueError as e:
            print(f"Error: {e}")

    elif creator == "torchvision":
        try:
            info = tv_mapping.get_dataset_info(name)
            print(f"Dataset: {name}")
            print(f"Category: {info.get('category', 'N/A')}")
            print(f"Input Size: {info.get('input_size', 'N/A')}")
            print(f"Classes: {info.get('num_classes', 'N/A')}")

            models = info.get("models", [])
            if models:
                print(f"\nRecommended Models ({len(models)}):")
                for model in models:
                    print(f"  • {model}")

            # Check if downloaded
            downloaded = tv_loader.list_downloaded_pairs()
            matching = [p for p in downloaded if p["dataset"] == name]
            if matching:
                print(f"\n✓ Downloaded: {len(matching)} model pairs")
            else:
                print(
                    f"\n⚠ Not downloaded (use --download {name} --creator torchvision)"
                )
        except ValueError as e:
            print(f"Error: {e}")

    print(f"\n{'=' * 80}\n")


def cmd_download(name: str, creator: str):
    """Download dataset/category."""
    print(f"\n{'=' * 80}")
    print(f"DOWNLOADING: {name} ({creator.upper()})")
    print(f"{'=' * 80}\n")

    if creator == "vnnlib":
        try:
            result = vnnlib_loader.download_vnnlib_category(name)

            if result["status"] == "success":
                print(f"✓ Successfully downloaded: {name}")
                print(f"  Location: {result['category_path']}")
                print(f"  Instances: {result['num_instances']}")
            else:
                print(f"✗ Download failed: {result['message']}")
                print(
                    f"\nNote: VNNLIB benchmarks must be downloaded manually from VNN-COMP."
                )
                print(f"Expected location: data/vnnlib/{name}/")
                print(f"\nManual steps:")
                print(
                    f"  1. Visit: https://github.com/ChristopherBrix/vnncomp_benchmarks"
                )
                print(f"  2. Download '{name}' benchmark")
                print(f"  3. Extract to: data/vnnlib/{name}/")
                print(f"  4. Ensure structure:")
                print(f"     - onnx/         (ONNX model files)")
                print(f"     - vnnlib/       (VNNLIB property files)")
                print(f"     - instances.csv (benchmark instances)")
        except Exception as e:
            print(f"✗ Download error: {e}")

    elif creator == "torchvision":
        try:
            info = tv_mapping.get_dataset_info(name)
            models = info.get("models", [])

            if not models:
                print(f"⚠ No models available for {name}")
                return

            print(f"Downloading {name} with {len(models)} models...\n")

            success_count = 0
            for model in models:
                result = tv_loader.download_dataset_model_pair(name, model)
                if result["status"] == "success":
                    print(f"✓ {name} + {model}")
                    success_count += 1
                else:
                    print(f"✗ {name} + {model} - {result['message']}")

            print(f"\n{'=' * 80}")
            print(f"Downloaded {success_count}/{len(models)} model pairs")
            print(f"{'=' * 80}")
        except Exception as e:
            print(f"✗ Download error: {e}")

    print()


def cmd_list_downloaded(creator: str):
    """List downloaded data-model pairs."""
    print(f"\n{'=' * 80}")
    print(f"DOWNLOADED DATA-MODEL PAIRS ({creator.upper()})")
    print(f"{'=' * 80}\n")

    if creator == "vnnlib":
        downloaded = vnnlib_loader.list_downloaded_pairs()
        if downloaded:
            # Group by category
            categories = {}
            for item in downloaded:
                cat = item["category"]
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(item)

            print(f"VNNLIB Downloads ({len(downloaded)} instances):")
            print("-" * 80)
            for cat in sorted(categories.keys()):
                instances = categories[cat]
                print(f"  {cat:30s} ({len(instances)} instances)")
                if len(instances) <= 5:
                    for inst in instances:
                        print(
                            f"    └─ {inst['instance_id']}: {inst['onnx_model']} + {inst['vnnlib_spec']}"
                        )
        else:
            print("No VNNLIB downloads found")
            print("Use --download <category> to download benchmarks")

    elif creator == "torchvision":
        downloaded = tv_loader.list_downloaded_pairs()
        if downloaded:
            # Group by dataset
            datasets = {}
            for item in downloaded:
                ds = item["dataset"]
                if ds not in datasets:
                    datasets[ds] = []
                datasets[ds].append(item["model"])

            print(f"TorchVision Downloads ({len(downloaded)} pairs):")
            print("-" * 80)
            for ds in sorted(datasets.keys()):
                models = datasets[ds]
                print(f"  {ds:30s} ({len(models)} models)")
                for model in sorted(models):
                    print(f"    └─ {model}")
        else:
            print("No TorchVision downloads found")
            print(
                "Use --download <dataset> --creator torchvision to download data-model pairs"
            )

    print(f"\n{'=' * 80}\n")


# ============================================================================
# Fuzzing Commands
# ============================================================================


def cmd_fuzz(args):
    """Run ACTFuzzer."""
    print_header()

    # Determine creator
    creator = args.creator
    print(f"📦 Using spec creator: {creator.upper()}")
    if args.strict_mode:
        print(f"⚠️  Strict mode enabled: Errors will be raised on constraint violations")
    print()

    # Load configuration from YAML with CLI overrides
    overrides: dict[str, Any] = dict(
        max_iterations=args.iterations,
        timeout_seconds=args.timeout,
        save_counterexamples=not args.no_save,
        output_dir=Path(args.output),
        report_interval=args.report_interval,
        # Tracing configuration
        trace_level=args.trace_level,
        trace_sample_rate=args.trace_sample,
        trace_storage=args.trace_storage,
        trace_output=Path(args.trace_output) if args.trace_output else None,
    )
    config = FuzzingConfig.from_yaml(**overrides)

    # Create spec creator and load data-model pairs
    print(f"{'=' * 80}")
    print(f"STEP 1: Loading Data-Model Pairs")
    print(f"{'=' * 80}\n")

    spec_results = []
    initial_seeds = []

    try:
        if creator == "vnnlib":
            spec_creator = VNNLibSpecCreator()

            if args.category:
                # Specific category
                categories = [args.category]
            else:
                # Use all downloaded categories
                downloaded = vnnlib_loader.list_downloaded_pairs()
                if not downloaded:
                    print("❌ No VNNLIB categories downloaded!")
                    print("Use: python -m act.pipeline --download <category>")
                    return
                categories = list(set(p["category"] for p in downloaded))

            print(f"Loading {len(categories)} VNNLIB category(ies):")
            for cat in categories:
                print(f"  • {cat}")
            print()

            spec_results = spec_creator.create_specs_for_data_model_pairs(
                categories=categories, max_instances=args.max_instances
            )

        elif creator == "torchvision":
            spec_creator = TorchVisionSpecCreator()

            if args.dataset:
                # Specific dataset
                datasets = [args.dataset]
            else:
                # Use all downloaded datasets
                downloaded = tv_loader.list_downloaded_pairs()
                if not downloaded:
                    print("❌ No TorchVision datasets downloaded!")
                    print(
                        "Use: python -m act.pipeline --download <dataset> --creator torchvision"
                    )
                    return
                datasets = list(set(p["dataset"] for p in downloaded))

            print(f"Loading {len(datasets)} TorchVision dataset(s):")
            for ds in datasets:
                print(f"  • {ds}")
            print()

            # Get models for each dataset
            if args.model:
                # Specific model for all datasets
                model_names = [args.model]
            else:
                # Use first available model for each dataset
                downloaded = tv_loader.list_downloaded_pairs()
                model_names = []
                for ds in datasets:
                    ds_models = [p["model"] for p in downloaded if p["dataset"] == ds]
                    if ds_models:
                        model_names.append(ds_models[0])

            if not model_names:
                print("❌ No models found for selected datasets!")
                return

            spec_results = spec_creator.create_specs_for_data_model_pairs(
                dataset_names=datasets,
                model_names=model_names,
                num_samples=args.num_samples,
            )

    except Exception as e:
        print(f"❌ Error loading data-model pairs: {e}")
        import traceback

        traceback.print_exc()
        return

    if not spec_results:
        print("❌ No spec results generated!")
        return

    print(f"✓ Generated {len(spec_results)} spec result(s)\n")

    # Synthesize models
    print(f"{'=' * 80}")
    print(f"STEP 2: Model Synthesis")
    print(f"{'=' * 80}\n")

    # Set strict mode for all VerifiableModel instances
    from act.front_end.verifiable_model import VerifiableModel

    VerifiableModel.set_strict_mode(args.strict_mode)

    try:
        wrapped_models = synthesize_models_from_specs(spec_results)
    except Exception as e:
        print(f"❌ Model synthesis failed: {e}")
        import traceback

        traceback.print_exc()
        return

    if not wrapped_models:
        print("❌ No models synthesized!")
        return

    print(f"✓ Synthesized {len(wrapped_models)} wrapped model(s)\n")

    # Extract initial seeds
    print(f"{'=' * 80}")
    print(f"STEP 3: Seed Extraction")
    print(f"{'=' * 80}\n")

    # Single model only; mixing seeds across spec_results breaks SeedCorpus(torch.cat).
    _, _, _, labeled_tensors, _ = spec_results[0]
    initial_seeds.extend(labeled_tensors)

    if not initial_seeds:
        print("❌ No initial seeds extracted!")
        return

    print(f"✓ Extracted {len(initial_seeds)} initial seeds\n")

    # Run fuzzing on first model
    print(f"{'=' * 80}")
    print(f"STEP 4: Fuzzing")
    print(f"{'=' * 80}\n")

    model_id = list(wrapped_models.keys())[0]
    wrapped_model = wrapped_models[model_id]

    print(f"Fuzzing model: {model_id}\n")

    try:
        fuzzer = ACTFuzzer(
            wrapped_model=wrapped_model, initial_seeds=initial_seeds, config=config
        )

        report = fuzzer.fuzz()

        # Print final results
        print(f"\n{'=' * 80}")
        print(f"FUZZING COMPLETE")
        print(f"{'=' * 80}")
        print(f"Iterations: {report.total_iterations}")
        print(f"Time: {report.total_time:.1f}s")
        print(f"Counterexamples: {len(report.counterexamples)}")
        print(f"Coverage: {report.neuron_coverage:.2%}")
        print(f"Seeds explored: {report.seeds_explored}")
        print(f"{'=' * 80}\n")

    except Exception as e:
        print(f"❌ Fuzzing failed: {e}")
        import traceback

        traceback.print_exc()
        return


# ============================================================================
# Verification Commands
# ============================================================================


def cmd_list_verifications():
    """List available verification tests."""
    print(f"\n{'=' * 80}")
    print(f"AVAILABLE VERIFICATION TESTS")
    print(f"{'=' * 80}\n")

    tests = [
        ("act2torch", "ACT→PyTorch conversion validation (model_factory)"),
        ("torch2act", "PyTorch→ACT conversion validation (torch2act)"),
        ("validate_verifier", "Verifier correctness validation with concrete tests"),
        ("all", "Run all verification tests"),
    ]

    for name, description in tests:
        print(f"  {name:25s} - {description}")

    print(f"\n{'=' * 80}\n")


def _run_soundness_check(tag: str, vm, net, results, validator, solver: str):
    vm = vm.to(validator.device, validator.dtype).eval()
    summary = validator.validate_results_soundness(
        tag, vm, results, solver=solver, act_net=net
    )
    for result in summary["results"]:
        status = result["validation_status"]
        ce_label = "FOUND" if result["concrete_counterexample"] else "NOT_FOUND"
        verifier = result["verifier_result"].name
        print(
            f"  [soundness] {result['network']}: {status} "
            f"(concrete_ce={ce_label}, verifier={verifier})"
        )
    return summary


def _print_soundness_summary(summary: dict[str, Any]) -> None:
    print(
        f"SOUNDNESS SUMMARY: total={summary['total']} passed={summary['passed']} "
        f"acceptable={summary['acceptable']} inconclusive={summary['inconclusive']} "
        f"failed={summary['failed']} unknown={summary['unknown']}"
    )


def _run_vnnlib_verify(args) -> bool:
    """Drive ``verify_once`` over a VNNLIB benchmark end-to-end.

    Bridges the front-end load → ACT-Net path that ``act.back_end --verify
    --network`` does not provide: ``VNNLibSpecCreator`` →
    ``synthesize_models_from_specs`` → ``TorchToACT`` → ``verify_once``.

    Single-mode per invocation, matching the ``act.back_end --verify`` CLI
    contract: uses the first element of ``--tf-modes`` (default
    ``"interval"``) and ``--solvers`` (default ``"torchlp"``).  Multi-mode
    sweeps are the caller's job — invoke once per (tf-mode, solver) cell.
    Dual ignores ``--tf-modes`` because it's a backward Solver.

    Solver=``hybridz`` routes to ``verify_once_hz`` + ``HZVerifier`` for the
    HZ-native verdict path; see ``_run_vnnlib_verify_hybridz``.
    """
    from act.front_end.vnnlib_loader.create_specs import VNNLibSpecCreator
    from act.front_end.model_synthesis import synthesize_models_from_specs
    from act.pipeline.verification.torch2act import TorchToACT
    from act.back_end.verifier import verify_once
    from act.back_end.transfer_functions import (
        set_solver_mode,
        set_transfer_function_mode,
    )
    from act.pipeline.verification.validate_verifier import VerificationValidator

    if not args.category:
        raise ValueError("--verify vnnlib requires --category (e.g. --category acasxu_2023)")

    tf_mode = (args.tf_modes or ["interval"])[0]
    solver = (args.solvers or ["torchlp"])[0]

    if solver == "hybridz":
        return _run_vnnlib_verify_hybridz(args)

    set_solver_mode(solver)
    if solver != "dual":
        set_transfer_function_mode(tf_mode)
    label = solver if solver == "dual" else f"{tf_mode}/{solver}"
    print(f"[vnnlib] category={args.category} max_instances={args.max_instances} mode={label}")

    spec_results = VNNLibSpecCreator().create_specs_for_data_model_pairs(
        categories=[args.category], max_instances=args.max_instances,
    )
    if not spec_results:
        raise RuntimeError(f"VNNLibSpecCreator produced no spec_results for category={args.category!r}")

    wrapped = synthesize_models_from_specs(spec_results)
    if not wrapped:
        raise RuntimeError("synthesize_models_from_specs produced no VerifiableModels")

    validator = None
    soundness_summary = None
    if args.validate_soundness:
        dtype = torch.float64 if args.dtype == "float64" else torch.float32
        validator = VerificationValidator(device=args.device, dtype=dtype)
    for mid, vm in wrapped.items():
        tag = "/".join(str(p) for p in mid)
        net = TorchToACT(vm).run()
        if getattr(args, "bab", False):
            status = _run_bab_on_net(net, args)
            label = f"BaB[{args.bab_solver_tier}]"
            print(f"  {tag}: {label} → {status}")
        else:
            results = verify_once(net)
            statuses = [r.status.name for r in results]
            print(f"  {tag}: {statuses}")
            if args.validate_soundness:
                assert validator is not None
                soundness_summary = _run_soundness_check(
                    tag, vm, net, results, validator, solver
                )

    if args.validate_soundness:
        assert validator is not None and soundness_summary is not None
        soundness_summary = validator._compute_summary(validation_type="counterexample")
        _print_soundness_summary(soundness_summary)
        return soundness_summary["failed"] > 0
    return False


_HYZOR_DEFAULT_ENV = {
    "HYZOR_V8_ATTACK_FIRST": "0",
    "HYZOR_V8_PGD_PORTFOLIO": "0",
    "HYZOR_LARGE_CLS_SAT_PREFLIGHT": "0",
    "HYZOR_DISABLE_TYPEB_BOX_CENTER": "1",
    "HYZOR_PURE_HZ_MODE": "1",
    "HYZOR_SAT_SIDECAR": "1",
    "HYZOR_DISPATCH_GUARD_GB": "13",
    "HYZOR_RELU_MEM_BUDGET_GB": "13",
    "HYZOR_CONV_MEM_BUDGET_GB": "13",
    "HYZOR_V8_MEM_BUDGET_GB": "13",
    "HYZOR_V8_MEM_RESERVE_GB": "1",
    "HYZOR_PEE_GPU_QR": "1",
    "HYZOR_LARGE_CLS_EQ_LAYERS": "1",
    "HYZOR_TF_MODE": "interval",
    "HYZOR_USE_ACT": "1",
}


def _normalize_hz_status(status_str: str) -> str:
    """HZVerifier emits UNSAT/SAT/UNKNOWN; normalize to verdict vocabulary."""
    return {"UNSAT": "CERTIFIED", "SAT": "FALSIFIED"}.get(status_str, status_str)


def _known_unsupported_as_unknown(exc: Exception) -> bool:
    """Fail closed on known unsupported front-end patterns.

    These are not verifier crashes and must not be promoted to a math
    verdict. Returning UNKNOWN is more honest than ERROR for VNN-COMP
    accounting while preserving the diagnostic in ``instance_error``.
    Keep this list narrow so real bugs remain visible as ERROR.
    """
    import os as _os
    if _os.environ.get("ACT_UNSUPPORTED_AS_UNKNOWN", "1").strip().lower() not in (
        "1", "true", "yes", "on",
    ):
        return False
    msg = f"{type(exc).__name__}: {exc}"
    return (
        "OnnxSlice at" in msg
        and (
            "cannot resolve starts/ends" in msg
            or "fixed-shape LUT_BOUNDS would be unsound" in msg
        )
    )


def _singleton_box_from_queries(queries: list):
    """Return the shared singleton input if every query has the same BOX.

    This is an exact degenerate-HZ case. It deliberately rejects anything
    beyond a shared zero-width BOX so extra input polytopes cannot be
    accidentally ignored.
    """
    import torch as _torch
    x_ref = None
    for in_spec, _out_spec in queries:
        if str(getattr(in_spec, "kind", "")).upper() != "BOX":
            return None
        lb = getattr(in_spec, "lb", None)
        ub = getattr(in_spec, "ub", None)
        if lb is None or ub is None:
            return None
        lb_t = lb.detach().cpu() if hasattr(lb, "detach") else _torch.tensor(lb)
        ub_t = ub.detach().cpu() if hasattr(ub, "detach") else _torch.tensor(ub)
        if lb_t.shape != ub_t.shape:
            return None
        if not (_torch.isfinite(lb_t).all() and _torch.isfinite(ub_t).all()):
            return None
        if not _torch.equal(lb_t, ub_t):
            return None
        x = lb_t.reshape(-1).to(dtype=_torch.float64)
        if x_ref is None:
            x_ref = x
        elif x.shape != x_ref.shape or not _torch.equal(x, x_ref):
            return None
    return x_ref.numpy() if x_ref is not None else None


def _ort_eval_once(onnx_path: str, x_flat):
    """Run one strict ORT forward for singleton-query fast paths."""
    import numpy as _np
    import onnxruntime as _ort

    sess = _ort.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )
    in_meta = sess.get_inputs()[0]
    in_shape = list(in_meta.shape)
    if in_shape and (not isinstance(in_shape[0], int) or in_shape[0] <= 0):
        in_shape[0] = 1
    x_in = _np.asarray(x_flat, dtype=_np.float32).reshape(in_shape)
    return sess.run(None, {in_meta.name: x_in})[0].ravel()


def _all_output_specs_safe_strict(y, queries: list) -> bool:
    """Evaluate all unsafe queries on a concrete singleton output.

    Uses the same zero-tolerance unsafe predicate as HZ witness replay.
    """
    import numpy as _np
    from act.front_end.verifiable_model import OutputSpecLayer
    from act.back_end.solver.solver_hz import _eval_unsafe_strict

    if not _np.isfinite(y).all():
        return False
    n_out = int(len(y))
    out_vars = list(range(n_out))
    for _in_spec, out_spec in queries:
        assert_layer = OutputSpecLayer(spec=out_spec).to_act_layers(
            0, out_vars, B=1,
        )[0][0]
        if _eval_unsafe_strict(y, assert_layer):
            return False
    return True


def aggregate_query_statuses(q_statuses: list) -> str:
    """Pure function: aggregate per-query HZVerifier statuses to an
    instance-level status under the disjunctive UNSAFE-set semantic.

    For a VNNLIB spec whose top-level OR blocks Cartesian-product into
    ``len(q_statuses)`` queries, the whole spec is satisfied IFF some
    per-query UNSAFE set is satisfied. Therefore:

        ANY q == "SAT"   → "SAT"
        ALL q == "UNSAT" → "UNSAT"
        otherwise        → "UNKNOWN"

    Empty input is treated as "UNKNOWN" (no decision possible without
    at least one query). Statuses outside {SAT, UNSAT, UNKNOWN} are
    treated as UNKNOWN.

    Pinned by `tests/test_cli_query_aggregation.py` after advisor
    2026-05-24 Round 3 finding that the inline aggregation logic at
    `_run_vnnlib_verify_hybridz` lacked a regression test.
    """
    if not q_statuses:
        return "UNKNOWN"
    if any(s == "SAT" for s in q_statuses):
        return "SAT"
    if all(s == "UNSAT" for s in q_statuses):
        return "UNSAT"
    return "UNKNOWN"


class IncompleteFormalAuditError(RuntimeError):
    """ROUND 6 (advisor 2026-05-24): raised when a formal-mode CLI run
    finishes with one or more instances in ``ERROR_RECEIPT_*`` or
    ``ERROR_INTERNAL_INCONSISTENCY``.

    The internal solver verdicts ARE preserved (math truth) and the
    counts dict still reflects them — this exception fires AFTER the
    final summary is printed. Its sole purpose is to propagate a
    non-zero exit code to upstream schedulers (cron, CI, batch
    runners) so they treat the run as INCOMPLETE rather than PASSED.

    The message body lists the offending counts so an operator can
    triage without re-reading the per-instance log.
    """


def compute_run_status(counts: dict, formal_mode: bool) -> str:
    """Pure function: derive a CLI-level run status from per-bench counts.

    Returns one of:
        "PASSED"                    — no verifier or formal-audit errors
        "FAILED"                    — at least one generic verifier/config error
        "INCOMPLETE_FORMAL_AUDIT"   — formal mode AND at least one of
                                      ERROR_RECEIPT/INTERNAL_INCONSISTENCY > 0

    Generic per-instance exceptions are converted into the ``ERROR`` bucket
    inside the hybridz driver; they must fail the run here because otherwise
    the surrounding ``cmd_verify`` sees a normal return and incorrectly
    prints ``PASSED``.
    """
    if int(counts.get("ERROR", 0) or 0) > 0:
        return "FAILED"
    if not formal_mode:
        return "PASSED"
    receipt_errors = int(counts.get("ERROR_RECEIPT", 0) or 0)
    inconsistency = int(counts.get("ERROR_INTERNAL_INCONSISTENCY", 0) or 0)
    if receipt_errors > 0 or inconsistency > 0:
        return "INCOMPLETE_FORMAL_AUDIT"
    return "PASSED"


def aggregate_reportable_verdicts(q_reportables: list) -> str:
    """Pure function: aggregate per-query REPORTABLE verdicts to an
    instance-level reportable verdict (advisor 2026-05-24 Round 4).

    Per-query inputs come from
    ``solver_hz.reportable_verdict_for_cli`` and are one of:
        FALSIFIED, CERTIFIED, UNKNOWN, ERROR_RECEIPT_*,
        ERROR_INTERNAL_INCONSISTENCY, ERROR_NO_FORMAL_RESULT,
        ERROR_UNEXPECTED_*.

    Aggregation rule (preserves the disjunctive UNSAFE-set semantic
    but propagates first witness-bearing ERROR honestly):
        ANY q == FALSIFIED                  → FALSIFIED
        else ANY q starts with "ERROR_"     → that ERROR (first one)
        else ALL q == CERTIFIED             → CERTIFIED
        else                                 → UNKNOWN

    Why FALSIFIED beats ERROR: a single REPORTABLE_FALSIFIED query
    suffices to falsify the whole spec (the disjunction is satisfied).
    The ERROR from a DIFFERENT query doesn't change that. Only when no
    query reports FALSIFIED does an ERROR propagate.
    """
    if not q_reportables:
        return "UNKNOWN"
    if any(v == "FALSIFIED" for v in q_reportables):
        return "FALSIFIED"
    for v in q_reportables:
        if isinstance(v, str) and v.startswith("ERROR_"):
            return v
    if all(v == "CERTIFIED" for v in q_reportables):
        return "CERTIFIED"
    return "UNKNOWN"


def remaining_instance_query_budget(timeout_s: float, elapsed_s: float) -> float:
    """Return the remaining per-instance budget available to another query.

    The hybridz VNNLIB path may need to verify many disjunctive queries for
    one official instance. Each query previously received the full CLI
    ``--timeout`` value, allowing an instance with hundreds of queries to run
    for hundreds of timeout periods. This helper gives the query loop an
    aggregate, fail-closed budget: once exhausted, an unvisited query becomes
    UNKNOWN and the instance cannot be incorrectly promoted to CERTIFIED.

    This is cooperative accounting between queries. A single long-running
    ``analyze()`` call still requires a process-level watchdog for hard wall
    interruption.
    """
    return max(0.0, float(timeout_s) - max(0.0, float(elapsed_s)))


def select_pairs_by_official_ids(pairs: list, instance_ids: str | None) -> list:
    """Select stable VNN-COMP instances for formal sentinel/audit runs.

    ``official_instance_id`` is the row index propagated by
    ``list_downloaded_pairs`` after header detection.  Selection is by that
    id, not by the filtered loop index, and follows the order requested on
    the command line so an audit command is reproducible verbatim.
    """
    if not instance_ids:
        return list(pairs)
    raw_ids = [part.strip() for part in instance_ids.split(",") if part.strip()]
    if not raw_ids:
        raise ValueError("--instance-ids must contain at least one integer id")
    try:
        wanted = [int(part) for part in raw_ids]
    except ValueError as exc:
        raise ValueError("--instance-ids must be comma-separated integers") from exc
    if len(set(wanted)) != len(wanted) or any(i < 0 for i in wanted):
        raise ValueError("--instance-ids must be distinct non-negative integers")
    by_id = {int(p["official_instance_id"]): p for p in pairs}
    missing = [i for i in wanted if i not in by_id]
    if missing:
        raise ValueError(f"official instance ids not found: {missing}")
    return [by_id[i] for i in wanted]


def _run_vnnlib_verify_hybridz(args) -> None:
    """HZ-native verification path: ``HZVerifier`` + ``verify_once_hz``.

    Stock ``--solvers torchlp|gurobi|dual`` consume ACT's BatchLPProblem;
    HZVerifier instead walks the cons IR via ``consume_cons`` (HZ-native),
    so it's wired through its own driver. Per-instance subprocess isolation
    is unnecessary here because each call to ``verify_once_hz`` operates on
    a fresh ``HZVerifier`` and the analyzer state is rebuilt per net.

    Reads ``--category``, ``--max-instances``, ``--timeout``, ``--device``,
    ``--dtype``. Defaults the HYZOR_* environment knobs to the values used
    by the v120 reference recipe, but only when the caller hasn't set them
    explicitly (env precedence preserved).
    """
    import os
    import time as _time
    import traceback as _tb
    from pathlib import Path as _Path

    import torch as _torch

    from act.front_end.vnnlib_loader.data_model_loader import (
        list_downloaded_pairs, load_vnnlib_pair,
    )
    from act.front_end.vnnlib_loader.vnnlib_parser import parse_vnnlib_queries
    from act.front_end.verifiable_model import (
        InputLayer, InputSpecLayer, OutputSpecLayer, VerifiableModel,
    )
    from act.pipeline.verification.torch2act import TorchToACT
    from act.back_end.solver.solver_hz import (
        HZVerifier, verify_once_hz, reportable_verdict_for_cli,
    )

    if not args.category:
        raise ValueError("--verify vnnlib requires --category (e.g. --category acasxu_2023)")

    for k, v in _HYZOR_DEFAULT_ENV.items():
        os.environ.setdefault(k, v)

    device = getattr(args, "device", None) or "cuda"
    dtype_str = getattr(args, "dtype", "float64")
    dtype = _torch.float64 if dtype_str == "float64" else _torch.float32
    timeout_s = float(getattr(args, "timeout", 300.0) or 300.0)
    max_instances = getattr(args, "max_instances", None)
    formal_mode = (
        os.environ.get("ACT_FAL_RECEIPT_FORMAL", "").strip().lower()
        in ("1", "true", "yes", "on")
    )
    results_dir = (
        os.environ.get("ACT_FORMAL_RESULTS_DIR")
        or os.environ.get("ACT_FAL_RECEIPT_DIR")
    )
    if formal_mode and not results_dir:
        raise IncompleteFormalAuditError(
            "INCOMPLETE_FORMAL_AUDIT: formal mode requires "
            "ACT_FORMAL_RESULTS_DIR or ACT_FAL_RECEIPT_DIR before execution; "
            "no unlogged formal run is permitted."
        )

    # ROUND 4 (advisor 2026-05-24): allow operators to point ACT at the
    # canonical VNN-COMP benchmark root instead of duplicating data under
    # ACT/data/vnnlib. Frozen-baseline audits should run against the
    # SHA-pinned canonical files, not whatever was redundantly downloaded.
    _vnnlib_root = os.environ.get("ACT_VNNLIB_ROOT")
    pairs = [p for p in list_downloaded_pairs(root_dir=_vnnlib_root)
             if p["category"] == args.category]
    if not pairs:
        raise RuntimeError(
            f"No downloaded VNNLIB instances for category={args.category!r}; "
            f"run `python -m act.pipeline --download {args.category}` first"
        )
    pairs = select_pairs_by_official_ids(
        pairs, getattr(args, "instance_ids", None)
    )
    if max_instances is not None:
        pairs = pairs[:int(max_instances)]

    total = len(pairs)
    print(f"[vnnlib] category={args.category} max_instances={total} "
          f"mode=hybridz (full TOP1_ROBUST via labeled_tensor collapse; "
          f"device={device}, timeout={timeout_s}s)")

    counts = {"CERTIFIED": 0, "FALSIFIED": 0, "UNKNOWN": 0, "ERROR": 0}
    # ROUND 6 (advisor 2026-05-24): structured per-instance log so
    # downstream audits don't depend on stdout scraping. Each row carries
    # official_instance_id (audit key), internal/reportable verdict,
    # query count, receipt path (if any), and wall.
    per_instance: list[dict] = []
    t_start = _time.time()
    for i, p in enumerate(pairs):
        onnx_p = p["paths"]["onnx"]
        vnn_p = p["paths"]["vnnlib"]
        tag = f"{p['category']}/{p['onnx_model']}@{p['vnnlib_spec']}"
        t0 = _time.time()
        internal_status = "ERROR"
        reportable_status = "ERROR"
        q_statuses: list[str] = []
        q_reportables: list[str] = []
        q_receipts: list[Optional[str]] = []
        q_solver_stats: list[dict] = []
        instance_error: Optional[str] = None
        instance_budget_start = _time.monotonic()
        instance_budget_exhausted = False
        # Initialized here (pre-try) so the end-of-iter env-restore block is
        # safe even if `try:` raises before the per-query loop populates it.
        _iid_env_restore: Dict[str, Optional[str]] = {}
        # cifar-endcap profile: per-iid tmp snapshot dir; declared at
        # per-iid scope so the end-of-iid cleanup block always has it.
        _cifar_endcap_snap_dir: Optional[str] = None
        _mlp_endcap_snap_dir: Optional[str] = None
        try:
            pair = load_vnnlib_pair(
                category=p["category"], onnx_model=p["onnx_model"],
                vnnlib_spec=p["vnnlib_spec"], auto_download=False,
                # ROUND 5 (advisor 2026-05-24): pass the SAME canonical
                # root that drove enumeration, so loader doesn't fall
                # back to ACT default root and miss benchmarks that
                # only exist under the canonical mirror.
                root_dir=_vnnlib_root,
            )
            model = pair["model"].to(dtype).eval()
            in_shape = pair["labeled_tensor"].tensor.shape
            # CRITICAL: pass labeled_tensor so the parser collapses 199
            # individual (Y_j >= Y_t) disjuncts into a single TOP1_ROBUST
            # query with M=199. Without this we'd silently fall back to
            # the legacy single-disjunct (UNSAFE_LINEAR M=1) shortcut.
            queries = parse_vnnlib_queries(_Path(vnn_p), labeled_tensor=pair["labeled_tensor"])
            # SOUNDNESS GATE (2026-05-24 advisor review): the parser emits
            # the Cartesian product of all top-level OR blocks as
            # ``queries``. Semantically, the whole VNNLIB spec describes
            # the UNSAFE set as the *union* over per-query unsafe sets, so:
            #   * ANY query → FALSIFIED  ⇒ instance FALSIFIED  (one
            #     witness in any per-query unsafe set is a real adversary)
            #   * ALL queries → CERTIFIED ⇒ instance CERTIFIED  (every
            #     branch of the disjunction proved infeasible)
            #   * otherwise → UNKNOWN
            # Taking only queries[0] would silently miss per-query unsafe
            # branches and mis-report CERTIFIED on multi-OR specs.
            # T2 ablation knob: ACT_HZ_EQ_LAYERS overrides the default
            # last-N eq_lagr_v8 layer count for large_cls scheduling.
            # HYZOR_LARGE_CLS_EQ_LAYERS is set by _HYZOR_DEFAULT_ENV for
            # the reference recipe; honor it when the newer ACT_* knob is
            # absent so the documented default actually reaches HZVerifier.
            _eq_layers = int(os.environ.get(
                "ACT_HZ_EQ_LAYERS",
                os.environ.get("HYZOR_LARGE_CLS_EQ_LAYERS", "3"),
            ))
            x_single = None
            if os.environ.get("ACT_HZ_SINGLETON_FASTPATH", "1").strip().lower() in (
                "1", "true", "yes", "on",
            ):
                x_single = _singleton_box_from_queries(queries)
            if x_single is not None:
                try:
                    y_single = _ort_eval_once(onnx_p, x_single)
                    if _all_output_specs_safe_strict(y_single, queries):
                        q_statuses = ["UNSAT"] * len(queries)
                        q_reportables = ["CERTIFIED"] * len(queries)
                        q_receipts = [None] * len(queries)
                        internal_status = "UNSAT"
                        reportable_status = "CERTIFIED"
                        status_str = "UNSAT"
                        instance_error = None
                        elapsed = _time.time() - t0
                        normalized = _normalize_hz_status(status_str)
                        key = normalized if normalized in counts else "UNKNOWN"
                        counts[key] = counts.get(key, 0) + 1
                        per_instance.append({
                            "official_instance_id": int(p.get('official_instance_id', i)),
                            "benchmark": p["category"],
                            "onnx_model": p["onnx_model"],
                            "vnnlib_spec": p["vnnlib_spec"],
                            "model_path": onnx_p,
                            "spec_path": vnn_p,
                            "internal_status": internal_status,
                            "reportable_status": reportable_status,
                            "cli_normalized": normalized,
                            "count_bucket": key,
                            "queries": [
                                {
                                    "query_index": q_idx,
                                    "internal_status": "UNSAT",
                                    "reportable_status": "CERTIFIED",
                                    "receipt_path": None,
                                }
                                for q_idx in range(len(queries))
                            ],
                            "q_statuses": q_statuses,
                            "q_reportables": q_reportables,
                            "q_receipts": q_receipts,
                            "error": instance_error,
                            "wall_s": float(elapsed),
                            "timeout_s": timeout_s,
                            "instance_budget_exhausted": False,
                            "singleton_query_fastpath": True,
                        })
                        print(
                            f"  [{i + 1:3d}/{total}] {tag}: {normalized} "
                            f"({elapsed:.1f}s)  V={counts['CERTIFIED']} "
                            f"A={counts['FALSIFIED']} U={counts['UNKNOWN']} "
                            f"E={counts['ERROR']}  R={counts.get('ERROR_RECEIPT', 0)}",
                            flush=True,
                        )
                        continue
                except Exception as e:
                    # Fail closed to the normal HZ path if the singleton
                    # shortcut cannot establish a strict all-safe result.
                    if os.environ.get("ACT_VERIFY_VERBOSE") == "1":
                        print(
                            "singleton_query_fastpath_error: "
                            f"{type(e).__name__}: {e}",
                            flush=True,
                        )
            in_layer = InputLayer(
                labeled_input=pair["labeled_tensor"],
                shape=tuple(int(s) for s in in_shape),
                dtype=dtype,
            )
            for q_idx, (in_spec, out_spec) in enumerate(queries):
                remaining_s = remaining_instance_query_budget(
                    timeout_s, _time.monotonic() - instance_budget_start
                )
                if remaining_s <= 0.0:
                    # There are unexplored UNSAFE branches. Fail closed as
                    # UNKNOWN instead of certifying an incompletely checked
                    # instance.
                    q_statuses.append("UNKNOWN")
                    q_reportables.append("UNKNOWN")
                    q_receipts.append(None)
                    instance_budget_exhausted = True
                    break
                vm = VerifiableModel(
                    input_layer=in_layer,
                    input_spec=InputSpecLayer(spec=in_spec),
                    model=model,
                    output_spec=OutputSpecLayer(spec=out_spec),
                )
                net = TorchToACT(vm).run()
                remaining_s = remaining_instance_query_budget(
                    timeout_s, _time.monotonic() - instance_budget_start
                )
                if remaining_s <= 0.0:
                    q_statuses.append("UNKNOWN")
                    q_reportables.append("UNKNOWN")
                    q_receipts.append(None)
                    instance_budget_exhausted = True
                    break
                # Witness profile: small-dense benches where LP-witness
                # extraction + ORT zero-tol replay produced sound V/A in
                # 2026-05/06 audits. See:
                #   project_safenlp_b14_frozen (+22 over B14)
                #   project_sat_relu_reopened_20260601 (+15 FAL)
                #   project_audit_final_consolidated_20260601 (dist_shift +5V,
                #     malbeware +2V, metaroom +2V)
                _small_dense_witness_profile = p['category'] in {
                    "linearizenn_2024",
                    "tllverifybench_2023",
                    "acasxu_2023",
                    "safenlp_2024",
                    "sat_relu",
                    "dist_shift_2023",
                    "malbeware",
                    "metaroom_2023",
                }
                _small_dense_dag_profile = p['category'] == "cersyve"
                # nn4sys lindex/lindex_deep specs have thousands of
                # independent UNSAFE_LINEAR rows over a one-dimensional box.
                # The direct query path avoids re-parsing all disjuncts for
                # every ACT query, the bound cache avoids repeated initial LP
                # bound solves, and the stable-affine fast path exactly closes
                # boxes where interval propagation fixes every ReLU side. All
                # three are fail-closed and remain forward-only continuous LP
                # checks; they do not introduce fallback/BaB/sampling.
                _nn4sys_lindex_profile = (
                    p['category'] == "nn4sys"
                    and "lindex" in str(vnn_p).lower()
                )
                # 2026-06-02 cifar100_2024 narrow profile: factor-aware ADD
                # + L38 FLATTEN snapshot + end-cap LP witness sidecar.
                # Validated by a 200-iid sweep that produced 15 sound FAL
                # (0 LOST, 0 ERR, 15/15 receipts independently passing
                # input_box_holds + vnnlib_query_holds + spec_zero_tol_holds).
                # All four knobs are env-gated and forward-only:
                #   ACT_HZ_FACTOR_ID_SGM        - ResNet residual factor-aware ADD
                #   ACT_HZ_ENDCAP_SNAPSHOT_DIR  - persist L38 HZ to disk
                #   ACT_HZ_ENDCAP_SNAPSHOT_KIND - "FLATTEN" for this profile
                #   ACT_HZ_CIFAR_ENDCAP_WITNESS - post-UNK xi_root + ORT replay
                # User opt-out: ACT_HZ_CIFAR_ENDCAP_PROFILE=0
                _cifar_endcap_profile = (
                    p['category'] == "cifar100_2024"
                    and os.environ.get(
                        "ACT_HZ_CIFAR_ENDCAP_PROFILE", "1"
                    ).strip().lower() not in ("0", "false", "no", "off")
                )
                # 2026-06-03 generic MLP end-cap profile — STRUCTURAL gate.
                # Logic extracted to act.back_end.profiles for test coverage;
                # see tests/test_generic_mlp_endcap_gate.py (14/14 cases).
                # The gate enforces tail shape + final out_dim + top-1 robust
                # vnnlib + CIFAR-narrow exclusion; the sidecar itself runs a
                # SECOND fail-closed check on snapshot root provenance
                # (research/generic_mlp_endcap_reuse.py lines 221/225). CERT
                # promotion is DISABLED by default (opt-in via
                # ACT_HZ_MLP_ENDCAP_ALLOW_CERT=1).
                from act.back_end.profiles import (
                    supports_generic_mlp_endcap as _supports_mlp_endcap,
                )
                _mlp_gate_result = _supports_mlp_endcap(
                    layers=net.layers,
                    pair=pair,
                    cifar_endcap_active=bool(_cifar_endcap_profile),
                )
                _generic_mlp_endcap_profile = _mlp_gate_result.enabled
                _mlp_gate_diag = {
                    "category": p['category'],
                    "tail_supported": _mlp_gate_result.tail_supported,
                    "tail_kinds": (
                        list(_mlp_gate_result.tail_kinds)
                        if _mlp_gate_result.tail_kinds is not None
                        else None
                    ),
                    "final_out_dim": _mlp_gate_result.final_out_dim,
                    "is_top1_robust": _mlp_gate_result.is_top1_robust,
                    "cifar_endcap_active": _mlp_gate_result.cifar_endcap_active,
                    "enabled": _mlp_gate_result.enabled,
                }
                # 2026-06-03 residual sparse-conv profile. This is a
                # structure-triggered profile for detector-style conv
                # residual nets whose tails remain convolutional rather than
                # entering a dense classifier MLP. It enables exact
                # factor-aware ADD plus sparse pre-conv materialisation so
                # repeated residual factors are merged instead of duplicated.
                # It is deliberately NOT enabled for CIFAR/Tiny classifier
                # tails, which have their own end-cap profiles above.
                _kinds_for_profile = [str(L.kind).upper() for L in net.layers]
                _n_conv_for_profile = sum(
                    1 for _k in _kinds_for_profile if _k in ("CONV", "CONV2D")
                )
                _n_add_for_profile = sum(
                    1 for _k in _kinds_for_profile if _k == "ADD"
                )
                _has_dense_tail_for_profile = any(
                    _k in ("DENSE", "GEMM", "MATMUL")
                    for _k in _kinds_for_profile
                )
                _last_non_assert_for_profile = next(
                    (
                        L for L in reversed(net.layers)
                        if str(L.kind).upper() != "ASSERT"
                    ),
                    None,
                )
                _out_dim_for_profile = (
                    len(_last_non_assert_for_profile.out_vars)
                    if _last_non_assert_for_profile is not None else 0
                )
                _residual_sparse_conv_profile = (
                    os.environ.get(
                        "ACT_HZ_RESIDUAL_SPARSE_PROFILE", "1"
                    ).strip().lower() not in ("0", "false", "no", "off")
                    and not _cifar_endcap_profile
                    and not _generic_mlp_endcap_profile
                    and _n_conv_for_profile >= 6
                    and _n_add_for_profile >= 2
                    and not _has_dense_tail_for_profile
                    and _out_dim_for_profile >= 1024
                )
                # _iid_env_restore (declared above the per-iid try) tracks
                # env vars THIS iid sets so we can restore them at end-of-
                # iid (avoids profile leaking into later iids in the same
                # Python process when cli is run directly with multiple
                # --instance-ids; watchdog_runner spawns one subprocess
                # per iid so it's unaffected, but we shouldn't rely on it).
                if _nn4sys_lindex_profile:
                    for _k in (
                        "ACT_HZ_SMALL_DENSE_DIRECT_QUERY",
                        "ACT_HZ_SPECAWARE_BOUND_CACHE",
                        "ACT_HZ_STABLE_AFFINE_FASTPATH",
                    ):
                        if _k not in _iid_env_restore:
                            _iid_env_restore[_k] = os.environ.get(_k)
                        os.environ.setdefault(_k, "1")
                if _cifar_endcap_profile:
                    # Track env vars in _iid_env_restore so a later iid
                    # in the same Python process (or a different
                    # benchmark in a script) does NOT inherit these.
                    for _k in (
                        "ACT_HZ_FACTOR_ID_SGM",
                        "ACT_HZ_ENDCAP_SNAPSHOT_DIR",
                        "ACT_HZ_ENDCAP_SNAPSHOT_KIND",
                        "ACT_HZ_CIFAR_ENDCAP_WITNESS",
                    ):
                        if _k not in _iid_env_restore:
                            _iid_env_restore[_k] = os.environ.get(_k)
                    # Per-iid temp dir for the L38 snapshot. The .pkl is
                    # cleaned up at end-of-iid (see end of try block).
                    import tempfile as _tf
                    _cifar_endcap_snap_dir = _tf.mkdtemp(
                        prefix=f"cifar_endcap_iid{p.get('official_instance_id', i)}_"
                    )
                    os.environ.setdefault("ACT_HZ_FACTOR_ID_SGM", "1")
                    os.environ.setdefault(
                        "ACT_HZ_ENDCAP_SNAPSHOT_DIR", _cifar_endcap_snap_dir
                    )
                    os.environ.setdefault("ACT_HZ_ENDCAP_SNAPSHOT_KIND", "FLATTEN")
                    os.environ.setdefault("ACT_HZ_CIFAR_ENDCAP_WITNESS", "1")
                if _generic_mlp_endcap_profile:
                    for _k in (
                        "ACT_HZ_FACTOR_ID_SGM",
                        "ACT_HZ_R05_AFFINE_INHERIT",
                        "ACT_HZ_CONV_FALLBACK_SAFE",
                        "ACT_HZ_GIRARD_PRESERVE_ROOT",
                        "ACT_HZ_ENDCAP_SNAPSHOT_DIR",
                        "ACT_HZ_ENDCAP_SNAPSHOT_KIND",
                        "ACT_HZ_GENERIC_MLP_ENDCAP_WITNESS",
                    ):
                        if _k not in _iid_env_restore:
                            _iid_env_restore[_k] = os.environ.get(_k)
                    import tempfile as _tf
                    _mlp_endcap_snap_dir = _tf.mkdtemp(
                        prefix=(
                            "mlp_endcap_"
                            f"{p['category']}_iid{p.get('official_instance_id', i)}_"
                        )
                    )
                    os.environ.setdefault("ACT_HZ_FACTOR_ID_SGM", "1")
                    os.environ.setdefault("ACT_HZ_R05_AFFINE_INHERIT", "1")
                    os.environ.setdefault("ACT_HZ_CONV_FALLBACK_SAFE", "1")
                    os.environ.setdefault("ACT_HZ_GIRARD_PRESERVE_ROOT", "1")
                    os.environ.setdefault(
                        "ACT_HZ_ENDCAP_SNAPSHOT_DIR", _mlp_endcap_snap_dir
                    )
                    os.environ.setdefault("ACT_HZ_ENDCAP_SNAPSHOT_KIND", "FLATTEN")
                    os.environ.setdefault(
                        "ACT_HZ_GENERIC_MLP_ENDCAP_WITNESS", "1"
                    )
                if _residual_sparse_conv_profile:
                    for _k in (
                        "ACT_HZ_FACTOR_ID_SGM",
                        "ACT_HZ_R05_AFFINE_INHERIT",
                        "ACT_HZ_CONV_FALLBACK_SAFE",
                        "ACT_HZ_GIRARD_PRESERVE_ROOT",
                        "ACT_HZ_PRECONV_SPARSE",
                        "ACT_HZ_PRECONV_BUDGET_MIB",
                        "ACT_HZ_DENSE_TO_SPARSE",
                        "ACT_HZ_SPARSE_GC_DENSITY",
                    ):
                        if _k not in _iid_env_restore:
                            _iid_env_restore[_k] = os.environ.get(_k)
                    os.environ.setdefault("ACT_HZ_FACTOR_ID_SGM", "1")
                    os.environ.setdefault("ACT_HZ_R05_AFFINE_INHERIT", "1")
                    os.environ.setdefault("ACT_HZ_CONV_FALLBACK_SAFE", "1")
                    os.environ.setdefault("ACT_HZ_GIRARD_PRESERVE_ROOT", "1")
                    os.environ.setdefault("ACT_HZ_PRECONV_SPARSE", "1")
                    os.environ.setdefault("ACT_HZ_PRECONV_BUDGET_MIB", "512")
                    os.environ.setdefault("ACT_HZ_DENSE_TO_SPARSE", "1")
                    os.environ.setdefault("ACT_HZ_SPARSE_GC_DENSITY", "1.0")
                _small_dense_lp_mode = os.environ.get("ACT_HZ_SMALL_DENSE_LP")
                if _small_dense_lp_mode is None:
                    if _small_dense_dag_profile:
                        _small_dense_lp_mode = "smalldense_dag"
                    elif _small_dense_witness_profile:
                        _small_dense_lp_mode = "witness"
                    else:
                        _small_dense_lp_mode = "specaware"
                _small_dense_lp_refinement = os.environ.get(
                    "ACT_HZ_SMALL_DENSE_LP_REFINEMENT_PASSES"
                )
                if _small_dense_lp_refinement is None:
                    # ACASXu: 50 passes proven sufficient (autoprofile, 14/15 A
                    # recovered). New audit benches use 80 (full audit setting
                    # that produced sat_relu +15A, safenlp +22).
                    if p['category'] == "acasxu_2023":
                        _small_dense_lp_refinement = "50"
                    elif p['category'] in {
                        "safenlp_2024", "sat_relu",
                        "dist_shift_2023", "malbeware", "metaroom_2023",
                        # linearizenn_2024 needs 80 passes too — under default
                        # 20-pass profile it gives 0V whereas with 80 passes +
                        # 20s LP it gives 16V on full bench. The historical
                        # 46V is NOT reproducible (see
                        # project_linearizenn_regression_p0_investigation),
                        # but 16V is real and worth keeping.
                        "linearizenn_2024",
                    }:
                        _small_dense_lp_refinement = "80"
                    elif _small_dense_witness_profile:
                        _small_dense_lp_refinement = "20"
                    else:
                        _small_dense_lp_refinement = "0"
                _small_dense_lp_time_limit = os.environ.get(
                    "ACT_HZ_SMALL_DENSE_LP_TIME_LIMIT_S"
                )
                if _small_dense_lp_time_limit is None:
                    # ACASXu: 15s default. New audit benches: 20s (matches
                    # the audit run that produced their gains).
                    if p['category'] == "acasxu_2023":
                        _small_dense_lp_time_limit = "15.0"
                    elif p['category'] in {
                        "safenlp_2024", "sat_relu",
                        "dist_shift_2023", "malbeware", "metaroom_2023",
                        "linearizenn_2024",
                    }:
                        _small_dense_lp_time_limit = "20.0"
                    else:
                        _small_dense_lp_time_limit = "5.0"
                solver = HZVerifier(
                    device=device, dtype=dtype, timeout_s=remaining_s,
                    strict_replay=True, onnx_path=onnx_p,
                    vnnlib_path=vnn_p,
                    relu_method=os.environ.get("ACT_HZ_RELU_METHOD", "eq_lagr_v8"),
                    # Use official_instance_id (row position in instances.csv
                    # after correct header detection), NOT the filtered-loop
                    # index ``i``. The two differ when filter / max-instances
                    # is applied or when the CSV is headerless. Per advisor
                    # 2026-05-24 Round 3, receipts MUST carry official ids.
                    instance_id=int(p.get('official_instance_id', i)),
                    query_index=q_idx,
                    benchmark=p['category'],
                    large_cls_eq_layers=_eq_layers,
                    girard_cap=int(os.environ.get("ACT_HZ_GIRARD_CAP", "6000")),
                    tail_preserve_dim=int(
                        os.environ.get("ACT_HZ_TAIL_PRESERVE_DIM", "0")
                    ),
                    constraint_keep_weight=float(
                        os.environ.get("ACT_HZ_CONSTRAINT_KEEP_WEIGHT", "0.0")
                    ),
                    sigmoid_K=int(os.environ.get("ACT_HZ_SIGMOID_K", "2")),
                    tanh_K=int(os.environ.get("ACT_HZ_TANH_K", "2")),
                    small_dense_lp=_small_dense_lp_mode,
                    small_dense_lp_root=os.environ.get("ACT_HZ_SMALL_DENSE_LP_ROOT"),
                    small_dense_lp_time_limit_s=float(
                        _small_dense_lp_time_limit
                    ),
                    small_dense_lp_refinement_passes=int(
                        _small_dense_lp_refinement
                    ),
                    small_dense_lp_fallback_on_unknown=(
                        os.environ.get("ACT_HZ_SMALL_DENSE_LP_FALLBACK", "0") == "1"
                    ),
                )
                q_status, _, _ = verify_once_hz(
                    net=net, solver=solver, timelimit=remaining_s
                )
                q_internal = str(q_status)
                # A1 narrow sidecar (env: ACT_HZ_CIFAR_ENDCAP_WITNESS=1,
                # default OFF, cifar100_2024 only). When the verifier
                # returns UNKNOWN AND a FLATTEN snapshot is available
                # (via the ACT_HZ_ENDCAP_SNAPSHOT_DIR research hook),
                # attempt the factor-aware-endcap-LP-root-xi witness:
                # extract LP xi_root, reconstruct concrete input, ORT-
                # replay the raw ONNX, and check the vnnlib unsafe
                # condition at strict zero tolerance. If all 3 checks
                # pass, upgrade the verdict to SAT/FALSIFIED with a
                # receipt. Otherwise leave UNKNOWN.
                # Advisor 2026-06-03 A+: new env knob
                # `ACT_HZ_TOPK_RIVAL_WITNESS=K` selects topK candidate
                # rivals (legacy ∪ topK by LP UB descending). Legacy
                # `ACT_HZ_CIFAR_ENDCAP_WITNESS=1` defaults K=5 and is
                # kept as an alias for backward compatibility.
                _topk_raw = os.environ.get(
                    "ACT_HZ_TOPK_RIVAL_WITNESS", "0").strip()
                try:
                    _topk_val = int(_topk_raw)
                except ValueError:
                    _topk_val = 0
                _legacy_on = os.environ.get(
                    "ACT_HZ_CIFAR_ENDCAP_WITNESS", "0"
                ).strip().lower() in ("1", "true", "yes", "on")
                if _topk_val >= 1:
                    _witness_on = True
                    _topk_K = _topk_val
                elif _legacy_on:
                    _witness_on = True
                    _topk_K = 5
                else:
                    _witness_on = False
                    _topk_K = 0
                _witness_receipt_path = None
                _cifar_witness_diag = {
                    "witness_on": bool(_witness_on),
                    "topk_raw": _topk_raw,
                    "topk_K": int(_topk_K),
                    "legacy_on": bool(_legacy_on),
                    "q_internal_before": q_internal,
                    "category": p['category'],
                    "snap_dir_env": os.environ.get(
                        "ACT_HZ_ENDCAP_SNAPSHOT_DIR", ""),
                    "snap_glob_count": None,
                }
                if (
                    _witness_on
                    and q_internal == "UNKNOWN"
                    and p['category'] == "cifar100_2024"
                ):
                    _snap_dir = os.environ.get(
                        "ACT_HZ_ENDCAP_SNAPSHOT_DIR", "")
                    if _snap_dir:
                        try:
                            from pathlib import Path as _PWit
                            _snap_glob = list(
                                _PWit(_snap_dir).glob("L*_FLATTEN.pkl")
                            )
                            _cifar_witness_diag["snap_glob_count"] = len(
                                _snap_glob)
                            if _snap_glob:
                                import sys as _swit
                                _swit.path.insert(0, "/data1/Kane/HyZor")
                                from receipt_factor_aware_endcap_lp import (
                                    _parse_vnnlib_full,
                                    _extract_xi_root_candidates,
                                    _disjunct_holds_zero_tol,
                                    _ort_eval,
                                    _validate_cifar_top1_robust,
                                )
                                import pickle as _pkw, json as _jsw
                                import numpy as _npw
                                _snap_path = str(_snap_glob[0])
                                _lb_v, _ub_v, _disjuncts = _parse_vnnlib_full(
                                    str(vnn_p))
                                # Fail-closed scope guard: only CIFAR
                                # top-1 robust shape can produce a
                                # receipt from this sidecar. Any
                                # structural mismatch raises here and
                                # the outer except converts to UNKNOWN.
                                _validate_cifar_top1_robust(
                                    _lb_v, _ub_v, _disjuncts)
                                with open(_snap_path, "rb") as _f_pkw:
                                    _snap_data = _pkw.load(_f_pkw)
                                # Hardening (advisor 2026-06-02
                                # post-TinyImageNet scout): the
                                # xi_root → x_cand reconstruction
                                # assumes root factors map bijectively
                                # to input pixels. TinyImageNet's
                                # snapshot shows root_ng > ng, which
                                # means reduction has compressed the
                                # root factor space — the
                                # reconstruction would be unsound. Two
                                # explicit preconditions:
                                #   root_ng == input_dim  (every input
                                #     pixel has its own factor)
                                #   root_ng <= ng         (no root
                                #     factors got dropped)
                                # Failure raises and the outer except
                                # converts to UNKNOWN (no receipt).
                                _root_ng = int(_snap_data.get("root_ng", -1))
                                _snap_ng = int(_snap_data.get("ng", -1))
                                _input_dim = int(_lb_v.shape[0])
                                if _root_ng != _input_dim:
                                    raise RuntimeError(
                                        f"sidecar root_ng={_root_ng} != "
                                        f"input_dim={_input_dim}; "
                                        f"witness reconstruction unsound"
                                    )
                                if _root_ng > _snap_ng:
                                    raise RuntimeError(
                                        f"sidecar root_ng={_root_ng} > "
                                        f"ng={_snap_ng}; root factors "
                                        f"compressed by reduction"
                                    )
                                # Advisor 2026-06-03 A+: iterate up to K
                                # candidates (legacy ∪ topK by LP UB
                                # descending). First one whose strict
                                # ORT replay satisfies the vnnlib unsafe
                                # condition wins; remaining candidates
                                # are skipped.
                                _cands, _y_t = _extract_xi_root_candidates(
                                    _snap_data, str(onnx_p), str(vnn_p),
                                    K=_topk_K, include_legacy=True,
                                )
                                _c_box = (_lb_v + _ub_v) / 2.0
                                _half = (_ub_v - _lb_v) / 2.0
                                _cand_log = []
                                _all_pass = False
                                _wr = -1
                                _lp_m = float("nan")
                                _winning_attempt = None
                                _y_ort = None
                                for _idx, _cand in enumerate(_cands):
                                    _xi = _npw.clip(
                                        _cand["xi_root"], -1.0, 1.0)
                                    _x_cand = _c_box + _half * _xi
                                    _in_box = bool(
                                        _npw.all(_x_cand >= _lb_v - 1e-12)
                                        and _npw.all(_x_cand <= _ub_v + 1e-12)
                                    )
                                    _y_ort = _ort_eval(
                                        str(onnx_p),
                                        _x_cand.reshape((1, 3, 32, 32))
                                        .astype(_npw.float64),
                                    )
                                    _qh = any(
                                        _disjunct_holds_zero_tol(_y_ort, _d)
                                        for _d in _disjuncts
                                    )
                                    _attempt_pass = bool(_in_box and _qh)
                                    _cand_log.append({
                                        "attempt_idx": _idx,
                                        "rival": int(_cand["rival"]),
                                        "source_rank": _cand["source_rank"],
                                        "lp_min": float(_cand["lp_min"]),
                                        "lp_upper_bound": float(
                                            _cand["lp_upper_bound"]),
                                        "input_box_holds": _in_box,
                                        "vnnlib_query_holds": _qh,
                                        "ort_y_argmax": int(
                                            _npw.argmax(_y_ort)),
                                        "ort_y_true_logit": float(
                                            _y_ort[_y_t]),
                                        "ort_y_rival_logit": float(
                                            _y_ort[int(_cand["rival"])]),
                                        "all_checks_pass": _attempt_pass,
                                    })
                                    if _attempt_pass:
                                        _all_pass = True
                                        _wr = int(_cand["rival"])
                                        _lp_m = float(_cand["lp_min"])
                                        _winning_attempt = _idx
                                        break
                                _receipt = {
                                    "source": "factor_aware_endcap_lp_topk_root_xi",
                                    "advisor_profile": "ACT_HZ_TOPK_RIVAL_WITNESS",
                                    "K": int(_topk_K),
                                    "snapshot_path": _snap_path,
                                    "onnx_path": str(onnx_p),
                                    "vnnlib_path": str(vnn_p),
                                    "y_true_from_vnnlib": int(_y_t),
                                    "n_candidates_tried": len(_cand_log),
                                    "winning_attempt_idx": _winning_attempt,
                                    "worst_rival": int(_wr),
                                    "lp_worst_min": float(_lp_m),
                                    "input_box_holds": (
                                        _cand_log[_winning_attempt][
                                            "input_box_holds"]
                                        if _winning_attempt is not None
                                        else False
                                    ),
                                    "vnnlib_query_holds": _all_pass,
                                    "spec_zero_tol_holds": _all_pass,
                                    "all_checks_pass": _all_pass,
                                    "ort_y_argmax": (
                                        int(_npw.argmax(_y_ort))
                                        if _y_ort is not None else -1),
                                    "ort_y_true": (
                                        float(_y_ort[_y_t])
                                        if _y_ort is not None
                                        else float("nan")),
                                    "ort_y_worst": (
                                        float(_y_ort[_wr])
                                        if _y_ort is not None and _wr >= 0
                                        else float("nan")),
                                    "candidate_log": _cand_log,
                                }
                                # Write receipt next to per_instance.
                                # 2026-06-03 (advisor Phase 1.5): include iid
                                # in the filename so per-iid receipts don't
                                # overwrite each other when multiple iids run
                                # into the same out-dir (the calibration sweep
                                # uses one out-dir per cap, not per iid).
                                _rdir = (
                                    os.environ.get("ACT_FAL_RECEIPT_DIR")
                                    or results_dir
                                    or _snap_dir
                                )
                                _iid_for_path = p.get(
                                    "official_instance_id", i)
                                _receipt_path = (
                                    Path(_rdir)
                                    / (f"endcap_witness_iid{_iid_for_path}"
                                       f"_q{len(q_statuses)}.json")
                                )
                                _receipt_path.parent.mkdir(
                                    parents=True, exist_ok=True)
                                with open(_receipt_path, "w") as _f_jsw:
                                    _jsw.dump(_receipt, _f_jsw, indent=2,
                                              default=float)
                                if _all_pass:
                                    q_internal = "SAT"
                                    _witness_receipt_path = str(_receipt_path)
                                    # Honor the formal-mode receipt contract:
                                    # this witness has full audit trail
                                    # (receipt JSON with input_box_holds /
                                    # vnnlib_query_holds / spec_zero_tol_holds
                                    # all True, independent ORT replay).
                                    # Mark it as REPORTABLE_FALSIFIED so
                                    # reportable_verdict_for_cli returns
                                    # FALSIFIED in both formal and non-formal
                                    # modes.
                                    solver._stats["formal_result"] = (
                                        "REPORTABLE_FALSIFIED"
                                    )
                                    solver._stats["fal_receipt_path"] = (
                                        str(_receipt_path)
                                    )
                                solver._stats[
                                    "cifar_endcap_witness_result"
                                ] = _receipt
                        except Exception as _e:
                            solver._stats[
                                "cifar_endcap_witness_error"
                            ] = f"{type(_e).__name__}: {_e}"
                _generic_mlp_on = os.environ.get(
                    "ACT_HZ_GENERIC_MLP_ENDCAP_WITNESS", "0"
                ).strip().lower() in ("1", "true", "yes", "on")
                # Diagnostic: record whether the sidecar would even consider
                # firing, and (if not) why. Useful when a structural gate
                # passes but no FAL is produced.
                _mlp_sidecar_diag = {
                    "generic_mlp_on_env": _generic_mlp_on,
                    "q_internal": q_internal,
                    "snap_dir_env": os.environ.get(
                        "ACT_HZ_ENDCAP_SNAPSHOT_DIR", ""),
                    "snap_glob_count": None,
                }
                if _generic_mlp_on and q_internal == "UNKNOWN":
                    _snap_dir = os.environ.get(
                        "ACT_HZ_ENDCAP_SNAPSHOT_DIR", "")
                    if _snap_dir:
                        try:
                            from pathlib import Path as _PMlp
                            _snap_glob = list(
                                _PMlp(_snap_dir).glob("L*_FLATTEN.pkl")
                            )
                            _mlp_sidecar_diag["snap_glob_count"] = len(
                                _snap_glob)
                            if _snap_glob:
                                import json as _jmlp
                                from types import SimpleNamespace as _NSMlp
                                from research.generic_mlp_endcap_reuse import (
                                    run as _run_generic_mlp_endcap,
                                )
                                _rdir = (
                                    os.environ.get("ACT_FAL_RECEIPT_DIR")
                                    or results_dir
                                    or _snap_dir
                                )
                                _safe_bench = str(p['category']).replace(
                                    "/", "_")
                                _safe_iid = int(p.get(
                                    'official_instance_id', i))
                                _result_path = (
                                    _PMlp(_rdir)
                                    / (
                                        f"generic_mlp_endcap_{_safe_bench}"
                                        f"_iid{_safe_iid}"
                                        f"_q{len(q_statuses)}.json"
                                    )
                                )
                                _args_mlp = _NSMlp(
                                    snapshot=str(_snap_glob[0]),
                                    onnx=str(onnx_p),
                                    vnnlib=str(vnn_p),
                                    out=str(_result_path),
                                    time_limit_s=float(os.environ.get(
                                        "ACT_HZ_MLP_ENDCAP_LP_TIME_LIMIT_S",
                                        "15.0",
                                    )),
                                    max_rivals=int(os.environ.get(
                                        "ACT_HZ_MLP_ENDCAP_MAX_RIVALS", "0"
                                    )),
                                    cert_tol=float(os.environ.get(
                                        "ACT_HZ_MLP_ENDCAP_CERT_TOL", "1e-8"
                                    )),
                                    fal_replay_threshold=float(os.environ.get(
                                        "ACT_HZ_MLP_ENDCAP_FAL_REPLAY_THRESHOLD",
                                        "0.0",
                                    )),
                                    progress_every=0,
                                )
                                _res_mlp = _run_generic_mlp_endcap(_args_mlp)
                                _PMlp(_result_path).parent.mkdir(
                                    parents=True, exist_ok=True)
                                with open(_result_path, "w") as _f_mlp:
                                    _jmlp.dump(_res_mlp, _f_mlp, indent=2,
                                               default=float)
                                solver._stats[
                                    "generic_mlp_endcap_result"
                                ] = {
                                    "verdict": _res_mlp.get("verdict"),
                                    "n_positive": _res_mlp.get("n_positive"),
                                    "n_rivals": _res_mlp.get("n_rivals"),
                                    "lp_min": _res_mlp.get("lp_min"),
                                    "worst_rival": _res_mlp.get("worst_rival"),
                                    "wall_s": _res_mlp.get("wall_s"),
                                    "result_path": str(_result_path),
                                }
                                # CERT promotion is DISABLED by default
                                # (advisor 2026-06-03): the canonical HZ
                                # verifier already handles CERT for the
                                # CIFAR/Tiny tail family; the LP-based CERT
                                # path is new attack surface that needs a
                                # separate audit. Opt-in via
                                # ACT_HZ_MLP_ENDCAP_ALLOW_CERT=1 only when
                                # research validation is in place.
                                _mlp_allow_cert = os.environ.get(
                                    "ACT_HZ_MLP_ENDCAP_ALLOW_CERT", "0"
                                ).strip().lower() in (
                                    "1", "true", "yes", "on"
                                )
                                if (_res_mlp.get("verdict") == "CERT"
                                        and _mlp_allow_cert):
                                    q_internal = "UNSAT"
                                    solver._stats["formal_result"] = (
                                        "REPORTABLE_CERTIFIED"
                                    )
                                elif _res_mlp.get("verdict") == "FAL":
                                    _fal = _res_mlp.get("fal_receipt") or {}
                                    if bool(_fal.get("all_checks_pass")):
                                        q_internal = "SAT"
                                        _witness_receipt_path = str(
                                            _result_path)
                                        solver._stats["formal_result"] = (
                                            "REPORTABLE_FALSIFIED"
                                        )
                                        solver._stats["fal_receipt_path"] = (
                                            str(_result_path)
                                        )
                        except Exception as _e:
                            solver._stats[
                                "generic_mlp_endcap_error"
                            ] = f"{type(_e).__name__}: {_e}"
                q_statuses.append(q_internal)
                q_reportables.append(
                    reportable_verdict_for_cli(solver, q_internal)
                )
                q_receipts.append(
                    _witness_receipt_path
                    or solver._stats.get("fal_receipt_path")
                )
                q_solver_stats.append({
                    "small_dense_lp_dispatch": solver._stats.get(
                        "small_dense_lp_dispatch"),
                    "small_dense_lp_backend": solver._stats.get(
                        "small_dense_lp_backend"),
                    "small_dense_lp_verdict": solver._stats.get(
                        "small_dense_lp_verdict"),
                    "small_dense_lp_elapsed_s": solver._stats.get(
                        "small_dense_lp_elapsed_s"),
                    "small_dense_lp_refinement_passes": solver._stats.get(
                        "small_dense_lp_refinement_passes"),
                    "small_dense_lp_direct_query": solver._stats.get(
                        "small_dense_lp_direct_query"),
                    "small_dense_lp_direct_query_kind": solver._stats.get(
                        "small_dense_lp_direct_query_kind"),
                    "small_dense_lp_direct_query_n_rows": solver._stats.get(
                        "small_dense_lp_direct_query_n_rows"),
                    "small_dense_lp_direct_query_unavailable": solver._stats.get(
                        "small_dense_lp_direct_query_unavailable"),
                    "small_dense_lp_direct_query_unavailable_reason": (
                        solver._stats.get(
                            "small_dense_lp_direct_query_unavailable_reason")
                    ),
                    "generic_mlp_endcap_result": solver._stats.get(
                        "generic_mlp_endcap_result"),
                    "generic_mlp_endcap_error": solver._stats.get(
                        "generic_mlp_endcap_error"),
                    "generic_mlp_endcap_gate": _mlp_gate_diag,
                    "generic_mlp_endcap_sidecar_diag": _mlp_sidecar_diag,
                    # 2026-06-03 P2.1 snapshot-writer diagnostics. Surfaces
                    # the hz_out type + attrs at every snapshot-attempt
                    # layer, plus any silent failures with traceback.
                    # Diagnoses the soundnessbench case where the structural
                    # gate fires but `snap_glob_count=0` because hz_out is
                    # SparseGcZ and the HZono-only writer raises
                    # AttributeError inside its own try/except.
                    "endcap_snapshot_diag": {
                        _k: solver._stats[_k]
                        for _k in solver._stats
                        if _k.startswith((
                            "endcap_snapshot@",
                            "endcap_snapshot_attempt@",
                            "endcap_snapshot_fail@",
                            "endcap_snapshot_traceback@",
                        ))
                    },
                    # 2026-06-03 (advisor Phase 1.5): expose the CIFAR
                    # narrow-profile receipt summary so aggregators can
                    # see lp_worst_min / ort_y_* per iid without re-reading
                    # the separate `endcap_witness_iid<n>_q*.json` file.
                    "cifar_endcap_witness_result": solver._stats.get(
                        "cifar_endcap_witness_result"),
                    "cifar_endcap_witness_diag": _cifar_witness_diag,
                    "cifar_endcap_witness_error": solver._stats.get(
                        "cifar_endcap_witness_error"),
                })
                # Short-circuit on FALSIFIED — any per-query SAT is sufficient.
                if q_internal == "SAT":
                    break
            # Aggregate per the disjunctive UNSAFE-set semantic — see
            # ``aggregate_query_statuses`` (internal math) and
            # ``aggregate_reportable_verdicts`` (formal-mode reportable).
            # In formal mode the reportable verdict may be ERROR_RECEIPT_*
            # even when the math says SAT — honest accounting: real
            # adversary exists but audit ledger is incomplete.
            internal_status = aggregate_query_statuses(q_statuses)
            reportable_status = aggregate_reportable_verdicts(q_reportables)
            # The reportable verdict is what enters the count buckets so
            # ERROR_RECEIPT_* is visible to operators. The internal status
            # is preserved via _stats for diagnostic.
            status_str = reportable_status if (
                os.environ.get("ACT_FAL_RECEIPT_FORMAL", "").strip().lower()
                in ("1", "true", "yes", "on")
            ) else internal_status
        except Exception as e:
            instance_error = f"{type(e).__name__}: {e}"
            if _known_unsupported_as_unknown(e):
                status_str = "UNKNOWN"
                instance_error = "UNSUPPORTED_AS_UNKNOWN: " + instance_error
            else:
                status_str = f"ERROR_{type(e).__name__}"
            if os.environ.get("ACT_VERIFY_VERBOSE") == "1":
                _tb.print_exc()

        elapsed = _time.time() - t0
        # ROUND 4 honesty (advisor 2026-05-24): formal-mode reportables
        # may be REPORTABLE_FALSIFIED, REPORTABLE_CERTIFIED, ERROR_RECEIPT_*,
        # ERROR_INTERNAL_INCONSISTENCY, or any normal HZ verdict. We
        # bucket ERROR_RECEIPT_* into a separate "ERROR_RECEIPT" count
        # so phantom-FAL audit is visible without polluting FAL count.
        normalized = _normalize_hz_status(status_str)
        if normalized == "ERROR_INTERNAL_INCONSISTENCY":
            key = "ERROR_INTERNAL_INCONSISTENCY"
            counts.setdefault(key, 0)
        elif normalized.startswith("ERROR_RECEIPT"):
            key = "ERROR_RECEIPT"
            counts.setdefault("ERROR_RECEIPT", 0)
        elif normalized in counts:
            key = normalized
        else:
            key = "ERROR"
        counts[key] = counts.get(key, 0) + 1
        per_instance.append({
            "official_instance_id": int(p.get("official_instance_id", i)),
            "benchmark": p.get("category", args.category),
            "onnx_model": p.get("onnx_model", ""),
            "vnnlib_spec": p.get("vnnlib_spec", ""),
            "model_path": onnx_p,
            "spec_path": vnn_p,
            "internal_status": internal_status,
            "reportable_status": reportable_status,
            "cli_normalized": normalized,
            "count_bucket": key,
            "queries": [
                {
                    "query_index": q_idx,
                    "internal_status": q_status,
                    "reportable_status": q_reportables[q_idx],
                    "receipt_path": q_receipts[q_idx],
                    "solver_stats": q_solver_stats[q_idx]
                    if q_idx < len(q_solver_stats) else {},
                }
                for q_idx, q_status in enumerate(q_statuses)
            ],
            "q_statuses": q_statuses,
            "q_reportables": q_reportables,
            "q_receipts": q_receipts,
            "q_solver_stats": q_solver_stats,
            "error": instance_error,
            "wall_s": float(elapsed),
            "timeout_s": timeout_s,
            "instance_budget_exhausted": instance_budget_exhausted,
        })
        print(f"  [{i + 1:3d}/{total}] {tag}: {normalized} ({elapsed:.1f}s)  "
              f"V={counts['CERTIFIED']} A={counts['FALSIFIED']} "
              f"U={counts['UNKNOWN']} E={counts['ERROR']}  "
              f"R={counts.get('ERROR_RECEIPT', 0)}", flush=True)
        # Restore env mutations made by per-iid profile auto-on so they
        # don't leak into the next iid in the same Python process.
        for _k, _prev in _iid_env_restore.items():
            if _prev is None:
                os.environ.pop(_k, None)
            else:
                os.environ[_k] = _prev
        # Cifar-endcap profile cleanup: remove the per-iid snapshot dir
        # and any L*_FLATTEN.pkl inside. Receipt JSONs were written to
        # ACT_FAL_RECEIPT_DIR / per_instance, NOT here, so they survive.
        if _cifar_endcap_snap_dir is not None:
            try:
                import shutil as _sh
                _sh.rmtree(_cifar_endcap_snap_dir, ignore_errors=True)
            except Exception:
                pass
        if _mlp_endcap_snap_dir is not None and results_dir:
            try:
                import shutil as _sh
                _sh.rmtree(_mlp_endcap_snap_dir, ignore_errors=True)
            except Exception:
                pass

    wall_min = (_time.time() - t_start) / 60.0
    print(f"\n[vnnlib/hybridz] FINAL — total={total} wall={wall_min:.1f} min")
    for k, v in counts.items():
        if v:
            print(f"  {k:12s} {v}")

    # ROUND 6 (advisor 2026-05-24): structured per-instance log + formal
    # exit-code contract. Two outputs:
    #   (a) Always: write per_instance JSON to ACT_FORMAL_RESULTS_DIR if
    #       set (or to ACT_FAL_RECEIPT_DIR/per_instance.json as fallback).
    #   (b) Formal mode: if R>0 or any INTERNAL_INCONSISTENCY, raise
    #       IncompleteFormalAuditError so cmd_verify maps to FAILED →
    #       sys.exit(1). The math verdicts are PRESERVED in the log.
    if results_dir:
        try:
            import json as _json
            import datetime as _datetime
            results_path = _Path(results_dir)
            results_path.mkdir(parents=True, exist_ok=True)
            ts = _datetime.datetime.now(_datetime.timezone.utc).strftime(
                "%Y%m%dT%H%M%S%fZ"
            )
            payload = {
                "schema_version": 1,
                "benchmark": args.category,
                "formal_mode": formal_mode,
                "timestamp_utc": ts,
                "canonical_root": _vnnlib_root,
                "receipt_dir": os.environ.get("ACT_FAL_RECEIPT_DIR"),
                "wall_min": wall_min,
                "counts": counts,
                "run_status": compute_run_status(counts, formal_mode),
                "per_instance": per_instance,
            }
            out_json = results_path / f"per_instance_{args.category}_{ts}.json"
            if out_json.exists():
                raise FileExistsError(f"refusing to overwrite {out_json}")
            tmp_json = out_json.with_suffix(out_json.suffix + ".tmp")
            tmp_json.write_text(_json.dumps(payload, indent=2, default=str))
            os.replace(tmp_json, out_json)
            print(f"  [structured] per-instance log → {out_json}")
        except Exception as e:
            print(f"  [structured] WARN: failed to write per-instance log: {e}",
                  flush=True)
            if formal_mode:
                raise

    run_status = compute_run_status(counts, formal_mode)
    if run_status == "INCOMPLETE_FORMAL_AUDIT":
        r_n = counts.get("ERROR_RECEIPT", 0)
        ii_n = counts.get("ERROR_INTERNAL_INCONSISTENCY", 0)
        raise IncompleteFormalAuditError(
            f"INCOMPLETE_FORMAL_AUDIT: ERROR_RECEIPT={r_n}, "
            f"ERROR_INTERNAL_INCONSISTENCY={ii_n}. "
            "Per-instance log preserved; internal math verdicts intact. "
            "Re-check receipt directory + instance_id propagation."
        )
    if run_status == "FAILED":
        raise RuntimeError(
            f"VERIFICATION_RUN_FAILED: ERROR={counts.get('ERROR', 0)}. "
            "Per-instance log preserved; inspect instance error fields."
        )


@contextmanager
def _sliced_net_view(net, sample_idx: int, batch_size: int):
    """Yield a per-sample view of ``net`` with spec/assert/input layers sliced.

    On exit, original layer params/out_vars are restored. Safer than inline
    try/finally because mutation surface is encapsulated.
    """
    from act.back_end.verifier import (
        find_entry_layer_id,
        gather_input_spec_layers,
        get_assert_layer,
    )

    assert_layer = get_assert_layer(net)
    spec_layers = gather_input_spec_layers(net)
    input_layer = net.by_id[find_entry_layer_id(net)]
    full_input_ids = list(input_layer.out_vars)
    input_dim = len(full_input_ids) // batch_size
    if len(full_input_ids) != input_dim * batch_size:
        raise RuntimeError(
            f"InputLayer.out_vars ({len(full_input_ids)}) not divisible by B={batch_size}"
        )

    orig_assert_params = deepcopy(assert_layer.params)
    orig_spec_params = [deepcopy(spec_layer.params) for spec_layer in spec_layers]
    orig_input_outvars = list(input_layer.out_vars)
    try:
        for key in OutputSpec.SLICEABLE_PARAM_KEYS:
            val = orig_assert_params.get(key)
            if (
                val is not None
                and hasattr(val, "dim")
                and val.dim() >= 1
                and val.shape[0] == batch_size
            ):
                assert_layer.params[key] = val[sample_idx : sample_idx + 1]

        for spec_layer, sp_orig in zip(spec_layers, orig_spec_params):
            for sp_key, sp_val in sp_orig.items():
                if (
                    hasattr(sp_val, "dim")
                    and sp_val.dim() >= 1
                    and sp_val.shape[0] == batch_size
                ):
                    spec_layer.params[sp_key] = sp_val[sample_idx : sample_idx + 1]

        input_layer.out_vars = full_input_ids[
            sample_idx * input_dim : (sample_idx + 1) * input_dim
        ]
        yield net
    finally:
        assert_layer.params = orig_assert_params
        for spec_layer, sp_orig in zip(spec_layers, orig_spec_params):
            spec_layer.params = sp_orig
        input_layer.out_vars = orig_input_outvars


def _run_bab_on_net(net, args, bab_first_sample_only: bool = False):
    """Verify an ACT Net via verify_bab_batched.

    For single-sample wrappers (B=1) returns one status string.
    For multi-sample wrappers (B>1, e.g. TorchVision), the behavior depends
    on ``bab_first_sample_only``:
      - True  → only sample 0 is verified (one local-robustness instance —
                the BaB-natural unit), returning a single status string.
      - False → all B samples are verified via per-sample iteration,
                returning a list of status strings.
    """
    from act.back_end.bab.bab import verify_bab_batched
    from act.back_end.config import BaBConfig
    from act.back_end.solver.solver_torchlp import TorchLPSolver
    from act.back_end.verifier import (
        gather_input_spec_layers,
        seed_from_input_specs,
    )

    config = BaBConfig(
        solver_tier=args.bab_solver_tier,
        max_depth=args.bab_max_depth,
        max_nodes=args.bab_max_nodes,
        branching_method=getattr(args, "bab_branching_method", "random"),
        bounding_method=getattr(args, "bab_bounding_method", "random"),
        bounding_order=getattr(args, "bab_bounding_order", "depth_lb"),
        sa_cooling_rate=getattr(args, "bab_sa_cooling_rate", 0.99),
        frontier_cap=getattr(args, "bab_frontier_cap", 0),
        input_split_fanout=getattr(args, "bab_input_split_fanout", 2),
        per_class_alpha=(
            str(getattr(args, "bab_per_class_alpha", "true")).lower() == "true"
        ),
        incremental_start_enabled=not getattr(args, "bab_no_incremental_start", False),
        provenance_enabled=getattr(args, "bab_provenance", False),
    )
    budget = float(getattr(args, "timeout", 60.0) or 60.0)

    spec_layers = gather_input_spec_layers(net)
    seed_bounds = seed_from_input_specs(spec_layers)
    B = seed_bounds.lb.shape[0] if seed_bounds.lb.dim() >= 2 else 1

    if B <= 1:
        result = verify_bab_batched(
            net=net,
            solver_factory=TorchLPSolver,
            config=config,
            max_batch_size=None,
            time_budget_s=budget,
        )
        return result.status.name

    sample_range = range(1) if bab_first_sample_only else range(B)

    statuses = []
    for sample_idx in sample_range:
        with _sliced_net_view(net, sample_idx, B) as sliced_net:
            result = verify_bab_batched(
                net=sliced_net,
                solver_factory=TorchLPSolver,
                config=config,
                max_batch_size=None,
                time_budget_s=budget,
            )
            statuses.append(result.status.name)
    return statuses[0] if bab_first_sample_only and statuses else statuses


def _run_torchvision_verify(args) -> bool:
    """Drive ``verify_once`` over a TorchVision dataset-model pair end-to-end.

    Bridges the front-end load → ACT-Net path for TorchVision the same way
    ``_run_vnnlib_verify`` does for VNNLIB benchmarks:
    ``TorchVisionSpecCreator`` → ``synthesize_models_from_specs`` →
    ``TorchToACT`` → ``verify_once``.  Single-mode per invocation, matching
    the ``act.back_end --verify`` CLI contract.

    All three solvers (interval+torchlp, hybridz+torchlp, dual) are
    supported on TorchVision smoke (MNIST + simple_cnn at 224×224). The
    dual track auto-falls back to interval-only at layers whose input
    dim exceeds ``_DENSE_LIN_BOUND_MAX_DIM`` (see ``tf_forward.py``) to
    avoid materializing the dense linear-bound matrix at high dims.
    """
    from act.front_end.torchvision_loader.create_specs import TorchVisionSpecCreator
    from act.front_end.model_synthesis import synthesize_models_from_specs
    from act.pipeline.verification.torch2act import TorchToACT
    from act.back_end.verifier import verify_once
    from act.back_end.transfer_functions import (
        set_solver_mode,
        set_transfer_function_mode,
    )
    from act.pipeline.verification.validate_verifier import VerificationValidator

    if not args.dataset:
        raise ValueError("--verify torchvision requires --dataset (e.g. --dataset MNIST)")

    tf_mode = (args.tf_modes or ["interval"])[0]
    solver = (args.solvers or ["torchlp"])[0]

    set_solver_mode(solver)
    if solver != "dual":
        set_transfer_function_mode(tf_mode)
    label = solver if solver == "dual" else f"{tf_mode}/{solver}"
    model_label = args.model or "<all>"
    print(
        f"[torchvision] dataset={args.dataset} model={model_label} "
        f"num_samples={args.num_samples} mode={label}"
    )

    spec_results = TorchVisionSpecCreator().create_specs_for_data_model_pairs(
        dataset_names=[args.dataset],
        model_names=[args.model] if args.model else None,
        num_samples=args.num_samples,
    )
    if not spec_results:
        raise RuntimeError(
            f"TorchVisionSpecCreator produced no spec_results for "
            f"dataset={args.dataset!r}, model={args.model!r}"
        )

    wrapped = synthesize_models_from_specs(spec_results)
    if not wrapped:
        raise RuntimeError("synthesize_models_from_specs produced no VerifiableModels")

    if getattr(args, "bab", False):
        local_robust = [
            (mid, vm) for mid, vm in wrapped.items() if "LINF_BALL" in tuple(str(p) for p in mid)
        ]
        if not local_robust:
            local_robust = list(wrapped.items())
        mid, vm = local_robust[0]
        tag = "/".join(str(p) for p in mid)
        net = TorchToACT(vm).run()
        status = _run_bab_on_net(net, args, bab_first_sample_only=True)
        label = f"BaB[{args.bab_solver_tier}]"
        print(f"  {tag} (sample 0 / local-robustness): {label} → {status}")
        return False

    validator = None
    soundness_summary = None
    if args.validate_soundness:
        dtype = torch.float64 if args.dtype == "float64" else torch.float32
        validator = VerificationValidator(device=args.device, dtype=dtype)
    for mid, vm in wrapped.items():
        tag = "/".join(str(p) for p in mid)
        net = TorchToACT(vm).run()
        results = verify_once(net)
        statuses = [r.status.name for r in results]
        print(f"  {tag}: {statuses}")
        if args.validate_soundness:
            assert validator is not None
            soundness_summary = _run_soundness_check(
                tag, vm, net, results, validator, solver
            )

    if args.validate_soundness:
        assert validator is not None and soundness_summary is not None
        soundness_summary = validator._compute_summary(validation_type="counterexample")
        _print_soundness_summary(soundness_summary)
        return soundness_summary["failed"] > 0
    return False


def cmd_verify(target: str, args):
    """Run verification tests from the verification submodule."""
    print_header()

    from act.pipeline.verification import model_factory, torch2act

    tests_to_run = []
    if target == "all":
        tests_to_run = ["act2torch", "torch2act"]
    else:
        tests_to_run = [target]

    results = {}

    for test_name in tests_to_run:
        print(f"\n{'=' * 80}")
        if test_name == "act2torch":
            print(f"VERIFICATION TEST: ACT→PyTorch Conversion")
            print(f"{'=' * 80}\n")
            try:
                model_factory.main()
                results[test_name] = "PASSED"
            except Exception as e:
                print(f"\n❌ Test failed: {e}")
                import traceback

                traceback.print_exc()
                results[test_name] = "FAILED"

        elif test_name == "torch2act":
            print(f"VERIFICATION TEST: PyTorch→ACT Conversion")
            print(f"{'=' * 80}\n")
            try:
                torch2act.main()
                results[test_name] = "PASSED"
            except Exception as e:
                print(f"\n❌ Test failed: {e}")
                import traceback

                traceback.print_exc()
                results[test_name] = "FAILED"

        elif test_name == "vnnlib":
            print(f"VERIFICATION TEST: VNNLIB → VerifiableModel → verify_once")
            print(f"{'=' * 80}\n")
            try:
                soundness_failed = _run_vnnlib_verify(args)
                results[test_name] = "FAILED" if soundness_failed else "PASSED"
            except Exception as e:
                print(f"\n❌ Test failed: {e}")
                import traceback

                traceback.print_exc()
                results[test_name] = "FAILED"

        elif test_name == "torchvision":
            print(f"VERIFICATION TEST: TorchVision → VerifiableModel → verify_once")
            print(f"{'=' * 80}\n")
            try:
                soundness_failed = _run_torchvision_verify(args)
                results[test_name] = "FAILED" if soundness_failed else "PASSED"
            except Exception as e:
                print(f"\n❌ Test failed: {e}")
                import traceback

                traceback.print_exc()
                results[test_name] = "FAILED"

    # Print summary
    print(f"\n{'=' * 80}")
    print(f"VERIFICATION TEST SUMMARY")
    print(f"{'=' * 80}")
    for test_name, result in results.items():
        status = "✅" if result == "PASSED" else "❌"
        print(f"  {status} {test_name:25s} {result}")
    print(f"{'=' * 80}\n")

    # Exit with error if any test failed
    if any(r == "FAILED" for r in results.values()):
        sys.exit(1)


def _resolve_batch_sizes(cli_value):
    """CLI flag > YAML ``validate.batch_sizes`` > built-in default ``[None]``.

    The ``[None]`` fallback means "validate each network at its native
    batch size from JSON only" (no batchification).
    """
    if cli_value:
        return cli_value
    try:
        import yaml
        from act.util.path_config import get_project_root
        cfg_path = (
            Path(get_project_root())
            / "act/back_end/examples/config_gen_act_net.yaml"
        )
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text()) or {}
            yaml_val = (cfg.get("validate") or {}).get("batch_sizes")
            if yaml_val:
                return yaml_val
    except Exception as e:
        # Intentional: optional YAML override; missing/malformed files fall through to default [None].
        logger.debug("suppressed: %s", e)
    return [None]


def cmd_validate_verifier(args):
    """Run verifier validation with specified mode.

    Args:
        mode: validation mode (counterexample, bounds, comprehensive)
        networks: list of networks to validate (default: all)
        solvers: list of solvers to use (default: gurobi torchlp)
        tf_modes: list of transfer function modes to use (default: interval)
        samples: number of samples to use (default: 10)
        per_neuron_topk: number of worst per-neuron violations to report
    """
    import torch
    from act.pipeline.verification.validate_verifier import VerificationValidator

    print_header()

    # Convert dtype string to torch dtype
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    # Create validator
    validator = VerificationValidator(device=args.device, dtype=dtype)

    # Parse networks if specified
    networks = args.networks.split(",") if args.networks else None

    try:
        per_neuron_config = PerNeuronCheckConfig(topk=int(args.per_neuron_topk))
        batch_sizes = _resolve_batch_sizes(getattr(args, "batch_sizes", None))
        if args.mode == "counterexample":
            summary = validator.validate_counterexamples(
                networks=networks,
                solvers=args.solvers,
                tf_modes=args.tf_modes,
                batch_sizes=batch_sizes,
            )
            exit_code = (
                0
                if args.ignore_errors
                else (
                    1 if (summary["failed"] > 0 or summary.get("errors", 0) > 0) else 0
                )
            )
        elif args.mode == "bounds":
            summary = validator.validate_bounds(
                networks=networks,
                tf_modes=args.tf_modes,
                num_samples=args.samples,
                per_neuron_config=per_neuron_config,
                batch_sizes=batch_sizes,
            )
            exit_code = (
                0
                if args.ignore_errors
                else (
                    1 if (summary["failed"] > 0 or summary.get("errors", 0) > 0) else 0
                )
            )
        else:
            combined = validator.validate_comprehensive(
                networks=networks,
                solvers=args.solvers,
                tf_modes=args.tf_modes,
                num_samples=args.samples,
                per_neuron_config=per_neuron_config,
                batch_sizes=batch_sizes,
            )
            exit_code = (
                0
                if args.ignore_errors
                else (1 if combined["overall_status"] in ("FAILED", "ERROR") else 0)
            )

        sys.exit(exit_code)

    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m act.pipeline",
        description="ACT Pipeline: Inference-based whitebox fuzzing for neural networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available VNNLIB categories
  python -m act.pipeline --list
  
  # Search for benchmarks
  python -m act.pipeline --search acas
  
  # Get detailed information
  python -m act.pipeline --info acasxu_2023
  
  # Download data-model pairs
  python -m act.pipeline --download acasxu_2023
  
  # List downloaded pairs
  python -m act.pipeline --list-downloaded
  
  # Fuzz VNNLIB benchmark
  python -m act.pipeline --fuzz --category acasxu_2023 --iterations 5000
  
  # Fuzz TorchVision dataset
  python -m act.pipeline --fuzz --creator torchvision --dataset MNIST
  
  # Run verification tests
  python -m act.pipeline --verify act2torch --device cpu
  python -m act.pipeline --verify torch2act --device cpu
  python -m act.pipeline --verify all --device cpu

  # Run verifier on a VNNLIB benchmark end-to-end (load → ACT → verify_once).
  # Single (tf, solver) per invocation; matrix sweeps by repeated calls.
  python -m act.pipeline --verify vnnlib --category acasxu_2023 --max-instances 3 --tf-modes interval --solvers torchlp
  python -m act.pipeline --verify vnnlib --category acasxu_2023 --max-instances 3 --tf-modes hybridz --solvers torchlp
  python -m act.pipeline --verify vnnlib --category acasxu_2023 --max-instances 3                          --solvers dual
  python -m act.pipeline --verify vnnlib --category tinyimagenet_2024 --max-instances 3                    --solvers hybridz   # HZ-native verifier (verify_once_hz)

  # Run verifier on a TorchVision dataset-model pair end-to-end.
  python -m act.pipeline --verify torchvision --dataset MNIST --model simple_cnn --num-samples 2 --tf-modes interval --solvers torchlp
  python -m act.pipeline --verify torchvision --dataset MNIST --model simple_cnn --num-samples 2 --tf-modes hybridz  --solvers torchlp
  python -m act.pipeline --verify torchvision --dataset MNIST --model simple_cnn --num-samples 2                     --solvers dual
  
  # Run verifier validation (comprehensive by default)
  python -m act.pipeline --validate-verifier --device cpu --dtype float64
  python -m act.pipeline --validate-verifier --mode counterexample
  python -m act.pipeline --validate-verifier --mode bounds --input-samples 20
  python -m act.pipeline --validate-verifier --mode bounds --per-neuron-topk 20
        """,
    )

    # Command selection (mutually exclusive)
    cmd_group = parser.add_mutually_exclusive_group(required=True)
    cmd_group.add_argument(
        "--list", "-l", action="store_true", help="List available datasets/categories"
    )
    cmd_group.add_argument(
        "--search",
        "-s",
        type=str,
        metavar="QUERY",
        help="Search for datasets/categories",
    )
    cmd_group.add_argument(
        "--info", "-i", type=str, metavar="NAME", help="Show detailed information"
    )
    cmd_group.add_argument(
        "--download", "-d", type=str, metavar="NAME", help="Download dataset/category"
    )
    cmd_group.add_argument(
        "--list-downloaded",
        action="store_true",
        help="List downloaded data-model pairs",
    )
    cmd_group.add_argument("--fuzz", "-f", action="store_true", help="Run ACTFuzzer")
    cmd_group.add_argument(
        "--verify",
        type=str,
        metavar="TARGET",
        choices=["act2torch", "torch2act", "vnnlib", "torchvision", "all"],
        help="Run verification tests: act2torch, torch2act, vnnlib, torchvision, "
        "or all. The 'vnnlib' target runs the verifier on a VNNLIB benchmark "
        "end-to-end (requires --category); 'torchvision' does the same for a "
        "TorchVision dataset-model pair (requires --dataset, optionally --model). "
        "Both read the FIRST element of --tf-modes / --solvers (single mode per "
        "invocation; matrix sweeps by repeated calls).",
    )
    cmd_group.add_argument(
        "--validate-verifier",
        action="store_true",
        help="Run verifier validation (counterexample and bounds checking)",
    )
    cmd_group.add_argument(
        "--list-verifications",
        action="store_true",
        help="List available verification tests",
    )

    # Creator selection
    parser.add_argument(
        "--creator",
        "-c",
        type=str,
        choices=["vnnlib", "torchvision"],
        default="vnnlib",
        help="Spec creator (default: vnnlib)",
    )

    # VNNLIB-specific options
    vnnlib_group = parser.add_argument_group("VNNLIB Options")
    vnnlib_group.add_argument(
        "--category", type=str, help="VNNLIB category to fuzz (e.g., acasxu_2023)"
    )
    vnnlib_group.add_argument(
        "--max-instances",
        type=int,
        default=10,
        help="Max VNNLIB instances to load (default: 10)",
    )
    vnnlib_group.add_argument(
        "--instance-ids",
        type=str,
        help=(
            "Comma-separated official instance ids for reproducible VNNLIB "
            "sentinel/audit runs (hybridz path); applied before --max-instances"
        ),
    )

    # TorchVision-specific options
    tv_group = parser.add_argument_group("TorchVision Options")
    tv_group.add_argument(
        "--dataset", type=str, help="TorchVision dataset to fuzz (e.g., MNIST)"
    )
    tv_group.add_argument(
        "--model", type=str, help="TorchVision model to fuzz (e.g., simple_cnn)"
    )
    tv_group.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to load (default: 10)",
    )

    bab_group = parser.add_argument_group("Branch-and-Bound Options (--verify {vnnlib,torchvision})")
    bab_group.add_argument(
        "--bab",
        action="store_true",
        help="Run BaB (verify_bab_batched) instead of single-shot verify_once",
    )
    bab_group.add_argument(
        "--bab-solver-tier",
        type=str,
        default="dual_alpha_eta",
        choices=list(VALID_SOLVER_TIERS),
        help=(
            "BaB solver tier when --bab is set (default: dual_alpha_eta). "
            "'lp' uses the existing LP/MILP backend; 'dual' uses DualSolver "
            "single-pass; 'dual_alpha' adds Lagrange-relaxed lower-slope "
            "optimization; 'dual_alpha_eta' adds joint slope + split-constraint "
            "KKT multipliers."
        ),
    )
    bab_group.add_argument(
        "--bab-max-depth",
        type=int,
        default=8,
        help="Maximum BaB tree depth (default: 8)",
    )
    bab_group.add_argument(
        "--bab-max-nodes",
        type=int,
        default=100,
        help="Maximum BaB nodes to expand (default: 100)",
    )
    bab_group.add_argument(
        "--bab-branching-method",
        type=str,
        default="random",
        choices=["random", "babsr", "fsb"],
        help="BaB branching strategy when --bab is set (default: random)",
    )
    bab_group.add_argument(
        "--bab-bounding-method",
        type=str,
        default="random",
        choices=["random", "topk"],
        help=(
            "Pool selection when subproblems exceed the batch size: 'random' or "
            "'topk' (keep the top-k by depth + lower-bound). Default: random."
        ),
    )
    bab_group.add_argument(
        "--bab-bounding-order",
        type=str,
        default="depth_lb",
        choices=["depth_lb", "greedy", "sa"],
        help="TopKBounding order policy (default: depth_lb)",
    )
    bab_group.add_argument(
        "--bab-sa-cooling-rate",
        type=float,
        default=0.99,
        help="Cooling rate for --bab-bounding-order sa (default: 0.99)",
    )
    bab_group.add_argument(
        "--bab-per-class-alpha",
        type=str,
        default="true",
        choices=["true", "false"],
        help=(
            "Per-spec α tensor (True; tighter bounds, M× memory) vs shared α "
            "across specs (False; looser, 1× memory). Default: true."
        ),
    )
    bab_group.add_argument(
        "--bab-no-incremental-start",
        action="store_true",
        help="Disable parent→child α/η incremental-start propagation (debugging / ablation).",
    )
    bab_group.add_argument(
        "--bab-frontier-cap",
        type=int,
        default=0,
        help="Maximum pending BaB frontier leaves to retain; 0 disables eviction (default: 0)",
    )
    bab_group.add_argument(
        "--bab-input-split-fanout",
        type=int,
        default=2,
        help="Uniform fanout for input splits; 2 preserves binary splitting (default: 2)",
    )
    bab_group.add_argument(
        "--bab-provenance",
        action="store_true",
        help="Enable node_id/parent_id provenance sidecar (requires --bab-bounding-method topk).",
    )

    # Fuzzing configuration
    fuzz_group = parser.add_argument_group("Fuzzing Options")
    fuzz_group.add_argument(
        "--iterations",
        type=int,
        default=10000,
        help="Max fuzzing iterations (default: 10000)",
    )
    fuzz_group.add_argument(
        "--timeout",
        type=float,
        default=3600.0,
        help="Timeout in seconds (default: 3600)",
    )
    fuzz_group.add_argument(
        "--output",
        type=str,
        default="fuzzing_results",
        help="Output directory (default: fuzzing_results)",
    )
    fuzz_group.add_argument(
        "--no-save", action="store_true", help="Don't save counterexamples to disk"
    )
    fuzz_group.add_argument(
        "--report-interval",
        type=int,
        default=100,
        help="Report progress every N iterations (default: 100)",
    )
    fuzz_group.add_argument(
        "--strict-mode",
        action="store_true",
        help="Enable strict mode: raise errors on input/output constraint violations (default: False)",
    )

    # Tracing options
    trace_group = parser.add_argument_group("Execution Tracing Options")
    trace_group.add_argument(
        "--trace-level",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
        help="Tracing detail level: 0=disabled (default), 1=basic (iteration metrics + inputs), "
        "2=full (+ layer activations), 3=debug (+ gradients and loss)",
    )
    trace_group.add_argument(
        "--trace-sample",
        type=int,
        default=1,
        metavar="N",
        help="Capture every Nth iteration (default: 1 = all iterations). "
        "Use higher values to reduce overhead (e.g., 10 = every 10th iteration)",
    )
    trace_group.add_argument(
        "--trace-storage",
        type=str,
        choices=["hdf5", "json"],
        default="json",
        help="Storage backend: json=text/readable (default), hdf5=binary/compressed",
    )
    trace_group.add_argument(
        "--trace-output",
        type=str,
        help="Custom trace output path (default: <output-dir>/traces.{hdf5|json})",
    )

    # Validation options
    validation_group = parser.add_argument_group("Validation Options")
    validation_group.add_argument(
        "--mode",
        type=str,
        choices=["counterexample", "bounds", "comprehensive"],
        default="comprehensive",
        help="Validation mode (default: comprehensive)",
    )
    validation_group.add_argument(
        "--networks",
        type=str,
        help="Comma-separated list of networks to validate (default: all)",
    )
    validation_group.add_argument(
        "--solvers",
        nargs="+",
        default=["gurobi", "torchlp"],
        help="Solvers for Level 1 validation (default: gurobi torchlp)",
    )
    validation_group.add_argument(
        "--tf-modes",
        nargs="+",
        default=["interval"],
        help="Transfer function modes for Level 2 bounds validation: interval, hybridz, dual (default: interval)",
    )
    validation_group.add_argument(
        "--input-samples",
        type=int,
        default=10,
        dest="samples",
        help="Number of input samples for Level 2 bounds validation (default: 10)",
    )
    validation_group.add_argument(
        "--per-neuron-topk",
        type=int,
        default=10,
        metavar="K",
        help="Number of worst per-neuron violations to report (default: 10). "
        "The bounds check itself is zero-tolerance — any deviation outside "
        "[lb, ub] is flagged as unsound.",
    )
    validation_group.add_argument(
        "--batch-sizes",
        type=lambda s: [
            (None if (b.strip() == "" or b.strip().lower() == "none") else int(b))
            for b in s.split(",")
        ],
        default=None,
        metavar="B1,B2,...",
        help="Batch sizes to validate at, e.g. '1,4'. Use 'none' for the "
        "network's native batch (from JSON). When omitted, falls back to "
        "the ``validate.batch_sizes`` list in config_gen_act_net.yaml, "
        "then to ``[None]`` (native only).",
    )
    validation_group.add_argument(
        "--ignore-errors",
        action="store_true",
        help="Always exit 0 (ignore failures and errors for CI)",
    )

    verify_group = parser.add_argument_group("Verify Options")
    verify_group.add_argument(
        "--validate-soundness",
        action="store_true",
        help="After --verify vnnlib/torchvision, run concrete-counterexample soundness validation on the same instances",
    )

    # Add standard device/dtype arguments (shared across all ACT CLIs)
    add_device_args(parser)

    args = parser.parse_args()

    # Initialize device manager from CLI arguments
    initialize_from_args(args)

    # Handle --dataset as alias for --category (for VNNLIB)
    # This provides a more intuitive interface: python -m act.pipeline --fuzz --dataset cifar100_2024
    if args.creator == "vnnlib" and args.dataset and not args.category:
        args.category = args.dataset

    # Execute command
    try:
        if args.list:
            cmd_list_available(args.creator)
        elif args.search:
            cmd_search(args.search, args.creator)
        elif args.info:
            cmd_info(args.info, args.creator)
        elif args.download:
            cmd_download(args.download, args.creator)
        elif args.list_downloaded:
            cmd_list_downloaded(args.creator)
        elif args.fuzz:
            cmd_fuzz(args)
        elif args.verify:
            cmd_verify(args.verify, args)
        elif args.validate_verifier:
            cmd_validate_verifier(args)
        elif args.list_verifications:
            cmd_list_verifications()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
