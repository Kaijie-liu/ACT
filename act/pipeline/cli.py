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
import logging
from pathlib import Path
from typing import List, Optional
import sys

from act.util.cli_utils import add_device_args, initialize_from_args

logger = logging.getLogger(__name__)
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
    overrides = dict(
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


def _run_vnnlib_verify(args) -> None:
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

    for mid, vm in wrapped.items():
        tag = "/".join(str(p) for p in mid)
        net = TorchToACT(vm).run()
        results = verify_once(net)
        statuses = [r.status.name for r in results]
        print(f"  {tag}: {statuses}")


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
        instance_error: Optional[str] = None
        instance_budget_start = _time.monotonic()
        instance_budget_exhausted = False
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
            # T2 ablation knob: env ACT_HZ_EQ_LAYERS overrides the default
            # last-N eq_lagr_v8 layer count for large_cls scheduling. Default 3.
            _eq_layers = int(os.environ.get("ACT_HZ_EQ_LAYERS", "3"))
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
                solver = HZVerifier(
                    device=device, dtype=dtype, timeout_s=remaining_s,
                    strict_replay=True, onnx_path=onnx_p,
                    vnnlib_path=vnn_p,
                    # Use official_instance_id (row position in instances.csv
                    # after correct header detection), NOT the filtered-loop
                    # index ``i``. The two differ when filter / max-instances
                    # is applied or when the CSV is headerless. Per advisor
                    # 2026-05-24 Round 3, receipts MUST carry official ids.
                    instance_id=int(p.get('official_instance_id', i)),
                    query_index=q_idx,
                    benchmark=p['category'],
                    large_cls_eq_layers=_eq_layers,
                    small_dense_lp=os.environ.get("ACT_HZ_SMALL_DENSE_LP", "auto"),
                    small_dense_lp_root=os.environ.get("ACT_HZ_SMALL_DENSE_LP_ROOT"),
                    small_dense_lp_time_limit_s=float(
                        os.environ.get("ACT_HZ_SMALL_DENSE_LP_TIME_LIMIT_S", "5.0")
                    ),
                    small_dense_lp_refinement_passes=int(
                        os.environ.get("ACT_HZ_SMALL_DENSE_LP_REFINEMENT_PASSES", "0")
                    ),
                    small_dense_lp_fallback_on_unknown=(
                        os.environ.get("ACT_HZ_SMALL_DENSE_LP_FALLBACK", "0") == "1"
                    ),
                )
                q_status, _, _ = verify_once_hz(
                    net=net, solver=solver, timelimit=remaining_s
                )
                q_internal = str(q_status)
                q_statuses.append(q_internal)
                q_reportables.append(
                    reportable_verdict_for_cli(solver, q_internal)
                )
                q_receipts.append(solver._stats.get("fal_receipt_path"))
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
            status_str = f"ERROR_{type(e).__name__}"
            instance_error = f"{type(e).__name__}: {e}"
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
                }
                for q_idx, q_status in enumerate(q_statuses)
            ],
            "q_statuses": q_statuses,
            "q_reportables": q_reportables,
            "q_receipts": q_receipts,
            "error": instance_error,
            "wall_s": float(elapsed),
            "timeout_s": timeout_s,
            "instance_budget_exhausted": instance_budget_exhausted,
        })
        print(f"  [{i + 1:3d}/{total}] {tag}: {normalized} ({elapsed:.1f}s)  "
              f"V={counts['CERTIFIED']} A={counts['FALSIFIED']} "
              f"U={counts['UNKNOWN']} E={counts['ERROR']}  "
              f"R={counts.get('ERROR_RECEIPT', 0)}", flush=True)

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


def _run_torchvision_verify(args) -> None:
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

    for mid, vm in wrapped.items():
        tag = "/".join(str(p) for p in mid)
        net = TorchToACT(vm).run()
        results = verify_once(net)
        statuses = [r.status.name for r in results]
        print(f"  {tag}: {statuses}")


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
                _run_vnnlib_verify(args)
                results[test_name] = "PASSED"
            except Exception as e:
                print(f"\n❌ Test failed: {e}")
                import traceback

                traceback.print_exc()
                results[test_name] = "FAILED"

        elif test_name == "torchvision":
            print(f"VERIFICATION TEST: TorchVision → VerifiableModel → verify_once")
            print(f"{'=' * 80}\n")
            try:
                _run_torchvision_verify(args)
                results[test_name] = "PASSED"
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
