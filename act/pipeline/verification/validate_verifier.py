#!/usr/bin/env python3
#===- act/pipeline/validate_verifier.py - Verifier Correctness Validation ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Unified verification validation framework with two validation levels:
#
#   Level 1: Counterexample/Soundness Validation
#     - Validates that verifier doesn't claim CERTIFIED when concrete 
#       counterexamples exist
#
#   Level 2: Bounds/Numerical Validation
#     - Validates that abstract bounds correctly overapproximate concrete 
#       activation values
#
#===---------------------------------------------------------------------===#
#
# Level 1: Counterexample/Soundness Validation
# ============================================
#
# Key Insight:
#   Concrete execution provides ground truth - if we find a real counterexample
#   at runtime, the formal verifier cannot claim the property is certified.
#   This is a soundness check for the verification backend.
#
# Validation Strategy:
#   1. For each network, generate strategic test cases:
#      - Center: Input at center of input spec (typically safe)
#      - Boundary: Input near boundary of input spec (risky)
#      - Random: Random input within input spec (varied)
#
#   2. Run concrete execution to find violations
#   3. If counterexample found, run formal verification
#   4. Cross-validate using matrix below
#
# Validation Matrix (Level 1):
#   ┌─────────────────────────┬────────────────────────────────────┬──────────────┐
#   │ Concrete Counterexample │ Verifier Result                    │ Validation   │
#   ├─────────────────────────┼────────────────────────────────────┼──────────────┤
#   │ FOUND                   │ CERTIFIED                          │ ❌ FAILED    │
#   │                         │ (Soundness Bug - false negative)   │              │
#   ├─────────────────────────┼────────────────────────────────────┼──────────────┤
#   │ FOUND                   │ FALSIFIED                          │ ✅ PASSED    │
#   │                         │ (Correct - verifier found issue)   │              │
#   ├─────────────────────────┼────────────────────────────────────┼──────────────┤
#   │ FOUND                   │ UNKNOWN                            │ ⚠️ ACCEPTABLE│
#   │                         │ (Incomplete but sound)             │              │
#   ├─────────────────────────┼────────────────────────────────────┼──────────────┤
#   │ NOT FOUND               │ Any Result                         │ ❓ INCONC.   │
#   │                         │ (Cannot validate - no ground truth)│              │
#   └─────────────────────────┴────────────────────────────────────┴──────────────┘
#
#   Legend:
#     FAILED       - Critical soundness bug (false negative)
#     PASSED       - Verifier correct
#     ACCEPTABLE   - Verifier incomplete but sound (conservative)
#     INCONCLUSIVE - No concrete counterexample to validate against
#
#===---------------------------------------------------------------------===#
#
# Level 2: Bounds/Numerical Validation
# ====================================
#
# Key Insight:
#   Abstract interpretation must overapproximate concrete values. If any
#   concrete activation value falls outside its abstract bounds [lb, ub],
#   the transfer function is unsound.
#
# Validation Strategy:
#   1. Sample concrete inputs from input specification
#   2. Run concrete forward pass through PyTorch model → get concrete activations
#   3. Run abstract analysis through ACT → get abstract bounds for each layer
#   4. Check: concrete_value ∈ [lb, ub] for all layers and all neurons
#
# Validation Matrix (Level 2):
#   ┌──────────────────────┬────────────────────────┬──────────────┐
#   │ Concrete Values      │ Abstract Bounds        │ Validation   │
#   ├──────────────────────┼────────────────────────┼──────────────┤
#   │ value ∈ [lb, ub]     │ All layers/neurons     │ ✅ PASSED    │
#   │ (Sound bounds)       │                        │              │
#   ├──────────────────────┼────────────────────────┼──────────────┤
#   │ value ∉ [lb, ub]     │ Any layer/neuron       │ ❌ FAILED    │
#   │ (Unsound bounds)     │ (Transfer function bug)│              │
#   └──────────────────────┴────────────────────────┴──────────────┘
#
#   Legend:
#     PASSED - All concrete values within abstract bounds (sound)
#     FAILED - Concrete value outside bounds (unsound transfer function)
#
#===---------------------------------------------------------------------===#
#
# Usage:
#   # Via CLI (recommended):
#   python -m act.pipeline --validate-verifier --mode comprehensive
#   python -m act.pipeline --validate-verifier --mode counterexample
#   python -m act.pipeline --validate-verifier --mode bounds
#   
#   # With device and dtype specification:
#   python -m act.pipeline --validate-verifier --device cpu --dtype float64
#   python -m act.pipeline --validate-verifier --device cuda --dtype float32
#   
#   # Test specific networks:
#   python -m act.pipeline --validate-verifier --networks mnist_mlp_small
#   python -m act.pipeline --validate-verifier --networks mnist_mlp_small,mnist_cnn_small
#   
#   # Test with specific solvers (Level 1):
#   python -m act.pipeline --validate-verifier --mode counterexample --solvers gurobi
#   python -m act.pipeline --validate-verifier --mode counterexample --solvers gurobi torchlp
#   
#   # Test with transfer function modes (Level 2):
#   python -m act.pipeline --validate-verifier --mode bounds --tf-modes interval
#   python -m act.pipeline --validate-verifier --mode bounds --tf-modes interval hybridz
#   
#   # Adjust number of samples for bounds validation:
#   python -m act.pipeline --validate-verifier --mode bounds --samples 20
#   
#   # Ignore errors and always exit 0 (useful for CI):
#   python -m act.pipeline --validate-verifier --ignore-errors
#   
#   # Combined options:
#   python -m act.pipeline --validate-verifier --mode comprehensive \
#       --networks mnist_mlp_small,mnist_cnn_small \
#       --solvers gurobi --tf-modes interval --samples 10 \
#       --device cpu --dtype float64
#
#   # Direct execution (legacy):
#   python act/pipeline/verification/validate_verifier.py
#   python act/pipeline/verification/validate_verifier.py --mode bounds --samples 5
#
# Exit Codes:
#   0 - All validations passed (no failures or errors)
#   0 - With --ignore-errors flag (always succeed regardless of results)
#   1 - Failures detected (verifier bugs) OR errors detected (backend bugs)
#
#===---------------------------------------------------------------------===#

import os
import copy
import time
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from act.pipeline.verification.model_factory import ModelFactory
from act.pipeline.verification.torch2act import TorchToACT
from act.pipeline.verification.activations import collect_concrete_activations
from act.pipeline.verification.bounds_core import compute_abstract_bounds
from act.pipeline.verification.layer_alignment import align_activations_to_act_layers
from act.pipeline.verification.bounds_compare import compare_bounds_per_neuron
from act.back_end.verifier import verify_once, gather_input_spec_layers, seed_from_input_specs, get_input_ids, get_assert_layer, find_entry_layer_id
from act.back_end.analyze import analyze
from act.back_end.solver.solver_gurobi import GurobiSolver
from act.back_end.solver.solver_torch import TorchLPSolver
from act.util.options import PerformanceOptions
from act.front_end.specs import OutKind

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VerificationValidator:
    """Unified verification validation framework with counterexample and bounds validation."""
    
    def __init__(
        self,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float64
    ):
        """
        Initialize verification validator.
        
        Args:
            device: Device for computation ('cpu' or 'cuda')
            dtype: Data type for computation (float32 or float64)
        """
        self.factory = ModelFactory()
        self.device = device
        self.dtype = dtype
        self.validation_results = []
        
        # Initialize debug file (GUARDED)
        if PerformanceOptions.debug_tf:
            debug_file = PerformanceOptions.debug_output_file
            with open(debug_file, 'w') as f:
                f.write(f"ACT Verification Debug Log\n")
                f.write(f"Device: {device}, Dtype: {dtype}\n")
                f.write(f"{'='*80}\n\n")
            logger.info(f"Debug logging to: {debug_file}")
            
    def find_concrete_counterexample(
        self,
        name: str,
        model: torch.nn.Module,
        max_random: int = 64,
    ) -> Optional[Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Try to find a concrete counterexample via concrete execution.
        Returns (input_tensor, results_dict) if found, else None.
        """
        if max_random < 0:
            raise ValueError(f"max_random must be >= 0, got {max_random}")
        was_training = bool(getattr(model, "training", False))
        model.eval()

        try:
            act_net = self.factory.get_act_net(name)
            input_shape = None
            shape_prod = None
            if act_net is not None:
                for layer in getattr(act_net, "layers", []):
                    if getattr(layer, "kind", None) == "INPUT":
                        shp = (layer.meta or {}).get("shape", None)
                        if (isinstance(shp, (list, tuple)) and shp and all(isinstance(x, int) and x > 0 for x in shp)):
                            input_shape = tuple(shp)
                            shape_prod = int(torch.tensor(input_shape).prod().item())
                        break

            spec_lb = spec_ub = None
            if act_net is not None:
                specs = gather_input_spec_layers(act_net)
                if specs:
                    seed = seed_from_input_specs(specs)
                    lb = seed.lb.to(device=self.device, dtype=self.dtype).flatten()
                    ub = seed.ub.to(device=self.device, dtype=self.dtype).flatten()
                    if lb.shape == ub.shape and lb.numel() > 0 and (not torch.any(lb > ub)):
                        spec_lb, spec_ub = lb, ub

            if spec_lb is None or spec_ub is None:
                return None

            delta = spec_ub - spec_lb
            dim = int(spec_lb.numel())

            # center
            x_flat = spec_lb + 0.5 * delta
            x = x_flat.reshape(*input_shape) if (input_shape and shape_prod == x_flat.numel()) else x_flat.reshape(1, -1)
            x = x.to(device=self.device, dtype=self.dtype)
            with torch.no_grad():
                res = model(x)
            if isinstance(res, dict) and res.get("input_satisfied", False) and (not res.get("output_satisfied", True)):
                logger.info("  🔴 Counterexample found (spec_center)")
                logger.info("     Input explanation:  %s", res.get("input_explanation"))
                logger.info("     Output explanation: %s", res.get("output_explanation"))
                return x, res

            # per-dimension edges (dim<=16)
            if dim <= 16:
                base = spec_lb + 0.5 * delta
                for i in range(dim):
                    for val, tag in ((spec_lb[i], "lb"), (spec_ub[i], "ub")):
                        x_edge = base.clone()
                        x_edge[i] = val
                        x = x_edge.reshape(*input_shape) if (input_shape and shape_prod == x_edge.numel()) else x_edge.reshape(1, -1)
                        x = x.to(device=self.device, dtype=self.dtype)
                        with torch.no_grad():
                            res = model(x)
                        if isinstance(res, dict) and res.get("input_satisfied", False) and (not res.get("output_satisfied", True)):
                            logger.info("  🔴 Counterexample found (spec_per_dim_%s_%d)", tag, i)
                            logger.info("     Input explanation:  %s", res.get("input_explanation"))
                            logger.info("     Output explanation: %s", res.get("output_explanation"))
                            return x, res

            # random in [lb, ub]
            for k in range(max_random):
                r = torch.rand_like(spec_lb)
                x_flat = spec_lb + r * delta
                x = x_flat.reshape(*input_shape) if (input_shape and shape_prod == x_flat.numel()) else x_flat.reshape(1, -1)
                x = x.to(device=self.device, dtype=self.dtype)
                with torch.no_grad():
                    res = model(x)
                if isinstance(res, dict) and res.get("input_satisfied", False) and (not res.get("output_satisfied", True)):
                    logger.info("  🔴 Counterexample found (spec_random_%d)", k)
                    logger.info("     Input explanation:  %s", res.get("input_explanation"))
                    logger.info("     Output explanation: %s", res.get("output_explanation"))
                    return x, res
                
            return None

        finally:
            if was_training:
                model.train()
    
    def validate_counterexamples(
        self, 
        networks: Optional[List[str]] = None,
        solvers: List[str] = ['gurobi', 'torchlp']
    ) -> Dict[str, Any]:
        """
        Level 1: Validate verifier soundness using concrete counterexamples.
        
        Args:
            networks: List of network names (None = all networks)
            solvers: List of solver names to test
            
        Returns:
            Summary dictionary with validation results
        """
        if networks is None:
            networks = self.factory.list_networks()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"LEVEL 1: COUNTEREXAMPLE/SOUNDNESS VALIDATION")
        logger.info(f"{'='*80}")
        logger.info(f"Testing {len(networks)} networks with {len(solvers)} solvers")
        logger.info(f"Device: {self.device}, Dtype: {self.dtype}")
        logger.info(f"{'='*80}\n")
        
        for network in networks:
            for solver in solvers:
                try:
                    self._validate_counterexample_single(network, solver)
                except Exception as e:
                    logger.error(f"Validation failed for {network}/{solver}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Add error result if not already added
                    error_result = {
                        'network': network,
                        'solver': solver,
                        'validation_type': 'counterexample',
                        'status': 'ERROR',
                        'error': f"Outer exception: {str(e)}",
                        'concrete_counterexample': False
                    }
                    self.validation_results.append(error_result)
        
        return self._compute_summary(validation_type='counterexample')
    
    def _validate_counterexample_single(
        self, 
        name: str, 
        solver: str
    ) -> Dict[str, Any]:
        """
        Validate verifier correctness for a single network (Level 1).
        
        Args:
            name: Network name from examples_config.yaml
            solver: 'gurobi' or 'torchlp'
            
        Returns:
            Validation result dictionary with status and details
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Validating: {name} (solver: {solver})")
        logger.info(f"{'='*80}")
        
        # Step 1: Load ACT Net from factory
        act_net = self.factory.get_act_net(name)
        
        # Step 2: Create PyTorch model for concrete execution
        model = self.factory.create_model(name, load_weights=True)
        model = model.to(device=self.device, dtype=self.dtype)
        counterexample = self.find_concrete_counterexample(name, model)
        
        # Step 3: Run formal verifier on ACT Net
        logger.info(f"\n  🔍 Running formal verifier ({solver})...")
        
        try:
            if solver == 'gurobi':
                solver_instance = GurobiSolver()
            elif solver == 'torchlp':
                solver_instance = TorchLPSolver()
            else:
                raise ValueError(f"Unknown solver: {solver}")
            
            verify_result = verify_once(act_net, solver=solver_instance)
            verifier_status = verify_result.status
            logger.info(f"     Verifier result: {verifier_status}")
            
            # If verifier found counterexample, validate it with model
            if verify_result.counterexample is not None:
                logger.info(f"     Verifier counterexample shape: {verify_result.counterexample.shape}")
                # Reshape CE to the model's expected input shape (avoid conv2d shape errors)
                ce_raw = verify_result.counterexample
                input_shape = None
                for layer in act_net.layers:
                    if getattr(layer, "kind", None) == "INPUT":
                        input_shape = layer.meta.get("shape")
                        break
                try:
                    if input_shape is not None:
                        ce_tensor = ce_raw.view(*input_shape)
                    else:
                        ce_tensor = ce_raw.unsqueeze(0)
                except Exception as reshape_err:
                    logger.warning(f"     CE reshape failed, using vector: {reshape_err}")
                    ce_tensor = ce_raw.unsqueeze(0)
                ce_tensor = ce_tensor.to(device=self.device, dtype=self.dtype)
                ce_results = model(ce_tensor)
                if isinstance(ce_results, dict):
                    logger.info(f"     CE validation: input_sat={ce_results['input_satisfied']}, "
                              f"output_sat={ce_results['output_satisfied']}")
            
        except Exception as e:
            logger.error(f"     Verifier failed: {e}")
            import traceback
            traceback.print_exc()
            error_result = {
                'network': name,
                'solver': solver,
                'validation_type': 'counterexample',
                'status': 'ERROR',
                'error': str(e),
                'concrete_counterexample': counterexample is not None
            }
            self.validation_results.append(error_result)
            return error_result
        
        # Step 4: Cross-validate results
        validation = self._cross_validate_counterexample(
            network_name=name,
            solver_name=solver,
            concrete_counterexample=counterexample,
            verifier_status=verifier_status
        )
        
        self.validation_results.append(validation)
        return validation
    
    def _cross_validate_counterexample(
        self,
        network_name: str,
        solver_name: str,
        concrete_counterexample: Optional[Tuple],
        verifier_status: str
    ) -> Dict[str, Any]:
        """
        Cross-validate concrete inference vs formal verification (Level 1).
        
        Validation Rules:
        1. If concrete counterexample found → verifier MUST report FALSIFIED or UNKNOWN
        2. If no concrete counterexample → verifier can report anything (testing incomplete)
        """
        result = {
            'network': network_name,
            'solver': solver_name,
            'validation_type': 'counterexample',
            'concrete_counterexample': concrete_counterexample is not None,
            'verifier_result': verifier_status,
            'validation_status': None,
            'explanation': None
        }
        
        if concrete_counterexample is not None:
            # We found a real counterexample - verifier MUST NOT claim CERTIFIED
            input_tensor, inference_results = concrete_counterexample
            
            if verifier_status == 'CERTIFIED':
                # CRITICAL BUG: Verifier claims safe, but we have a counterexample!
                result['validation_status'] = 'FAILED'
                result['explanation'] = (
                    f"🚨 SOUNDNESS BUG DETECTED! Verifier claims CERTIFIED but "
                    f"concrete counterexample exists. This is a false negative."
                )
                logger.error(f"\n  {result['explanation']}")
                logger.error(f"     Counterexample input: {input_tensor.shape}, "
                            f"range=[{input_tensor.min():.4f}, {input_tensor.max():.4f}]")
                logger.error(f"     Output violation: {inference_results['output_explanation']}")
                
            elif verifier_status == 'FALSIFIED':
                # CORRECT: Verifier correctly identified the issue
                result['validation_status'] = 'PASSED'
                result['explanation'] = (
                    f"✅ CORRECT - Verifier correctly reported FALSIFIED "
                    f"(matches concrete execution)"
                )
                logger.info(f"\n  {result['explanation']}")
                
            elif verifier_status == 'UNKNOWN':
                # ACCEPTABLE: Verifier couldn't decide (incomplete but sound)
                result['validation_status'] = 'ACCEPTABLE'
                result['explanation'] = (
                    f"⚠️ INCOMPLETE - Verifier returned UNKNOWN, but concrete "
                    f"counterexample exists (verifier is sound but incomplete)"
                )
                logger.warning(f"\n  {result['explanation']}")
                
            else:
                result['validation_status'] = 'UNKNOWN'
                result['explanation'] = f"Unknown verifier result: {verifier_status}"
                logger.warning(f"\n  {result['explanation']}")
        
        else:
            # No concrete counterexample found in testing
            result['validation_status'] = 'INCONCLUSIVE'
            result['explanation'] = (
                f"⚪ INCONCLUSIVE - No counterexample found in concrete testing. "
                f"Verifier result: {verifier_status} (cannot validate with this test)"
            )
            logger.info(f"\n  {result['explanation']}")
        
        return result
    
    def validate_bounds(
        self,
        networks: Optional[List[str]] = None,
        tf_modes: List[str] = ['interval'],
        num_samples: int = 10
    ) -> Dict[str, Any]:
        """
        Level 2: Validate abstract bounds overapproximate concrete values.
        
        Args:
            networks: List of network names (None = all networks)
            tf_modes: Transfer function modes to test ('interval', 'hybridz')
            num_samples: Number of concrete inputs to sample per network
            
        Returns:
            Summary dictionary with validation results
        """
        if networks is None:
            networks = self.factory.list_networks()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"LEVEL 2: BOUNDS/NUMERICAL VALIDATION")
        logger.info(f"{'='*80}")
        logger.info(f"Testing {len(networks)} networks with {len(tf_modes)} TF modes")
        logger.info(f"Samples per network: {num_samples}")
        logger.info(f"Device: {self.device}, Dtype: {self.dtype}")
        logger.info(f"{'='*80}\n")
        
        for network in networks:
            for tf_mode in tf_modes:
                try:
                    self._validate_bounds_single(network, tf_mode, num_samples)
                except Exception as e:
                    logger.error(f"Bounds validation failed for {network}/{tf_mode}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Add error result if not already added
                    error_result = {
                        'network': network,
                        'tf_mode': tf_mode,
                        'validation_type': 'bounds',
                        'status': 'ERROR',
                        'error': f"Outer exception: {str(e)}",
                        'samples_processed': 0
                    }
                    self.validation_results.append(error_result)
        
        return self._compute_summary(validation_type='bounds')
    
    def _validate_bounds_single(
        self,
        name: str,
        tf_mode: str,
        num_samples: int
    ) -> Dict[str, Any]:
        """
        Validate bounds for a single network (Level 2).
        
        Args:
            name: Network name
            tf_mode: Transfer function mode ('interval' or 'hybridz')
            num_samples: Number of concrete inputs to sample
            
        Returns:
            Validation result dictionary
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Validating bounds: {name} (tf_mode: {tf_mode})")
        logger.info(f"{'='*80}")
        
        # Step 1: Load ACT Net and PyTorch model
        act_net = self.factory.get_act_net(name)
        model = self.factory.create_model(name, load_weights=True)
        model = model.to(device=self.device, dtype=self.dtype)
        
        # Step 2: Set transfer function mode globally
        from act.back_end.transfer_functions import set_transfer_function_mode
        set_transfer_function_mode(tf_mode)
        
        # Step 3: Sample concrete inputs
        violations = []
        total_checks = 0
        
        def _get_input_bounds_from_act(act_net_inner):
            from act.back_end.core import Bounds
            for layer in act_net_inner.layers:
                if layer.kind != "INPUT_SPEC":
                    continue

                params = layer.params or {}
                meta = layer.meta or {}

                # 1) Prefer BOX
                if "lb" in params and "ub" in params:
                    return Bounds(
                        lb=params["lb"].flatten().to(device=self.device, dtype=self.dtype),
                        ub=params["ub"].flatten().to(device=self.device, dtype=self.dtype),
                    )

                # 2) LINF_BALL: center + eps
                if "center" in params and "eps" in meta:
                    center = params["center"].flatten().to(device=self.device, dtype=self.dtype)
                    eps = meta["eps"]
                    if not torch.is_tensor(eps):
                        eps = torch.tensor(eps, device=self.device, dtype=self.dtype)
                    else:
                        eps = eps.to(device=self.device, dtype=self.dtype)
                    return Bounds(lb=center - eps, ub=center + eps)
            return None
        
        spec_bounds = _get_input_bounds_from_act(act_net)
        
        for sample_idx in range(num_samples):
            # Generate random input within spec
            input_tensor = self.factory.generate_test_input(name, 'random')
            input_tensor = input_tensor.to(device=self.device, dtype=self.dtype)
            
            # Step 4: Get concrete activations via forward hooks
            concrete_activations = self._get_concrete_activations(model, input_tensor, act_net)
            
            # Step 5: Prepare entry fact from input tensor
            from act.back_end.core import Fact, Bounds, ConSet
            # entry_id = 0  # INPUT layer is typically layer 0
            entry_id = find_entry_layer_id(act_net)
            if spec_bounds is not None:
                input_bounds = spec_bounds
            else:
                input_bounds = Bounds(lb=input_tensor.flatten(), ub=input_tensor.flatten())
            # Use an empty constraint set for inputs so downstream analysis
            # never iterates over a None cons field.
            entry_fact = Fact(bounds=input_bounds, cons=None)
            
            # Step 6: Run abstract analysis
            try:
                before, after, globalC = analyze(act_net, entry_id, entry_fact)
                
                # Step 7: Check bounds containment
                for layer_id, concrete_vals in concrete_activations.items():
                    if layer_id not in after:
                        continue
                    
                    abstract_bounds = after[layer_id].bounds
                    lb = abstract_bounds.lb
                    ub = abstract_bounds.ub
                    
                    # Flatten concrete values to match ACT's 1D representation
                    concrete_vals_flat = concrete_vals.flatten()
                    
                    # Ensure shapes match (ACT may have different neuron counts)
                    if concrete_vals_flat.shape != lb.shape:
                        logger.warning(f"  ⚠️ Shape mismatch at layer {layer_id}: "
                                     f"concrete={concrete_vals_flat.shape}, abstract={lb.shape}. Skipping.")
                        continue
                    
                    # Check if concrete values are within bounds
                    violations_mask = (concrete_vals_flat < lb) | (concrete_vals_flat > ub)
                    num_violations = violations_mask.sum().item()
                    total_checks += concrete_vals_flat.numel()
                    
                    if num_violations > 0:
                        violation_info = {
                            'sample_idx': sample_idx,
                            'layer_id': layer_id,
                            'num_violations': num_violations,
                            'total_neurons': concrete_vals_flat.numel(),
                            'concrete_min': concrete_vals_flat.min().item(),
                            'concrete_max': concrete_vals_flat.max().item(),
                            'abstract_lb': lb.min().item(),
                            'abstract_ub': ub.max().item()
                        }
                        violations.append(violation_info)
                        logger.error(f"  ❌ Bounds violation at layer {layer_id}: "
                                   f"{num_violations}/{concrete_vals_flat.numel()} neurons")
            
            except Exception as e:
                logger.error(f"  ⚠️ Abstract analysis failed for sample {sample_idx}: {e}")
                error_result = {
                    'network': name,
                    'tf_mode': tf_mode,
                    'validation_type': 'bounds',
                    'status': 'ERROR',
                    'error': str(e),
                    'samples_processed': sample_idx
                }
                self.validation_results.append(error_result)
                return error_result
        
        # Step 6: Summarize results
        if len(violations) > 0:
            result = {
                'network': name,
                'tf_mode': tf_mode,
                'validation_type': 'bounds',
                'validation_status': 'FAILED',
                'explanation': f"🚨 UNSOUND BOUNDS: {len(violations)} violations found across {num_samples} samples",
                'total_checks': total_checks,
                'violations': violations
            }
            logger.error(f"\n  {result['explanation']}")
        else:
            result = {
                'network': name,
                'tf_mode': tf_mode,
                'validation_type': 'bounds',
                'validation_status': 'PASSED',
                'explanation': f"✅ SOUND BOUNDS: All {total_checks} checks passed across {num_samples} samples",
                'total_checks': total_checks,
                'violations': []
            }
            logger.info(f"\n  {result['explanation']}")
        
        self.validation_results.append(result)
        return result

    def validate_bounds_per_neuron(
        self,
        networks: Optional[List[str]] = None,
        tf_modes: List[str] = ['interval'],
        num_samples: int = 1,
        *,
        atol: float = 1e-6,
        rtol: float = 0.0,
        topk: int = 10,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Level 2 (strict): Per-neuron bounds validation with alignment.
        """
        if networks is None:
            networks = self.factory.list_networks()

        logger.info(f"\n{'='*80}")
        logger.info(f"LEVEL 2: BOUNDS PER-NEURON VALIDATION (STRICT)")
        logger.info(f"{'='*80}")
        logger.info(f"Testing {len(networks)} networks with {len(tf_modes)} TF modes")
        logger.info(f"Samples per network: {num_samples}")
        logger.info(f"Device: {self.device}, Dtype: {self.dtype}")
        logger.info(f"{'='*80}\n")

        for network in networks:
            for tf_mode in tf_modes:
                try:
                    self._validate_bounds_per_neuron_single(
                        network,
                        tf_mode,
                        num_samples,
                        atol=atol,
                        rtol=rtol,
                        topk=topk,
                        strict=strict,
                    )
                except Exception as e:
                    logger.error(f"Bounds-per-neuron validation failed for {network}/{tf_mode}: {e}")
                    import traceback
                    traceback.print_exc()
                    error_result = {
                        'network': network,
                        'tf_mode': tf_mode,
                        'validation_type': 'bounds_per_neuron',
                        'status': 'ERROR',
                        'error': f"Outer exception: {str(e)}",
                        'samples_processed': 0,
                        'strict': bool(strict),
                    }
                    self.validation_results.append(error_result)

        return self._compute_summary(validation_type='bounds_per_neuron')

    def _validate_bounds_per_neuron_single(
        self,
        name: str,
        tf_mode: str,
        num_samples: int,
        *,
        atol: float,
        rtol: float,
        topk: int,
        strict: bool,
    ) -> Dict[str, Any]:
        """
        Validate per-neuron bounds for a single network (strict alignment).
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Validating bounds per-neuron: {name} (tf_mode: {tf_mode})")
        logger.info(f"{'='*80}")
        network_start = time.perf_counter()

        from act.back_end.core import Bounds, Fact

        act_net = self.factory.get_act_net(name)
        model = self.factory.create_model(name, load_weights=True)
        model = model.to(device=self.device, dtype=self.dtype)

        was_training = bool(getattr(model, "training", False))
        model.eval()

        def _get_input_bounds_from_act(act_net_inner):
            for layer in act_net_inner.layers:
                if layer.kind != "INPUT_SPEC":
                    continue

                params = layer.params or {}
                meta = layer.meta or {}

                if "lb" in params and "ub" in params:
                    return Bounds(
                        lb=params["lb"].flatten().to(device=self.device, dtype=self.dtype),
                        ub=params["ub"].flatten().to(device=self.device, dtype=self.dtype),
                    )

                if "center" in params and "eps" in meta:
                    center = params["center"].flatten().to(device=self.device, dtype=self.dtype)
                    eps = meta["eps"]
                    if not torch.is_tensor(eps):
                        eps = torch.tensor(eps, device=self.device, dtype=self.dtype)
                    else:
                        eps = eps.to(device=self.device, dtype=self.dtype)
                    return Bounds(lb=center - eps, ub=center + eps)
            return None

        spec_bounds = _get_input_bounds_from_act(act_net)
        total_checks = 0
        total_violations = 0
        alignment_meta: Optional[Dict[str, Any]] = None
        errors: List[str] = []
        warnings: List[str] = []

        stats_by_layer: Dict[int, Dict[str, Any]] = {}
        candidate_violations: List[Dict[str, Any]] = []
        samples_processed = 0
        samples_failed = 0
        samples_error = 0
        worst_gap = 0.0
        worst_layer: Optional[Dict[str, Any]] = None
        missing_bounds_all: List[int] = []

        try:
            for sample_idx in range(num_samples):
                sample_start = time.perf_counter()
                samples_processed = sample_idx + 1
                input_tensor = self.factory.generate_test_input(name, 'random')
                input_tensor = input_tensor.to(device=self.device, dtype=self.dtype)

                if spec_bounds is not None:
                    input_bounds = spec_bounds
                else:
                    input_bounds = Bounds(
                        lb=input_tensor.flatten(),
                        ub=input_tensor.flatten(),
                    )
                entry_fact = Fact(bounds=input_bounds, cons=None)

                bounds_by_layer, bounds_errors = compute_abstract_bounds(
                    act_net,
                    entry_fact,
                    tf_mode=tf_mode,
                )
                sample_errors: List[str] = []
                sample_warnings: List[str] = []
                missing_bounds: List[int] = []
                align_events = 0
                align_mapped = 0
                align_drop_batch = 0
                align_err = 0
                align_warn = 0
                if bounds_errors:
                    sample_errors.extend([f"sample {sample_idx}: {e}" for e in bounds_errors])
                    if strict:
                        errors.extend(sample_errors)
                        samples_error += 1
                        sample_status = "ERROR"
                        sample_checks = 0
                        sample_viol = 0
                        sample_max_gap = 0.0
                        sample_time = time.perf_counter() - sample_start
                        logger.info(
                            "\n  ❌ [L2][bounds_per_neuron] net=%s tf=%s sample=%d/%d "
                            "status=%s checks=%d viol=%d max_gap=%.2e "
                            "align={events=%d mapped=%d drop_batch=%d err=%d warn=%d} time=%.2fs",
                            name,
                            tf_mode,
                            samples_processed,
                            num_samples,
                            sample_status,
                            sample_checks,
                            sample_viol,
                            sample_max_gap,
                            align_events,
                            align_mapped,
                            align_drop_batch,
                            align_err,
                            align_warn,
                            sample_time,
                        )
                        logger.error("❌ [L2][bounds_per_neuron] net=%s tf=%s ERROR details:", name, tf_mode)
                        for e in sample_errors[:3]:
                            logger.error("  - %s", e)
                        break
                    errors.extend(sample_errors)

                events, event_errors, event_warnings = collect_concrete_activations(
                    model,
                    input_tensor,
                )
                if event_errors:
                    sample_errors.extend([f"sample {sample_idx}: {e}" for e in event_errors])
                    if strict:
                        errors.extend(sample_errors)
                        samples_error += 1
                        sample_status = "ERROR"
                        sample_checks = 0
                        sample_viol = 0
                        sample_max_gap = 0.0
                        sample_time = time.perf_counter() - sample_start
                        logger.info(
                            "\n  ❌ [L2][bounds_per_neuron] net=%s tf=%s sample=%d/%d "
                            "status=%s checks=%d viol=%d max_gap=%.2e "
                            "align={events=%d mapped=%d drop_batch=%d err=%d warn=%d} time=%.2fs",
                            name,
                            tf_mode,
                            samples_processed,
                            num_samples,
                            sample_status,
                            sample_checks,
                            sample_viol,
                            sample_max_gap,
                            len(events),
                            0,
                            0,
                            len(sample_errors),
                            len(event_warnings),
                            sample_time,
                        )
                        logger.error("❌ [L2][bounds_per_neuron] net=%s tf=%s ERROR details:", name, tf_mode)
                        for e in sample_errors[:3]:
                            logger.error("  - %s", e)
                        break
                    errors.extend(sample_errors)
                if event_warnings:
                    sample_warnings.extend([f"sample {sample_idx}: {w}" for w in event_warnings])
                    warnings.extend(sample_warnings)

                alignment = align_activations_to_act_layers(
                    act_net,
                    events,
                    mode="hookable_order_strict",
                )
                if alignment_meta is None:
                    alignment_meta = alignment.meta
                if alignment.meta:
                    align_events = int(alignment.meta.get("hookable_events", len(events)))
                    align_mapped = int(alignment.meta.get("hookable_layers", len(alignment.mapping)))
                    align_drop_batch = int(alignment.meta.get("drop_batch", 0))
                    align_err = int(alignment.meta.get("err", len(alignment.errors)))
                    align_warn = int(
                        alignment.meta.get(
                            "warn",
                            len(event_warnings) + len(alignment.warnings),
                        )
                    )
                else:
                    align_events = len(events)
                    align_mapped = len(alignment.mapping)
                    align_drop_batch = 0
                    align_err = len(alignment.errors)
                    align_warn = len(event_warnings) + len(alignment.warnings)
                if alignment.errors:
                    sample_errors.extend([f"sample {sample_idx}: {e}" for e in alignment.errors])
                    if strict:
                        errors.extend(sample_errors)
                        samples_error += 1
                        sample_status = "ERROR"
                        sample_checks = 0
                        sample_viol = 0
                        sample_max_gap = 0.0
                        sample_time = time.perf_counter() - sample_start
                        logger.info(
                            "\n  ❌ [L2][bounds_per_neuron] net=%s tf=%s sample=%d/%d "
                            "status=%s checks=%d viol=%d max_gap=%.2e "
                            "align={events=%d mapped=%d drop_batch=%d err=%d warn=%d} time=%.2fs",
                            name,
                            tf_mode,
                            samples_processed,
                            num_samples,
                            sample_status,
                            sample_checks,
                            sample_viol,
                            sample_max_gap,
                            align_events,
                            align_mapped,
                            align_drop_batch,
                            align_err,
                            align_warn,
                            sample_time,
                        )
                        logger.error("❌ [L2][bounds_per_neuron] net=%s tf=%s ERROR details:", name, tf_mode)
                        for e in sample_errors[:3]:
                            logger.error("  - %s", e)
                        logger.error(
                            "  alignment: events=%d mapped=%d drop_batch=%d err=%d warn=%d",
                            align_events,
                            align_mapped,
                            align_drop_batch,
                            align_err,
                            align_warn,
                        )
                        break
                    errors.extend(sample_errors)

                concrete_by_layer = {lid: ev.tensor for lid, ev in alignment.mapping.items()}
                missing_bounds = [lid for lid in concrete_by_layer.keys() if lid not in bounds_by_layer]
                if missing_bounds:
                    missing_bounds_all.extend(missing_bounds)
                    sample_errors.append(
                        f"sample {sample_idx}: missing bounds for layer_ids={sorted(missing_bounds)}"
                    )
                    if strict:
                        errors.extend(sample_errors)
                        samples_error += 1
                        sample_status = "ERROR"
                        sample_checks = 0
                        sample_viol = 0
                        sample_max_gap = 0.0
                        sample_time = time.perf_counter() - sample_start
                        logger.info(
                            "\n  ❌ [L2][bounds_per_neuron] net=%s tf=%s sample=%d/%d "
                            "status=%s checks=%d viol=%d max_gap=%.2e "
                            "align={events=%d mapped=%d drop_batch=%d err=%d warn=%d} time=%.2fs",
                            name,
                            tf_mode,
                            samples_processed,
                            num_samples,
                            sample_status,
                            sample_checks,
                            sample_viol,
                            sample_max_gap,
                            align_events,
                            align_mapped,
                            align_drop_batch,
                            align_err,
                            align_warn,
                            sample_time,
                        )
                        logger.error("❌ [L2][bounds_per_neuron] net=%s tf=%s ERROR details:", name, tf_mode)
                        for e in sample_errors[:3]:
                            logger.error("  - %s", e)
                        logger.error("  missing_bounds_layer_ids=%s", sorted(missing_bounds))
                        logger.error(
                            "  alignment: events=%d mapped=%d drop_batch=%d err=%d warn=%d",
                            align_events,
                            align_mapped,
                            align_drop_batch,
                            align_err,
                            align_warn,
                        )
                        break
                    errors.extend(sample_errors)

                bounds_for_compare = {lid: bounds_by_layer[lid] for lid in concrete_by_layer.keys()}
                compare = compare_bounds_per_neuron(
                    bounds_by_layer=bounds_for_compare,
                    concrete_by_layer=concrete_by_layer,
                    atol=atol,
                    rtol=rtol,
                    topk=topk,
                    nan_policy="error" if strict else "warn",
                )
                if compare.get("status") == "ERROR":
                    sample_errors.extend([f"sample {sample_idx}: {e}" for e in compare.get("errors", [])])
                    if strict:
                        errors.extend(sample_errors)
                        samples_error += 1
                        sample_status = "ERROR"
                        sample_checks = 0
                        sample_viol = 0
                        sample_max_gap = 0.0
                        sample_time = time.perf_counter() - sample_start
                        logger.info(
                            "\n  ❌ [L2][bounds_per_neuron] net=%s tf=%s sample=%d/%d "
                            "status=%s checks=%d viol=%d max_gap=%.2e "
                            "align={events=%d mapped=%d drop_batch=%d err=%d warn=%d} time=%.2fs",
                            name,
                            tf_mode,
                            samples_processed,
                            num_samples,
                            sample_status,
                            sample_checks,
                            sample_viol,
                            sample_max_gap,
                            align_events,
                            align_mapped,
                            align_drop_batch,
                            align_err,
                            align_warn,
                            sample_time,
                        )
                        logger.error("❌ [L2][bounds_per_neuron] net=%s tf=%s ERROR details:", name, tf_mode)
                        for e in sample_errors[:3]:
                            logger.error("  - %s", e)
                        break
                    errors.extend(sample_errors)
                    samples_error += 1
                    sample_status = "ERROR"
                    sample_checks = 0
                    sample_viol = 0
                    sample_max_gap = 0.0
                    sample_time = time.perf_counter() - sample_start
                    logger.info(
                        "\n  ❌ [L2][bounds_per_neuron] net=%s tf=%s sample=%d/%d "
                        "status=%s checks=%d viol=%d max_gap=%.2e "
                        "align={events=%d mapped=%d drop_batch=%d err=%d warn=%d} time=%.2fs",
                        name,
                        tf_mode,
                        samples_processed,
                        num_samples,
                        sample_status,
                        sample_checks,
                        sample_viol,
                        sample_max_gap,
                        align_events,
                        align_mapped,
                        align_drop_batch,
                        align_err,
                        align_warn,
                        sample_time,
                    )
                    logger.error("❌ [L2][bounds_per_neuron] net=%s tf=%s ERROR details:", name, tf_mode)
                    for e in sample_errors[:3]:
                        logger.error("  - %s", e)
                    continue

                total_violations += int(compare.get("violations_total", 0))
                candidate_violations.extend(compare.get("violations_topk", []))

                sample_stats = compare.get("layerwise_stats", [])
                sample_total = sum(int(s.get("num_neurons", 0)) for s in sample_stats)
                total_checks += int(sample_total)
                sample_max_gap = 0.0
                for s in sample_stats:
                    sample_max_gap = max(sample_max_gap, float(s.get("max_gap", 0.0)))
                worst_gap = max(worst_gap, sample_max_gap)
                for s in sample_stats:
                    lid = int(s["layer_id"])
                    entry = stats_by_layer.get(lid)
                    if entry is None:
                        entry = {
                            "layer_id": lid,
                            "kind": s.get("kind"),
                            "shape": s.get("shape"),
                            "num_neurons": int(s.get("num_neurons", 0)),
                            "num_violations": 0,
                            "max_gap": 0.0,
                            "mean_gap": 0.0,
                            "lb_min": float("inf"),
                            "lb_max": float("-inf"),
                            "ub_min": float("inf"),
                            "ub_max": float("-inf"),
                            "concrete_min": float("inf"),
                            "concrete_max": float("-inf"),
                            "_sum_gap": 0.0,
                            "_sum_violations": 0,
                        }
                        stats_by_layer[lid] = entry

                    entry["num_violations"] += int(s.get("num_violations", 0))
                    entry["max_gap"] = max(entry["max_gap"], float(s.get("max_gap", 0.0)))
                    entry["_sum_gap"] += float(s.get("mean_gap", 0.0)) * int(s.get("num_violations", 0))
                    entry["_sum_violations"] += int(s.get("num_violations", 0))
                    entry["lb_min"] = min(entry["lb_min"], float(s.get("lb_min", 0.0)))
                    entry["lb_max"] = max(entry["lb_max"], float(s.get("lb_max", 0.0)))
                    entry["ub_min"] = min(entry["ub_min"], float(s.get("ub_min", 0.0)))
                    entry["ub_max"] = max(entry["ub_max"], float(s.get("ub_max", 0.0)))
                    entry["concrete_min"] = min(entry["concrete_min"], float(s.get("concrete_min", 0.0)))
                    entry["concrete_max"] = max(entry["concrete_max"], float(s.get("concrete_max", 0.0)))

                if sample_errors:
                    sample_status = "ERROR"
                    samples_error += 1
                elif int(compare.get("violations_total", 0)) > 0:
                    samples_failed += 1
                    sample_status = "FAILED"
                    if sample_stats:
                        worst_layer_sample = max(
                            [x for x in sample_stats if int(x.get("num_violations", 0)) > 0],
                            key=lambda x: float(x.get("max_gap", 0.0)),
                            default=None,
                        )
                        if worst_layer_sample is not None:
                            worst_layer = worst_layer_sample
                else:
                    sample_status = "PASS"

                sample_time = time.perf_counter() - sample_start
                logger.info(
                    "\n  %s [L2][bounds_per_neuron] net=%s tf=%s sample=%d/%d "
                    "status=%s checks=%d viol=%d max_gap=%.2e "
                    "align={events=%d mapped=%d drop_batch=%d err=%d warn=%d} time=%.2fs",
                    "🚨" if sample_status == "FAILED" else ("❌" if sample_status == "ERROR" else "✅"),
                    name,
                    tf_mode,
                    samples_processed,
                    num_samples,
                    sample_status,
                    sample_total,
                    int(compare.get("violations_total", 0)),
                    float(sample_max_gap),
                    align_events,
                    align_mapped,
                    align_drop_batch,
                    align_err,
                    align_warn,
                    sample_time,
                )

                if sample_status == "ERROR":
                    logger.error("❌ [L2][bounds_per_neuron] net=%s tf=%s ERROR details:", name, tf_mode)
                    for e in sample_errors[:3]:
                        logger.error("  - %s", e)
                    if missing_bounds:
                        logger.error("  missing_bounds_layer_ids=%s", sorted(missing_bounds))
                    logger.error(
                        "  alignment: events=%d mapped=%d drop_batch=%d err=%d warn=%d",
                        align_events,
                        align_mapped,
                        align_drop_batch,
                        align_err,
                        align_warn,
                    )

                if sample_status == "FAILED":
                    if worst_layer is not None:
                        logger.error(
                            "🚨 [L2][bounds_per_neuron] worst_layer: "
                            "layer_id=%s kind=%s shape=%s viol=%s/%s max_gap=%.2e",
                            worst_layer.get("layer_id"),
                            worst_layer.get("kind"),
                            worst_layer.get("shape"),
                            worst_layer.get("num_violations"),
                            worst_layer.get("num_neurons"),
                            float(worst_layer.get("max_gap", 0.0)),
                        )
                    topk_violations = compare.get("violations_topk", [])[: int(topk)]
                    for v in topk_violations:
                        val = float(v.get("concrete", 0.0))
                        lb = float(v.get("lb", 0.0))
                        ub = float(v.get("ub", 0.0))
                        if val < lb:
                            direction = "below_lb"
                        elif val > ub:
                            direction = "above_ub"
                        else:
                            direction = "in_bounds"
                        logger.error(
                            "  topk: layer_id=%s idx=%s gap=%.2e val=%.6g lb=%.6g ub=%.6g %s",
                            v.get("layer_id"),
                            v.get("neuron_index"),
                            float(v.get("gap", 0.0)),
                            val,
                            lb,
                            ub,
                            direction,
                        )
                    for s in sample_stats:
                        if int(s.get("num_violations", 0)) == 0:
                            continue
                        logger.error(
                            "  layer: id=%s kind=%s shape=%s viol=%s lb=[%.6g, %.6g] "
                            "ub=[%.6g, %.6g] concrete=[%.6g, %.6g]",
                            s.get("layer_id"),
                            s.get("kind"),
                            s.get("shape"),
                            s.get("num_violations"),
                            float(s.get("lb_min", 0.0)),
                            float(s.get("lb_max", 0.0)),
                            float(s.get("ub_min", 0.0)),
                            float(s.get("ub_max", 0.0)),
                            float(s.get("concrete_min", 0.0)),
                            float(s.get("concrete_max", 0.0)),
                        )

        finally:
            if was_training:
                model.train()

        if errors and strict:
            if PerformanceOptions.debug_tf:
                logger.warning(
                    "bounds_per_neuron strict errors (first 3): %s",
                    errors[:3],
                )
            logger.info(
                "\n  ❌ [L2][bounds_per_neuron] net=%s tf=%s RESULT=ERROR samples=%d checks=%d "
                "viol=%d worst_gap=%.2e time=%.2fs",
                name,
                tf_mode,
                samples_processed,
                int(total_checks),
                int(total_violations),
                float(worst_gap),
                time.perf_counter() - network_start,
            )
            for e in errors[:3]:
                logger.error("  - %s", e)
            if missing_bounds_all:
                logger.error("  missing_bounds_layer_ids=%s", sorted(set(missing_bounds_all)))
            if alignment_meta:
                align_events = int(alignment_meta.get("hookable_events", 0))
                align_mapped = int(alignment_meta.get("hookable_layers", 0))
                align_drop_batch = int(alignment_meta.get("drop_batch", 0))
                align_err = int(alignment_meta.get("err", 0))
                align_warn = int(alignment_meta.get("warn", 0))
                logger.error(
                    "  alignment: events=%d mapped=%d drop_batch=%d err=%d warn=%d",
                    align_events,
                    align_mapped,
                    align_drop_batch,
                    align_err,
                    align_warn,
                )
            result = {
                'network': name,
                'tf_mode': tf_mode,
                'validation_type': 'bounds_per_neuron',
                'status': 'ERROR',
                'validation_status': 'ERROR',
                'errors': errors,
                'warnings': warnings,
                'alignment': alignment_meta or {},
                'total_checks': int(total_checks),
                'violations_total': 0,
                'violations_topk': [],
                'layerwise_stats': [],
                'samples_processed': int(samples_processed),
                'strict': bool(strict),
            }
            self.validation_results.append(result)
            return result

        layerwise_stats = []
        for lid, s in stats_by_layer.items():
            total_v = int(s.pop("_sum_violations", 0))
            sum_gap = float(s.pop("_sum_gap", 0.0))
            if total_v > 0:
                mean_gap = sum_gap / total_v
            else:
                mean_gap = 0.0
            s["mean_gap"] = float(mean_gap)
            s["layer_status"] = "FAIL" if s["num_violations"] > 0 else "PASS"
            if s["lb_min"] == float("inf"):
                s["lb_min"] = 0.0
                s["lb_max"] = 0.0
                s["ub_min"] = 0.0
                s["ub_max"] = 0.0
                s["concrete_min"] = 0.0
                s["concrete_max"] = 0.0
            layerwise_stats.append(s)

        candidate_violations.sort(key=lambda x: x.get("gap", 0.0), reverse=True)
        topk_violations = candidate_violations[: int(topk)]

        if total_violations > 0:
            validation_status = 'FAILED'
        else:
            validation_status = 'PASSED'

        if samples_error > 0:
            result_label = "ERROR"
            result_emoji = "❌"
        elif total_violations > 0:
            result_label = "FAILED"
            result_emoji = "🚨"
        else:
            result_label = "PASS"
            result_emoji = "✅"
        logger.info(
            "\n  %s [L2][bounds_per_neuron] net=%s tf=%s RESULT=%s samples=%d checks=%d "
            "viol=%d worst_gap=%.2e time=%.2fs",
            result_emoji,
            name,
            tf_mode,
            result_label,
            samples_processed,
            int(total_checks),
            int(total_violations),
            float(worst_gap),
            time.perf_counter() - network_start,
        )
        if result_label == "ERROR":
            validation_status = "ERROR"
            for e in errors[:3]:
                logger.error("  - %s", e)
            if missing_bounds_all:
                logger.error("  missing_bounds_layer_ids=%s", sorted(set(missing_bounds_all)))
            if alignment_meta:
                align_events = int(alignment_meta.get("hookable_events", 0))
                align_mapped = int(alignment_meta.get("hookable_layers", 0))
                align_drop_batch = int(alignment_meta.get("drop_batch", 0))
                align_err = int(alignment_meta.get("err", 0))
                align_warn = int(alignment_meta.get("warn", 0))
                logger.error(
                    "  alignment: events=%d mapped=%d drop_batch=%d err=%d warn=%d",
                    align_events,
                    align_mapped,
                    align_drop_batch,
                    align_err,
                    align_warn,
                )
        if validation_status == "FAILED":
            if worst_layer is not None:
                logger.error(
                    "🚨 [L2][bounds_per_neuron] worst_layer: "
                    "layer_id=%s kind=%s shape=%s viol=%s/%s max_gap=%.2e",
                    worst_layer.get("layer_id"),
                    worst_layer.get("kind"),
                    worst_layer.get("shape"),
                    worst_layer.get("num_violations"),
                    worst_layer.get("num_neurons"),
                    float(worst_layer.get("max_gap", 0.0)),
                )
            for v in topk_violations[: int(topk)]:
                val = float(v.get("concrete", 0.0))
                lb = float(v.get("lb", 0.0))
                ub = float(v.get("ub", 0.0))
                if val < lb:
                    direction = "below_lb"
                elif val > ub:
                    direction = "above_ub"
                else:
                    direction = "in_bounds"
                logger.error(
                    "  topk: layer_id=%s idx=%s gap=%.2e val=%.6g lb=%.6g ub=%.6g %s",
                    v.get("layer_id"),
                    v.get("neuron_index"),
                    float(v.get("gap", 0.0)),
                    val,
                    lb,
                    ub,
                    direction,
                )
            for s in layerwise_stats:
                if int(s.get("num_violations", 0)) == 0:
                    continue
                logger.error(
                    "  layer: id=%s kind=%s shape=%s viol=%s lb=[%.6g, %.6g] "
                    "ub=[%.6g, %.6g] concrete=[%.6g, %.6g]",
                    s.get("layer_id"),
                    s.get("kind"),
                    s.get("shape"),
                    s.get("num_violations"),
                    float(s.get("lb_min", 0.0)),
                    float(s.get("lb_max", 0.0)),
                    float(s.get("ub_min", 0.0)),
                    float(s.get("ub_max", 0.0)),
                    float(s.get("concrete_min", 0.0)),
                    float(s.get("concrete_max", 0.0)),
                )

        result = {
            'network': name,
            'tf_mode': tf_mode,
            'validation_type': 'bounds_per_neuron',
            'validation_status': validation_status,
            'status': "ERROR" if result_label == "ERROR" else None,
            'total_checks': int(total_checks),
            'violations_total': int(total_violations),
            'violations_topk': topk_violations,
            'layerwise_stats': layerwise_stats,
            'alignment': alignment_meta or {},
            'errors': errors,
            'warnings': warnings,
            'strict': bool(strict),
        }
        self.validation_results.append(result)
        return result
    
    def _get_concrete_activations(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        act_net=None
    ) -> Dict[int, torch.Tensor]:
        """
        Get concrete activation values by running forward pass with hooks.
        
        Args:
            model: PyTorch model
            input_tensor: Input tensor
            act_net: Optional ACT Net to align hooks to ACT layer ids
            
        Returns:
            Dictionary mapping layer_id to activation tensor
        """
        activations = {}
        hooks = []
        collected = []
        
        def make_hook(layer_id):
            def hook(module, input, output):
                collected.append(output.detach().clone())
            return hook
        
        hook_kinds = {
            "DENSE": torch.nn.Linear,
            "CONV2D": torch.nn.Conv2d,
            "RELU": torch.nn.ReLU,
            "FLATTEN": torch.nn.Flatten,
        }
        
        # Register hooks on relevant torch modules; map to ACT ids after forward
        layer_id = 0
        for module in model.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ReLU, torch.nn.Flatten)):
                hook = module.register_forward_hook(make_hook(layer_id))
                hooks.append(hook)
                layer_id += 1
        
        # Run forward pass
        with torch.no_grad():
            model(input_tensor)
        
        # Align collected activations with ACT layer ids so shapes match
        if act_net is not None:
            act_ids = [layer.id for layer in act_net.layers if layer.kind in hook_kinds]
            if len(act_ids) != len(collected):
                logger.warning(
                    "  ⚠️ Hook count mismatch: torch collected %d, ACT hookable layers=%d; aligning by position.",
                    len(collected), len(act_ids),
                )
            for idx, act_id in enumerate(act_ids[:len(collected)]):
                activations[act_id] = collected[idx]
        else:
            for idx, tensor in enumerate(collected):
                activations[idx] = tensor
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return activations
    
    def validate_comprehensive(
        self,
        networks: Optional[List[str]] = None,
        solvers: List[str] = ['gurobi', 'torchlp'],
        tf_modes: List[str] = ['interval'],
        num_samples: int = 10
    ) -> Dict[str, Any]:
        """
        Run both Level 1 and Level 2 validations.
        
        Args:
            networks: List of network names (None = all networks)
            solvers: List of solver names for Level 1
            tf_modes: Transfer function modes for Level 2
            num_samples: Number of samples for Level 2
            
        Returns:
            Combined summary dictionary
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"COMPREHENSIVE VERIFICATION VALIDATION")
        logger.info(f"{'='*80}")
        logger.info(f"Running both Level 1 (Counterexample) and Level 2 (Bounds) validation")
        logger.info(f"Device: {self.device}, Dtype: {self.dtype}")
        logger.info(f"{'='*80}\n")
        
        # Run Level 1
        summary_l1 = self.validate_counterexamples(networks=networks, solvers=solvers)
        
        # Run Level 2
        summary_l2 = self.validate_bounds(networks=networks, tf_modes=tf_modes, num_samples=num_samples)
        
        # Combine summaries - FAILED if any failures OR errors
        has_failures = (summary_l1.get('failed', 0) > 0 or summary_l2.get('failed', 0) > 0)
        has_errors = (summary_l1.get('errors', 0) > 0 or summary_l2.get('errors', 0) > 0)
        
        if has_failures:
            overall_status = 'FAILED'  # Critical: verifier is unsound
        elif has_errors:
            overall_status = 'ERROR'   # Backend bugs prevent validation
        else:
            overall_status = 'PASSED'  # All tests passed
        
        combined = {
            'level1_counterexample': summary_l1,
            'level2_bounds': summary_l2,
            'overall_status': overall_status
        }
        
        self._print_comprehensive_summary(combined)
        return combined
    
    def _compute_summary(self, validation_type: str) -> Dict[str, Any]:
        """
        Compute validation summary statistics for specific validation type.
        
        Args:
            validation_type: 'counterexample' or 'bounds'
        """
        results = [r for r in self.validation_results if r.get('validation_type') == validation_type]
        total = len(results)
        
        if total == 0:
            return {
                'validation_type': validation_type,
                'total': 0,
                'passed': 0,
                'failed': 0,
                'acceptable': 0,
                'inconclusive': 0,
                'errors': 0,
                'results': [],
                'error_message': 'No validation results (all tests encountered errors)'
            }
        
        passed = sum(1 for r in results if r.get('validation_status') == 'PASSED')
        failed = sum(1 for r in results if r.get('validation_status') == 'FAILED')
        acceptable = sum(1 for r in results if r.get('validation_status') == 'ACCEPTABLE')
        inconclusive = sum(1 for r in results if r.get('validation_status') == 'INCONCLUSIVE')
        errors = sum(1 for r in results if r.get('status') == 'ERROR')
        
        summary = {
            'validation_type': validation_type,
            'total': total,
            'passed': passed,
            'failed': failed,
            'acceptable': acceptable,
            'inconclusive': inconclusive,
            'errors': errors,
            'results': results
        }
        
        if validation_type == 'counterexample':
            summary['counterexamples_found'] = sum(1 for r in results if r.get('concrete_counterexample', False))
            summary['critical_bugs'] = failed
        elif validation_type == 'bounds':
            summary['total_checks'] = sum(r.get('total_checks', 0) for r in results)
            summary['total_violations'] = sum(len(r.get('violations', [])) for r in results)
        elif validation_type == 'bounds_per_neuron':
            summary['total_checks'] = sum(r.get('total_checks', 0) for r in results)
            summary['total_violations'] = sum(r.get('violations_total', 0) for r in results)
        
        self._print_summary(summary)
        return summary
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print validation summary for specific validation type."""
        validation_type = summary.get('validation_type', 'unknown')
        
        print("\n" + "="*80)
        print(f"VALIDATION SUMMARY - {validation_type.upper()}")
        print("="*80)
        
        if summary['total'] == 0:
            print()
            print("⚠️  No validation tests completed successfully")
            if 'error_message' in summary:
                print(f"   {summary['error_message']}")
            print("="*80)
            return
        
        print(f"\nTotal validation tests: {summary['total']}")
        
        if validation_type == 'counterexample':
            print(f"Concrete counterexamples found: {summary.get('counterexamples_found', 0)}")
        elif validation_type in ('bounds', 'bounds_per_neuron'):
            print(f"Total bound checks: {summary.get('total_checks', 0)}")
            print(f"Total violations: {summary.get('total_violations', 0)}")
        
        print()
        print(f"✅ PASSED:       {summary['passed']}")
        if validation_type == 'counterexample':
            print(f"⚠️  ACCEPTABLE:   {summary['acceptable']}")
            print(f"⚪ INCONCLUSIVE: {summary['inconclusive']}")
        print(f"❌ ERRORS:       {summary['errors']}")
        print(f"🚨 FAILED:       {summary['failed']}")
        print("="*80)
        
        if summary['failed'] > 0:
            print(f"\n🚨 CRITICAL: {validation_type.upper()} validation failed!")
            if validation_type == 'counterexample':
                print("Soundness bugs detected in the following networks:")
            else:
                print("Unsound bounds detected in the following networks:")
            for result in summary['results']:
                if result.get('validation_status') == 'FAILED':
                    if validation_type == 'counterexample':
                        print(f"  - {result['network']} ({result['solver']})")
                    else:
                        print(f"  - {result['network']} ({result['tf_mode']})")
            print()
        elif summary['errors'] > 0:
            print(f"\n⚠️  All {validation_type} validation tests encountered errors!")
            print("This indicates pre-existing bugs in the verification backend.")
            print()
        else:
            print(f"\n✅ {validation_type.upper()} validation PASSED!")
        
        print("="*80)
    
    def _print_comprehensive_summary(self, combined: Dict[str, Any]):
        """Print comprehensive summary for both validation levels."""
        print("\n" + "="*80)
        print("COMPREHENSIVE VALIDATION SUMMARY")
        print("="*80)
        
        l1 = combined['level1_counterexample']
        l2 = combined['level2_bounds']
        
        print(f"\nLevel 1 (Counterexample): {l1['passed']}/{l1['total']} passed, {l1['failed']} failed, {l1['errors']} errors")
        print(f"Level 2 (Bounds):         {l2['passed']}/{l2['total']} passed, {l2['failed']} failed, {l2['errors']} errors")
        print()
        print(f"Overall Status: {combined['overall_status']}")
        print("="*80)


def main():
    """Run verification validation test suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description='ACT Verification Validator')
    parser.add_argument('--mode', choices=['counterexample', 'bounds', 'bounds_per_neuron', 'comprehensive'],
                       default='comprehensive', help='Validation mode')
    parser.add_argument('--device', default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('--dtype', default='float64', choices=['float32', 'float64'],
                       help='Data type')
    parser.add_argument('--networks', nargs='+', help='Specific networks to test')
    parser.add_argument('--solvers', nargs='+', default=['gurobi', 'torchlp'],
                       help='Solvers for Level 1')
    parser.add_argument('--tf-modes', nargs='+', default=['interval'],
                       help='Transfer function modes for Level 2')
    parser.add_argument('--samples', type=int, default=10,
                       help='Number of samples for Level 2')
    parser.add_argument('--atol', type=float, default=1e-6,
                       help='Absolute tolerance for bounds_per_neuron')
    parser.add_argument('--rtol', type=float, default=0.0,
                       help='Relative tolerance for bounds_per_neuron')
    parser.add_argument('--topk', type=int, default=10,
                       help='Top-k violations for bounds_per_neuron')
    parser.add_argument('--no-strict', action='store_false', dest='strict', default=True,
                       help='Disable strict bounds_per_neuron checks')
    parser.add_argument('--ignore-errors', action='store_true',
                       help='Always exit 0 (ignore failures and errors for CI)')
    
    args = parser.parse_args()
    
    # Convert dtype string to torch dtype
    dtype = torch.float64 if args.dtype == 'float64' else torch.float32
    
    # Create validator
    validator = VerificationValidator(device=args.device, dtype=dtype)
    
    # Run validation
    if args.mode == 'counterexample':
        summary = validator.validate_counterexamples(
            networks=args.networks,
            solvers=args.solvers
        )
        # Exit 1 if any failures OR errors detected
        exit_code = 1 if (summary['failed'] > 0 or summary['errors'] > 0) else 0
    elif args.mode == 'bounds':
        summary = validator.validate_bounds(
            networks=args.networks,
            tf_modes=args.tf_modes,
            num_samples=args.samples
        )
        # Exit 1 if any failures OR errors detected
        exit_code = 1 if (summary['failed'] > 0 or summary['errors'] > 0) else 0
    elif args.mode == 'bounds_per_neuron':
        summary = validator.validate_bounds_per_neuron(
            networks=args.networks,
            tf_modes=args.tf_modes,
            num_samples=args.samples,
            atol=args.atol,
            rtol=args.rtol,
            topk=args.topk,
            strict=args.strict,
        )
        exit_code = 1 if (summary['failed'] > 0 or summary['errors'] > 0) else 0
    else:  # comprehensive
        combined = validator.validate_comprehensive(
            networks=args.networks,
            solvers=args.solvers,
            tf_modes=args.tf_modes,
            num_samples=args.samples
        )
        # Exit 1 for both FAILED (verification bugs) and ERROR (backend bugs)
        exit_code = 1 if combined['overall_status'] in ['FAILED', 'ERROR'] else 0
    
    # Override exit code if --ignore-errors is set
    if args.ignore_errors:
        exit_code = 0
    
    # Print debug file location (GUARDED)
    if PerformanceOptions.debug_tf:
        logger.info(f"\n📝 Debug log written to: {PerformanceOptions.debug_output_file}")
    
    return exit_code


if __name__ == "__main__":
    import sys
    sys.exit(main())
