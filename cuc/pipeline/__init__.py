# Pipeline module for PyTorch model generation and testing utilities.
# Provides tools for converting between PyTorch models and Nets,
# verifier validation, and utility functions.

"""Pipeline Module - Model Generation and Testing Utilities.

This module provides utilities for PyTorch model generation, conversion,
verifier validation, and performance analysis.

Key Components:
    - ModelFactory: Create PyTorch models from YAML configurations
    - TorchToCUC: Convert PyTorch models to the internal representation
    - VerifierValidator: Validate verifier correctness with concrete tests
    - PerformanceProfiler: Profile execution time and memory usage

All verification utilities are located in the verification/ submodule.

Example:
    # Create PyTorch model from Net JSONs (generated via config_gen_cuc_net.yaml)
    from cuc.pipeline import ModelFactory, TorchToCUC
    
    factory = ModelFactory()
    model = factory.create_model("mnist_mlp_small", load_weights=True)
    
    # Convert to the internal format
    converter = TorchToCUC()
    act_net = converter.convert(model, input_shape=(1, 784))
"""

# Core imports
from cuc.pipeline.verification.model_factory import ModelFactory
from cuc.pipeline.verification.torch2cuc import TorchToCUC

# Import utilities
try:
    from cuc.pipeline.verification.utils import (
        PerformanceProfiler,
        ParallelExecutor,
        print_memory_usage,
        clear_torch_cache,
        setup_logging,
        ProgressTracker,
    )
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    PerformanceProfiler = None
    ParallelExecutor = None
    print_memory_usage = None
    clear_torch_cache = None
    setup_logging = None
    ProgressTracker = None

__all__ = [
    # Core model factory and conversion
    'ModelFactory',
    'TorchToCUC',
    
    # Utilities
    'PerformanceProfiler',
    'ParallelExecutor',
    'print_memory_usage',
    'clear_torch_cache',
    'setup_logging',
    'ProgressTracker',
    
    # Availability flags
    'UTILS_AVAILABLE',
]
