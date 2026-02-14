# Front End module providing unified specification system and utilities
# for DNN verification. Integrates with spec creators for data/model/spec
# loading and synthesis.

"""
Front End Module

Key Features:
- Unified specification system (InputSpec/OutputSpec)
- Spec creators (TorchVision, VNNLib)
- Device-aware tensor management
- Verifiable model wrappers

Usage:
    >>> from cuc.front_end import InputSpec, OutputSpec, InKind, OutKind
    >>> 
    >>> # Create specifications
    >>> input_spec = InputSpec(kind=InKind.LINF_BALL, center=data, eps=0.1)
    >>> output_spec = OutputSpec(kind=OutKind.SAFETY, ...)
"""

# Core specification system
from cuc.front_end.specs import InputSpec, OutputSpec, InKind, OutKind

# Device management - import only when needed to avoid triggering argparse at import time
# from cuc.util.device_manager import get_default_device, get_default_dtype, get_current_settings

__all__ = [
    # Specifications
    'InputSpec', 'OutputSpec', 'InKind', 'OutKind',
    
    # Device management (available via act.util.device_manager)
    # 'get_default_device', 'get_default_dtype', 'get_current_settings',
]
