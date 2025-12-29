"""
ConfigNet package: sampling configs/specs, building nets via existing templates,
and JSONL logging utilities. Lives under pipeline.verification.confignet.
"""

from act.pipeline.verification.confignet.configs import ModelConfig, SpecConfig, canonical_hash
from act.pipeline.verification.confignet.sampler import sample_configs
from act.pipeline.verification.confignet.seeds import set_global_seeds
from act.pipeline.verification.confignet.builders import build_act_net, build_torch_model
from act.pipeline.verification.confignet.jsonl import write_jsonl_records
from act.pipeline.verification.confignet.utils import extract_effective_spec

__all__ = [
    "ModelConfig",
    "SpecConfig",
    "sample_configs",
    "set_global_seeds",
    "build_act_net",
    "build_torch_model",
    "write_jsonl_records",
    "canonical_hash",
    "extract_effective_spec",
]
