"""Original multi-layer author-model execution intake, not a verifier lowering.

Routes are observed during the SAME full forward, never frozen and replayed as
an output-level model. Interval/domain verification is deliberately fail-closed
until dependent histories and intervening backbone operations are lowered.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import time

import torch
from torch import nn


class LayeredAuthorIntake(nn.Module):
    def __init__(self, model, *, family, source_hashes):
        super().__init__()
        if family not in {'rome', 'robust_experts'}:
            raise ValueError('unsupported layered family')
        if any(m.training for m in model.modules()):
            raise ValueError('explicit eval model required; intake never changes mode')
        if not source_hashes:
            raise ValueError('frozen source inventory required')
        self.model = model
        self.family = family
        self.source_hashes = dict(source_hashes)
        self.validate_sources()
        self.gates = []
        for name, module in model.named_modules():
            if family == 'rome':
                is_gate = (type(module).__module__ in {'models.lora_moe_mlp', 'models.lora_moe_linear'}
                    and type(module).__name__ in {'LoRAMoEAdapterMLP', 'LoRAMoEAdapterLinear'})
            else:
                is_gate = (type(module).__module__ == 'src.models.nn.moe.gate.topk'
                           and type(module).__name__ == 'TopKGate')
            if is_gate:
                source = str(Path(inspect.getsourcefile(type(module))).resolve())
                if source not in source_hashes or 'forward' in module.__dict__:
                    raise ValueError('unbound or overridden gate source')
                self.gates.append((name, module))
        if not self.gates:
            raise ValueError('no reviewed author routing layers found')
        self.eval()

    def validate_sources(self):
        for filename, expected in self.source_hashes.items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
                raise ValueError('author source binding mismatch: ' + filename)

    def forward(self, x):
        output = self.model(x)
        return output[0] if isinstance(output, (list, tuple)) else output

    def forward_with_trace(self, x):
        if any(m.training for m in self.model.modules()):
            raise ValueError('training/cobatch semantics not accepted')
        self.validate_sources()
        events, handles = [], []
        start = time.monotonic()
        def hook(name):
            def record(module, inputs, output):
                if self.family == 'rome':
                    weights = output[1]['gate_scores']
                    mechanism = 'dense_all_expert_lora; local/global gate; native float32 softmax'
                    k = int(module.num_lora)
                else:
                    weights = output[1]
                    mechanism = 'intermediate_topk; native scores/(selected_sum+1e-5)' if module.normalize_routing else 'intermediate_unnormalized_topk'
                    k = int(module.k)
                if not torch.isfinite(weights).all():
                    raise ValueError('nonfinite routed weights')
                events.append({'layer': name, 'call_index': len(events), 'mechanism': mechanism,
                    'expert_count': int(weights.shape[-1]), 'configured_k_or_dense_count': k,
                    'input_shape': list(inputs[0].shape), 'weight_shape': list(weights.shape),
                    'weight_dtype': str(weights.dtype), 'weights': weights.detach().cpu().clone(),
                    'observed_nonzero_weight_counts': (weights != 0).sum(-1).detach().cpu().clone(),
                    'weight_sums': weights.sum(-1).detach().cpu().clone()})
            return record
        try:
            for name, module in self.gates:
                handles.append(module.register_forward_hook(hook(name)))
            output = self.forward(x)
            if not torch.isfinite(output).all():
                raise ValueError('nonfinite complete output')
        finally:
            for handle in handles:
                handle.remove()
        return output, {'family': self.family, 'events': events,
            'executed_gate_calls': len(events), 'registered_gate_modules': len(self.gates),
            'trace_seconds': time.monotonic() - start,
            'semantics': 'observed full-forward history only, NOT reachable-domain enumeration',
            'evidence_grade': 'CONCRETE_EXECUTION_CONTROL', 'route_frozen': False}

    def verify_box(self, *args, **kwargs):
        # Crucial: feeding the clean observed history to Route A would omit
        # other possible histories, and is not a compatibility implementation.
        return {'status': 'UNSUPPORTED', 'evidence_grade': 'NONE', 'source_complete': False,
            'reason': 'dependent multilayer route/backbone lowering is not implemented',
            'family': self.family, 'clean_trace_is_not_domain_coverage': True}
