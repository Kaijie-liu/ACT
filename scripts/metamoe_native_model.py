"""Pinned native PyTorch loader for the author's backend Customized entry.

No ONNX, BN folding, dtype conversion or preprocessing. This bypass is an
explicit front-end compatibility variant, not original export reproduction.
"""
import hashlib
from pathlib import Path
import sys


def load_component(repo, checkpoint, expected_sha256):
    path = Path(checkpoint)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('checkpoint identity')
    for suffix in ['', 'src/Vision_Transformer_Pytorch']:
        sys.path.insert(0, str(Path(repo) / suffix))
    import torch
    # Public full-module pickle bytes/types were reviewed in the parent freeze.
    model = torch.load(path, map_location='cpu', weights_only=False).eval()
    if type(model).__name__ != 'ModelWrapper' or type(model.model).__name__ != 'UltraVerifiableCNN':
        raise ValueError('unexpected original checkpoint type')
    return model.model.eval()
