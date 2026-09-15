"""Local, hash-bound logical evidence records; pure standard-library I/O."""
from pathlib import Path
from portable_proof.runtime import digest,strict_json


def loader(root,tick=lambda:None):
    root=Path(root).resolve()
    def load(ref):
        tick();p=root/ref['file']
        if p.is_symlink() or p.resolve().parent!=root:raise ValueError('nonlocal proof reference')
        raw=p.read_bytes()
        if digest(raw)!=ref['sha256']:raise ValueError('proof file identity mismatch')
        result=strict_json(raw);tick();return result
    return load
