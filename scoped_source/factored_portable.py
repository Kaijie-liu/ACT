"""Publish code/envelope around an already generated factored proof tree."""
from pathlib import Path
from scoped_proof.io import ROOT, save, sha, tick
from scoped_source.factored_verify import CODE, MEMBER_LIMIT, ENVELOPE_LIMIT, TOTAL_LIMIT
from source_enclosure.format import compact


def publish(root,source_sha,proof_sha,mode,invocation,deadline):
    root=Path(root); tick(deadline)
    for name in CODE:
        target=root/'code'/name; target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f: f.write((ROOT/name).read_bytes())
        tick(deadline)
    with (root/'verify.py').open('xb') as f: f.write((ROOT/'scoped_source/factored_verify.py').read_bytes())
    files={}; total=0
    for p in sorted(root.rglob('*')):
        tick(deadline)
        if p.is_symlink(): raise ValueError('symlink publication')
        if p.is_file():
            size=p.stat().st_size
            if not 0<size<=MEMBER_LIMIT: raise ValueError('publication member size')
            files[str(p.relative_to(root))]={'bytes':size,'sha256':sha(p)}; total+=size
    m={'schema':'HF_PORTABLE_V1','source_manifest_sha256':source_sha,'proof_manifest_sha256':proof_sha,
        'mode':mode,'invocation':invocation,'files':files}
    raw=compact(m)
    if len(raw)>ENVELOPE_LIMIT or total+len(raw)>TOTAL_LIMIT: raise ValueError('portable size admission')
    tick(deadline); record=save(root/'manifest.json',m); tick(deadline)
    return {**record,'source_manifest_sha256':source_sha,'proof_manifest_sha256':proof_sha,
            'mode':mode,'invocation':invocation,'bundle_bytes':total+len(raw)}
