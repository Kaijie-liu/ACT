"""Package H1's existing checker without importing its producer at verification."""
from pathlib import Path
from scoped_proof.io import ROOT, save, sha, tick

CODE = ('scoped_source/graph.py', 'scoped_source/sparse_ir.py',
        'scoped_source/sparse_check.py', 'source_enclosure/format.py',
        'upstream_source/checker.py', 'router_source/checker.py')


def pack(root, doc, proof, source_hash, invocation, deadline):
    root = Path(root); tick(deadline); root.mkdir(exist_ok=False)
    files = {}
    for name, value in [('source.json', doc), ('proof.json', proof)]:
        files[name] = save(root/name, value)['sha256']; tick(deadline)
    for name in CODE:
        target = root/'code'/name; target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream: stream.write((ROOT/name).read_bytes())
        files['code/'+name] = sha(target); tick(deadline)
    with (root/'verify.py').open('xb') as stream:
        stream.write((ROOT/'scoped_source/sparse_verify.py').read_bytes())
    files['verify.py'] = sha(root/'verify.py')
    manifest = {'schema': 'H1_PORTABLE_DECLARED_SOURCE_V1', 'source_sha256': source_hash,
                'invocation': invocation, 'files': files,
                'scope': 'declared real Linear/ReLU weighted top2; not native floating execution'}
    record = save(root/'manifest.json', manifest); tick(deadline)
    return {**record, 'source_sha256': source_hash, 'invocation': invocation,
            'bundle_bytes': sum(p.stat().st_size for p in root.rglob('*') if p.is_file())}
