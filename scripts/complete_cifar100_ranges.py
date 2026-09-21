"""Separate missing-range retrieval; never overwrite the previous attempts."""
import hashlib
import json
from pathlib import Path
import tarfile
import time
import urllib.request


if __name__ == '__main__':
    began = time.monotonic()
    base = Path('/data1/Kane/MOE/baseline_data')
    old = base / 'robust_experts_20260921_r2'
    root = base / 'robust_experts_20260921_r3'
    root.mkdir(exist_ok=False)
    total, i, n = 169001437, 5, 8
    first, last = total*i//n, total*(i+1)//n-1
    request = urllib.request.Request('https://cave.cs.toronto.edu/kriz/cifar-100-python.tar.gz',
        headers={'Range': f'bytes={first}-{last}', 'If-Match': '"a12c1dd-47ffe10fc4937"'})
    part = root / 'part05'
    with urllib.request.urlopen(request, timeout=45) as response, part.open('xb') as out:
        if response.status != 206 or response.headers.get('Content-Range') != f'bytes {first}-{last}/{total}':
            raise ValueError('range identity mismatch')
        while block := response.read(1024*1024):
            out.write(block)
    paths = [part if j == 5 else old / f'part{j:02d}' for j in range(n)]
    hashes = {}
    for j, path in enumerate(paths):
        if path.stat().st_size != total*(j+1)//n-total*j//n:
            raise ValueError('required completed range absent/truncated; no partial acceptance')
        hashes[str(path)] = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
    archive = root / 'cifar-100-python.tar.gz'
    with archive.open('xb') as out:
        for path in paths:
            with path.open('rb') as stream:
                while block := stream.read(1024*1024):
                    out.write(block)
    md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    if archive.stat().st_size != total or hashlib.file_digest(archive.open('rb'), 'md5').hexdigest() != md5:
        raise ValueError('full public archive checksum mismatch')
    with tarfile.open(archive) as tar:
        tar.extractall(root, filter='data')
    result = {'status': 'PUBLIC_DATA_INTEGRITY_PASS', 'reused_range_files_sha256': hashes,
        'published_torchvision_md5': md5,
        'archive_sha256': hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest(),
        'files': {name: hashlib.file_digest((root/'cifar-100-python'/name).open('rb'), 'sha256').hexdigest()
                  for name in ['train', 'test', 'meta']}, 'seconds': time.monotonic()-began}
    with (root/'manifest.json').open('x') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
    print(json.dumps(result, indent=2))
