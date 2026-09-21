"""Bounded public Toronto download; exact Range and torchvision MD5 checks.

New directory only. Eight normal HTTP range requests, no credentials or alternate
dataset. Preserve every partial part and the previous serial attempt unchanged.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tarfile
import time
import urllib.request


if __name__ == '__main__':
    began = time.monotonic()
    root = Path('/data1/Kane/MOE/baseline_data/robust_experts_20260921_r2')
    root.mkdir(exist_ok=False)
    total = 169001437
    md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    # This is the observed Location of torchvision's canonical Toronto URL.
    url = 'https://cave.cs.toronto.edu/kriz/cifar-100-python.tar.gz'
    n = 8
    def fetch(i):
        first, last = total*i//n, total*(i+1)//n-1
        request = urllib.request.Request(url, headers={'Range': f'bytes={first}-{last}',
            'If-Match': '"a12c1dd-47ffe10fc4937"'})
        path = root / f'part{i:02d}'
        with urllib.request.urlopen(request, timeout=45) as response, path.open('xb') as out:
            if response.status != 206 or response.headers.get('Content-Range') != f'bytes {first}-{last}/{total}':
                raise ValueError('server did not honor exact requested range')
            while block := response.read(1024*1024):
                out.write(block)
        if path.stat().st_size != last-first+1:
            raise ValueError('truncated range')
        return {'part': i, 'start': first, 'end': last, 'sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()}
    with ThreadPoolExecutor(max_workers=n) as pool:
        parts = list(pool.map(fetch, range(n)))
    archive = root / 'cifar-100-python.tar.gz'
    with archive.open('xb') as out:
        for i in range(n):
            with (root / f'part{i:02d}').open('rb') as stream:
                while block := stream.read(1024*1024):
                    out.write(block)
    if archive.stat().st_size != total or hashlib.file_digest(archive.open('rb'), 'md5').hexdigest() != md5:
        raise ValueError('complete archive does not match torchvision published MD5')
    with tarfile.open(archive) as tar:
        tar.extractall(root, filter='data')
    files = {name: hashlib.file_digest((root/'cifar-100-python'/name).open('rb'), 'sha256').hexdigest()
             for name in ['train', 'test', 'meta']}
    result = {'status': 'PUBLIC_DATA_INTEGRITY_PASS', 'url': url, 'canonical_url':
        'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz', 'bytes': total,
        'published_torchvision_md5': md5, 'archive_sha256': hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest(),
        'parts': parts, 'files': files, 'seconds': time.monotonic()-began}
    with (root/'manifest.json').open('x') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
    print(json.dumps(result, indent=2))
