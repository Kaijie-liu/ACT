"""Download only the preselected public CIFAR10 MAX checkpoint; retain failures."""
import hashlib
import json
import os
from pathlib import Path

if __name__ == '__main__':
    import gdown
    listing = Path('/data1/Kane/MOE/baseline_runs/rome_public_listing_20260921_r1/listing.json')
    listed = json.loads(listing.read_text())
    chosen = [e for e in listed['entries'] if e['path'] == 'checkpoints/ViT_CIFAR10/MAX_apgd_cifar10.pth']
    if len(chosen) != 1 or chosen[0]['id'] != '1uXcDiPfY8VyN9JjbQf4hN7rik6_ZzSQ4':
        raise ValueError('public listing target mismatch')
    root = Path('/data1/Kane/MOE/baseline_weights/rome_20260921')
    root.mkdir(exist_ok=False)
    partial = root / 'MAX_apgd_cifar10.pth.partial'
    out = gdown.download(id=chosen[0]['id'], output=str(partial), quiet=False, use_cookies=False)
    if out != str(partial) or not 1000000 < partial.stat().st_size < 1024**3:
        raise ValueError('failed/invalid bounded weight download')
    h = hashlib.file_digest(partial.open('rb'), 'sha256').hexdigest()
    final = root / 'MAX_apgd_cifar10.pth'
    os.link(partial, final)
    partial.unlink()
    value = {'public_author_folder': '1OnuDO-y3Bf3_F_SQKEHVW7zU9icniPEN', 'entry': chosen[0],
        'listing_sha256': hashlib.sha256(listing.read_bytes()).hexdigest(), 'path': str(final),
        'bytes': final.stat().st_size, 'local_sha256': h,
        'publisher_checksum_available': False, 'scope': 'public author-linked weight, not evaluated yet'}
    with (root / 'manifest.json').open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
    print(json.dumps(value, indent=2))
