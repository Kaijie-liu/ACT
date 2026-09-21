"""Inventory public author Google Drive links without downloading a whole folder."""
import argparse
import json
from pathlib import Path

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import gdown
    entries = gdown.download_folder(url='https://drive.google.com/drive/folders/1OnuDO-y3Bf3_F_SQKEHVW7zU9icniPEN',
        output=str(a.output.parent / 'not_downloaded'), quiet=False, skip_download=True, remaining_ok=True,
        use_cookies=False)
    if not entries:
        raise ValueError('no public folder inventory returned')
    values = [e._asdict() if hasattr(e, '_asdict') else vars(e) for e in entries]
    with a.output.open('x') as f:
        json.dump({'author_link': 'RoME README', 'downloaded_weights': False, 'entries': values}, f, indent=2)
        f.write('\n')
    print(json.dumps(values, indent=2))
