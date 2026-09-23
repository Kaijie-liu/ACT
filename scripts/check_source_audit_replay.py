"""Re-read saved input audit, separating historical facts from live-source tags.

The original --check requires even its informational current-source hashes to
remain unchanged. Later implementation work can change those tags. We retain
that old ledger and compare EVERY other field exactly, while separately
verifying the fresh tags against the actual files. No model/solver is run.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLOTS = ('current_sha256', 'current_equals_frozen')


def compare(saved, fresh, root):
    adjusted = copy.deepcopy(fresh)
    old_paths = saved['frozen_sources']['reviewed_call_path']
    new_paths = fresh['frozen_sources']['reviewed_call_path']
    if old_paths.keys() != new_paths.keys():
        raise ValueError('reviewed path roster changed')
    changes = []
    for name, record in new_paths.items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
            raise ValueError('outside/linked source')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if (record['current_sha256'] != actual or
                record['current_equals_frozen'] is not (actual == record['frozen_sha256'])):
            raise ValueError('invalid live source identity')
        for key in SLOTS:
            if old_paths[name][key] != record[key]:
                changes.append(dict(path=name, field=key, saved=old_paths[name][key], fresh=record[key]))
            adjusted['frozen_sources']['reviewed_call_path'][name][key] = old_paths[name][key]
    if adjusted != saved:
        raise ValueError('historical source-audit evidence changed')
    return dict(status='PASS', historical_evidence_matches=True,
                ignored_scientific_fields=[], separately_validated_live_identity_changes=changes,
                original_ledger_overwritten=False, independent_network_proof=False)


def main():
    spec = importlib.util.spec_from_file_location('saved_source_audit', ROOT / 'scripts/audit_moe_main_source.py')
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    saved = json.loads(audit.OUTPUT.read_bytes())
    fresh, elapsed = audit.analyze()
    result = compare(saved, fresh, ROOT)
    result.update(seconds=elapsed, packages=fresh['packages_decoded'],
                  old_ledger_sha256=hashlib.sha256(audit.OUTPUT.read_bytes()).hexdigest())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
