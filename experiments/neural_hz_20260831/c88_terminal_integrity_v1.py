"""Read-only frozen-source/artifact audit; publish a new exclusive record."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import (
    _sha256, _atomic_exclusive_json)

EXP = Path(__file__).resolve().parent


def main():
    started = time.monotonic()
    reports = []
    for name, count in [('c88_inline_tile_20260913_v1', 2080)]:
        run = EXP/'results'/name
        prereg = json.loads((run/'preregistered.json').read_text())
        final = json.loads((run/'exit.json').read_text())
        if (not final['all_declared_stages_passed'] or final['tests_exit'] != 0
                or final['tests_count'] != count or final['source_drift'] or final['provenance_drift']
                or _provenance(ROOT) != prereg['provenance']):
            raise ValueError('terminal/configuration/source qualification differs')
        for relative, digest in prereg['source_sha256'].items():
            if _sha256(EXP/relative) != digest:
                raise ValueError('frozen source mismatch: '+relative)
        for relative, digest in final['artifacts'].items():
            if _sha256(run/relative) != digest:
                raise ValueError('saved artifact mismatch: '+relative)
        reports.append(dict(run=name, source_files=len(prereg['source_sha256']),
            saved_artifacts=len(final['artifacts']), exit_sha256=_sha256(run/'exit.json'),
            result_sha256=_sha256(run/'result.json'), all_sources_and_artifacts_match=True,
            terminal_tests=count))
    manifest = EXP/'manifests/tinyimagenet_2024_universe_v1.json'
    if _sha256(manifest) != 'a8a0dc7504af2c6b89d099fd5c74f27aa5ae98458c5c151cbb6d0da2ef5c1f59':
        raise ValueError('frozen original universe mismatch')
    original = json.loads(manifest.read_text())
    models = {v['model_relative_path']: v['model_sha256'] for v in original['instances']}
    for name, expected in models.items():
        if _sha256(Path(original['source_benchmark_root'])/original['family']/name) != expected:
            raise ValueError('original model mismatch')
    names = {'run_c88_inline_tile_supervisor_v1.py', 'c88_inline_tile_worker_v1.py'}
    live = []
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit():
            continue
        try:
            argv = (folder/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if len(argv) > 1 and Path(argv[1].decode(errors='replace')).name in names:
            live.append(int(folder.name))
    if live:
        raise ValueError('a registered C88 process is still live')
    prior = json.loads((EXP/'C88_PRIOR_HISTORY_INTEGRITY_20260913.json').read_text())
    result = dict(completed=True, terminal_runs=reports, live_registered_processes=live,
        prior_history_check_reused_without_repeating_unchanged_hash_scan=prior,
        original_models_rechecked=len(models), provenance=_provenance(ROOT),
        sealed_c69_source_report_sha256=_sha256(EXP/'results/c69_prepared_finite_20260913_v1/actual/result.json'),
        wall_s=time.monotonic()-started, historical_writes=False, formal_gain=0)
    _atomic_exclusive_json(EXP/'C88_TERMINAL_INTEGRITY_20260913.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
