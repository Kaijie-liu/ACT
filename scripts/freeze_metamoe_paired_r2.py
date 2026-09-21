"""Materialize ordered clean-correct cohort BEFORE verification; separate smoke."""
import argparse
import json
from pathlib import Path
import sys
import time
from recent_moe_deployment import sha256
from recent_moe_env_inventory import inventory
from robust_experts_workflow_control import write


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    start = time.monotonic()
    import numpy as np
    import torch
    from metamoe_paired_model import load_full
    torch.set_num_threads(2)
    parent = json.loads(Path('configs/recent_moe/metamoe_full_intake_r2.json').read_text())
    backend = json.loads(Path('configs/recent_moe/metamoe_component_control_r4.json').read_text())
    suffix = 'smoke' if a.smoke else 'confirmatory'
    manifest = Path(f'configs/recent_moe/metamoe_paired_{suffix}_r2.json')
    tensors = Path(f'/data1/Kane/MOE/baseline_data/metamoe_paired_20260922_r2_{suffix}')
    tensors.mkdir(parents=True, exist_ok=False)
    if not a.smoke:
        audit = Path('docs/metamoe_paired_smoke_review_20260922_r2.json')
        if not audit.exists() or json.loads(audit.read_text()).get('execution_control_pass') is not True:
            raise ValueError('real paired smoke/audit gate')
    model = load_full(parent['repo'], parent['checkpoint'], parent['files'][parent['checkpoint']])
    sys.path.insert(0, str(Path(parent['repo']) / 'src/Formal_Neural_Network_Verification/alpha-beta-crown'))
    from create_vnnlib_specs import load_dataset
    requests = []
    for dataset, offset in [('CIFAR10', 0), ('MNIST', 10)]:
        images, labels, _ = load_dataset(dataset, parent['data_root'], 1 if a.smoke else 1000)
        count = 0
        for index, (image, label) in enumerate(zip(images, labels)):
            if not a.smoke and index == 0:
                continue
            x = image.unsqueeze(0).double()
            with torch.no_grad():
                out = model(x)[0]
            if not torch.isfinite(out).all():
                raise ValueError('undefined clean selection')
            if not a.smoke and int(out.argmax(1)) != label+offset:
                continue
            rid = f'{dataset.lower()}_{index}'
            file = tensors / f'{rid}.npz'
            np.savez(file, center=x.numpy(), lower=(x-2/255).clamp(-10, 10).numpy(),
                     upper=(x+2/255).clamp(-10, 10).numpy())
            requests.append({'id': rid, 'dataset': dataset, 'index': index, 'label': label+offset,
                             'clean_prediction': int(out.argmax(1)), 'tensor_file': str(file)})
            count += 1
            if count == (1 if a.smoke else 10):
                break
        if count != (1 if a.smoke else 10):
            raise ValueError('registered first1000 scan insufficient; no automatic selection expansion')
    files = {**parent['files'], **backend['execution_files'],
             **{str(tensors / (r['id']+'.npz')): sha256(tensors / (r['id']+'.npz')) for r in requests}}
    for name in ['scripts/metamoe_paired_model.py', 'scripts/metamoe_paired_execution_r2.py',
                 'scripts/freeze_metamoe_paired_r2.py', 'scripts/audit_metamoe_paired.py',
                 'tests/test_metamoe_paired_model.py', 'scripts/metamoe_functional_intake.py',
                 'tests/test_metamoe_functional_intake.py', 'scripts/replay_metamoe_paired.py', 'configs/recent_moe/metamoe_component_control_r4.json',
                 'docs/metamoe_native_control_review_20260922_r4.json']:
        files[name] = sha256(name)
    cfg = {'protocol': f'metamoe_matched_full_request_{suffix}_r2', 'requests': requests,
           'repo': parent['repo'], 'checkpoint': parent['checkpoint'], 'backend_repo': backend['backend_repo'],
           'python': {'act': parent['python'], 'author': backend['python']},
           'repositories': {parent['repo']: parent['commit'],
              backend['backend_repo']: backend['backend_commit'],
              str(Path(backend['backend_repo'])/'auto_LiRPA'): backend['lirpa_commit']},
           'environment': inventory([parent['python'], backend['python']]),
           'files': files, 'seconds': 300, 'epsilon': 2/255, 'margin': 1e-7,
           'output_root': f'/data1/Kane/MOE/baseline_runs/metamoe_paired_20260922_r2_{suffix}',
           'selection': 'old index0 controls' if a.smoke else 'first10 each domain in raw order, clean-correct on frozen float64 snapshot, exclude index0; no route/bound selection',
           'input_semantics': 'same materialized normalized CPU float64 box; exact decimal VNNLIB endpoints',
           'arms': ['ACT_functional_ReLU_intake_HZ_policy', 'author_native_backend_route_invariance_sufficient_adapter'],
           'scope': 'full identical model/property; real-abstraction numerical policies, NOT source-complete floating proof',
           'cost': '300s each including child imports/load/own route/translation/solve/result; archival independent audit separate',
           'failure': 'stop batch on ERROR; keep timeouts; no retry/config changes',
           'preparation_seconds': time.monotonic()-start}
    if not a.smoke:
        cfg['files'][str(audit)] = sha256(audit)
    write(manifest, cfg)
    print('FROZEN_NOT_EXECUTED', len(requests), sha256(manifest))

