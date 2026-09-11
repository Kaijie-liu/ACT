"""Endpoint-blind selection for the separately authorized 30-input confirmation.

Only ordered clean forwards run here, in deployment CPU/float64 batch-one
semantics. No router scores, route counts or verification endpoints select data.
"""
import argparse
import json
from pathlib import Path

import torch

from act.back_end.moe import load_output_moe_checkpoint
from act.pipeline.moe.experiment1 import PROJECT_ROOT, _sha256, _inside, WRITE_ROOT
from act.pipeline.moe.freeze_staged_multimodel_bundle import MODELS
from act.pipeline.moe.staged_verifier import _tensor_identity, _model_state_identity
from act.pipeline.moe.train import _load_dataset
from act.util.device_manager import initialize_device

COUNT = 30
START = 4000
EPSILON = 2 / 255
PREVIOUS = PROJECT_ROOT/'act/pipeline/moe/configs/staged_verifier_multimodel_fixed2_selection_r1.json'
ORIGINAL_OUTPUT = PROJECT_ROOT/'act/pipeline/moe/configs/schedule_confirmation_selection_r1.json'
OUTPUT = PROJECT_ROOT/'act/pipeline/moe/configs/schedule_confirmation_selection_r2.json'
INVENTORY = PROJECT_ROOT/'data/moe/results/schedule_confirmation_selection_20260912_r2/excluded_artifacts.json'
TENSOR_SEMANTICS = 'TORCH_DEFAULT_FLOAT64_BEFORE_TOTENSOR'


def index_fields(value):
    """Explicit dataset indices only; never treat expert/factor IDs as inputs."""
    result = set()
    if isinstance(value, dict):
        if 'dataset_index' in value:
            result.add(int(value['dataset_index']))
        for child in value.values():
            result.update(index_fields(child))
    elif isinstance(value, list):
        for child in value:
            result.update(index_fields(child))
    return result


def source_indices(path):
    values = ([json.loads(line) for line in path.read_text().splitlines()]
              if path.suffix == '.jsonl' else [json.loads(path.read_text())])
    indices = set()
    for value in values:
        if path.name == 'rows.jsonl' and value.get('method') not in {
                'staged', 'monolithic_f0', 'route_invariance', 'tier1_only', 'adaptive', 'monolithic'}:
            continue
        indices.update(index_fields(value))
        if path.name == 'sample_indices.json' and 'indices' in value:
            indices.update(int(i) for i in value['indices'])
    return indices


def inventory():
    raw = PROJECT_ROOT/'data/moe/results'
    configs = PROJECT_ROOT/'act/pipeline/moe/configs'
    paths = set(raw.rglob('sample_indices.json')) | set(raw.rglob('selection.json'))
    paths |= {p for p in configs.glob('*selection*.json') if p not in {OUTPUT, ORIGINAL_OUTPUT}}
    # Terminal ledgers catch previous watchdog deaths without full packages.
    paths |= set(raw.rglob('rows.jsonl'))
    paths |= set(raw.rglob('evidence.json'))
    records = []; union = set()
    for path in sorted(paths):
        indices = source_indices(path)
        if indices:
            if any(i < 0 or i >= 10000 for i in indices):
                raise ValueError(f'non-CIFAR10 index in exclusion source: {path}')
            records.append({'path': str(path), 'sha256': _sha256(path), 'indices': sorted(indices)})
            union.update(indices)
    if not any(r['path'] == str(PREVIOUS) for r in records):
        raise ValueError('old common cohort absent from exclusions')
    return records, union


def verify_exclusions(selection):
    record = selection['exclusion_inventory']
    inventory_path = _inside(Path(record['path']), WRITE_ROOT)
    if _sha256(inventory_path) != record['sha256']:
        raise ValueError('exclusion inventory drift')
    records = json.loads(inventory_path.read_text())
    union = set()
    for record in records:
        path = _inside(Path(record['path']), WRITE_ROOT)
        if _sha256(path) != record['sha256']:
            raise ValueError('frozen exclusion artifact drift')
        if sorted(source_indices(path)) != record['indices']:
            raise ValueError('exclusion indices do not reconstruct')
        union.update(record['indices'])
    if sorted(union) != selection['excluded_indices']:
        raise ValueError('exclusion union mismatch')
    chosen = [r['dataset_index'] for r in selection['samples']]
    if len(chosen) != COUNT or len(set(chosen)) != COUNT or chosen != sorted(chosen):
        raise ValueError('non-canonical confirmation selection')
    if union.intersection(chosen) or min(chosen) < START:
        raise ValueError('selection overlaps prior endpoints or starts early')
    if not all(r['dataset_index'] in union for r in selection['smoke_samples']):
        raise ValueError('smoke must use previously observed inputs')
    if set(chosen).intersection(r['dataset_index'] for r in selection['smoke_samples']):
        raise ValueError('smoke/confirmation overlap')
    return records


@torch.no_grad()
def generate(exclusions=None):
    torch.set_num_threads(1)
    # ToTensor uses torch's default dtype for uint8 scaling. Match the actual
    # CLI initialization order, not float32 scaling followed by a double cast.
    initialize_device('cpu', 'float64')
    if exclusions is None:
        records, excluded = inventory()
    else:
        records = exclusions
        excluded = {i for r in records for i in r['indices']}
    nets = {}; models = {}; dataset = None
    for name, frozen in MODELS.items():
        checkpoint = Path(frozen['checkpoint'])
        if _sha256(checkpoint) != frozen['checkpoint_sha256']:
            raise ValueError('checkpoint drift')
        net, payload = load_output_moe_checkpoint(checkpoint, map_location='cpu')
        if payload['dataset'] != 'CIFAR10': raise ValueError('unexpected dataset')
        net.cpu().double().eval(); nets[name] = net
        models[name] = {'checkpoint': str(checkpoint), 'checkpoint_sha256': frozen['checkpoint_sha256'],
                        'model_state': _model_state_identity(net), 'dataset': 'CIFAR10'}
        if dataset is None: dataset = _load_dataset('CIFAR10', False, download=False)
    def sample(index, rank):
        x, label = dataset[index]; center = x.unsqueeze(0).double()
        predictions = {name: int(net(center).argmax(1).item()) for name, net in nets.items()}
        return {'sample_rank': rank, 'dataset_index': index, 'label': int(label),
                'clean_predictions': predictions, 'center': _tensor_identity(center),
                'lower': _tensor_identity((center-EPSILON).clamp(0, 1)),
                'upper': _tensor_identity((center+EPSILON).clamp(0, 1))}
    samples = []; scanned = []
    for index in range(START, len(dataset)):
        if index in excluded: continue
        row = sample(index, len(samples))
        scanned.append({'dataset_index': index, 'label': row['label'], 'clean_predictions': row['clean_predictions']})
        if all(p == row['label'] for p in row['clean_predictions'].values()):
            samples.append(row)
            if len(samples) == COUNT: break
    if len(samples) != COUNT: raise ValueError('insufficient common clean-correct inputs')
    old = json.loads(PREVIOUS.read_text())
    smoke = [sample(old['samples'][0]['dataset_index'], 0)]
    if not all(p == smoke[0]['label'] for p in smoke[0]['clean_predictions'].values()):
        raise ValueError('registered smoke no longer clean-correct')
    raw = PROJECT_ROOT/'data/torchvision/CIFAR10/raw/cifar-10-batches-py/test_batch'
    return {'schema': 'schedule_confirmation_selection_v1', 'status': 'FROZEN_BEFORE_NEW_ENDPOINTS',
            'classification': 'NEW_ENDPOINT_COHORT_SAME_THREE_MODELS_NOT_NEVER_SEEN_IMAGES',
            'rule': {'start_index': START, 'sample_count': COUNT, 'ordering': 'ascending dataset index',
                     'selection_predicates_only': ['absent from exclusion union', 'all frozen models clean-correct'],
                     'clean_semantics': TENSOR_SEMANTICS},
            'models': models, 'dataset': {'name': 'CIFAR10', 'length': len(dataset), 'split': 'test',
                'raw_test_batch': str(raw), 'raw_test_batch_sha256': _sha256(raw)},
            'exclusion_inventory': {'path': str(INVENTORY), 'sha256': _sha256(INVENTORY), 'sources': len(records)},
            'excluded_indices': sorted(excluded),
            'scanned_clean_only': scanned, 'samples': samples, 'smoke_samples': smoke,
            'request': {'epsilon': EPSILON, 'boundary_search': False, 'route_instability_prefilter': False},
            'supersedes_selection': {'path': str(ORIGINAL_OUTPUT), 'sha256': _sha256(ORIGINAL_OUTPUT),
                                    'reason': 'R1 old-input smoke exposed preprocessing dtype identity mismatch; no new endpoint ran'}}


def audit_selection(path):
    selection = json.loads(Path(path).read_text())
    records = verify_exclusions(selection)
    rebuilt = generate(records)
    if rebuilt != selection: raise ValueError('selection does not reconstruct from clean-only forwards')
    return {'status': 'PASS', 'issues': [], 'selection_sha256': _sha256(path),
            'sample_count': COUNT, 'excluded_count': len(selection['excluded_indices']),
            'index_range': [selection['samples'][0]['dataset_index'], selection['samples'][-1]['dataset_index']],
            'scope': 'Separate clean-only reconstruction; no verification endpoint run.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        from act.pipeline.moe.paired_followup import save
        if OUTPUT.exists() or INVENTORY.parent.exists(): raise ValueError('no overwrite of frozen selection')
        # Repair uses the SAME frozen exclusion union and requires exactly the
        # SAME thirty selected indices. No result-based replacement is possible.
        original = json.loads(ORIGINAL_OUTPUT.read_text())
        records = verify_exclusions(original)
        INVENTORY.parent.mkdir()
        save(INVENTORY, records)
        value = generate(records)
        if [r['dataset_index'] for r in value['samples']] != [r['dataset_index'] for r in original['samples']]:
            raise ValueError('dtype repair changed chosen indices; stop for review, do not replace samples')
        save(OUTPUT, value)
        print(json.dumps({'selection': str(OUTPUT), 'count': len(value['samples']),
                          'excluded': len(value['excluded_indices']),
                          'indices': [r['dataset_index'] for r in value['samples']]}))
    elif args.audit:
        print(json.dumps(audit_selection(args.audit)))
    else:
        parser.error('choose --freeze or --audit')
