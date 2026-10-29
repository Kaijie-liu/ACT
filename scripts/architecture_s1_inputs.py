"""One fixed test prefix, materialized once without model-based selection."""
import time
ENTRY = time.monotonic()
import json
import os
from pathlib import Path
import signal
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from scoped_proof.io import load, save, sha


def fingerprint(tensor):
    import hashlib
    t = tensor.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(t.shape), str(t.dtype))).encode()+t.numpy().tobytes()).hexdigest()


def tensors_at(payload, index, numerator):
    import torch
    if type(index) is not int or not 0 <= index < 1000 or numerator not in (2, 8):
        raise ValueError('fixed S1 input/radius')
    center = payload['centers'][index:index+1].double()
    label = int(payload['labels'][index])
    rows = center.new_zeros((9, 10))
    for k, other in enumerate(c for c in range(10) if c != label):
        rows[k, label], rows[k, other] = 1., -1.
    epsilon = numerator/255
    return dict(center=center, lower=(center-epsilon).clamp_min(0),
        upper=(center+epsilon).clamp_max(1), rows=rows, thresholds=center.new_zeros(9))


def materialize(folder):
    from multilayer_train import setup
    folder.mkdir()
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
    signal.setitimer(signal.ITIMER_REAL, max(.001, 290-(time.monotonic()-ENTRY)))
    torch = setup(1)
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor
    spec = load(ROOT/'docs/architecture_s1_training_freeze_20261029.json')
    source = str(Path(spec['dataset_root'])/'cifar-10-batches-py/test_batch')
    if sha(source) != spec['data_files'][source]:
        raise ValueError('same CIFAR test source')
    dataset = CIFAR10(spec['dataset_root'], train=False, download=False, transform=ToTensor())
    values = [dataset[i] for i in range(1000)]
    payload = dict(centers=torch.stack([x for x, _ in values]),
        labels=torch.tensor([y for _, y in values], dtype=torch.int64))
    assert payload['centers'].shape == (1000, 3, 32, 32)
    assert payload['centers'].dtype == torch.float32
    assert bool(torch.isfinite(payload['centers']).all())
    assert bool(((payload['centers'] >= 0) & (payload['centers'] <= 1)).all())
    path = folder/'inputs.pt'
    with path.open('xb') as f:
        torch.save(payload, f); f.flush(); os.fsync(f.fileno())
    restored = torch.load(path, map_location='cpu', weights_only=True)
    assert all(torch.equal(payload[k], restored[k]) for k in payload)
    records = []
    for index in range(1000):
        domains = {}
        for numerator in (2, 8):
            tensors = tensors_at(restored, index, numerator)
            assert bool(((tensors['lower'] <= tensors['center']) &
                         (tensors['center'] <= tensors['upper'])).all())
            assert tensors['rows'].shape == (9, 10)
            assert bool((tensors['rows'].sum(1) == 0).all())
            assert bool((tensors['rows'][:, int(restored['labels'][index])] == 1).all())
            domains[str(numerator)] = {k: fingerprint(v) for k, v in tensors.items()}
        records.append(dict(index=index, label=int(restored['labels'][index]),
            complete=index < 200, domains=domains))
    save(folder/'catalog.json', dict(schema='S1_FIXED_TEST_PREFIX_V1', count=1000,
        complete_count=200, indices=list(range(1000)), radii_numerators=[2, 8],
        radius_denominator=255, input_file=str(path), input_sha256=sha(path),
        data_source=source, data_sha256=sha(source), samples=records,
        selection='dataset order only; no model forward or clean-correct selection',
        historical_exposure=True, old_experiments_unchanged=True,
        preprocessing='ToTensor float32, physically shared; float64 declared evaluation',
        bound_semantics='binary64 subtraction/clipping under numerical policy, not outward rational source proof'))
    save(folder/'complete.json', dict(status='COMPLETE', examples=1000, domains=2000,
        model_calls=0, verification_calls=0, seconds=time.monotonic()-ENTRY,
        input_sha256=sha(path), catalog_sha256=sha(folder/'catalog.json'),
        source_sha256=sha(__file__),
        bytes=sum(p.stat().st_size for p in folder.iterdir() if p.is_file())))
    signal.setitimer(signal.ITIMER_REAL, 0.)
    print(json.dumps(dict(status='COMPLETE', examples=1000, domains=2000, model_calls=0)))


if __name__ == '__main__':
    materialize(Path(sys.argv[1]))
