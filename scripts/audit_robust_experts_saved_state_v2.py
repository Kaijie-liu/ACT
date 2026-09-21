"""Explicit ChainedScheduler schema support; the frozen v1 auditor is unchanged.

v1 incorrectly required a top-level last_epoch on the author's PolyLR chain.
Both saved child positions must still equal one, as required by the control.
"""
import argparse
import json
from pathlib import Path
import audit_robust_experts_saved_state as legacy

_tensor_state_check = legacy.check_state


def check_chained_state(checkpoint, prediction, result):
    schedulers = checkpoint.get('lr_schedulers', [])
    if len(schedulers) != 1:
        raise ValueError('missing PolyLR state')
    chain = schedulers[0]
    children = chain.get('_schedulers', [])
    if (chain.get('T_max') != 1 or chain.get('exponent') != .9 or len(children) != 2
            or any(s.get('last_epoch') != 1 or s.get('base_lrs') != [.1] for s in children)
            or children[0].get('start_factor') != .01 or children[0].get('total_iters') != 0):
        raise ValueError('unexpected PolyLR structure or child positions')
    actual_lr = [g['lr'] for g in checkpoint['optimizer_states'][0]['param_groups']]
    if chain.get('_last_lr') != actual_lr or children[1].get('_last_lr') != actual_lr or actual_lr != [0.]:
        raise ValueError('optimizer and scheduler LR mismatch')
    # The legacy tensor/optimizer checker receives a validated abstract position.
    # Stored checkpoint bytes and the original composite state are not edited.
    normalized = {**checkpoint, 'lr_schedulers': [{'last_epoch': 1}]}
    value = _tensor_state_check(normalized, prediction, result)
    value.update(audit='SAVED_TENSORS_AND_CHAINED_STATE_PASS',
        scheduler_schema='author_PolyLR_ChainedScheduler', scheduler_state=chain,
        v1_rejection='our auditor assumed top-level last_epoch; both child positions equal1')
    return value


def collect(path):
    original = legacy.check_state
    try:
        legacy.check_state = check_chained_state
        value = legacy.collect(path)
    finally:
        legacy.check_state = original
    value['checker_source_sha256'] = legacy.sha256(__file__)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    value = collect(args.config)
    if args.check:
        if json.loads(args.output.read_text()) != value:
            raise ValueError('saved audit differs')
    else:
        with args.output.open('x') as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write('\n')
    print(value['audit'])
    if not value['accepted']:
        raise SystemExit(1)
