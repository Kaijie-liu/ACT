"""Read-only, independent saved-run review; no author runner/model/attack imports.

This checks recorded coverage and aggregates, NOT predictions by re-evaluation.
Writes only a new derived archive directory; frozen run files remain untouched.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import statistics
import time
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def state_digest(state):
    # Independently implement the documented tensor identity format.
    import torch
    h = hashlib.sha256()
    for name in sorted(state):
        t = state[name].detach().cpu().contiguous()
        require(bool(torch.isfinite(t).all()), 'nonfinite saved tensor')
        header = json.dumps([name, str(t.dtype), list(t.shape)], separators=(',', ':')).encode()
        data = t.reshape(-1).view(torch.uint8).numpy().tobytes()
        for item in (header, data):
            h.update(len(item).to_bytes(8, 'big'))
            h.update(item)
    return h.hexdigest()


def trajectory(folder, train, recipe):
    epochs = train['epochs']
    require(epochs == recipe['trainer']['max_epochs'], 'epoch configuration')
    observed = train['observations']
    require(observed['updates'] == train['global_step'], 'update count')
    require(set(observed['epoch_batches']) == {str(e) for e in range(epochs)}, 'batch epoch roster')
    require({p.name for p in folder.glob('epoch*.json')} ==
            {f'epoch{e:03d}.json' for e in range(epochs)}, 'journal roster')
    with (folder/'local/train/metrics.csv').open() as f:
        rows = list(csv.DictReader(f))
    grouped = {}
    for row in rows:
        if not row.get('epoch'):
            continue
        epoch = int(row['epoch'])
        group = grouped.setdefault(epoch, {})
        for key, value in row.items():
            if value:
                number = float(value)
                require(math.isfinite(number), 'nonfinite trajectory')
                require(key not in group or group[key] == number, 'conflicting CSV epoch metric')
                group[key] = number
    require(set(grouped) == set(range(epochs)), 'CSV epoch roster')
    output, updates, previous = [], 0, 0.0
    for epoch in range(epochs):
        journal = read(folder/f'epoch{epoch:03d}.json')
        batches = journal['train_batch_sizes']
        require(batches == observed['epoch_batches'][str(epoch)] == [640]*62+[320],
                'full 40000-example training epoch coverage')
        updates += len(batches)
        require(journal['epoch'] == epoch and journal['global_step'] == updates, 'epoch cursor')
        lr = journal['lr']
        target = recipe['model']['optimizer']['lr'] * (1-(epoch+1)/epochs)**.9
        require(len(lr) == 1 and math.isclose(lr[0], target, rel_tol=1e-12, abs_tol=1e-15),
                'polynomial LR trajectory')
        elapsed = journal['elapsed_seconds']
        require(math.isfinite(elapsed) and elapsed > previous, 'nonmonotone epoch time')
        m = grouped[epoch]
        keys = ['train/acc', 'val/acc', 'loss', 'main_loss', 'aux_loss',
                'val/loss', 'val/main_loss', 'val/aux_loss']
        require(all(k in m for k in keys), 'missing loss/accuracy trajectory')
        require(m['step'] == updates-1, 'CSV step/journal mismatch')
        for prefix in ('', 'val/'):
            require(math.isclose(m[prefix+'loss'], m[prefix+'main_loss']+m[prefix+'aux_loss'],
                                 abs_tol=3e-6, rel_tol=1e-6), 'loss decomposition')
        require(0 <= m['train/acc'] <= 1 and 0 <= m['val/acc'] <= 1, 'accuracy range')
        output.append({'epoch_completed': epoch+1, 'global_step': updates,
                       'lr_after_epoch': lr[0], 'epoch_interval_seconds': elapsed-previous,
                       **{k: m[k] for k in keys}})
        previous = elapsed
    require(updates == train['global_step'], 'final update cursor')
    return output


def evaluation(folder, train, ev, checkpoint_hash):
    require(ev['full_test_set'] is True and ev['empirical_only'] is True and ev['formal_SAFE'] is False,
            'evaluation scope')
    require([r['kind'] for r in ev['records']] == ['clean', 'PGD20', 'APGD20'], 'test roster')
    with (folder/'local/evaluate/metrics.csv').open() as f:
        csv_rows = list(csv.DictReader(f))
    output = {}
    for r, prefix in zip(ev['records'], ('test/', 'attack/pgd/', 'attack/apgd/')):
        require(r == read(folder/f"evaluation_{r['kind']}.json"), 'per-evaluation source')
        require(r['examples'] == 10000 and r['batches'] == (16 if r['kind']=='clean' else 40),
                'test denominator/batches')
        require(r['checkpoint_sha256'] == checkpoint_hash and r['state_digest'] == train['state_digest'],
                'evaluation state identity')
        require(len(r['metrics']) == 1, 'ambiguous aggregate')
        metrics = r['metrics'][0]
        require(set(metrics) == {prefix+s for s in ('acc', 'loss', 'main_loss', 'aux_loss')},
                'evaluation metric roster')
        require(all(math.isfinite(float(v)) for v in metrics.values()), 'nonfinite evaluation')
        for k, value in metrics.items():
            found = [float(row[k]) for row in csv_rows if row.get(k)]
            require(found == [value], 'evaluation JSON/CSV inconsistency')
        accuracy = metrics[prefix+'acc']
        require(0 <= accuracy <= 1, 'test accuracy range')
        # Integer count is reconstructed from a float32 aggregate, not a replay.
        correct = round(accuracy*10000)
        require(abs(accuracy*10000-correct) < .001, 'non-count-compatible accuracy')
        output[r['kind']] = {'recorded_accuracy': accuracy, 'correct_from_aggregate': correct,
                            'examples': 10000, 'accuracy_percent': correct/100,
                            'worker_seconds': r['seconds'], 'metrics': metrics}
    return output


def checkpoint_review(path, train, binding, recipe):
    import torch
    torch.set_num_threads(2)
    # Only locally generated, explicitly user-scoped checkpoints are unpickled.
    state = torch.load(path, map_location='cpu', weights_only=False)
    require(state['epoch'] == train['epochs'] and state['global_step'] == train['global_step'],
            'final checkpoint cursor')
    require(state['reproduction_binding'] == binding, 'checkpoint provenance')
    require(state_digest(state['state_dict']) == train['state_digest'], 'saved tensor identity')
    require(state['reproduction_rng'] and state['reproduction_cuda_rng'], 'RNG missing')
    require(len(state['optimizer_states']) == 1 and len(state['lr_schedulers']) == 1, 'optimizer roster')
    opt = state['optimizer_states'][0]
    for group in opt['param_groups']:
        for k in ('momentum', 'weight_decay', 'nesterov'):
            require(group[k] == recipe['model']['optimizer'][k], 'optimizer parameter: '+k)
        require(group['lr'] == 0 and group['initial_lr'] == .01, 'final/initial optimizer LR')
    buffers = [v['momentum_buffer'] for v in opt['state'].values() if 'momentum_buffer' in v]
    require(buffers and all(bool(torch.isfinite(v).all()) for v in buffers), 'momentum state')
    sch = state['lr_schedulers'][0]
    require(sch['T_max'] == train['epochs'] and sch['exponent'] == .9 and sch['_last_lr'] == [0.],
            'scheduler horizon/final LR')
    require(len(sch['_schedulers']) == 2 and all(s['last_epoch'] == train['epochs'] for s in sch['_schedulers']),
            'scheduler child cursors')
    return {'finite_state_tensors': len(state['state_dict']), 'finite_momentum_buffers': len(buffers),
            'optimizer_groups': [{k:v for k,v in g.items() if k != 'params'} for g in opt['param_groups']],
            'scheduler': sch, 'exact_gpu_resume_established': False}


def review(config_path):
    started = time.monotonic()
    cfg = read(config_path)
    require(cfg['mode'] == 'training' and cfg['execution_freeze_approved'], 'not frozen training')
    root, binding = Path(cfg['output_root']), digest(config_path)
    for path, expected in cfg['files'].items():
        require(digest(path) == expected, 'frozen input changed: '+path)
    print(f"Checked {len(cfg['files'])} frozen source/data identities", flush=True)
    summary = read(root/'summary.json')
    require(summary['accepted'] and summary['config_sha256'] == binding and
            [r['arm'] for r in summary['records']] == ['dense', 'convmoe'], 'summary roster')
    arms, trajectories = [], {}
    inventory = {str(config_path): binding, str(root/'summary.json'): digest(root/'summary.json')}
    for arm in ('dense', 'convmoe'):
        began = time.monotonic()
        folder = root/arm
        term, receipt, inner = [read(folder/n) for n in ('terminal.json', 'receipt.json', 'inner_terminal.json')]
        require(term in summary['records'] and term['accepted'] and term['error'] is None and
                term['status'] == 'FINAL_TRAINING_AND_EVALUATION_AUDITED', 'terminal status')
        require(receipt['status'] == 'COMPLETED' and term['execution_seconds'] <= cfg['seconds_per_arm'],
                'outer status/deadline')
        require(term['execution_seconds'] == receipt['execution_including_preflight_seconds'] and
                term['with_postflight_seconds'] == receipt['total_with_postflight_seconds'], 'receipt costs')
        require(inner['status'] == 'PIPELINE_COMPLETED' and inner['config_sha256'] == binding and
                inner['arm'] == arm and set(inner['stage_hashes']) == {'train', 'evaluate', 'audit'}, 'stage roster')
        for kind in ('stdout', 'stderr'):
            require(digest(folder/(kind+'.txt')) == receipt[kind+'_sha256'], 'outer log identity')
        checkpoint_hash = digest(folder/'final_epoch.ckpt')
        stages, costs = {}, {}
        for stage in ('train', 'evaluate', 'audit'):
            stages[stage] = r = read(folder/(stage+'.json'))
            require(r['status'] == 'COMPLETED' and r['config_sha256'] == binding and r['arm'] == arm and
                    r['checkpoint_sha256'] == checkpoint_hash, 'stage binding')
            require(digest(folder/(stage+'.json')) == inner['stage_hashes'][stage], 'stage hash')
            finished = read(folder/(stage+'_finished.json'))
            require(finished['returncode'] == 0, 'failed subprocess')
            costs[stage] = finished['seconds']
        require(sum(costs.values()) <= term['execution_seconds']+.05, 'stage cost overflow')
        train, ev, audit = [stages[s] for s in ('train','evaluate','audit')]
        for path, expected in audit['source_files'].items():
            require(digest(path) == expected, 'saved audit source hash')
        prepared = read(folder/'prepared.json')
        require(prepared['config_sha256'] == binding and prepared['arm'] == arm and
                prepared['mode'] == 'training', 'prepared binding')
        recipe = cfg['recipes'][arm]
        for k in ('model', 'attack_model', 'datamodule'):
            expected = copy.deepcopy(recipe[k])
            if k == 'model':
                expected['scheduler']['T_max'] = 200  # frozen execution resolves the symbolic horizon
            require(prepared['symbolic_config'][k] == expected, 'resolved scientific recipe drift: '+k)
        require(not any(k.startswith('limit_') or k in ('fast_dev_run', 'max_steps')
                        for k in prepared['symbolic_config']['trainer']), 'hidden batch/step limit')
        expected_names = {f'epoch{e:03d}.ckpt' for e in range(train['epochs'])}
        require(set(train['epoch_checkpoints']) == expected_names and
                {p.name for p in (folder/'checkpoints').glob('*.ckpt')} == expected_names, 'checkpoint roster')
        for name, expected in train['epoch_checkpoints'].items():
            require(digest(folder/'checkpoints'/name) == expected, 'epoch checkpoint changed: '+name)
        saved = checkpoint_review(folder/'final_epoch.ckpt', train,
                                  {'arm': arm, 'config_sha256': binding}, recipe)
        logging = read(folder/'logging.json')
        require(logging['separate_stages'] and set(logging['files']) ==
                {str(folder/'local'/s/'metrics.csv') for s in ('train','evaluate')}, 'log roster')
        for path, expected in logging['files'].items():
            require(digest(path) == expected, 'retained CSV identity')
        trajectories[arm] = curve = trajectory(folder, train, recipe)
        metrics = evaluation(folder, train, ev, checkpoint_hash)
        maximum = max(curve, key=lambda r:r['val/acc'])
        arms.append({'arm': arm, 'epochs': train['epochs'], 'updates': train['global_step'],
            'epoch_checkpoint_hashes_checked': len(expected_names), 'checkpoint_sha256': checkpoint_hash,
            'checkpoint_bytes': (folder/'final_epoch.ckpt').stat().st_size,
            'saved_state': saved, 'evaluations': metrics, 'stage_outer_seconds': costs,
            'execution_seconds': term['execution_seconds'],
            'journal_epoch_interval_median_seconds': statistics.median(r['epoch_interval_seconds'] for r in curve),
            'best_observed_validation': {'epoch': maximum['epoch_completed'], 'accuracy': maximum['val/acc'],
                                        'not_used_for_checkpoint_selection': True},
            'last20_validation_mean': statistics.mean(r['val/acc'] for r in curve[-20:]),
            'milestones': [r for r in curve if r['epoch_completed'] in (1,20,50,100,150,180,200)],
            'audit_seconds': time.monotonic()-began})
        # Retain a relocatable list of small source receipts and original identity.
        for path in sorted(folder.glob('*.json')):
            inventory[str(path)] = digest(path)
        for path in logging['files']:
            inventory[path] = digest(path)
        inventory[str(folder/'final_epoch.ckpt')] = checkpoint_hash
        print(f"{arm}: 200 epoch hashes, final state, trajectories and all evaluations checked", flush=True)
    require(sum(r['with_postflight_seconds'] for r in summary['records']) <=
            summary['whole_launch_with_postflight_seconds'], 'whole launch accounting')
    result = {'status': 'INDEPENDENT_SAVED_TRAINING_REVIEW_PASS', 'config_sha256': binding,
              'run_root': str(root), 'reviewer_sha256': digest(__file__),
              'frozen_input_files_checked': len(cfg['files']), 'arms': arms,
              'whole_launch_with_postflight_seconds': summary['whole_launch_with_postflight_seconds'],
              'separate_review_seconds': time.monotonic()-started,
              'scope': 'independent saved-file/tensor/coverage/aggregate consistency review; no new inference or attack',
              'prediction_replay': False, 'formal_SAFE': False, 'exact_gpu_resume': False,
              'per_input_prediction_records_available': False,
              'evidence_limit': 'counts are runtime receipts and reconstructed aggregates, not independently replayed per-input predictions',
              'source_inventory': inventory}
    return result, trajectories


def save_plots(folder, trajectories):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    panels = [('val/acc', 'Clean validation accuracy (%)', 100),
              ('train/acc', 'PGD-7 augmented training accuracy (%)', 100),
              ('main_loss', 'Training classification loss (not total loss)', 1),
              ('aux_loss', 'Training auxiliary loss (negative entropy)', 1)]
    for ax, (key, title, scale) in zip(axes.flat, panels):
        for arm, curve in trajectories.items():
            ax.plot([r['epoch_completed'] for r in curve], [scale*r[key] for r in curve], label=arm)
        ax.set_title(title)
        ax.set_xlabel('Completed epoch')
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle('Frozen Robust Experts R1: all 200 epochs; final-epoch selection unchanged')
    fig.tight_layout()
    fig.savefig(folder/'trajectory.svg')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(exist_ok=False)
    try:
        result, curves = review(args.config.resolve())
        with (args.output/'trajectory.csv').open('x', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['arm', *next(iter(curves.values()))[0]])
            writer.writeheader()
            writer.writerows({'arm':arm, **row} for arm, rows in curves.items() for row in rows)
        save_plots(args.output, curves)
        result['derived_files'] = {name:digest(args.output/name) for name in ('trajectory.csv','trajectory.svg')}
        with (args.output/'review.json').open('x') as f:
            json.dump(result, f, indent=2, allow_nan=False)
            f.write('\n')
        print(result['status'], flush=True)
    except BaseException as exc:
        with (args.output/'review_failed.json').open('x') as f:
            json.dump({'status':'REVIEW_FAILED', 'error':repr(exc)}, f)
        raise


if __name__ == '__main__':
    main()
