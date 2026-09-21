"""Freeze SCIENTIFIC training choices, not an untested GPU execution identity."""
import json
from pathlib import Path
import sys
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write


if __name__ == '__main__':
    parent = json.loads(Path('configs/recent_moe/robust_experts_resume_r1.json').read_text())
    audit_path = Path('docs/robust_experts_resume_archive_20260922_r1.json')
    if json.loads(audit_path.read_text())['audit'] != 'INDEPENDENT_SAVED_STATE_REREAD_PASS':
        raise ValueError('CPU continuation control missing')
    repo = Path(parent['repo'])
    sys.path.insert(0, str(repo))
    import hydra
    from omegaconf import OmegaConf, open_dict
    from src import run as native  # author resolver registration
    recipes = {}
    for name, experiment in [('dense', 'pgd_adv_train'), ('convmoe', 'pgd_adv_train_resnet_conv_moe')]:
        with hydra.initialize_config_dir(config_dir=str(repo/'configs'), version_base='1.1'):
            cfg = hydra.compose(config_name='config', overrides=[
                'experiment=cifar100-resnet18/'+experiment, 'logger=csv', 'callbacks=default'])
        with open_dict(cfg):
            cfg.trainer.min_epochs = cfg.trainer.max_epochs = 200
            cfg.trainer.gpus = 1
            cfg.model.optimizer.lr = .01
            cfg.datamodule.data_dir = parent['data_root']
            cfg.datamodule.num_workers = 2
            cfg.datamodule.pin_memory = True
            cfg.use_clearml = False
            cfg.wandb = {}
            if name == 'convmoe':
                cfg.model.model.k = 2
            cfg.callbacks.pop('early_stopping', None)
            cfg.callbacks.model_checkpoint.every_n_epochs = 1
            cfg.callbacks.model_checkpoint.save_last = True
            cfg.callbacks.model_checkpoint.save_top_k = 0
        recipes[name] = OmegaConf.to_container(cfg, resolve=False)
    write('configs/recent_moe/robust_experts_paper_training_recipe_r1.json', {
        'status': 'SCIENTIFIC_RECIPE_FIXED_GPU_EXECUTION_NOT_FROZEN', 'recipes': recipes,
        'data': 'native CIFAR100 split and five native train transforms, identical seed12345',
        'checkpoint_selection': 'completed epoch200 only; no validation/certification-based selection',
        'fixed_changes_from_source_default': ['200 instead of100 epochs', '.01 instead of.1 LR',
            'ConvMoE E4/k2 instead of source experiment k1', 'no early stop; completed-epoch recovery state',
            'local logging and location-only data forwarding, workers2'],
        'native_invariants': ['SGD momentum.9/wd.0005', 'PolyLR exponent.9', 'batch640',
            'PGD7 eps.03137/alpha.00784313725', 'eval clean/PGD20/APGD20',
            'native STE/router math and balancing loss retained'],
        'compatibility': 'R3 explicit APGD eval-mode restoration variant and optional SyncBN import repair',
        'cpu_continuation_archive_sha256': sha256(audit_path),
        'remaining_execution_gates': ['dedicated Blackwell-capable full workflow environment',
            'both final architectures/batch sizes native GPU update+save controls',
            'resource/whole-run deadline and outer-supervisor execution freeze',
            'resolved local-only output paths and checkpoint binding; no accidental remote logger'],
        'automatic_launch': False, 'ACT_full_domain_certificates': 'NOT_ESTABLISHED'})
    print('SCIENTIFIC_RECIPE_FIXED; GPU execution deliberately gated')
