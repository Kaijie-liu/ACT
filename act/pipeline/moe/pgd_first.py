"""Default-off fixed B1 PGD recipe and strict full-model replay, never SAFE."""
import math
from act.pipeline.moe.request_budget import BudgetExhausted

RECIPE='B1_CE_20X3_V1'


def validate(config):
    cfg=config.get('pgd_first')
    if cfg is None:return None
    if not isinstance(cfg,dict) or type(cfg.get('enabled',False)) is not bool:
        raise ValueError('PGD-first explicit boolean configuration')
    if not cfg.get('enabled',False):return None
    if (cfg.get('recipe')!=RECIPE or type(cfg.get('seed')) is not int
            or not 0<=cfg['seed']<2**63 or type(cfg.get('seconds')) not in (int,float)
            or not math.isfinite(cfg['seconds']) or not 0<cfg['seconds']<=300
            or config.get('route_complexity_schedule',{}).get('total_seconds')!=300):
        raise ValueError('fixed PGD recipe/seed/sub-budget within shared request')
    return cfg


def strict_replay(model,point,*,lower,upper,label,budget):
    import torch
    budget.check('pgd_replay_before')
    if (point.shape!=lower.shape[1:] or not torch.isfinite(point).all()
            or not bool(((point>=lower[0])&(point<=upper[0])).all())):
        return dict(valid=False,reason='STRICT_DOMAIN_REJECTED')
    with torch.no_grad():output,route=model.forward_with_routing(point.unsqueeze(0))
    if not torch.isfinite(output).all():
        return dict(valid=False,reason='NONFINITE_DYNAMIC_OUTPUT')
    prediction=int(output.argmax(dim=1).item())
    value=dict(valid=prediction!=label,prediction=prediction,
               topk_set=sorted(int(v) for v in route.indices[0].tolist()))
    budget.check('pgd_replay_accept')
    return value


def attack(model,center,lower,upper,*,label,epsilon,cfg,budget):
    import torch
    import torch.nn.functional as F
    start=budget.clock();deadline=min(budget.deadline,start+float(cfg['seconds']))
    generator=torch.Generator(device='cpu').manual_seed(cfg['seed'])
    record=dict(recipe=RECIPE,seed=cfg['seed'],seconds_limit=cfg['seconds'],
        steps=20,restarts=3,step_size_fraction=.25,completed_steps=0,started_restarts=0,
        replay_calls=0,status='NO_COUNTEREXAMPLE_FOUND',
        grade='NATIVE_FLOAT64_FULL_MODEL_REPLAY_NOT_NETWORK_PROOF')
    witness=None
    def check():
        budget.check('pgd_first')
        return budget.clock()<deadline
    try:
        for restart in range(3):
            if not check():record['status']='PGD_SUBBUDGET_EXHAUSTED';break
            point=lower+(upper-lower)*torch.rand(center.shape,generator=generator,dtype=center.dtype)
            record['started_restarts']+=1
            for step in range(20):
                if not check():record['status']='PGD_SUBBUDGET_EXHAUSTED';break
                point.requires_grad_(True)
                loss=F.cross_entropy(model(point),torch.tensor([label]))
                gradient,=torch.autograd.grad(loss,point)
                if not torch.isfinite(gradient).all():
                    record['status']='NONFINITE_GRADIENT';return record,None
                if not check():record['status']='PGD_SUBBUDGET_EXHAUSTED';break
                with torch.no_grad():
                    point=torch.maximum(lower,torch.minimum(upper,
                        point+float(epsilon)*.25*gradient.sign())).detach()
                record['completed_steps']+=1;record['replay_calls']+=1
                replay=strict_replay(model,point[0],lower=lower,upper=upper,label=label,budget=budget)
                if budget.clock()>=deadline:record['status']='PGD_SUBBUDGET_EXHAUSTED';break
                if replay['valid']:
                    witness=point[0].clone();record.update(status='UNSAFE_REPLAYED',replay=replay)
                    return record,witness
            if record['status']=='PGD_SUBBUDGET_EXHAUSTED':break
    except BudgetExhausted:
        record['status']='REQUEST_BUDGET_EXHAUSTED'
    except RuntimeError as exc:
        record.update(status='CANDIDATE_GENERATION_ERROR',error=type(exc).__name__)
    finally:
        record['elapsed_seconds']=budget.clock()-start
    return record,witness
