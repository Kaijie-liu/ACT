"""Unchanged ordinary solve; bind its actual model and exact witness extension."""
from contextlib import contextmanager

from act.back_end.solver import solver_hz as backend
from experiments.neural_hz_20260831.c9_live_runtime_v1 import SelectedRejected
from experiments.neural_hz_20260831.c10_terminal_observer_v1 import observed
from experiments.neural_hz_20260831.c34_witness_reconstruction_v1 import recover
from experiments.neural_hz_20260831.c5_functional_transaction_v1 import measured_build
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool

_ACTIVE=False


@contextmanager
def installed(new,output,input_hz,input_shape,proof,*,enabled=False,emit=None,on_point=None):
    global _ACTIVE
    if not enabled:
        yield
        return
    if _ACTIVE:raise ValueError('nested terminal model/witness observation')
    original_lower=backend._lower_hz_milp;original_recover=backend.HZSolver._recover_input
    models=[];points=0;solves=0
    def event(data):
        if emit is not None:emit(data)
    def lower(actual,**kwargs):
        if (models or actual is not output or kwargs.get('project_inactive_cont') is not False
                or kwargs.get('fix_implied_binary') is not False or kwargs.get('prune_unused') is not True
                or kwargs.get('coalesce_rows') is not True):
            raise SelectedRejected('ordinary terminal changed actual source or enabled a rescue')
        model=original_lower(actual,**kwargs)
        if (model.n_cont!=proof['native_lowered_n_cont'] or model.n_bin!=proof['native_lowered_n_bin']
                or model.cont_eliminations or model.bin_fixes):
            raise SelectedRejected('actual final lowering differs from independently admitted dimensions')
        models.append(model)
        event(dict(event='c34_actual_ordinary_model_bound',n_cont=model.n_cont,n_bin=model.n_bin,
            rows=model.A.shape[0],predicate_nnz=model.A.nnz,no_base_feasibility_bypass=True))
        return model
    def recovered(model,x,inp,shape,lane):
        nonlocal points
        if len(models)!=1 or model is not models[0] or inp is not input_hz or tuple(shape)!=tuple(input_shape) or points:
            raise SelectedRejected('witness does not belong to the single actual bound terminal')
        points+=1
        try:
            if on_point is not None:on_point(model,x)
            (result,report),construction=measured_build(lambda:recover(new,model,x,inp,shape,lane,original_recover,
                pool=WorkPool(256_000_000),enabled=True))
            event(dict(event='c34_exact_witness_reconstruction_passed',proof=report,construction=construction))
            return result
        except Exception as exc:
            event(dict(event='c34_exact_witness_reconstruction_rejected',reason=str(exc),type=type(exc).__name__))
            raise SelectedRejected('exact witness reconstruction rejected: '+str(exc)) from exc
    _ACTIVE=True;backend._lower_hz_milp=lower;backend.HZSolver._recover_input=staticmethod(recovered)
    try:
        with observed(event):
            delegate=backend.milp
            def solve(*args,**kwargs):
                nonlocal solves
                solves+=1
                event(dict(event='c34_ordinary_milp_start',call=solves,
                    configured_call_limit_s=float(kwargs['options']['time_limit']),
                    first_base_call=solves==1))
                return delegate(*args,**kwargs)
            backend.milp=solve
            try:yield
            finally:backend.milp=delegate
    finally:
        backend._lower_hz_milp=original_lower;backend.HZSolver._recover_input=staticmethod(original_recover);_ACTIVE=False
