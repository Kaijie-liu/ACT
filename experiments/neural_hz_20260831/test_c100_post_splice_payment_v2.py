"""Actual runtime writer adapter uses the fully proved post-splice payload."""
import pytest
from experiments.neural_hz_20260831.test_c28_consumer_discovery_v1 import source
from experiments.neural_hz_20260831.test_c30_first_write_v1 import view_from
from experiments.neural_hz_20260831.c30_append_discovery_v1 import discover_append
from experiments.neural_hz_20260831.c99_writer_bound_v1 import writer_bound
from experiments.neural_hz_20260831.c100_live_runtime_v2 import write_with_bound
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c10_fused_rows_v1 import WorkPool as CoupledPool


@pytest.mark.parametrize('subtract',[False,True])
def test_runtime_adapter_post_splice_bound_and_exact_payment(subtract):
    c,h,overlay,*_=source(mixed=True,subtract=subtract)
    view=view_from(c,h)
    plans,_=discover_append(c,view,overlay,pool=WorkPool(256_000_000),enabled=True)
    bill=writer_bound(view,plans,pool=WorkPool(256_000_000),enabled=True)
    budget=CoupledPool(0,0)
    new,report,payment=write_with_bound(view,plans,budget,bill)
    before=2*sum(m.nnz for n in ('Ac','Ab','Auc','Aub') for m in view.blocks(n))+view.n_eq+view.n_ineq
    assert before-bill['native_payload_upper']==5*len(plans)>0
    assert payment.native.used==payment.native.cap==bill['native_payload_upper']
    assert budget.used==bill['incremental_upper'] and new.exact
