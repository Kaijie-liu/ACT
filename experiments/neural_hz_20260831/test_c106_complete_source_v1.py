# SPDX-License-Identifier: AGPL-3.0-or-later
"""Full original nonconvex source and every owner/inverse/report remain equal."""
import numpy as np
import pytest
import scipy.sparse as sp
from act.back_end.hybridz_tf.tf_cnn import SparseHZAffineExpr,SparseHZAffineTerm
from experiments.neural_hz_20260831.c104_birth_emission_v1 import lift as original
from experiments.neural_hz_20260831.c106_birth_emission_v1 import lift
from experiments.neural_hz_20260831.c91_physical_archive_v1 import fingerprint
from experiments.neural_hz_20260831.test_c98_fresh_circuit_v1 import expression
from experiments.neural_hz_20260831.test_c62_physical_boundary_v2 import complete


@pytest.mark.parametrize('kind',['ordinary','repeated','shared'])
def test_complete_nonconvex_HZ_exact_source_identity(kind):
    expr=expression();term=expr.terms[0]
    if kind=='repeated':
        expr=SparseHZAffineExpr((term,term),expr.bias,expr.n_out,expr.frame_id)
    elif kind=='shared':
        operators=term.operators[:-1]+(sp.diags(np.full(expr.n_out,.25),format='csr'),)
        expr=SparseHZAffineExpr((term,SparseHZAffineTerm(term.source,operators)),expr.bias,expr.n_out,expr.frame_id)
    keep=np.ones(expr.n_out,bool)
    a,b=original(expr,keep,enabled=True),lift(expr,keep,enabled=True)
    assert fingerprint(a['state'])==fingerprint(b['state'])
    assert a['state']['fields']['report']==b['state']['fields']['report']
    assert b['state']['fields']['hz'].n_bin==1 and b['state']['fields']['hz'].exact
    for key in ('eq_uids','ineq_uids'):assert np.array_equal(a['construction'][key],b['construction'][key])
    assert a['construction']['selected']==b['construction']['selected']
    assert len(a['construction']['circuits'])==len(b['construction']['circuits'])
    for x,y in zip(a['construction']['circuits'],b['construction']['circuits']):
        for k in x:
            if isinstance(x[k],np.ndarray):assert x[k].dtype==y[k].dtype and x[k].tobytes()==y[k].tobytes()
            else:assert x[k]==y[k]


@pytest.mark.parametrize('kind',['chain','shared','conv_disjoint'])
def test_original_no_circuit_plan_rejection_remains(kind):
    _,saved=complete(kind)
    for fn in (original,lift):
        with pytest.raises(ValueError,match='no strictly smaller whole circuit plan'):
            fn(saved['expression'],saved['keep'],enabled=True)


def test_default_off_does_not_inspect_inputs():
    assert lift(object(),object()) is None
