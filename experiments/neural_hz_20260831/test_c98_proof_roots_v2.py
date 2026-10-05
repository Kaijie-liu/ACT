"""Explicit complete fresh graph/UID/packet roots at source proof boundary."""
import numpy as np
from experiments.neural_hz_20260831.test_c98_fresh_circuit_v1 import expression
from experiments.neural_hz_20260831.c97_birth_emission_v1 import lift as old_lift
from experiments.neural_hz_20260831.c98_birth_emission_v1 import lift
from experiments.neural_hz_20260831.c98_worker_v2 import proof_root_layout
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout


def test_full_proof_roots_include_graph_packets_and_all_prior_sources():
    expr=expression();keep=np.ones(expr.n_out,bool);pool=WorkPool(256_000_000)
    old=old_lift(expr,keep,enabled=True);draft=lift(expr,keep,enabled=True)
    data=dict(prior_source=old)
    complete=proof_root_layout(data,draft,pool)
    physical=numeric_layout(dict(prior_source=old,source=draft['state']),pool)
    assert complete['complete_proof_root_entries']>physical.resident_entries
    assert complete['new_graph_arrays']==4*len(draft['construction']['nodes'])
    assert complete['complete_packets']==len(draft['construction']['circuits'])

