"""Read-only whole-recipe cost rejection for first-affine tensor aliases.

No checkpoint, input, HZ source, solver, native package or GPU is loaded.
The conditional bounds concern the pinned V3 algorithms, not all verifiers.
"""
import argparse
import ast
import hashlib
import json
from math import comb, ceil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANCHORS = {
    "scoped_source/scaled_references.py": "534761cad44cbdf62ed31ee4d999fce0a5bd51aed4461b98d8f986b75d50d75d",
    "scoped_source/check_scaled_references.py": "e49210d311a7b4de1281fc7674ff946facf2f4bbd365b717068d8621d7a64696",
    "scoped_source/check_source_dag.py": "ae8f619784601d8b4864c8394165988181bea5275ade31cbb87bf5c245e7cd4c",
    "scoped_source/check_affine_references.py": "a0d688ee82bf88c42945edfd3abc9591a46998c01cf236cbf752ad5b549e9c62",
    "scoped_source/scaled_endpoints.py": "674f92c7c0034759b6cc6a6c7ab09037f65b1932902ba7f8558ac2fcf8c3cf40",
    "scoped_source/source_dag.py": "38863bafe654242077732bac9d9862d928ceb62eb5a77dcc44991cf2fab934e0",
    "scoped_source/affine_references.py": "0f0d10033640210e6ce79642fa340a61c6e4293cb76779e4851a1f0f5cf582d8",
    "scoped_source/forest_endpoints.py": "86396e9c662ba8c91f6ac3921dc7f410d9b977937df38a9fb782f82991c50a4c",
    "scoped_source/hz_request_store.py": "7cb2be7647039f59d3c896b7856f634f43be9c532d9d2fd43632be09ab3230e0",
    "act/back_end/moe/scaled_forest_endpoints.py": "118e88e3484ff4f4977f3b2535fa3aba8bdf7b1b1b7598da8232c0fe0ea57af1",
    "act/pipeline/moe/configs/experiment1_multiseed_training_r1.json": "1ad3cfcc4aaa181d29320dfea57f40d4113e67943684e352edb8da5cbae04f4b",
    "docs/hz_real_intake_20261001_r1.json": "9f5f7f240bbe6757206b52ea2754d30051ebd98d829ab8d6341823f25c87a8ab",
    "docs/hz_scaled_dense_20261013.json": "2c7415cd2a56a2b94f74484c9db17a96692cb5eedd22c63ba1951744d67b431d"
}
CAP = 8000000


def partitions(n, minimum=1):
    """All integer partitions: equivalence-class sizes, not queried routes."""
    if n == 0:
        yield ()
    else:
        for first in range(minimum, n+1):
            for tail in partitions(n-first, first):
                yield (first, *tail)


def consumer_floor(inputs, experts, classes, linear_outputs, nondegenerate, width=8):
    pairs = comb(experts, 2)
    if not (inputs > 0 and 2 <= experts <= 8 and 2 <= classes <= 10
            and 0 <= nondegenerate <= pairs and 1 <= width <= 8):
        raise ValueError('registered dimension/case range')
    guards = pairs * 2 * (experts-2)
    nodes = inputs + linear_outputs + guards
    properties = classes-1
    endpoints = (pairs+nondegenerate)*properties
    tiles = (pairs-nondegenerate)*ceil(properties/width) + nondegenerate*ceil(2*properties/width)
    factor_loops = 6*inputs*endpoints
    node_loops = nodes*(pairs+endpoints+tiles)
    return dict(nondegenerate_pairs=nondegenerate, required_properties=pairs*properties,
                endpoints=endpoints, tiles=tiles, factor_floor=inputs, node_floor=nodes,
                unconditional_factor_work=factor_loops, unconditional_node_work=node_loops,
                complete_consumer_work_floor=factor_loops+node_loops)


def anchors(blobs):
    if set(blobs) != set(ANCHORS): raise ValueError('complete reviewed-code inventory')
    for name, expected in ANCHORS.items():
        if hashlib.sha256(blobs[name]).hexdigest() != expected:
            raise ValueError('reviewed source changed: '+name)


def functions_with_lines(name, names):
    tree=ast.parse((ROOT/name).read_bytes()); found={}
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names:
            found[node.name]=dict(start=node.lineno,end=node.end_lineno)
    if set(found)!=set(names): raise ValueError('named cost seam missing')
    return dict(file=name,functions=found)


def analyze():
    blobs={name:(ROOT/name).read_bytes() for name in ANCHORS}; anchors(blobs)
    training=json.loads(blobs['act/pipeline/moe/configs/experiment1_multiseed_training_r1.json'])['training']
    prior=json.loads(blobs['docs/hz_real_intake_20261001_r1.json'])['dense_recipe']
    dense=json.loads(blobs['docs/hz_scaled_dense_20261013.json'])
    if dense['production_work']!=7351435 or dense['receipt']['result']['status']!='UNKNOWN_NONPOSITIVE':
        raise ValueError('closed dense result changed')
    n=prior['input_dimensions']; e=training['num_experts']; c=prior['classes']
    r=[n,*training['router_hidden'],e]; x=[n,*training['expert_hidden'],c]
    if r!=prior['router_widths'] or x!=prior['expert_widths'] or e!=prior['experts']:
        raise ValueError('registered model recipe disagreement')
    linear=sum(r[1:])+e*sum(x[1:])
    first=n*(r[1]+e*x[1])
    later=sum(a*b for a,b in zip(r[1:],r[2:]))+e*sum(a*b for a,b in zip(x[1:],x[2:]))
    cases=[]
    for groups in partitions(e):
        p=comb(e,2)-sum(comb(g,2) for g in groups)
        cases.append(dict(group_sizes=list(groups),**consumer_floor(n,e,c,linear,p)))
    zero=consumer_floor(n,e,c,linear,0)
    distinct=consumer_floor(n,e,c,linear,comb(e,2))
    positive_pairs=[case['nondegenerate_pairs'] for case in cases if case['nondegenerate_pairs']]
    pmin=min(positive_pairs); first_nonidentical=consumer_floor(n,e,c,linear,pmin)
    return dict(schema='TENSOR_ALIAS_COMPLETE_COST_REFUSAL_V1',
        reviewed_head='7684307c18a5c8269358c86cb56e5e96d19747e3',
        code_and_metadata=ANCHORS,work_cap=CAP,
        scope='registered seed1/2 layer shapes; no new seed0 topology check or actual router/weight observation',
        recipe=dict(inputs=n,experts=e,classes=c,router=r,expert=x,linear_outputs=linear,
                    pairs=comb(e,2),properties=c-1,first_weight_positions=first,later_weight_positions=later),
        actual_nondegenerate_pairs=None,actual_first_nnz=None,actual_nonzero_center_products=None,
        current_source_unconditional_first_visits=2*first,
        unchanged_consumer=dict(formula='6*F*E+N*(pairs+E+tiles)',
            assumptions=['all endpoints present', 'current full-factor loops and full-node pullbacks',
                         'first-weight aliases do not change node allocation or factor roster'],
            all_identical_router=zero,minimum_nonidentical_router=first_nonidentical,
            all_distinct_router=distinct,minimum_nonzero_pair_count=pmin,
            equivalence_partition_cases=cases,source_native_cost_assumed=0,
            actual_router_case_observed=False),
        optimistic_one_checker_visit_variant=dict(
            hypothesis='entire producer and native preparation are free; checker still visits each first weight once; current consumer unchanged',
            first_checker_visits=first,consumer_minimum=zero['complete_consumer_work_floor'],
            complete_partial_floor=first+zero['complete_consumer_work_floor']),
        conditional_exact_source=dict(
            definitions='Z=nonzero first weights; K=first positions with nonzero weight and nonzero input interval center',
            formula='at least 2*Z+2*K exact calls if current producer/checker norm and center computation retained',
            dense_zero_center_arithmetic_floor=2*first,
            dense_nonzero_center_arithmetic_floor=4*first,
            applies_to_actual_trained_weights='NOT_OBSERVED'),
        conditional_later_references=dict(
            dense_weight_positions=later,unchanged_reference_cap=65536,
            applies='distinct parent nodes and nonzero later weights; no actual nnz claim'),
        decision='STOP_FIRST_AFFINE_TENSOR_ALIASES_AS_STANDALONE_MECHANISM',
        real_execution_admitted=False,implementation_admitted=False,
        new_source_runs=0,new_bound_queries=0,checkpoint_loads=0,gpu_calls=0,
        scientific_conclusion='no new safety result, runtime comparison, or universal verification lower bound',
        next_mechanism='joint checked residual templates plus source-aware center/range and tensor operators requires full ledger first; not frozen or admitted',
        seams=[
            functions_with_lines('scoped_source/scaled_endpoints.py',['pullback','weighted_norm','_plan','prepare','accept']),
            functions_with_lines('scoped_source/scaled_references.py',['affine','local_norm']),
            functions_with_lines('scoped_source/check_scaled_references.py',['affine','local_norm_add']),
            functions_with_lines('scoped_source/source_dag.py',['input','node','finish']),
            functions_with_lines('scoped_source/affine_references.py',['affine','guard']),
        ])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--check-report',type=Path)
    args=parser.parse_args(); report=analyze()
    if args.check_report:
        if report!=json.loads(args.check_report.read_bytes()): raise ValueError('static report differs')
        print('PASS: frozen code, registered metadata and complete-query arithmetic; no source or solver run')
    else: print(json.dumps(report,sort_keys=True,indent=2))
