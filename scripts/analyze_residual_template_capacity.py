"""Static cost countercase for scoped residual templates, without model execution.

This does not observe nonzero weights, adjoints, ranges or candidate sparsity.
It cannot authorize implementation, a source intake or a numerical experiment.
"""
import argparse
import hashlib
import json
from math import ceil
from pathlib import Path

from scripts.analyze_tensor_alias_capacity import analyze as prior_analysis, ROOT

PARENT_SCRIPT_SHA = 'ecf037bdafb044e3173b8d560a58f7db8686bf0c673da49bd8ecf515c7844e3c'


def template_countercase(inputs, hidden, experts, properties, width=8):
    if not all(type(v) is int and v > 0 for v in (inputs, hidden, experts, properties, width)):
        raise ValueError('positive integer dimensions')
    if width > 8:
        raise ValueError('existing exact tile width at most eight')
    bank = inputs * hidden * experts
    products = bank * properties
    visits = bank * ceil(properties / width)
    return dict(first_expert_weight_positions=bank, templates=experts*properties,
                coefficient_column_products=products, exact_arithmetic=2*products,
                coefficient_visits=visits, partial_work=2*products+visits)


def source_floor(first_positions, nonzero_weights, nonzero_center_products):
    if not (0 <= nonzero_center_products <= nonzero_weights <= first_positions):
        raise ValueError('0 <= K <= Z <= N')
    return first_positions + nonzero_weights + nonzero_center_products


def analyze():
    parent = ROOT/'scripts/analyze_tensor_alias_capacity.py'
    if hashlib.sha256(parent.read_bytes()).hexdigest() != PARENT_SCRIPT_SHA:
        raise ValueError('parent static analysis identity')
    prior = prior_analysis()
    r = prior['recipe']; cap = prior['work_cap']
    first = r['first_weight_positions']; n = r['inputs']; e = r['experts']
    prop = r['properties']; hidden = r['expert'][1]
    cases = {str(w): template_countercase(n,hidden,e,prop,w) for w in (8,4,2,1)}
    biases = r['linear_outputs']
    source_values = first+r['later_weight_positions']+biases+n
    return dict(schema='RESIDUAL_TEMPLATE_STATIC_CAPACITY_DECISION_V1',
        reviewed_head='f5022871d5d3c458f02eb3961528a529c83244ce',
        parent_script_sha256=PARENT_SCRIPT_SHA,code_and_metadata=prior['code_and_metadata'],
        work_cap=cap,recipe=r,
        optimistic_source_checker=dict(first_visits=first,formula='N+Z+K',
            remaining_after_visits=cap-first,
            dense_zero_center=source_floor(first,first,0),
            dense_all_nonzero_center=source_floor(first,first,first),
            actual_Z=None,actual_K=None,
            hypothesis='producer and native free; one checker retains current exact row norms and centers'),
        template_countercase=dict(cases_by_width=cases,
            hypotheses=['dense nonzero first expert weights',
                        'all first hidden adjoints nonzero in every property template',
                        'current per-active-column exact multiply/add charging'],
            excluded_costs=['source','zero candidates','router','later layers','endpoint combination',
                            'native candidates','serialization','portable reception'],
            actual_candidate_sparsity_observed=False),
        necessary_sparse_case=dict(
            hypotheses='only one first source visit retained; active expert rows have 3072 nonzero entries; every other cost free',
            max_active_row_template_associations=(cap-first)//(2*n),
            total_possible_associations=e*hidden*prop,
            not_an_admission=True),
        symbolic_source_alternative=dict(
            mathematical_change='symbolic centers and independently checked coarse tensor magnitude ranges',
            one_source_value_scan_if_used=source_values,
            remainder_before_any_other_cost=cap-source_values,
            preserves_existing_enclosure=False,complete_ledger_available=False),
        decision='STOP_RESIDUAL_TEMPLATES_AS_SUFFICIENT_CAPACITY_FIX',
        implementation_admitted=False,real_execution_admitted=False,
        new_source_runs=0,new_bound_queries=0,checkpoint_loads=0,gpu_calls=0,
        scientific_limit='conditional code-path countercase, not actual trained cost or a universal verification lower bound')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-report',type=Path)
    args=parser.parse_args(); result=analyze()
    if args.check_report:
        if result != json.loads(args.check_report.read_bytes()):
            raise ValueError('static result differs')
        print('PASS: bound recipe and conditional arithmetic; no source, candidate or GPU execution')
    else:
        print(json.dumps(result,sort_keys=True,indent=2))
