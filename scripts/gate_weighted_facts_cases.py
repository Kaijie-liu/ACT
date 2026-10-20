"""Fixed standard-library controls. Backends are local stubs, never models/solvers."""
import ast
import builtins
import copy
from fractions import Fraction
import importlib.util
import math
from pathlib import Path
import time
from types import SimpleNamespace as NS

ROOT = Path("/data1/Kane/MOE/ACT")
CASES = (
    "default_and_config", "signed_not_safe", "gate_direction", "constants_and_dimensions",
    "tie_and_fixed_gate", "nonpositive_fallback", "scope_binding", "fact_binding",
    "gate_binding", "conditioned_router_binding", "support_fallback", "capture_alias",
    "missing_sources", "deadline_before_and_after", "old_positive_path",
    "staged_stub_differential", "monolithic_stub_differential",
    "source_audit_and_missing_obligation", "native_error_and_deadline",
)


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def functions(relative, names, namespace, imports):
    """Execute unchanged function ASTs with explicitly local backend doubles.

    No sys.modules replacement or global monkeypatch. Does not test heavyweight
    package imports, HZ propagation, optimizer behavior or deployed model semantics.
    """
    source = ast.parse((ROOT / relative).read_text())
    nodes = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    require({n.name for n in nodes} == set(names), "fixed function inventory")
    normal_import = builtins.__import__

    def local_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in imports:
            return imports[name]
        if name in {"time", "__future__"}:
            return normal_import(name, globals, locals, fromlist, level)
        raise ImportError("unregistered control dependency: " + name)

    namespace["__builtins__"] = dict(vars(builtins), __import__=local_import)
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
                      + nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(ROOT / relative), "exec"), namespace)
    return NS(**{name: namespace[name] for name in names})


def run(deadline, tick):
    g = load("act/pipeline/moe/gate_weighted_facts.py", "fixed_gate_facts")
    rb = load("act/pipeline/moe/request_budget.py", "fixed_budget")
    lp = load("act/back_end/solver/lp_certificate.py", "fixed_lp_check")
    work = g.Work(deadline)  # ONE ledger across all fixed controls.
    policy = dict(safe_positive_margin=1e-7, outward_absolute=1e-9, outward_relative=1e-9)
    config = dict(f0=dict(gate_weighted_facts=True), scoped_proof_reuse=True,
                  comparison_method="staged", route_complexity_schedule=dict(
                      version=1, total_seconds=300, multi_pair_tier1_fraction=.25))
    identity = dict(model_state=dict(sha256="a"*64), lower=dict(sha256="b"*64),
                    upper=dict(sha256="c"*64),
                    property=dict(kind="TOP1_ROBUST", classes=3, clean_prediction=0))
    scope = dict(identity, request_id=g.identity(identity), frame_id="f1", numerical_policy=policy,
                 gate="selected_softmax_top2", tie_policy="ANY_LEGAL_TOPK")
    a = dict(lower=[-4., 0., 10.], upper=[-4., 0., 10.])
    b = dict(lower=[2., 0., 10.], upper=[2., 0., 10.])
    branches = [dict(candidate=i, proof_output_bounds=copy.deepcopy(s))
                for i, s in enumerate((a, b, b))]
    support = dict(complete_exact=True, lower_status=["milp_optimal"],
                   upper_status=["milp_optimal"], gaps=[0.], solves=2, elapsed=.01)
    q = [1., -1., 0.]
    gate = g.gate_fact(scope, [0, 1], 0, [0., .25], [-4., -2.], support, "f1", work)
    row = g.propose_row(scope, [0, 1], 0, q, 0., [a, b], gate, work)
    require(row is not None and row["accepted_minimum"] > .49, "fixed positive control")
    old = functions("act/pipeline/moe/scoped_f0_proofs.py",
                    ("facts_from_branches", "reuse_property", "audit_reused_property", "make_scope"),
                    dict(math=math, check=lp.check, identity=lp.identity, rational=lp.rational,
                         hz_outward_slack=lambda *v: 1e-9 + 1e-9 * max(map(abs, v))),
                    {})

    def rejected(fn, types=(ValueError, KeyError, TimeoutError)):
        try:
            fn()
        except types as exc:
            return type(exc).__name__ + ": " + str(exc)[:180]
        raise AssertionError("invalid or late evidence accepted")

    def check(r=row, s=scope, sources=None, gt=gate):
        return g.check_row(r, scope=s, pair=[0, 1], index=0, q=q, constant=0.,
                           sources=sources or [a, b], gate=gt, work=work)

    def make_context(clock=None, supplied=branches):
        budget = rb.RequestBudget(300, clock=clock)
        return g.GateFacts(supplied, scope, budget, work=work)

    def gate_object(router, pair=(0, 1)):
        return NS(pair=pair, lambda_bounds=(0., .25), margin_bounds=(-4., -2.),
                  router_frame_id=router.frame_id, router_output_width=router.n_out,
                  conditioned_router=router, margin_support=NS())

    router = NS(frame_id="f1", n_out=3)
    saved = []
    pipeline_evidence = {}

    def pipeline(method, enabled, *, fault=None, all_positive=False):
        calls = dict(propagation=0, gate=0, build=0, solve=0)
        used_branches = ([
            dict(candidate=i, proof_output_bounds=dict(lower=[3., 0., 0.], upper=[3., 0., 0.]))
            for i in range(3)] if all_positive else branches)
        context = make_context(supplied=used_branches)
        budget = context.budget
        reuse = dict(scope=scope, facts=old.facts_from_branches(used_branches, scope))
        if enabled:
            reuse["gate_weighted"] = context

        def propagate(*args, **kwargs):
            calls["propagation"] += 1
            return NS(expert_a=NS(elapsed=0.), expert_b=NS(elapsed=0.), joint=NS())

        def compute(current, pair, **kwargs):
            calls["gate"] += 1
            return gate_object(current, pair)

        def build(*args, **kwargs):
            calls["build"] += 1
            return NS(margin_bounds=(-4., -2.), bounds=NS(
                lambda_lower=0., lambda_upper=.25, difference_lower=-10., difference_upper=10.),
                margin_support=NS(), difference_support=NS())

        def solve(*args, **kwargs):
            calls["solve"] += 1
            if fault == "exception":
                raise ValueError("fixed native failure")
            if fault == "deadline":
                raise rb.BudgetExhausted("fixed native deadline")
            return NS(status="UNKNOWN", reason="UNKNOWN_WEIGHTED_SOLVER_LIMIT",
                      solver_status=1, solver_gap=None, minimum=None, candidate_objective=None,
                      solver_certified_lower_bound=None, solver_bound_kind=None,
                      solver_primal_objective=None, solver_dual_objective=None,
                      candidate_input=None, pair_count=len(args[0]) if method == "monolithic" else 1)

        imports = {"act.pipeline.moe.scoped_f0_proofs": old}
        ns = dict(time=time, HybridZConfig=lambda **kw: NS(**kw),
                  linear_safety_rows=lambda spec, width: [(q, 0.), ([1., 0., -1.], 0.)],
                  condition_topk_set=lambda hz, pair: NS(hz=router),
                  guarded_input_topk_set=lambda *args: NS(hz=NS()),
                  shared_input_pair_propagation=propagate, compute_weighted_top2_gate_range=compute,
                  build_weighted_top2_f0=build, solve_weighted_top2_f0=solve,
                  solve_monolithic_weighted_top2_f0=solve,
                  _forward_validate=lambda *args, **kw: dict(valid=False, prediction=None, topk_set=None),
                  _support_record=lambda p: {}, _support_status=lambda p: copy.deepcopy(support),
                  BudgetExhausted=rb.BudgetExhausted, reuse_property=old.reuse_property,
                  asdict=lambda value: dict(vars(value)))
        for key in ("SAFE_WEIGHTED_RANGE", "UNSAFE_FULL_FORWARD_FALLBACK",
                    "UNKNOWN_WEIGHTED_SOLVER_LIMIT", "UNKNOWN_WEIGHTED_NUMERICAL",
                    "UNKNOWN_WEIGHTED_RELAXATION"):
            ns[key] = key
        if method == "staged":
            fs = functions("act/pipeline/moe/staged_verifier.py",
                           ("_pair_reason", "_f0_reason", "_run_f0_impl", "_run_f0"), ns, imports)
            entry = fs._run_f0
        else:
            fs = functions("act/pipeline/moe/paired_monolithic.py",
                           ("_run_monolithic", "run_monolithic"), ns, imports)
            entry = fs.run_monolithic
        internal = dict(program=NS(output_width=3, experts=[NS(), NS(), NS()]),
                        router=NS(output_hz=router, input_hz=NS()),
                        route_sets=NS(feasible=[(0, 1), (0, 2)]), lower=None, upper=None, output_spec=None)
        cfg = dict(support=dict(lp_neurons=2, milp_neurons=2, lp_time_limit=1., milp_time_limit=1.),
                   solver=dict(margin_support_seconds=1., difference_support_seconds=1.,
                               property_seconds=1., safety_tolerance=1e-7))
        result, witness = entry(model=None, center=NS(shape=(1, 3)), clean_prediction=0,
                                internal=internal, config=cfg, reuse=reuse, budget=budget)
        require(witness is None, "no fabricated witness")
        evidence = dict(identity=identity, request_id=scope["request_id"],
                        proof_reuse=dict(enabled=True, frame_id="f1"),
                        numerical_safety=policy, registered_budgets=config,
                        route_coverage=dict(feasible_route_sets=[[0, 1], [0, 2]],
                                            candidate_experts=[0, 1, 2], coverage_complete=True,
                                            candidate_set_minimal=True, route_sets_exact=True),
                        tier1=dict(branches=used_branches), tier2=result,
                        verdict=dict(status=result["status"],
                                     decision_tier="TIER2_F0" if method == "staged" else "MONOLITHIC_F0"))
        if enabled:
            evidence["gate_weighted_facts"] = context.record()
            g.audit_evidence(evidence, work=work)
        return calls, evidence

    def case(name):
        if name == "default_and_config":
            require(not g.validate_config({}) and g.validate_config(config), "default-off/config")
            bad = copy.deepcopy(config); bad["f0"]["gate_weighted_facts"] = 1
            return dict(rejected=rejected(lambda: g.validate_config(bad)))
        if name == "signed_not_safe":
            fact = g.signed_fact(scope, 0, 0, q, 0., a, work)
            require(fact["lower_bound"] < 0 and "status" not in fact, "signed fact is not SAFE")
            return dict(fact=fact, positive_combination=row)
        if name == "gate_direction":
            reverse = g.gate_fact(scope, [0, 1], 1, [.75, 1.], [2., 4.], support, "f1", work)
            r = g.propose_row(scope, [0, 1], 0, q, 0., [a, b], reverse, work)
            require(r["checked_lower_bound"] == row["checked_lower_bound"], "direction complement")
            return dict(reverse=reverse, bound=r["checked_lower_bound"])
        if name == "constants_and_dimensions":
            s = copy.deepcopy(scope); s["property"]["classes"] = 4
            changed_identity = copy.deepcopy(identity); changed_identity["property"]["classes"] = 4
            changed_identity["property"].update(kind="LINEAR_CONTROL",
                                                q=[1., -2., .5, 0.], constant=20.)
            s["property"] = copy.deepcopy(changed_identity["property"])
            s["request_id"] = g.identity(changed_identity)
            sources = [dict(lower=[1., 2., 3., 4.], upper=[2., 4., 5., 6.])] * 2
            gt = g.gate_fact(s, [0, 1], 0, [.2, .8], [-2., 2.], support, "f1", work)
            r = g.propose_row(s, [0, 1], 2, [1., -2., .5, 0.], 20., sources, gt, work)
            require(r is not None and 14.49 < r["accepted_minimum"] < 14.51, "constant counted once")
            return dict(row=r)
        if name == "tie_and_fixed_gate":
            out = []
            for bounds in ([.5, .5], [0., 0.], [1., 1.]):
                gt = g.gate_fact(scope, [0, 1], 0, bounds, [0., 0.], support, "f1", work)
                r = g.propose_row(scope, [0, 1], 0, q, 0., [a, b], gt, work)
                out.append(None if r is None else r["checked_lower_bound"])
            require(out[0] is None and out[1] is not None and out[2] is None, "fixed/tie range")
            return dict(bounds=out, semantics=[
                "tie at equal margins", "given-weight endpoint algebra only, not finite softmax",
                "given-weight endpoint algebra only, not finite softmax"])
        if name == "nonpositive_fallback":
            gt = g.gate_fact(scope, [0, 1], 0, [0., 1.], [-10., 10.], support, "f1", work)
            require(g.propose_row(scope, [0, 1], 0, q, 0., [a, b], gt, work) is None,
                    "unit gate must not prove unsafe expert interval")
            require(g.propose_row(scope, [0, 1], 1, [1., 0., -1.], 0., [a, b], gate, work) is None,
                    "two negative facts")
            return dict(fallbacks=2)
        if name == "scope_binding":
            errors = {}
            for field in ("request_id", "model_state", "lower", "upper", "property",
                          "frame_id", "numerical_policy", "gate", "tie_policy"):
                changed = copy.deepcopy(scope)
                changed[field] = {"wrong": True} if isinstance(changed[field], dict) else "wrong"
                errors[field] = rejected(lambda: check(s=changed))
            return errors
        if name == "fact_binding":
            errors = {}
            changes = dict(expert=2, property_index=1, q=[0., 1., -1.], constant=1.,
                           source=b, lower_bound=100., guarantee="STRICT_NETWORK_PROOF")
            for key, value in changes.items():
                r = copy.deepcopy(row); r["proof_sources"][0][key] = value
                errors[key] = rejected(lambda: check(r=r))
            for label, value in (("nan", math.nan), ("positive_inf", math.inf), ("negative_inf", -math.inf)):
                source = copy.deepcopy(a); source["lower"][0] = value
                errors[label] = rejected(lambda: g.signed_fact(scope, 0, 0, q, 0., source, work))
            return errors
        if name == "gate_binding":
            errors = {}
            for key, value in dict(scope_sha256="d"*64, pair=[1, 2], weight_expert=2,
                                   raw_lambda_bounds=[.8, .2], lambda_bounds=["1", "0"],
                                   frame_id="f2", margin_bounds=[2., -2.]).items():
                gt = copy.deepcopy(gate); gt[key] = value
                errors[key] = rejected(lambda: check(gt=gt))
            r = copy.deepcopy(row); r["accepted_minimum"] = 100.
            errors["forged_bound"] = rejected(lambda: check(r=r))
            for label, value in (("nan", math.nan), ("positive_inf", math.inf), ("negative_inf", -math.inf)):
                errors[label] = rejected(lambda: g.gate_fact(
                    scope, [0, 1], 0, [value, .25], [-4., -2.], support, "f1", work))
            return errors
        if name == "conditioned_router_binding":
            context = make_context()
            obj = gate_object(router)
            error = rejected(lambda: context.record_gate((0, 1), obj, support, NS(frame_id="f1", n_out=3)))
            obj.router_output_width = 4
            return dict(foreign_object=error,
                        width=rejected(lambda: context.record_gate((0, 1), obj, support, router)))
        if name == "support_fallback":
            out = []
            for change in (dict(complete_exact=False), dict(gaps=[None]), dict(lower_status=["fast_fallback"])):
                context = make_context(); sup = dict(support, **change)
                context.record_gate((0, 1), gate_object(router), sup, router)
                require(context.attempt((0, 1), 0, q, 0.) is None, "native support preserved")
                out.append(context.record())
            return dict(records=out)
        if name == "capture_alias":
            mutable = copy.deepcopy(branches); context = make_context(supplied=mutable)
            context.record_gate((0, 1), gate_object(router), support, router)
            mutable[0]["proof_output_bounds"]["lower"][0] = -1000.
            r = context.attempt((0, 1), 0, q, 0.)
            require(r["checked_lower_bound"] == row["checked_lower_bound"], "captured source aliased")
            return dict(changed_source=rejected(lambda: check(r=r, sources=[
                mutable[0]["proof_output_bounds"], b])))
        if name == "missing_sources":
            require(g.propose_row(scope, [0, 1], 0, q, 0., [None, b], gate, work) is None, "missing fallback")
            r = copy.deepcopy(row); r["proof_sources"].pop()
            duplicate = branches + [branches[0]]
            return dict(missing=rejected(lambda: check(r=r)),
                        duplicate=rejected(lambda: make_context(supplied=duplicate)))
        if name == "deadline_before_and_after":
            now = time.monotonic(); values = [now]
            context = make_context(clock=lambda: values[0])
            context.record_gate((0, 1), gate_object(router), support, router)
            values[0] += 301
            before = rejected(lambda: context.attempt((0, 1), 0, q, 0.))
            require(context.stats["discharged_pair_properties"] == 0, "late fact counted")
            values = [now, now, now, now, now + 301]
            def clock():
                return values.pop(0) if len(values) > 1 else values[0]
            context = make_context(clock=clock)
            context.record_gate((0, 1), gate_object(router), support, router)
            after = rejected(lambda: context.attempt((0, 1), 0, q, 0.))
            require(context.stats["attempted_pair_properties"] == 1
                    and context.stats["discharged_pair_properties"] == 0, "postcheck deadline counted")
            return dict(before=before, after=after)
        if name == "old_positive_path":
            good = [dict(candidate=i, proof_output_bounds=dict(
                lower=[3., 0., 0.], upper=[3., 0., 0.])) for i in (0, 1)]
            facts = old.facts_from_branches(good, scope)
            first = old.reuse_property(facts, scope, [0, 1], 0)
            context = make_context(supplied=good)
            # Old proof is chosen before any gate invocation.
            again = old.reuse_property(facts, scope, [0, 1], 0)
            require(first == again and not context.gates and context.stats["attempted_pair_properties"] == 0,
                    "old positive fast path")
            paths = {}
            for method in ("staged", "monolithic"):
                off_calls, off = pipeline(method, False, all_positive=True)
                on_calls, on = pipeline(method, True, all_positive=True)
                require(off_calls == on_calls == dict(propagation=0, gate=0, build=0, solve=0),
                        "positive proofs must skip propagation, gate and F0 in both paths")
                require(off["tier2"]["status"] == on["tier2"]["status"] == "SAFE",
                        "preserved complete old-positive verdict")
                paths[method] = dict(calls=on_calls, status=on["tier2"]["status"])
            return dict(proof=first, new_layer_attempts=0, function_stub_paths=paths)
        if name in ("staged_stub_differential", "monolithic_stub_differential"):
            method = name.split("_")[0]
            off_calls, off = pipeline(method, False)
            on_calls, on = pipeline(method, True)
            require(off["tier2"]["status"] == on["tier2"]["status"] == "UNKNOWN", "partial not SAFE")
            require(off_calls["propagation"] == on_calls["propagation"] == 2
                    and off_calls["gate"] == on_calls["gate"] == 2, "no extra query/reordered propagation")
            expected = (4, 2) if method == "staged" else (2, 1)
            require((off_calls["solve"], on_calls["solve"]) == expected, "fixed F0 stub counts")
            require(on["gate_weighted_facts"]["f0_solve_attempted"] == on_calls["solve"], "actual calls")
            pipeline_evidence[method] = on
            return dict(off_calls=off_calls, on_calls=on_calls, evidence=on,
                        scope="STUB_ONLY_NO_TIMING_OR_MODEL_EFFICACY")
        if name == "source_audit_and_missing_obligation":
            errors = {}
            for method, ev in pipeline_evidence.items():
                require(g.audit_evidence(ev, work=work) == 2, "nested/flat evidence")
                changed = copy.deepcopy(ev); changed["tier1"]["branches"][0]["proof_output_bounds"]["lower"][0] -= 1
                errors[method+"_source"] = rejected(lambda: g.audit_evidence(changed, work=work))
            imports = {"act.pipeline.moe.scoped_f0_proofs": old,
                       "act.pipeline.moe.gate_weighted_facts": g}
            audit = functions("act/pipeline/moe/audit_staged_evidence.py",
                ("_audit_safe_structure", "_audit_monolithic_partition", "_record_issue"),
                dict(math=math), imports)
            for method, ev in pipeline_evidence.items():
                partial = copy.deepcopy(ev); partial["tier2"]["status"] = "SAFE"
                if method == "staged":
                    for p in partial["tier2"]["pairs"]:
                        p["status"] = "SAFE"
                        p["property_rows"] = [r for r in p["property_rows"] if r["solver_bound_kind"] == g.KIND]
                else:
                    partial["tier2"]["property_rows"] = partial["tier2"]["property_rows"][:1]
                issues = []
                audit._audit_safe_structure(partial, issues, gate_work=work)
                require(bool(issues), "missing obligations accepted")
                errors[method+"_coverage"] = issues
            return errors
        if name == "native_error_and_deadline":
            calls, ev = pipeline("staged", True, fault="exception")
            require(ev["tier2"]["status"] == "UNKNOWN" and calls["solve"] == 2, "native error")
            out = dict(exception=ev)
            for method in ("staged", "monolithic"):
                calls, ev = pipeline(method, True, fault="deadline")
                require(ev["tier2"]["status"] == "TIMEOUT" and ev["tier2"]["partial_rows_censored"],
                        "partial cannot become SAFE")
                out[method] = ev
            return out
        raise AssertionError(name)

    for name in CASES:
        tick(); work.tick(); began = time.monotonic()
        try:
            detail = case(name)
            saved.append(dict(name=name, status="PASS", seconds=time.monotonic()-began, detail=detail))
        except Exception as exc:
            saved.append(dict(name=name, status="FAIL", seconds=time.monotonic()-began,
                              error=repr(exc)))
            return dict(status="FAIL", cases=saved, work_counts=work.counts)
    require(not any(n in __import__("sys").modules for n in ("torch", "numpy", "scipy")),
            "no heavy backend import")
    return dict(status="PASS", cases=saved, work_counts=work.counts,
                backend_scope="standard-library checks and unchanged-function AST with local stubs",
                model_executed=False, gpu_executed=False, native_solver_executed=False)
