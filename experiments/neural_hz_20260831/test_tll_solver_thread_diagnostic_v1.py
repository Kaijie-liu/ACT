import pytest

from experiments.neural_hz_20260831.run_tll_solver_thread_diagnostic_v1 import classify, compare, summarize


def test_rejected_unknown_proposal_is_not_reported_invalid_adv():
    result = classify({"verdict": "UNKNOWN", "concrete_validations": [{"valid": False}]})
    assert result["counted_verdict"] == "UNKNOWN"
    assert result["rejected_solver_proposals"] == 1
    assert result["reported_invalid_adv"] == 0


@pytest.mark.parametrize("validations", [[], [{"valid": False}], [{"valid": True, "input_ok": False, "violates": True}]])
def test_reported_adv_without_valid_original_witness_is_error(validations):
    result = classify({"verdict": "ADV", "output_hz_exact": True, "concrete_validations": validations})
    assert result["counted_verdict"] == "ERROR" and result["reported_invalid_adv"] == 1


def test_certificate_requires_exact_terminal_hz():
    with pytest.raises(ValueError, match="exact terminal"):
        classify({"verdict": "CERT"})


def test_summary_never_unions_prior_solved_or_promotes():
    jobs = [{"iid": 0, "baseline": "CERT", "prior_candidate_solved": True},
            {"iid": 1, "baseline": "UNKNOWN", "prior_candidate_solved": True}]
    summary = summarize([{"iid": 0, "counted_verdict": "CERT"}, {"iid": 1, "counted_verdict": "UNKNOWN"}], jobs)
    assert summary["solved"] == [0] and summary["lost_prior_candidate_solved"] == [1]
    assert summary["lost_formal_solved"] == []
    assert summary["formal_gain"] == 0 and not summary["promotion_passed"]


def test_comparison_does_not_equate_missing_or_partial_calls():
    first = [{"iid": 0, "counted_verdict": "UNKNOWN", "calls": [{"problem": {"sha256": "a"}}]}]
    second = [{"iid": 0, "counted_verdict": "CERT", "calls": [
        {"problem": {"sha256": "a"}}, {"problem": {"sha256": "b"}}]}]
    result = compare(first, second)[0]
    assert result["numeric_call_matches"] == [True] and not result["all_calls_identical"]
    assert compare(first, [{"iid": 0, "counted_verdict": "ERROR"}])[0]["compared_numeric_calls"] == 0
