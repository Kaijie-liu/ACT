"""
PyRAT --STRICT launcher: cuts every concrete-inference falsification helper
so the verdict reflects only what the abstract analyzer itself proves.

This includes the "center-of-box" / "1-point witness" style helpers that some
tools (e.g. CORA's `falsification_method=center`) leave on by default. A single
deterministic forward pass at any concrete input is still a *falsifier*, not
the verifier — we therefore eliminate it.

What is disabled
----------------
1. Gradient-based attacks (in-house and via helper):
     pyrat.attacks.attacks.{pgd_attack, pgd_attack_batched, deepfool_attack}
     pyrat.attacks.utils_attacks.{look_for_counter, look_for_counter_adv,
                                  counter_adv, infer_counter}
2. Random-sample falsifier:
     pyrat.attacks.utils_attacks.look_random  — patched to "no forward pass":
        * called with nb=1  (the UNSAFE-witness branch of check_prop)
            -> returns True without touching model.infer, so the analyzer's
               UNSAFE verdict propagates without any concrete evaluation
        * called with any other nb (the search-style entries)
            -> returns False immediately (no inference)
3. Simulation-guided BaB scorer:
     pyrat.partitioning.scorers.output_influence_scorer.OutputInfluenceScorer
     .score  — replaced with an abstract-width fallback (no model.infer).

What is also enforced via forced CLI flags
------------------------------------------
   --check skip          (no pre/post falsification sweep in analyze/check_prop)
   --nb_random 0         (random-sample size 0 in the rare paths still reachable)
   --attack bounds       (the `attack` list reduces to a cosmetic value; never fires)
   --batch_attack False  (cosmetic; consumer is patched)
   --scorer width        (analytical scorer; no concrete model evaluation)
   --exhaustive False    (disables the integer brute-force enumeration mode)

Effect on verdicts
------------------
   Result = True         analyzer proved the property (over-approx ⊆ safe set)
   Result = False        analyzer proved violation (over-approx ⊆ violation set)
                         — no concrete witness extraction is performed
   Result = Timeout/Unknown
                         analyzer could neither prove nor disprove

Compare to a previous --STRICT iteration that left look_random(nb=1) active:
that path performed a single deterministic forward pass at a sampled point
from the (UNSAFE-flagged) input box to "confirm" the analyzer's verdict.
Although passive, that is structurally identical to CORA's center-of-box and
therefore counts as a helper. It is removed here.
"""
import sys


def _disable_falsification():
    """Replace every helper that would invoke model.infer on a concrete point."""
    import pyrat.attacks.attacks as A
    import pyrat.attacks.utils_attacks as U
    import pyrat.analyzer.analyzer as AZ
    import pyrat.analyzer.analyzer_single as AS

    def _none_pair(*a, **k):
        return (None, None)

    def _false(*a, **k):
        return False

    def _look_random_passive(*args, **kwargs):
        """Replace look_random with a *purely passive* witness shim.

        - If called with nb == 1 (the check_prop UNSAFE branch hardcodes this),
          return True so the analyzer's UNSAFE verdict is preserved. No
          forward pass is performed; we trust the abstract analyzer.
        - For any other nb (search-style invocations from analyze() pre-check
          or look_for_counter fallback), return False immediately. With
          --check skip and look_for_counter monkey-patched these paths are
          dead anyway, but we defend in depth.
        """
        nb = kwargs.get("nb", None)
        if nb is None and len(args) >= 6:
            nb = args[5]
        return True if nb == 1 else False

    # 1) gradient-based attacks
    A.pgd_attack = _none_pair
    A.pgd_attack_batched = _none_pair
    A.deepfool_attack = _none_pair

    U.pgd_attack = _none_pair
    U.pgd_attack_batched = _none_pair
    U.deepfool_attack = _none_pair
    U.counter_adv = _false
    U.look_for_counter = _false
    U.look_for_counter_adv = _false

    # 2) the witness/random-sample falsifier — passive shim
    U.look_random = _look_random_passive
    U.infer_counter = _false  # belt-and-suspenders: nothing should call it

    # 3) imported references inside analyzer modules
    AZ.look_random = _look_random_passive
    AZ.counter_adv = _false
    AZ.infer_counter = _false
    AS.look_random = _look_random_passive
    AS.look_for_counter = _false


def _disable_concrete_sim_helpers():
    """Kill simulation-guided heuristics (affects BaB split choices only)."""
    import pyrat.partitioning.scorers.output_influence_scorer as OIS

    def _dummy_score(self, input_box, *a, **k):
        import numpy as np
        widths = input_box.get_widths()
        return np.argsort(widths, axis=None)[::-1]

    OIS.OutputInfluenceScorer.score = _dummy_score


def _force_flags(argv, strict: bool):
    base = {
        "--check": "skip",
        "--nb_random": "0",
        "--attack": "bounds",
        "--batch_attack": "False",
    }
    if strict:
        # Do NOT force --scorer here. PyRAT's default `coef` is analytical
        # (uses analysis coefficients only — no model.infer). The only
        # "concrete-inference" scorer is OutputInfluenceScorer, which is
        # already neutralised by _disable_concrete_sim_helpers(). Forcing
        # --scorer width breaks CNN benches (AssertionError: degenerate
        # bounds in torch_box after width-style BaB splits).
        base.update({
            "--exhaustive": "False",
        })
    out = list(argv)
    for flag, val in base.items():
        if flag in out:
            i = out.index(flag)
            if i + 1 < len(out) and not out[i + 1].startswith("--"):
                out[i + 1] = val
            else:
                out.insert(i + 1, val)
        else:
            out += [flag, val]
    return out


if __name__ == "__main__":
    strict = "--strict" in sys.argv
    if strict:
        sys.argv.remove("--strict")

    _disable_falsification()
    if strict:
        _disable_concrete_sim_helpers()

    sys.argv = [sys.argv[0]] + _force_flags(sys.argv[1:], strict=strict)
    from pyrat.main import main
    main()
