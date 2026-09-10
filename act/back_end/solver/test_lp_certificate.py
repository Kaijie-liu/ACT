import copy
from fractions import Fraction
import unittest
from unittest.mock import patch

from act.back_end.solver.lp_certificate import check, identity, propose


class LPCertificateTests(unittest.TestCase):
    def test_guard_support_with_equality_and_residual(self):
        lp = {"c": [1, 1], "A": [[-1, 0]], "b": [-.5],
              "E": [[0, 1]], "h": [.25], "lower": [0, 0], "upper": [1, 1]}
        certificate = propose(lp)
        with patch("scipy.optimize.linprog", side_effect=AssertionError("checker used solver")):
            result = check(lp, certificate)
        self.assertEqual(Fraction(result["checked_lower_bound"]), Fraction(3, 4))
        # Approximate stationarity is repaired exactly, not assumed to hold.
        certificate["inequality_dual"] = [-.9]
        certificate["claimed_lower_bound"] = .65
        self.assertGreaterEqual(Fraction(check(lp, certificate)["checked_lower_bound"]), Fraction(.65))

    def test_invalid_sign_identity_and_overclaim_rejected(self):
        lp = {"c": [1], "A": [[-1]], "b": [-.5], "lower": [0], "upper": [1]}
        original = propose(lp)
        for key, value in (("inequality_dual", [1]), ("lp_sha256", "wrong"),
                           ("claimed_lower_bound", .5000000001)):
            altered = copy.deepcopy(original)
            altered[key] = value
            with self.assertRaises(ValueError):
                check(lp, altered)

    def test_binary_float_semantics_not_decimal_rounding(self):
        lp = {"c": [1], "lower": [.1], "upper": [.1]}
        cert = {"lp_sha256": identity(lp), "inequality_dual": [], "equality_dual": [],
                "claimed_lower_bound": "1/10"}
        value = Fraction(check(lp, cert)["checked_lower_bound"])
        self.assertEqual(value, Fraction.from_float(.1))
        self.assertNotEqual(value, Fraction(1, 10))

    def test_unbounded_box_and_bad_shape_fail(self):
        for lp in ({"c": [1], "lower": [0], "upper": ["1/0"]},
                   {"c": [1], "lower": [0], "upper": [1], "A": [[1, 2]], "b": [1]}):
            cert = {"lp_sha256": identity(lp), "inequality_dual": [0] if "A" in lp else [],
                    "equality_dual": [], "claimed_lower_bound": 0}
            with self.assertRaises((ValueError, ZeroDivisionError)):
                check(lp, cert)


if __name__ == "__main__":
    unittest.main()
