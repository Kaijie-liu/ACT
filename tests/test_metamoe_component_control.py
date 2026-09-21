import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from metamoe_component_control import inspect_vnnlib, parse_backend_result
from audit_metamoe_component_control import check_spec


SPEC = """(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (>= X_0 -0.1))
(assert (<= X_0 0.1))
(assert (>= X_1 0.2))
(assert (<= X_1 0.3))
(assert
    (or
        (and (>= Y_1 Y_0))
    )
)
"""


class ComponentControls(unittest.TestCase):
    def test_complete_property(self):
        lo, hi = inspect_vnnlib(SPEC, 0, 2, 2)
        self.assertLess(lo[0], hi[0])

    def test_missing_bound(self):
        with self.assertRaises(ValueError):
            inspect_vnnlib(SPEC.replace("(assert (>= X_1 0.2))\n", ""), 0, 2, 2)

    def test_missing_property(self):
        with self.assertRaises(ValueError):
            inspect_vnnlib(SPEC.replace("(and (>= Y_1 Y_0))", ""), 0, 2, 2)

    def test_wrong_property_binding(self):
        with self.assertRaises(ValueError):
            inspect_vnnlib(SPEC, 1, 2, 2)

    def test_added_constraint(self):
        with self.assertRaises(ValueError):
            inspect_vnnlib(SPEC + "(assert false)\n", 0, 2, 2)

    def test_inverted_bounds(self):
        with self.assertRaises(ValueError):
            inspect_vnnlib(SPEC.replace(">= X_0 -0.1", ">= X_0 0.2"), 0, 2, 2)

    def output(self, status):
        return f"Result: {status} in 1.0000 seconds\nFinal verified acc: 100.0% (total 1 examples)\n"

    def test_native_positive_not_formal_safe(self):
        result = parse_backend_result(self.output("safe-incomplete"), 0, 1)
        self.assertEqual(result["status"], "BACKEND_POSITIVE")
        self.assertEqual(result["evidence_grade"], "AUTHOR_BACKEND_REPORTED_ONNX_RESULT")

    def test_unsafe_not_confused_with_safe(self):
        self.assertEqual(parse_backend_result(self.output("unsafe-bab"), 0, 1)["status"],
                         "BACKEND_UNSAFE_UNREPLAYED")

    def test_incomplete_and_duplicate_terminal(self):
        for output in ["", self.output("safe-incomplete") * 2]:
            self.assertEqual(parse_backend_result(output, 0, 1)["status"], "ERROR")

    def test_nonzero_exit_cannot_accept_positive(self):
        self.assertEqual(parse_backend_result(self.output("safe-incomplete"), 1, 1)["status"], "ERROR")

    def test_late_positive_not_accepted(self):
        for status, seconds in [("safe-incomplete", 301), ("safe-incomplete (timed out)", 1)]:
            self.assertEqual(parse_backend_result(self.output(status), 0, seconds)["status"], "TIMEOUT")

    def test_unknown_not_unsafe(self):
        self.assertEqual(parse_backend_result(self.output("unknown"), 0, 1)["status"], "UNKNOWN")

    def test_independent_parser_and_mutations(self):
        text = "\n".join([f"(declare-const X_{i} Real)" for i in range(3072)]
                         + [f"(declare-const Y_{i} Real)" for i in range(10)]
                         + [f"(assert (>= X_{i} -1))\n(assert (<= X_{i} 1))" for i in range(3072)]
                         + ["(assert (or " + " ".join(f"(and (>= Y_{i} Y_0))" for i in range(1, 10)) + "))"])
        lower, upper = check_spec(text, 0)
        self.assertEqual(len(lower), 3072)
        for changed in [text.replace("(assert (or", "(assert (and"),
                        text.replace("(and (>= Y_9 Y_0))", ""),
                        text + "\n(assert false)", text.replace("Y_9 Y_0", "Y_8 Y_0")]:
            with self.assertRaises((ValueError, IndexError)):
                check_spec(changed, 0)


if __name__ == "__main__":
    unittest.main()
