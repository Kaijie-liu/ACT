import unittest

from act.pipeline.moe.audit_certification_gap_evidence import validate_matrix


class AuditCertificationGapEvidenceTests(unittest.TestCase):
    def _values(self):
        protocol = {
            "coding": {
                "numerical_instantiation": ["FORMULA_ONLY"],
                "constant_protocol": ["SOUND"],
            },
            "required_semantic_fields": ["router"],
        }
        screening = {"included_records": [{"record_index": 7}]}
        evidence = {
            "url": "https://example.test/paper",
            "locator": "Theorem 1",
            "finding": "A formula is stated.",
        }
        matrix = {
            "status": "PARTIAL_RETRIEVAL_NO_PREVALENCE",
            "author_contact": "NOT_PERFORMED",
            "claim_limit": "This does not estimate prevalence.",
            "records": [
                {
                    "record_index": 7,
                    "primary_sources": ["https://example.test/paper"],
                    "dimensions": {
                        "numerical_instantiation": {
                            "code": "FORMULA_ONLY",
                            "evidence": [evidence],
                        },
                        "constant_protocol": {
                            "code": "SOUND",
                            "evidence": [evidence],
                        },
                    },
                    "semantics": {"router": "hard top-1"},
                }
            ],
        }
        return protocol, screening, matrix

    def test_complete_partial_matrix_passes(self):
        self.assertEqual(validate_matrix(*self._values()), [])

    def test_invalid_enum_and_missing_evidence_fail(self):
        protocol, screening, matrix = self._values()
        value = matrix["records"][0]["dimensions"]["constant_protocol"]
        value["code"] = "INVENTED"
        value["evidence"] = []
        issues = validate_matrix(protocol, screening, matrix)
        self.assertIn("record 7 has invalid enum for constant_protocol", issues)
        self.assertIn("record 7/constant_protocol has no primary-source evidence", issues)

    def test_screening_identity_drift_fails(self):
        protocol, screening, matrix = self._values()
        screening["included_records"][0]["record_index"] = 8
        self.assertIn(
            "matrix records differ from adjudicated included records",
            validate_matrix(protocol, screening, matrix),
        )


if __name__ == "__main__":
    unittest.main()
