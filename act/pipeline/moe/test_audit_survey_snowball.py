import unittest

from act.pipeline.moe.audit_survey_snowball import audit_document


class AuditSurveySnowballTests(unittest.TestCase):
    def _document(self):
        return {
            "status": "PARTIAL_RETRIEVAL_NO_PREVALENCE",
            "prevalence_claim_allowed": False,
            "zero_citation_counts_are_not_absence_evidence": True,
            "author_contact": "NOT_PERFORMED",
            "source_native_export_audit": [
                {
                    "source": f"source-{index}",
                    "endpoint": f"https://example.org/{index}",
                    "status": "LIMITED",
                    "limitation": "coverage limit",
                }
                for index in range(11)
            ],
            "snowball_candidates": [
                {
                    "dedup_key": f"key-{index}",
                    "primary_url": f"https://example.org/paper/{index}",
                    "preliminary_decision": "EXCLUDE",
                    "exclusion_code": "E_ATTACK_ONLY",
                    "discovery_edges": [{"seed": "seed", "direction": "BACKWARD"}],
                    "rationale": "no certificate",
                }
                for index in range(13)
            ],
            "snowball_coverage": {
                "non_seed_candidates": 13,
                "new_included_families": 0,
                "seed_families_reencountered_and_deduplicated": [1, 2, 3, 4],
            },
        }

    def test_valid_partial_artifact(self):
        self.assertEqual(audit_document(self._document()), [])

    def test_duplicate_and_prevalence_claim_are_rejected(self):
        document = self._document()
        document["prevalence_claim_allowed"] = True
        document["snowball_candidates"][1]["dedup_key"] = "key-0"
        issues = audit_document(document)
        self.assertIn("prevalence claims are not explicitly disabled", issues)
        self.assertIn("snowball candidates contain duplicate dedup keys", issues)


if __name__ == "__main__":
    unittest.main()
