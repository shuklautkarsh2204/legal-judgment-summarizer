import unittest
from types import SimpleNamespace

from backend.app.services.evidence_verification import (
    RoleAwareEvidenceOrganizer,
    UNCLASSIFIED,
    classify_role,
    map_transferred_function,
    retain_role_verified_claims,
    verify_claim,
)
from backend.app.services.summarization import HierarchicalSummarizer


def source(passage_id, text, *, page_numbers=None):
    passage = {
        "document_id": "synthetic-judgment",
        "passage_id": passage_id,
        "text": text,
        "original_position": {
            "start_char": passage_id * 100,
            "end_char": passage_id * 100 + len(text),
        },
    }
    if page_numbers is not None:
        passage["page_numbers"] = page_numbers
    return RoleAwareEvidenceOrganizer().organize([passage])[0]


class EvidenceVerificationTests(unittest.TestCase):
    """Controlled synthetic examples; these are not legal-accuracy evaluations."""

    def test_party_submission_cannot_support_a_court_finding(self):
        evidence = source(
            1, "The petitioner argued that the investigation was unlawful."
        )

        result = verify_claim(
            "The Court held that the investigation was unlawful.", [evidence]
        )

        self.assertEqual(
            result["verification_result"], "potential_role_attribution_problem"
        )
        self.assertEqual(result["supporting_source_passage_ids"], [1])

    def test_unqualified_claim_from_party_evidence_is_withheld(self):
        evidence = source(
            11, "The petitioner argued that the investigation was unlawful."
        )
        claim = "The investigation was unlawful."
        verification = verify_claim(claim, [evidence])

        retained, withheld = retain_role_verified_claims(
            claim, [verification]
        )

        self.assertEqual(retained, "")
        self.assertEqual(len(withheld), 1)
        self.assertEqual(
            withheld[0]["verification_result"],
            "potential_role_attribution_problem",
        )

    def test_respondent_argument_is_not_attributed_to_petitioner(self):
        evidence = source(
            2, "The respondent argued that the detention was unlawful."
        )

        result = verify_claim(
            "The petitioner argued that the detention was unlawful.", [evidence]
        )

        self.assertEqual(
            result["verification_result"], "potential_role_attribution_problem"
        )
        self.assertEqual(result["evidence_role"], ["respondent_argument"])

    def test_court_decision_is_supported_by_court_decision_passage(self):
        evidence = source(
            3, "The Court dismissed the appeal after reviewing the record."
        )

        result = verify_claim("The Court dismissed the appeal.", [evidence])

        self.assertEqual(
            result["verification_result"], "supported_by_identified_evidence"
        )
        self.assertEqual(result["claim_category"], "court_decision")

    def test_claim_without_identifiable_evidence_is_insufficient(self):
        evidence = source(
            4, "The Court dismissed the appeal after reviewing the record."
        )

        result = verify_claim(
            "The Court awarded damages of 500 rupees.", [evidence]
        )

        self.assertEqual(result["verification_result"], "insufficient_evidence")
        self.assertEqual(result["supporting_source_passage_ids"], [])

    def test_claim_can_collect_support_from_multiple_passages(self):
        evidence = [
            source(5, "The petitioner argued detention was unlawful."),
            source(
                6,
                "The petitioner argued the investigation was incomplete, "
                "warrant absent, and evidence inadmissible.",
            ),
            source(
                7,
                "The petitioner argued the order illegal and hearing unfair, "
                "procedure flawed.",
            ),
        ]

        result = verify_claim(
            "The petitioner argued detention unlawful investigation incomplete "
            "warrant absent evidence inadmissible order illegal hearing unfair "
            "procedure flawed notice deficient record incomplete process unfair.",
            evidence,
        )

        self.assertEqual(
            result["verification_result"], "supported_by_identified_evidence"
        )
        self.assertGreaterEqual(len(result["supporting_source_passage_ids"]), 2)

    def test_opposing_outcome_is_flagged_as_possible_contradiction(self):
        evidence = source(
            8, "The Court dismissed the appeal after reviewing the record."
        )

        result = verify_claim("The Court allowed the appeal.", [evidence])

        self.assertEqual(result["verification_result"], "possible_contradiction")

    def test_ambiguous_source_role_requires_human_review(self):
        evidence = source(
            9,
            "The petitioner argued that the claim was valid, while the Court "
            "held that the claim was valid.",
        )

        self.assertEqual(evidence["evidence_role"], UNCLASSIFIED)
        result = verify_claim("The Court held the claim was valid.", [evidence])
        self.assertEqual(result["verification_result"], "requires_human_review")

    def test_missing_page_and_paragraph_metadata_is_explicit(self):
        evidence = source(10, "The Court dismissed the appeal.")

        result = verify_claim("The Court dismissed the appeal.", [evidence])
        reference = result["supporting_source_references"][0]

        self.assertIsNone(reference["page_numbers"])
        self.assertIsNone(reference["paragraph_reference"])
        self.assertIn("page_numbers", reference["unavailable_metadata"])
        self.assertIn("paragraph_reference", reference["unavailable_metadata"])

    def test_transfer_mapping_is_approximate_and_never_assigns_speaker(self):
        mapping = map_transferred_function("Subsumtion")

        self.assertEqual(mapping["approximate_function"], "court_reasoning")
        self.assertFalse(mapping["establishes_speaker"])
        self.assertEqual(
            classify_role("The respondent argued for dismissal.")["role"],
            "respondent_argument",
        )

    def test_long_generation_input_is_chunked_without_truncation(self):
        class RecordingGenerator:
            tokenizer = SimpleNamespace(
                model_max_length=100,
                encode=lambda text, add_special_tokens=True: text.split(),
            )
            model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=80))

            def __init__(self):
                self.inputs = []
                self.options = []

            def __call__(self, text, **kwargs):
                self.inputs.append(text)
                self.options.append(kwargs)
                return [{"summary_text": "synthetic output"}]

        generator = RecordingGenerator()
        summarizer = object.__new__(HierarchicalSummarizer)
        summarizer._generator = generator
        summarizer._generation_stats = {
            "input_token_limit": None,
            "generation_chunk_count": 0,
            "truncation_enabled": False,
        }
        source_tokens = [f"sourceword{index}" for index in range(100)]

        summarizer._generate(" ".join(source_tokens))

        self.assertGreater(len(generator.inputs), 1)
        self.assertTrue(all(len(item.split()) <= 80 for item in generator.inputs))
        combined_input = " ".join(generator.inputs)
        self.assertTrue(all(token in combined_input for token in source_tokens))
        self.assertTrue(all(options["truncation"] is False for options in generator.options))
        self.assertEqual(
            summarizer._generation_stats["generation_chunk_count"],
            len(generator.inputs),
        )


if __name__ == "__main__":
    unittest.main()
