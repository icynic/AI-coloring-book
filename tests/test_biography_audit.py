import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from evaluation.audit_biographies import (
    check_integrity,
    estimated_syllables,
    readability_metrics,
    run,
    validate_annotation,
)
from source_text import text_sha256
from summary_validation import split_source_sentences


class BiographyAuditTest(unittest.TestCase):
    def setUp(self):
        self.source_text = "Ada was a scientist. Ada studied at Marburg."
        self.source = {
            "summary": self.source_text,
            "revision_id": 12,
            "source_text_sha256": text_sha256(self.source_text),
        }
        self.summary = {
            "summary": "Ada was a scientist. She studied at Marburg.",
            "supporting_source_sentence_ids": [1, 2],
            "supporting_source_sentences": split_source_sentences(self.source_text),
            "source_revision_id": 12,
            "source_text_sha256": text_sha256(self.source_text),
            "word_count": 8,
        }
        self.annotation = {
            "slug": "Ada",
            "summary_text_sha256": text_sha256(self.summary["summary"]),
            "source_text_sha256": text_sha256(self.source_text),
            "claims": [{"claim": "Ada was a scientist.", "status": "supported", "evidence_ids": [1]}],
        }

    def test_integrity_and_hash_mismatch(self):
        self.assertTrue(check_integrity(self.summary, self.source, [5, 20])["valid"])
        changed = {**self.summary, "source_text_sha256": "wrong"}
        result = check_integrity(changed, self.source, [5, 20])
        self.assertFalse(result["valid"])
        self.assertIn("summary_source_hash_matches_text", result["errors"])

    def test_readability_keeps_unicode_words(self):
        metrics = readability_metrics("Grimm wrote Weisthümer. He studied at Marburg.")
        self.assertEqual(metrics["lexical_word_count"], 7)
        self.assertEqual(metrics["sentence_count"], 2)
        self.assertEqual(metrics["max_sentence_words"], 4)
        self.assertTrue(metrics["contains_marburg"])
        self.assertGreaterEqual(estimated_syllables("Weisthümer"), 1)

    def test_stale_review_and_invalid_evidence_are_rejected(self):
        changed = {**self.summary, "summary": "A different biography."}
        with self.assertRaisesRegex(ValueError, "Stale claim audit"):
            validate_annotation(self.annotation, changed, self.source)
        invalid = {**self.annotation, "claims": [{"claim": "Ada studied.", "status": "supported", "evidence_ids": [99]}]}
        with self.assertRaisesRegex(ValueError, "Invalid reviewed evidence"):
            validate_annotation(invalid, self.summary, self.source)

    def test_support_labels_do_not_follow_from_generated_evidence(self):
        annotation = {
            **self.annotation,
            "claims": [{"claim": "Ada won a prize.", "status": "unsupported", "evidence_ids": []}],
        }
        result = validate_annotation(annotation, self.summary, self.source)
        self.assertEqual(result["strict_supported_claim_rate"], 0)
        self.assertEqual(result["status_counts"]["unsupported"], 1)

    def test_end_to_end_audit_preserves_input_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "summaries").mkdir()
            (root / "sources").mkdir()
            (root / "manifest.json").write_text(json.dumps({
                "configuration": {"summary_word_range": [5, 20]},
                "items": [{"slug": "Ada", "query": "Ada"}],
            }), encoding="utf-8")
            summary_path = root / "summaries" / "Ada.json"
            summary_path.write_text(json.dumps(self.summary), encoding="utf-8")
            (root / "sources" / "Ada.json").write_text(json.dumps(self.source), encoding="utf-8")
            annotation_path = root / "annotations.json"
            annotation_path.write_text(json.dumps({
                "provenance": {"review_type": "test"}, "records": [self.annotation],
            }), encoding="utf-8")
            original_bytes = summary_path.read_bytes()
            result = run(SimpleNamespace(
                flux_run=str(root), output_dir=str(root / "audit"), annotations=str(annotation_path)
            ))
            self.assertEqual(result["overview"]["reviewed_claims"], 1)
            self.assertEqual(result["overview"]["integrity_passed"], 1)
            self.assertEqual(summary_path.read_bytes(), original_bytes)
            self.assertTrue((root / "audit" / "biography_audit.md").exists())


if __name__ == "__main__":
    unittest.main()
