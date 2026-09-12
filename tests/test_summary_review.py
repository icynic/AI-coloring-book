import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from main import parse_args, run_pipeline
from refine_biographies import prepare_refinement
from source_text import text_sha256
from Summarizer import Summarizer
from summary_review import (
    SummaryReviewError, acceptance_problems, make_policy, refine_biography,
    validate_review, validate_stored_review,
)
from summary_validation import split_source_sentences


SOURCE = ("Ada was a scientist. She met Liebig in Germany. "
          "She studied at the University of Marburg.")
DRAFT = {"summary": "Ada was a scientist. She studied with Liebig.",
         "supporting_source_sentence_ids": [1, 2]}
FIXED = {"summary": SOURCE, "supporting_source_sentence_ids": [1, 2, 3]}


def row(index, source_index, status="supported", reason=""):
    return {"sentence_id": index, "status": status, "reason": reason,
            "evidence": [{"source_sentence_id": source_index,
                          "quote": split_source_sentences(SOURCE)[source_index - 1]}]}


BAD_REVIEW = {"sentence_reviews": [row(1, 1), row(2, 2, "unsupported", "Met does not entail studied with.")],
              "issues": [{"kind": "marburg_missing", "reason": "Include the supported study connection."}]}
GOOD_REVIEW = {"sentence_reviews": [row(i, i) for i in range(1, 4)], "issues": []}


def model_with_outputs(*outputs):
    model = Summarizer.__new__(Summarizer)
    model.model_name = "test-model"
    model.revision = "test-revision"
    model.quantization = "4bit"
    model.model_load_seconds = 0
    model.pipe = Mock(side_effect=[[{"generated_text": json.dumps(o)}] for o in outputs])
    return model


class ReviewValidationTest(unittest.TestCase):
    def test_exact_quote_and_complete_coverage(self):
        self.assertEqual(validate_review(GOOD_REVIEW, SOURCE, SOURCE), GOOD_REVIEW)

    def test_missing_duplicate_or_invalid_sentence_ids_are_rejected(self):
        for mutate in (
            lambda r: r["sentence_reviews"].pop(),
            lambda r: r["sentence_reviews"][1].update(sentence_id=1),
            lambda r: r["sentence_reviews"][0].update(sentence_id=True),
        ):
            review = copy.deepcopy(GOOD_REVIEW)
            mutate(review)
            with self.assertRaises(ValueError):
                validate_review(review, SOURCE, SOURCE)

    def test_fake_quotes_out_of_range_and_boolean_source_ids_are_rejected(self):
        for update in ({"quote": "She studied with Liebig in Germany."},
                       {"source_sentence_id": 999}, {"source_sentence_id": True}, {"quote": ""}):
            review = copy.deepcopy(GOOD_REVIEW)
            review["sentence_reviews"][0]["evidence"][0].update(update)
            with self.assertRaises(ValueError):
                validate_review(review, SOURCE, SOURCE)

    def test_source_conflict_requires_two_distinct_quotes(self):
        review = copy.deepcopy(GOOD_REVIEW)
        review["sentence_reviews"][0].update(status="source_conflict", reason="Dates disagree.")
        with self.assertRaises(ValueError):
            validate_review(review, SOURCE, SOURCE)
        review["sentence_reviews"][0]["evidence"].append(row(1, 2)["evidence"][0])
        validate_review(review, SOURCE, SOURCE)
        self.assertTrue(acceptance_problems(review, SOURCE, make_policy(SOURCE, "10-14", 8, 30)))

    def test_empty_evidence_unknown_status_and_invalid_issues_fail_closed(self):
        for mutate in (
            lambda r: r["sentence_reviews"][0].update(evidence=[]),
            lambda r: r["sentence_reviews"][0].update(status="probably_correct"),
            lambda r: r.update(issues="none"),
            lambda r: r.update(issues=[{"kind": "age_style", "reason": ""}]),
            lambda r: r.update(passed=True),
        ):
            review = copy.deepcopy(GOOD_REVIEW)
            mutate(review)
            with self.assertRaises(ValueError):
                validate_review(review, SOURCE, SOURCE)

    def test_marburg_keyword_requirement_is_not_left_to_the_model(self):
        text = "Ada was a scientist."
        review = {"sentence_reviews": [row(1, 1)], "issues": []}
        validate_review(review, text, SOURCE)
        self.assertIn("marburg_missing", acceptance_problems(
            review, text, make_policy(SOURCE, "10-14", 1, 30))[0])


class RefinementLoopTest(unittest.TestCase):
    def test_fact_revision_is_checked_again_in_fresh_chats_and_original_is_unchanged(self):
        initial = copy.deepcopy(DRAFT)
        model = model_with_outputs(BAD_REVIEW, FIXED, GOOD_REVIEW)
        result = model.refine_with_evidence(DRAFT, SOURCE, min_words=8, max_words=30)
        self.assertEqual(DRAFT, initial)
        self.assertEqual(result["summary"], SOURCE)
        self.assertEqual(result["summary_review"]["content_revisions"], 1)
        self.assertEqual(result["summary_review"]["initial_draft"], DRAFT)
        self.assertEqual([e["kind"] for e in result["summary_review"]["events"]],
                         ["verification", "revision", "verification"])
        for call in model.pipe.call_args_list:
            self.assertEqual([m["role"] for m in call.kwargs["text"]], ["system", "user"])
            self.assertFalse(call.kwargs["enable_thinking"])
            self.assertFalse(call.kwargs["generate_kwargs"]["do_sample"])
        validate_stored_review(result, SOURCE, "10-14", 8, 30)

    def test_passed_draft_does_not_need_a_revision_and_citations_are_remapped(self):
        model = model_with_outputs(GOOD_REVIEW)
        result = model.refine_with_evidence({**FIXED, "supporting_source_sentence_ids": [1]},
                                           SOURCE, min_words=8, max_words=30)
        self.assertEqual(model.pipe.call_count, 1)
        self.assertEqual(result["supporting_source_sentence_ids"], [1, 2, 3])
        self.assertEqual(result["summary_review"]["content_revisions"], 0)

    def test_malformed_review_retries_without_reusing_the_bad_answer(self):
        model = model_with_outputs({"passed": True}, GOOD_REVIEW)
        result = model.refine_with_evidence(FIXED, SOURCE, min_words=8, max_words=30)
        self.assertEqual(model.pipe.call_count, 2)
        self.assertIn("error", result["summary_review"]["events"][0])
        retry = model.pipe.call_args_list[1].kwargs["text"]
        self.assertEqual([m["role"] for m in retry], ["system", "user", "user"])
        self.assertNotIn('"passed": true', retry[-1]["content"][0]["text"])

    def test_revision_word_count_error_is_retried_with_measured_feedback(self):
        short = {"summary": "Ada worked.", "supporting_source_sentence_ids": [1]}
        model = model_with_outputs(BAD_REVIEW, short, FIXED, GOOD_REVIEW)
        result = model.refine_with_evidence(DRAFT, SOURCE, min_words=8, max_words=30)
        self.assertEqual(result["summary"], SOURCE)
        feedback = model.pipe.call_args_list[2].kwargs["text"][-1]["content"][0]["text"]
        self.assertIn("2 words; required 8-30", feedback)

    def test_unresolved_review_stops_at_the_content_revision_limit(self):
        model = model_with_outputs(BAD_REVIEW, DRAFT, BAD_REVIEW, DRAFT, BAD_REVIEW)
        with self.assertRaises(SummaryReviewError) as caught:
            model.refine_with_evidence(DRAFT, SOURCE, min_words=8, max_words=30)
        self.assertEqual(model.pipe.call_count, 5)
        self.assertIn("after 2 revisions", str(caught.exception))
        self.assertEqual(len(caught.exception.attempts), 5)

    def test_repeated_malformed_review_is_not_promoted_to_success(self):
        model = model_with_outputs({"passed": True}, {"passed": True})
        with self.assertRaises(SummaryReviewError) as caught:
            model.refine_with_evidence(FIXED, SOURCE, min_words=8, max_words=30)
        self.assertEqual(len(caught.exception.attempts), 2)

    def test_length_trimming_cannot_bypass_the_marburg_requirement(self):
        # The 16-word revision exceeds this synthetic 15-word limit, so its
        # final Marburg sentence is removed by the pre-existing prefix policy.
        trimmed_review = {"sentence_reviews": [row(1, 1), row(2, 2)], "issues": []}
        model = model_with_outputs(BAD_REVIEW, FIXED, trimmed_review)
        with self.assertRaises(SummaryReviewError) as caught:
            model.refine_with_evidence(DRAFT, SOURCE, min_words=8, max_words=15, max_revisions=1)
        self.assertIn("marburg_missing", str(caught.exception))
        self.assertEqual(model.pipe.call_count, 3)

    def test_zero_revision_mode_still_rejects_an_unresolved_review(self):
        model = model_with_outputs(BAD_REVIEW)
        with self.assertRaises(SummaryReviewError):
            model.refine_with_evidence(DRAFT, SOURCE, min_words=8, max_words=30, max_revisions=0)
        self.assertEqual(model.pipe.call_count, 1)

    def test_stale_summary_source_or_policy_and_missing_review_are_rejected(self):
        result = model_with_outputs(GOOD_REVIEW).refine_with_evidence(
            FIXED, SOURCE, min_words=8, max_words=30)
        for record, source, age, low in (
            ({**result, "summary": SOURCE + " She worked."}, SOURCE, "10-14", 8),
            (result, SOURCE + " She worked.", "10-14", 8),
            (result, SOURCE, "8-10", 8), (result, SOURCE, "10-14", 9),
            (FIXED, SOURCE, "10-14", 8),
            ({**result, "supporting_source_sentence_ids": [1]}, SOURCE, "10-14", 8),
        ):
            with self.assertRaises(ValueError):
                validate_stored_review(record, source, age, low, 30)

    def test_inference_failure_is_logged(self):
        generate = Mock(side_effect=RuntimeError("CUDA out of memory"))
        with self.assertRaises(SummaryReviewError) as caught:
            refine_biography(generate, FIXED, SOURCE, {}, min_words=8, max_words=30)
        self.assertIn("CUDA out of memory", caught.exception.attempts[0]["error"])


class SeparateRunTest(unittest.TestCase):
    def make_run(self, root):
        for directory in ("sources", "summaries", "generated_images", "pages"):
            (root / directory).mkdir(parents=True)
        source = {"title": "Ada", "summary": SOURCE, "revision_id": 123,
                  "source_policy_version": 2, "source_text_sha256": text_sha256(SOURCE),
                  "image_path": "/content/old/sources/images/Ada.jpg"}
        (root / "sources/Ada.json").write_text(json.dumps(source), encoding="utf-8")
        summary = {**DRAFT, "source_text_sha256": text_sha256(SOURCE), "source_revision_id": 123}
        (root / "summaries/Ada.json").write_text(json.dumps(summary), encoding="utf-8")
        Image.new("RGB", (512, 640), "white").save(root / "generated_images/Ada.png")
        (root / "coloring_book.pdf").write_bytes(b"frozen-book")
        manifest = {"schema_version": 1, "runtime": {"cuda_device": "Tesla T4"},
                    "book_path": "old/path/coloring_book.pdf", "items": [],
                    "configuration": {"names": ["Ada"], "qwen_model": "test-model",
                                      "qwen_revision": "test-revision", "qwen_quantization": "4bit",
                                      "target_age": "10-14", "summary_word_range": [8, 30]}}
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def snapshot(self, root):
        return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob("*") if p.is_file()}

    def test_read_only_preflight_has_no_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, derived = Path(tmp) / "frozen", Path(tmp) / "derived"
            self.make_run(source)
            before = self.snapshot(source)
            prepare_refinement(source, derived, check_only=True)
            self.assertEqual(self.snapshot(source), before)
            self.assertFalse(derived.exists())

    def test_separate_output_and_resume_preserve_frozen_files_without_network_or_flux(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, derived = Path(tmp) / "frozen", Path(tmp) / "derived"
            self.make_run(source)
            before = self.snapshot(source)
            prepare_refinement(source, derived)
            self.assertFalse((derived / "coloring_book.pdf").exists())
            self.assertEqual((derived / "original_summaries/Ada.json").read_bytes(),
                             (source / "summaries/Ada.json").read_bytes())
            model = model_with_outputs(BAD_REVIEW, FIXED, GOOD_REVIEW)
            pipe = model.pipe
            with patch("main.get_person_info", side_effect=AssertionError("network")), \
                 patch("main.image_generation_stage", side_effect=AssertionError("FLUX")), \
                 patch("Summarizer.Summarizer", return_value=model):
                result = run_pipeline(parse_args(["--repair-summaries", "--verify-summaries",
                                                  "--output-dir", str(derived)]))
            self.assertTrue(result["book_path"])
            self.assertEqual(pipe.call_count, 3)
            self.assertEqual(self.snapshot(source), before)
            self.assertEqual(result["configuration"]["qwen_quantization"], "4bit")
            prepare_refinement(source, derived)
            with patch("Summarizer.Summarizer", side_effect=AssertionError("reviewed cache")):
                resumed = run_pipeline(parse_args(["--repair-summaries", "--output-dir", str(derived)]))
            self.assertTrue(resumed["book_path"])
            self.assertEqual(self.snapshot(source), before)

    def test_failed_review_does_not_publish_an_old_or_unreviewed_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, derived = Path(tmp) / "frozen", Path(tmp) / "derived"
            self.make_run(source)
            prepare_refinement(source, derived)
            (derived / "coloring_book.pdf").write_bytes(b"old-derived-book")
            model = model_with_outputs(BAD_REVIEW, DRAFT, BAD_REVIEW, DRAFT, BAD_REVIEW)
            with patch("Summarizer.Summarizer", return_value=model):
                result = run_pipeline(parse_args(["--repair-summaries", "--verify-summaries",
                                                  "--output-dir", str(derived)]))
            self.assertIsNone(result["book_path"])
            self.assertEqual((derived / "coloring_book.pdf").read_bytes(), b"old-derived-book")
            failure = json.loads((derived / "summary_failures/Ada.json").read_text(encoding="utf-8"))
            self.assertEqual(len(failure["attempts"]), 5)
            self.assertTrue(failure["context"]["verify_summaries"])

    def test_same_nested_nonempty_or_changed_input_directories_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, derived = Path(tmp) / "frozen", Path(tmp) / "derived"
            self.make_run(source)
            for output in (source, source / "child", Path(tmp)):
                with self.assertRaises(ValueError):
                    prepare_refinement(source, output)
            occupied = Path(tmp) / "occupied"
            occupied.mkdir()
            (occupied / "note.txt").write_text("user file", encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare_refinement(source, occupied)
            prepare_refinement(source, derived)
            (source / "summaries/Ada.json").write_text(json.dumps(FIXED), encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare_refinement(source, derived)

    def test_mutated_derived_images_are_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, derived = Path(tmp) / "frozen", Path(tmp) / "derived"
            self.make_run(source)
            prepare_refinement(source, derived)
            (derived / "generated_images/Ada.png").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                prepare_refinement(source, derived)


if __name__ == "__main__":
    unittest.main()
