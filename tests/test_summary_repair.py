import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from Concatenator import Concatenator
from main import parse_args, resolve_summary_word_range, run_pipeline
from Summarizer import Summarizer, SummaryGenerationError
from summary_validation import parse_response, validate_summary


SOURCE = "Ada was a scientist. She worked on mathematical problems."
BIOGRAPHY = "Ada was a scientist who worked on mathematical problems."
VALID = {"summary": BIOGRAPHY, "supporting_source_sentence_ids": [1, 2]}


class SummaryValidationTest(unittest.TestCase):
    def test_reasoning_examples_and_invalid_types_are_rejected(self):
        bad = [
            'Thinking Process: Schema {"summary":"...","supporting_source_sentence_ids":[1,2]}',
            '<think>Schema {"summary":"...","supporting_source_sentence_ids":[1,2]}',
            '{"summary":"...","supporting_source_sentence_ids":[1,2]}',
            '{"summary":null,"supporting_source_sentence_ids":[1]}',
            '{"summary":["biography"],"supporting_source_sentence_ids":[1]}',
            json.dumps(VALID) + ' trailing draft',
            json.dumps(VALID) + json.dumps(VALID),
            '{"summary":"Ada worked.","supporting_source_sentence_ids":[true]}',
        ]
        for text in bad:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_response(text)

    def test_final_json_after_closed_thinking_block_and_fences(self):
        raw = '<think>Ignore a schema {"summary":"..."}</think>\n```json\n' + json.dumps(VALID) + '\n```'
        self.assertEqual(parse_response(raw), (BIOGRAPHY, [1, 2]))

    def test_word_count_and_evidence_are_actually_validated(self):
        self.assertEqual(validate_summary(VALID, SOURCE, 8, 20), 9)
        for record, lo, hi in [
            (VALID, 80, 110),
            ({**VALID, "supporting_source_sentence_ids": [3]}, 8, 20),
            ({**VALID, "word_count": 200}, 8, 20),
            ({**VALID, "supporting_source_sentences": ["Wrong source"]}, 8, 20),
        ]:
            with self.assertRaises(ValueError):
                validate_summary(record, SOURCE, lo, hi)

    def test_short_biography_requires_an_explicitly_wider_range(self):
        # Synthetic word-count fixture, not generated/evaluation content.
        record = {**VALID, "summary": " ".join(["word"] * 73), "word_count": 73}
        with self.assertRaises(ValueError):
            validate_summary(record, SOURCE)
        self.assertEqual(validate_summary(record, SOURCE, 60, 110), 73)
        with self.assertRaises(ValueError):
            validate_summary({**record, "supporting_source_sentence_ids": [999]}, SOURCE, 60, 110)

    def test_length_retry_revises_the_draft_with_measured_feedback(self):
        summarizer = Summarizer.__new__(Summarizer)
        summarizer.model_name = "test-model"
        summarizer.revision = "test-revision"
        summarizer.quantization = "none"
        summarizer.model_load_seconds = 0
        longer = {**VALID, "summary": BIOGRAPHY + " Her work involved mathematics."}
        summarizer.pipe = Mock(side_effect=[
            [{"generated_text": json.dumps(VALID)}],
            [{"generated_text": json.dumps(longer)}],
        ])
        result = summarizer.summarize_with_evidence(SOURCE, min_words=10, max_words=20)
        self.assertEqual(result["summary"], longer["summary"])
        calls = summarizer.pipe.call_args_list
        self.assertEqual(len(calls[0].kwargs["text"]), 2)
        messages = calls[1].kwargs["text"]
        self.assertEqual(messages[-2]["role"], "assistant")
        self.assertEqual(json.loads(messages[-2]["content"][0]["text"]), VALID)
        self.assertIn("9 words", messages[-1]["content"][0]["text"])
        self.assertIn("at least 1 more words", messages[-1]["content"][0]["text"])
        self.assertIn("Aim for 15", messages[-1]["content"][0]["text"])

    def test_generation_retries_and_correct_pipeline_parameter_routing(self):
        summarizer = Summarizer.__new__(Summarizer)
        summarizer.model_name = "test-model"
        summarizer.revision = "test-revision"
        summarizer.quantization = "4bit"
        summarizer.model_load_seconds = 0
        summarizer.pipe = Mock(side_effect=[
            [{"generated_text": '{"summary":"...","supporting_source_sentence_ids":[1]}'}],
            [{"generated_text": json.dumps(VALID)}],
        ])
        result = summarizer.summarize_with_evidence(SOURCE, min_words=8, max_words=20)
        self.assertEqual(result["summary"], BIOGRAPHY)
        self.assertEqual(len(result["generation_attempts"]), 2)
        for index, call in enumerate(summarizer.pipe.call_args_list):
            self.assertFalse(call.kwargs["enable_thinking"])
            self.assertFalse(call.kwargs["return_full_text"])
            self.assertEqual(call.kwargs["generate_kwargs"],
                             {"max_new_tokens": 1024 * (index + 1), "do_sample": False})
        summarizer.pipe = Mock(return_value=[{"generated_text": "truncated draft"}])
        with self.assertRaises(SummaryGenerationError) as caught:
            summarizer.summarize_with_evidence(SOURCE, min_words=8, max_words=20)
        self.assertEqual(len(caught.exception.attempts), 2)

    def test_installed_transformers_routes_thinking_to_template_not_generation(self):
        from transformers.pipelines.image_text_to_text import ImageTextToTextPipeline, ReturnType

        pipe = ImageTextToTextPipeline.__new__(ImageTextToTextPipeline)
        preprocess, forward, postprocess = pipe._sanitize_parameters(
            enable_thinking=False, return_full_text=False,
            generate_kwargs={"max_new_tokens": 1024, "do_sample": False},
        )
        self.assertEqual(preprocess, {"enable_thinking": False})
        self.assertEqual(forward, {"generate_kwargs": {"max_new_tokens": 1024, "do_sample": False}})
        self.assertEqual(postprocess["return_type"], ReturnType.NEW_TEXT)


class RepairRunTest(unittest.TestCase):
    def make_run(self, root):
        for directory in ("sources", "summaries", "generated_images", "pages"):
            (root / directory).mkdir()
        source = {"query": "Ada", "title": "Ada", "summary": SOURCE,
                  "image_path": "/content/old/path/Ada.jpg", "revision_id": 123}
        (root / "sources/Ada.json").write_text(json.dumps(source), encoding="utf-8")
        Image.new("RGB", (512, 640), "white").save(root / "generated_images/Ada.png")
        (root / "summaries/Ada.json").write_text(
            '{"summary":"...","word_count":1,"supporting_source_sentence_ids":[1,2]}', encoding="utf-8")
        (root / "coloring_book.pdf").write_bytes(b"old-placeholder-pdf")
        (root / "pages/Ada.pdf").write_bytes(b"old-page")
        manifest = {"runtime": {"cuda_device": "Tesla T4"}, "started_at": "original-time",
                    "configuration": {"names": ["Ada"], "qwen_quantization": "4bit",
                                      "summary_word_range": [8, 20]}, "items": []}
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    def snapshot(self, root):
        return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob("*") if p.is_file()}

    def test_check_only_does_not_write_or_load_models_or_use_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_run(root)
            before = self.snapshot(root)
            with patch("main.get_person_info", side_effect=AssertionError("network")), \
                 patch("Summarizer.Summarizer", side_effect=AssertionError("model loaded")), \
                 patch("main.image_generation_stage", side_effect=AssertionError("FLUX")):
                result = run_pipeline(parse_args(["--repair-summaries", "--check-only", "--output-dir", tmp]))
            self.assertEqual(result["invalid_summaries"], ["Ada"])
            self.assertEqual(self.snapshot(root), before)

    def test_word_range_defaults_and_partial_overrides(self):
        args = parse_args([])
        self.assertEqual([args.summary_min_words, args.summary_max_words], [80, 110])
        for flags, expected in [([], [80, 110]),
                                (["--summary-min-words", "60"], [60, 110]),
                                (["--summary-max-words", "120"], [80, 120])]:
            args = parse_args(["--repair-summaries", *flags])
            self.assertEqual(resolve_summary_word_range(args, [80, 110]), expected)

    def test_range_override_is_logged_persisted_and_reuses_valid_cached_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = self.make_run(root)
            original["configuration"]["summary_word_range"] = [10, 20]
            (root / "manifest.json").write_text(json.dumps(original), encoding="utf-8")
            (root / "summaries/Ada.json").write_text(json.dumps(VALID), encoding="utf-8")
            before = self.snapshot(root)
            command = ["--repair-summaries", "--output-dir", tmp, "--summary-min-words", "8"]
            with patch("main.get_person_info", side_effect=AssertionError("network")), \
                 patch("main.image_generation_stage", side_effect=AssertionError("FLUX")), \
                 patch("Summarizer.Summarizer", side_effect=AssertionError("valid cached biography")):
                check = run_pipeline(parse_args(command + ["--check-only"]))
                self.assertEqual(check["invalid_summaries"], [])
                self.assertEqual(self.snapshot(root), before)
                result = run_pipeline(parse_args(command))
                resumed = run_pipeline(parse_args(["--repair-summaries", "--output-dir", tmp]))
            self.assertTrue(result["book_path"])
            self.assertTrue(resumed["book_path"])
            self.assertEqual(result["configuration"]["summary_word_range"], [8, 20])
            repair = result["summary_repairs"][-1]
            self.assertEqual(repair["previous_summary_word_range"], [10, 20])
            self.assertEqual(repair["summary_word_range"], [8, 20])
            self.assertEqual(result["runtime"], original["runtime"])
            for path in ("sources/Ada.json", "generated_images/Ada.png", "summaries/Ada.json"):
                self.assertEqual(self.snapshot(root)[path], before[path])

    def test_invalid_override_is_rejected_before_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_run(root)
            before = self.snapshot(root)
            for minimum in ("0", "30"):
                with self.assertRaises(ValueError):
                    run_pipeline(parse_args(["--repair-summaries", "--output-dir", tmp,
                                             "--summary-min-words", minimum]))
            self.assertEqual(self.snapshot(root), before)

    def test_repair_preserves_images_source_provenance_and_original_backups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = self.make_run(root)
            before = self.snapshot(root)
            with patch("main.get_person_info", side_effect=AssertionError("network")), \
                 patch("main.image_generation_stage", side_effect=AssertionError("FLUX")), \
                 patch("Summarizer.Summarizer") as factory:
                factory.return_value.summarize_with_evidence.return_value = copy.deepcopy(VALID)
                result = run_pipeline(parse_args(["--repair-summaries", "--output-dir", tmp]))
            self.assertEqual(result["runtime"], original["runtime"])
            self.assertEqual(result["configuration"], original["configuration"])
            self.assertEqual(result["items"][0]["errors"], [])
            self.assertEqual(json.loads((root / "summaries/Ada.json").read_text())["summary"], BIOGRAPHY)
            after = self.snapshot(root)
            for path in ("sources/Ada.json", "generated_images/Ada.png"):
                self.assertEqual(after[path], before[path])
            backup = next((root / "backups").iterdir())
            self.assertEqual((backup / "coloring_book.pdf").read_bytes(), b"old-placeholder-pdf")
            self.assertIn('"summary":"..."', (backup / "summaries/Ada.json").read_text())
            self.assertTrue((root / "coloring_book.pdf").read_bytes().startswith(b"%PDF"))
            with patch("main.get_person_info", side_effect=AssertionError("network")), \
                 patch("main.image_generation_stage", side_effect=AssertionError("FLUX")), \
                 patch("Summarizer.Summarizer", side_effect=AssertionError("valid cache must be reused")):
                resumed = run_pipeline(parse_args(["--repair-summaries", "--output-dir", tmp]))
            self.assertTrue(resumed["book_path"])
            self.assertEqual(resumed["items"][0]["errors"], [])

    def test_failed_generation_cannot_fall_back_to_bad_cache_or_claim_pdf_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_run(root)
            with patch("Summarizer.Summarizer") as factory:
                factory.return_value.summarize_with_evidence.side_effect = SummaryGenerationError(
                    "still invalid", [{"error": "placeholder"}])
                result = run_pipeline(parse_args(["--repair-summaries", "--output-dir", tmp]))
            self.assertIsNone(result["book_path"])
            self.assertTrue(result["items"][0]["errors"])
            self.assertEqual((root / "coloring_book.pdf").read_bytes(), b"old-placeholder-pdf")
            self.assertTrue((root / "summary_failures/Ada.json").is_file())

    def test_pdf_rejects_placeholder_and_overflow_without_damaging_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.png"
            Image.new("RGB", (512, 640), "white").save(image)
            output = root / "book.pdf"
            output.write_bytes(b"original")
            for text in ("...", "long biography " * 500):
                self.assertFalse(Concatenator().create_page(image, text, output))
                self.assertEqual(output.read_bytes(), b"original")


if __name__ == "__main__":
    unittest.main()
