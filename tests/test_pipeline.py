import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from PIL import Image

from main import get_names, load_valid_summary, parse_args, run_configuration, run_pipeline, slugify
from source_text import text_sha256
from Summarizer import Summarizer
from summary_validation import fit_summary_length


class PipelineHelpersTest(unittest.TestCase):
    def test_model_self_review_is_not_part_of_the_public_api(self):
        args = parse_args([])
        self.assertFalse(hasattr(args, "verify_summaries"))
        self.assertFalse(hasattr(args, "max_review_revisions"))
        self.assertFalse(hasattr(args, "repair_summaries"))
        self.assertFalse(hasattr(args, "offline_repair"))
        self.assertFalse(hasattr(args, "refresh_source_text"))
        self.assertFalse(hasattr(Summarizer, "refine_with_evidence"))

    def test_default_subjects_use_the_single_current_eight_person_list(self):
        names = get_names(parse_args([]))
        self.assertEqual(len(names), 8)
        self.assertEqual(names[0], "Otto Hahn")
        self.assertEqual(names[5], "K. Ferdinand Braun")
        self.assertNotIn("Gertrud von Le Fort", names)

    def test_full_colab_notebook_has_valid_code_and_no_repair_cells(self):
        notebook_path = Path(__file__).resolve().parents[1] / "colab/AIColoringBook.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            transformed = "".join(
                " " * (len(line) - len(line.lstrip())) + "pass\n"
                if line.lstrip().startswith(("!", "%")) else line
                for line in source.splitlines(keepends=True)
            )
            ast.parse(transformed, filename=f"cell-{index}")
            for retired in ("REPAIR_SUMMARIES_ONLY", "REFRESH_SOURCE_TEXT", "--repair-summaries", "refine_biographies"):
                self.assertNotIn(retired, source)
        configuration = "".join(notebook["cells"][6]["source"])
        self.assertIn("evaluation/subjects.csv", configuration)
        self.assertIn("final_run_v2", configuration)
        self.assertEqual(parse_args([]).output_dir, "output/final_run_v2")

    def test_unknown_or_old_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "keep.txt").write_text("user result", encoding="utf-8")
            args = parse_args(["--output-dir", str(root)])
            with self.assertRaisesRegex(ValueError, "Nonempty"):
                run_pipeline(args)
            (root / "manifest.json").write_text('{"schema_version":1}', encoding="utf-8")
            before = {path.name: path.read_bytes() for path in root.iterdir()}
            with self.assertRaisesRegex(ValueError, "older code"):
                run_pipeline(args)
            self.assertEqual(before, {path.name: path.read_bytes() for path in root.iterdir()})

    def test_slugify_and_name_deduplication(self):
        args = parse_args(["--names", "Marie Curie", "marie curie", "Max Planck"])
        self.assertEqual(get_names(args), ["Marie Curie", "Max Planck"])
        self.assertEqual(slugify("Marie Curie"), "Marie_Curie")

    def test_grounded_summary_json_parser(self):
        summary, evidence = Summarizer._parse_json_response(
            '```json\n{"summary": "A short biography.", '
            '"supporting_source_sentence_ids": [2, 1, 2]}\n```'
        )
        self.assertEqual(summary, "A short biography.")
        self.assertEqual(evidence, [1, 2])

    def test_t4_safe_mode_applies_complete_preset(self):
        args = parse_args(
            [
                "--t4-safe-mode",
                "--qwen-quantization",
                "none",
                "--flux-quantization",
                "none",
                "--max-side",
                "1024",
                "--flux-offload",
            ]
        )
        self.assertEqual(args.qwen_quantization, "4bit")
        self.assertEqual(args.flux_quantization, "8bit")
        self.assertEqual(args.max_side, 640)
        self.assertEqual(args.max_sequence_length, 256)
        self.assertFalse(args.flux_offload)

    def test_cached_pipeline_builds_book_without_loading_models(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            for directory in [
                "sources/images",
                "summaries",
                "generated_images",
                "generation_metadata",
                "pages",
            ]:
                (run_dir / directory).mkdir(parents=True, exist_ok=True)

            source_image = run_dir / "sources/images/Test_Person.png"
            generated_image = run_dir / "generated_images/Test_Person.png"
            Image.new("RGB", (512, 640), "white").save(source_image)
            Image.new("RGB", (512, 640), "white").save(generated_image)

            source = {
                "query": "Test Person",
                "title": "Test Person",
                "summary": "Test Person made an important contribution to science.",
                "image_path": str(source_image),
                "page_url": "https://example.org/Test_Person",
                "revision_id": 123,
                "source_text_sha256": text_sha256("Test Person made an important contribution to science."),
                "source_policy_version": 2,
                "image_sha256": "image-hash",
                "image_artist": "Example Artist",
                "image_credit": None,
                "image_license": "Public domain",
            }
            (run_dir / "sources/Test_Person.json").write_text(
                json.dumps(source), encoding="utf-8"
            )
            generated_summary = {
                "title": "Test Person",
                "summary": "Test Person made an important contribution to science.",
                "supporting_source_sentence_ids": [1],
                "source_revision_id": 123,
                "source_text_sha256": source["source_text_sha256"],
            }
            (run_dir / "summaries/Test_Person.json").write_text(
                json.dumps(generated_summary), encoding="utf-8"
            )

            args = parse_args(
                [
                    "--names",
                    "Test Person",
                    "--output-dir",
                    str(run_dir),
                    "--skip-summarization",
                    "--skip-image-generation",
                    "--summary-min-words", "8",
                    "--summary-max-words", "20",
                ]
            )
            (run_dir / "manifest.json").write_text(json.dumps({
                "schema_version": 2,
                "configuration": run_configuration(args, get_names(args)),
            }), encoding="utf-8")
            manifest = run_pipeline(args)
            resumed = run_pipeline(args)

            self.assertTrue((run_dir / "coloring_book.pdf").exists())
            self.assertTrue((run_dir / "pages/Test_Person.pdf").exists())
            self.assertTrue((run_dir / "manifest.json").exists())
            self.assertEqual(manifest["items"][0]["errors"], [])
            self.assertEqual(resumed["items"][0]["errors"], [])
            self.assertFalse((run_dir / "backups").exists())
            before = (run_dir / "manifest.json").read_bytes()
            args.seed += 1
            with self.assertRaisesRegex(ValueError, "different settings"):
                run_pipeline(args)
            self.assertEqual((run_dir / "manifest.json").read_bytes(), before)
            for key in ("verify_summaries", "summary_review_version", "max_review_revisions"):
                self.assertNotIn(key, manifest["configuration"])


# The actual 115-word model output reported in the failed run; no new generation.
DRAFT = ("Baroness Gertrud von Le Fort (1876–1971) was a German writer born in Minden, the daughter of a Prussian colonel. "
         "She studied in Heidelberg, Marburg, and Berlin before making her home in Bavaria. "
         "Her career began with editing Ernst Troeltsch's posthumous work in 1925, followed by her conversion to Roman Catholicism. "
         "She published over 20 books, including the novella The Song at the Scaffold, which inspired Georges Bernanos' Dialogues des Carmélites. "
         "This work formed the basis for Francis Poulenc's 1956 opera. "
         "Le Fort won the Gottfried-Keller Prize in 1952 and was nominated by Hermann Hesse for the Nobel Prize. "
         "She died in Oberstdorf at age 95, leaving a legacy of depth and beauty in her writing.")
RECORD = {'summary': DRAFT, 'supporting_source_sentence_ids': [1]}


class LengthValidationTest(unittest.TestCase):
    def test_actual_115_word_answer_is_trimmed_only_at_sentence_boundary(self):
        result = fit_summary_length(RECORD, DRAFT, 60, 110)
        self.assertEqual(len(DRAFT.split()), 115)
        self.assertEqual(result['word_count'], 98)
        adjustment = result['length_adjustment']
        self.assertEqual(result['summary'] + adjustment['removed_tail'], DRAFT)
        self.assertTrue(result['summary'].endswith('Nobel Prize.'))
        self.assertIn('Marburg', result['summary'])
        self.assertEqual(adjustment['original_word_count'], 115)

    def test_invalid_evidence_short_and_wildly_overlong_answers_still_fail(self):
        for record in ({**RECORD, 'supporting_source_sentence_ids': [999]},
                       {**RECORD, 'summary': 'A short biography.'},
                       {**RECORD, 'summary': DRAFT * 5},
                       {**RECORD, 'word_count': 99}):
            with self.assertRaises(ValueError):
                fit_summary_length(record, DRAFT, 60, 110)

    def test_abbreviations_and_open_quotes_are_not_cut_points(self):
        for text in ('She studied with Dr. Robert Bunsen in Germany.',
                     'She called it "Important work. More discussion followed."'):
            with self.assertRaises(ValueError):
                fit_summary_length({'summary': text, 'supporting_source_sentence_ids': [1]}, text, 3, 7)

    def test_malformed_first_answer_then_115_words_succeeds_without_third_generation(self):
        model = Summarizer.__new__(Summarizer)
        model.model_name = 'test-model'
        model.revision = 'test-revision'
        model.quantization = 'none'
        model.model_load_seconds = 0
        model.pipe = Mock(side_effect=[[{'generated_text': 'not JSON'}],
                                       [{'generated_text': json.dumps(RECORD)}]])
        result = model.summarize_with_evidence(DRAFT, min_words=60, max_words=110)
        self.assertEqual(model.pipe.call_count, 2)
        self.assertEqual(result['word_count'], 98)
        self.assertEqual(json.loads(result['raw_model_response'])['summary'], DRAFT)


if __name__ == "__main__":
    unittest.main()
