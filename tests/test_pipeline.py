import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import torch
from transformers import GenerationConfig, GPT2Config, GPT2LMHeadModel
from transformers.pipelines.image_text_to_text import ImageTextToTextPipeline

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
        pipeline = "".join(notebook["cells"][7]["source"])
        self.assertIn("evaluation/subjects.csv", configuration)
        self.assertIn("final_run_v4", configuration)
        self.assertIn("SUMMARY_TARGET_WORDS = 95", configuration)
        self.assertIn("'--summary-target-words', str(SUMMARY_TARGET_WORDS)", pipeline)
        self.assertEqual(parse_args([]).output_dir, "output/final_run_v2")
        self.assertEqual(parse_args([]).summary_target_words, 95)

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
                    "--summary-target-words", "12",
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

    def test_missing_source_image_stops_before_qwen(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            source_text = " ".join(["biographical"] * 100)
            source = {
                "title": "Test Person",
                "summary": source_text,
                "image_path": None,
                "revision_id": 123,
                "source_text_sha256": text_sha256(source_text),
                "source_policy_version": 2,
            }
            args = parse_args([
                "--names", "Test Person", "Other Person",
                "--output-dir", str(run_dir),
            ])

            with patch("main.get_person_info", return_value=source), \
                    patch("Summarizer.Summarizer") as qwen, \
                    patch("main.image_generation_stage") as flux_stage, \
                    patch("main.pdf_stage") as pdf_stage:
                with self.assertRaisesRegex(
                    RuntimeError, r"\[fetch\].*stopping before loading Qwen"
                ):
                    run_pipeline(args)

            qwen.assert_not_called()
            flux_stage.assert_not_called()
            pdf_stage.assert_not_called()
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertIsNone(manifest["book_path"])
            self.assertIn("No source image was retrieved.", manifest["items"][0]["errors"])

    def test_failed_summary_stops_before_flux(self):
        from Summarizer import SummaryGenerationError

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_dir = root / "run"
            source_image = root / "input.png"
            Image.new("RGB", (64, 64), "white").save(source_image)
            source_text = " ".join(["biographical"] * 100)
            source = {
                "title": "Test Person",
                "summary": source_text,
                "image_path": str(source_image),
                "image_sha256": "image-hash",
                "revision_id": 123,
                "source_text_sha256": text_sha256(source_text),
                "source_policy_version": 2,
            }
            args = parse_args([
                "--names", "Test Person", "Other Person",
                "--output-dir", str(run_dir),
            ])
            qwen = Mock()
            qwen.summarize_with_evidence.side_effect = SummaryGenerationError(
                "No valid biography after 3 attempts", []
            )

            other_source = {**source, "title": "Other Person", "revision_id": 456}
            with patch("main.get_person_info", side_effect=[source, other_source]), \
                    patch("Summarizer.Summarizer", return_value=qwen), \
                    patch("main.image_generation_stage") as flux_stage, \
                    patch("main.pdf_stage") as pdf_stage:
                with self.assertRaisesRegex(
                    RuntimeError, r"\[summarize\].*stopping before loading FLUX"
                ):
                    run_pipeline(args)

            qwen.cleanup.assert_called_once_with()
            qwen.summarize_with_evidence.assert_called_once()
            flux_stage.assert_not_called()
            pdf_stage.assert_not_called()
            self.assertTrue((run_dir / "summary_failures/Test_Person.json").is_file())
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertIsNone(manifest["book_path"])
            self.assertIn("Summarization failed", manifest["items"][0]["errors"][0])
            self.assertIn("not attempted", manifest["items"][1]["errors"][0])


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
    def make_summarizer(self, answers):
        model = Summarizer.__new__(Summarizer)
        model.model_name = 'test-model'
        model.revision = 'test-revision'
        model.quantization = 'none'
        model.model_load_seconds = 0
        model.pipe = Mock(side_effect=[[{'generated_text': answer}] for answer in answers])
        model.pipe.generation_config = GenerationConfig(
            max_length=20, do_sample=True, temperature=0.7, top_p=0.8,
            top_k=20, eos_token_id=[248044, 248046], pad_token_id=None,
        )
        return model

    def test_generation_config_is_independent_and_contains_all_overrides(self):
        model = self.make_summarizer([])
        base = model.pipe.generation_config
        config = model._generation_config(1024)
        self.assertIsNot(config, base)
        self.assertEqual(base.max_length, 20)
        self.assertTrue(base.do_sample)
        self.assertIsNone(base.pad_token_id)
        self.assertIsNone(config.max_length)
        self.assertEqual(config.max_new_tokens, 1024)
        self.assertFalse(config.do_sample)
        self.assertEqual(config.pad_token_id, 248044)
        self.assertEqual(config.eos_token_id, [248044, 248046])
        self.assertEqual((config.temperature, config.top_p, config.top_k), (1.0, 1.0, 50))
        base.pad_token_id = 123
        self.assertEqual(model._generation_config(2048).pad_token_id, 123)

    def test_real_transformers_forward_accepts_config_without_generation_warnings(self):
        # Tiny random CPU model: checks the installed generation API, not Qwen quality.
        model = Summarizer.__new__(Summarizer)
        language_model = GPT2LMHeadModel(GPT2Config(
            vocab_size=16, n_positions=32, n_embd=16, n_layer=1, n_head=2,
            bos_token_id=1, eos_token_id=2, pad_token_id=2,
        )).eval()
        model.pipe = SimpleNamespace(
            model=language_model,
            generation_config=GenerationConfig(max_length=20, eos_token_id=2),
        )
        config = model._generation_config(3)
        inputs = {'text': 'test prompt', 'input_ids': torch.tensor([[1, 3]]),
                  'attention_mask': torch.ones((1, 2), dtype=torch.long)}
        with self.assertNoLogs('transformers.generation.utils', level='WARNING'):
            result = ImageTextToTextPipeline._forward(
                model.pipe, inputs, generate_kwargs={'generation_config': config},
            )
        self.assertEqual(result['generated_sequence'].shape[0], 1)

    def test_malformed_json_then_160_words_gets_a_bounded_length_correction(self):
        overlong = ' '.join(['detail'] * 159 + ['End.'])
        accepted = fit_summary_length(RECORD, DRAFT, 80, 110)['summary']
        model = self.make_summarizer([
            'Here is the biography: not JSON',
            json.dumps({'summary': overlong, 'supporting_source_sentence_ids': [1]}),
            json.dumps({'summary': accepted, 'supporting_source_sentence_ids': [1]}),
        ])
        result = model.summarize_with_evidence(DRAFT)
        self.assertEqual(model.pipe.call_count, 3)
        self.assertEqual(result['word_count'], 98)
        self.assertEqual(result['prompt_version'], 3)
        self.assertEqual(result['prompt_constraints'], {
            'target_words': 95, 'sentence_count': 5, 'words_per_sentence': [18, 20],
        })
        calls = model.pipe.call_args_list
        self.assertEqual([call.kwargs['generate_kwargs']['generation_config'].max_new_tokens
                          for call in calls], [1024, 2048, 1024])
        for call in calls:
            self.assertEqual(set(call.kwargs['generate_kwargs']), {'generation_config'})
            self.assertFalse(call.kwargs['enable_thinking'])
        first_prompt = calls[0].kwargs['text'][1]['content'][0]['text']
        self.assertIn('first character must be {', first_prompt)
        self.assertIn('5 short sentences', first_prompt)
        final_feedback = calls[2].kwargs['text'][-1]['content'][0]['text']
        self.assertIn('160 words, 50 over the hard limit', final_feedback)
        self.assertIn('Marburg connection', final_feedback)
        self.assertIn('supported', first_prompt)

    def test_persistent_bad_answers_fail_after_three_attempts(self):
        from Summarizer import SummaryGenerationError

        model = self.make_summarizer(['not JSON'] * 3)
        with self.assertRaises(SummaryGenerationError) as failure:
            model.summarize_with_evidence(DRAFT)
        self.assertEqual(model.pipe.call_count, 3)
        self.assertEqual(len(failure.exception.attempts), 3)

    def test_soft_target_is_independent_from_the_accepted_minimum(self):
        accepted = fit_summary_length(RECORD, DRAFT, 50, 110)['summary']
        model = self.make_summarizer([
            json.dumps({'summary': accepted, 'supporting_source_sentence_ids': [1]}),
        ])
        result = model.summarize_with_evidence(
            DRAFT, min_words=50, max_words=110, target_words=95,
        )
        self.assertEqual(result['prompt_constraints'], {
            'target_words': 95, 'sentence_count': 5, 'words_per_sentence': [18, 20],
        })
        prompt = model.pipe.call_args.kwargs['text'][1]['content'][0]['text']
        self.assertIn('50-110 whitespace-separated words', prompt)
        self.assertIn('Aim for 95 words, using 5 short sentences', prompt)

    def test_repeated_short_draft_gets_fresh_rewrite_with_full_source(self):
        short = 'Jacob Grimm ' + ' '.join(['detail'] * 44) + ' End.'
        accepted = fit_summary_length(RECORD, DRAFT, 50, 110)['summary']
        short_json = json.dumps({
            'summary': short,
            'supporting_source_sentence_ids': [1],
        })
        model = self.make_summarizer([
            short_json,
            short_json,
            json.dumps({'summary': accepted, 'supporting_source_sentence_ids': [1]}),
        ])

        result = model.summarize_with_evidence(
            DRAFT, min_words=50, max_words=110, target_words=95,
        )

        self.assertEqual(result['word_count'], 98)
        calls = model.pipe.call_args_list
        self.assertEqual([message['role'] for message in calls[1].kwargs['text']],
                         ['system', 'user', 'assistant', 'user'])
        self.assertEqual([message['role'] for message in calls[2].kwargs['text']],
                         ['system', 'user', 'user'])
        for call in calls:
            original_user_prompt = call.kwargs['text'][1]['content'][0]['text']
            self.assertIn('SOURCE:', original_user_prompt)
            self.assertIn('Baroness Gertrud von Le Fort', original_user_prompt)
            self.assertIn('END SOURCE', original_user_prompt)
        fresh_feedback = calls[2].kwargs['text'][-1]['content'][0]['text']
        self.assertIn('FRESH REWRITE REQUIRED', fresh_feedback)
        self.assertIn('from scratch', fresh_feedback)
        self.assertIn('aim for 95 words', fresh_feedback)
        self.assertFalse(result['generation_attempts'][0]['repeated_invalid_output'])
        self.assertTrue(result['generation_attempts'][1]['repeated_invalid_output'])

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
