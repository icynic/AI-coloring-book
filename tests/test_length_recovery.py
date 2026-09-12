import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from main import parse_args, run_pipeline
from source_text import text_sha256
from Summarizer import Summarizer
from summary_recovery import recover_failed_summary
from summary_validation import fit_summary_length


# The actual 115-word model output reported in the failed run; no new generation.
DRAFT = ("Baroness Gertrud von Le Fort (1876–1971) was a German writer born in Minden, the daughter of a Prussian colonel. "
         "She studied in Heidelberg, Marburg, and Berlin before making her home in Bavaria. "
         "Her career began with editing Ernst Troeltsch's posthumous work in 1925, followed by her conversion to Roman Catholicism. "
         "She published over 20 books, including the novella The Song at the Scaffold, which inspired Georges Bernanos' Dialogues des Carmélites. "
         "This work formed the basis for Francis Poulenc's 1956 opera. "
         "Le Fort won the Gottfried-Keller Prize in 1952 and was nominated by Hermann Hesse for the Nobel Prize. "
         "She died in Oberstdorf at age 95, leaving a legacy of depth and beauty in her writing.")
RECORD = {'summary': DRAFT, 'supporting_source_sentence_ids': [1]}


class LengthRecoveryTest(unittest.TestCase):
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

    def make_run(self, root, with_context=False):
        (root / 'summary_failures').mkdir()
        (root / 'sources').mkdir()
        (root / 'summaries').mkdir()
        (root / 'generated_images').mkdir()
        source = {'summary': DRAFT, 'revision_id': 123, 'title': 'Gertrud',
                  'source_policy_version': 2, 'source_text_sha256': text_sha256(DRAFT)}
        record = {'slug': 'Gertrud', 'query': 'Gertrud', 'source': source}
        failure = {'created_at': '2026-09-06T10:01:00+00:00', 'error': '115 words',
                   'attempts': [{'raw_model_response': json.dumps(RECORD), 'max_new_tokens': 2048}]}
        config = {'names': ['Gertrud'], 'qwen_model': 'test-model', 'qwen_revision': 'test-revision',
                  'qwen_quantization': '4bit', 'target_age': '10-14', 'summary_word_range': [60, 110]}
        if with_context:
            failure['context'] = {'query': 'Gertrud', 'source_revision_id': 123,
                                  'source_text_sha256': text_sha256(DRAFT), 'model_id': 'test-model',
                                  'model_revision': 'test-revision', 'quantization': '4bit',
                                  'target_age': '10-14', 'requested_word_range': [60, 110]}
        repair = {'started_at': '2026-09-06T10:00:00+00:00', 'completed_at': '2026-09-06T10:02:00+00:00',
                  'summary_word_range': [60, 110], 'items': [{'slug': 'Gertrud', 'query': 'Gertrud',
                  'source_revision_id': 123, 'source_text_sha256': text_sha256(DRAFT), 'errors': ['115 words']}]}
        for path, value in [('summary_failures/Gertrud.json', failure), ('sources/Gertrud.json', source),
                            ('repair_manifest.json', repair), ('manifest.json', {'configuration': config})]:
            (root / path).write_text(json.dumps(value), encoding='utf-8')
        return record, failure, repair

    def test_legacy_run_interval_and_direct_context_recovery(self):
        for with_context in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                record, _, _ = self.make_run(root, with_context)
                result = recover_failed_summary(record, root, 60, 110)
                self.assertEqual(result['word_count'], 98)
                self.assertEqual(result['source_text_sha256'], text_sha256(DRAFT))
                for key, value in [('summary', DRAFT + ' Changed source.'), ('revision_id', 124)]:
                    changed = copy.deepcopy(record)
                    changed['source'][key] = value
                    with self.assertRaises(ValueError):
                        recover_failed_summary(changed, root, 60, 110)

    def test_stale_legacy_log_is_not_promoted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record, failure, _ = self.make_run(root)
            failure['created_at'] = '2026-09-05T10:00:00+00:00'
            (root / 'summary_failures/Gertrud.json').write_text(json.dumps(failure), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'time interval'):
                recover_failed_summary(record, root, 60, 110)

    def test_offline_pipeline_cannot_load_any_model_or_use_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_run(root)
            # PDF is mocked here; only the recovery path and validation are under test.
            (root / 'generated_images/Gertrud.png').write_bytes(b'unchanged-image')
            with patch('Summarizer.Summarizer', side_effect=AssertionError('Qwen')), \
                 patch('main.image_generation_stage', side_effect=AssertionError('FLUX')), \
                 patch('main.get_person_info', side_effect=AssertionError('network')), \
                 patch('main.pdf_stage', return_value=root / 'coloring_book.pdf'):
                result = run_pipeline(parse_args(['--repair-summaries', '--offline-repair', '--output-dir', tmp]))
                self.assertEqual(result['items'][0]['errors'], [])
                saved = json.loads((root / 'summaries/Gertrud.json').read_text(encoding='utf-8'))
                self.assertEqual(saved['word_count'], 98)
                # A stale log must fail offline rather than silently starting Qwen.
                (root / 'summaries/Gertrud.json').unlink()
                result = run_pipeline(parse_args(['--repair-summaries', '--offline-repair', '--output-dir', tmp]))
                self.assertTrue(result['items'][0]['errors'])


if __name__ == '__main__':
    unittest.main()
