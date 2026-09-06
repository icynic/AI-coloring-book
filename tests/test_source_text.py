import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests
from PIL import Image

from Fetcher import _fetch_article_text, _fetch_page_metadata, refresh_person_text
from main import load_valid_summary, ensure_run_directories, parse_args, run_pipeline
from source_text import select_article_text, text_sha256


HTML = '''<div class="mw-parser-output"><table class="infobox"><tr><td><p>Wrong table data.</p></td></tr></table>
<div class="hatnote"><p>Not biography.</p></div><p>Ada was a writer.<sup class="reference">[1]</sup></p>
<h2><span>Life</span><span class="mw-editsection">edit</span></h2>
<p>She studied at Marburg. She wrote novels and poems.</p>
<h3>Career</h3><p>She published many books. Her work received an award.</p>
<h2>References</h2><p>Exclude reference details.</p><h3>Books</h3><p>Exclude nested references.</p>
<h2>Legacy</h2><p>Readers remember her books.</p></div>'''


class SourceSelectionTest(unittest.TestCase):
    def test_body_headings_cleaning_and_budget(self):
        result = select_article_text(HTML)
        self.assertEqual(result['lead_summary'], 'Ada was a writer.')
        self.assertIn('Marburg', result['summary'])
        self.assertIn('published', result['summary'])
        self.assertIn('Readers remember', result['summary'])
        for excluded in ('Wrong table', 'Not biography', 'Exclude', '[1]', 'edit'):
            self.assertNotIn(excluded, result['summary'])
        self.assertIn('Life / Career', [p['section'] for p in result['source_passages']])
        self.assertEqual(result['source_text_sha256'], text_sha256(result['summary']))
        small = select_article_text(HTML, max_words=12)
        self.assertLessEqual(small['source_word_count'], 12)
        self.assertIn('Marburg', small['summary'])
        self.assertTrue(small['summary'].endswith('.'))
        self.assertEqual(select_article_text(HTML), result)

    def test_long_article_is_bounded_and_duplicate_sentences_removed(self):
        paragraphs = ''.join(f'<p>She published book number {i} about history and education.</p>' for i in range(300))
        result = select_article_text('<p>Ada was a writer.</p><h2>Career</h2>' + paragraphs + '<p>Ada was a writer.</p>')
        self.assertLessEqual(result['source_word_count'], 800)
        self.assertEqual(result['summary'].count('Ada was a writer.'), 1)
        with self.assertRaises(ValueError):
            select_article_text('<table><tr><td>No usable prose</td></tr></table>')

    def test_api_reads_exact_revision_and_rejects_mismatches(self):
        response = Mock()
        response.json.return_value = {'parse': {'pageid': 10, 'revid': 123, 'text': HTML}}
        with patch('Fetcher._get', return_value=response) as get:
            result = _fetch_article_text(Mock(), 123, 10)
            self.assertIn('Marburg', result['summary'])
            params = get.call_args.kwargs['params']
            self.assertEqual(params['oldid'], 123)
            self.assertNotIn('exintro', params)
            response.json.return_value['parse']['revid'] = 124
            with self.assertRaises(ValueError):
                _fetch_article_text(Mock(), 123, 10)

    def test_normal_fetch_also_uses_article_prose(self):
        response = Mock()
        response.json.return_value = {'query': {'pages': [{'pageid': 10, 'title': 'Ada',
            'revisions': [{'revid': 123}], 'fullurl': 'https://en.wikipedia.org/wiki/Ada'}]}}
        with patch('Fetcher._get', return_value=response) as get, \
             patch('Fetcher._fetch_article_text', return_value=select_article_text(HTML)) as article:
            result = _fetch_page_metadata(Mock(), 'Ada')
            self.assertIn('Marburg', result['summary'])
            self.assertEqual(article.call_args.args[1:], (123, 10))
            self.assertNotIn('extracts', get.call_args.kwargs['params']['prop'])

    def test_refresh_does_not_replace_image_metadata(self):
        original = {'revision_id': 123, 'page_id': 10, 'image_path': '/old/image.jpg',
                    'image_sha256': 'original-image', 'image_license': 'Public domain'}
        with patch('Fetcher._fetch_article_text', return_value=select_article_text(HTML)):
            result = refresh_person_text(original)
        for key, value in original.items():
            self.assertEqual(result[key], value)


class SourceRecoveryTest(unittest.TestCase):
    def make_run(self, root):
        (root / 'sources').mkdir()
        (root / 'summaries').mkdir()
        source = {'title': 'Ada', 'page_id': 10, 'revision_id': 123, 'summary': 'Ada was a writer.',
                  'image_path': '/content/old/image.jpg', 'image_sha256': 'original-image'}
        source['source_text_sha256'] = text_sha256(source['summary'])
        (root / 'sources/Ada.json').write_text(json.dumps(source), encoding='utf-8')
        summary = {'summary': 'Ada was a writer.', 'supporting_source_sentence_ids': [1],
                   'source_revision_id': 123}
        (root / 'summaries/Ada.json').write_text(json.dumps(summary), encoding='utf-8')
        (root / 'coloring_book.pdf').write_bytes(b'original-book')
        (root / 'image.png').write_bytes(b'original-image')
        manifest = {'configuration': {'names': ['Ada'], 'summary_word_range': [4, 20]},
                    'runtime': {'cuda_device': 'Tesla T4'}, 'book_path': 'old-book.pdf',
                    'items': [{'query': 'Ada', 'pdf_path': 'old-page.pdf', 'errors': []}]}
        (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        return source

    def snapshot(self, root):
        return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()}

    def test_refresh_invalidates_same_revision_summary_and_is_resumable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = self.make_run(root)
            before = self.snapshot(root)
            replacement = {**original, **select_article_text(HTML)}
            with patch('main.refresh_person_text', return_value=replacement) as fetch, \
                 patch('main.summarization_stage', side_effect=AssertionError('Qwen')), \
                 patch('main.image_generation_stage', side_effect=AssertionError('FLUX')):
                check = run_pipeline(parse_args(['--refresh-source-text', '--check-only', '--output-dir', tmp]))
                self.assertEqual(check['pending_source_refreshes'], ['Ada'])
                self.assertEqual(self.snapshot(root), before)
                self.assertFalse(fetch.called)
                result = run_pipeline(parse_args(['--refresh-source-text', '--output-dir', tmp]))
                self.assertEqual(result['errors'], [])
                run_pipeline(parse_args(['--refresh-source-text', '--output-dir', tmp]))
                self.assertEqual(fetch.call_count, 1)
            for path in ('image.png', 'coloring_book.pdf', 'summaries/Ada.json'):
                self.assertEqual((root / path).read_bytes(), before[path])
            manifest = json.loads((root / 'manifest.json').read_text())
            self.assertIsNone(manifest['book_path'])
            self.assertEqual(manifest['runtime']['cuda_device'], 'Tesla T4')
            backup_sources = list((root / 'backups').glob('*/sources/Ada.json'))
            self.assertEqual(backup_sources[0].read_bytes(), before['sources/Ada.json'])
            args = parse_args(['--summary-min-words', '4', '--summary-max-words', '20'])
            with self.assertRaisesRegex(ValueError, 'Source text changed'):
                load_valid_summary({'slug': 'Ada', 'source': replacement}, ensure_run_directories(root, False), args)

    def test_failed_refresh_keeps_source_and_does_not_run_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_run(root)
            before = self.snapshot(root)
            with patch('main.refresh_person_text', side_effect=requests.ConnectionError('offline')), \
                 patch('main.repair_summaries', side_effect=AssertionError('must not load Qwen')):
                with self.assertRaises(SystemExit):
                    run_pipeline(parse_args(['--refresh-source-text', '--repair-summaries', '--output-dir', tmp]))
            for path in ('sources/Ada.json', 'summaries/Ada.json', 'coloring_book.pdf', 'image.png'):
                self.assertEqual((root / path).read_bytes(), before[path])
            report = json.loads((root / 'source_refresh_manifest.json').read_text())
            self.assertIn('offline', report['errors'][0])

    def test_combined_refresh_repair_records_new_hash_and_skips_flux(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = self.make_run(root)
            (root / 'generated_images').mkdir()
            image = root / 'generated_images/Ada.png'
            Image.new('RGB', (512, 640), 'white').save(image)
            image_before = image.read_bytes()
            replacement = {**original, **select_article_text(HTML)}
            with patch('main.refresh_person_text', return_value=replacement), \
                 patch('main.get_person_info', side_effect=AssertionError('image retrieval')), \
                 patch('main.image_generation_stage', side_effect=AssertionError('FLUX')), \
                 patch('Summarizer.Summarizer') as factory:
                factory.return_value.summarize_with_evidence.return_value = {
                    'summary': 'Ada was a writer.', 'supporting_source_sentence_ids': [1]}
                result = run_pipeline(parse_args(['--refresh-source-text', '--repair-summaries', '--output-dir', tmp]))
            self.assertTrue(result['book_path'])
            self.assertEqual(result['items'][0]['errors'], [])
            saved = json.loads((root / 'summaries/Ada.json').read_text())
            self.assertEqual(saved['source_text_sha256'], replacement['source_text_sha256'])
            self.assertEqual(saved['source_policy_version'], 2)
            self.assertEqual(image.read_bytes(), image_before)
            with patch('main.refresh_person_text', side_effect=AssertionError('source already upgraded')), \
                 patch('Summarizer.Summarizer', side_effect=AssertionError('summary already repaired')):
                resumed = run_pipeline(parse_args(['--refresh-source-text', '--repair-summaries', '--output-dir', tmp]))
            self.assertTrue(resumed['book_path'])

    def test_rate_limit_defers_remaining_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_run(root)
            (root / 'sources/Bob.json').write_text(json.dumps({**source, 'title': 'Bob'}), encoding='utf-8')
            manifest = json.loads((root / 'manifest.json').read_text())
            manifest['configuration']['names'].append('Bob')
            (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            error = requests.HTTPError('429 rate limit', response=Mock(status_code=429))
            with patch('main.refresh_person_text', side_effect=error) as fetch:
                result = run_pipeline(parse_args(['--refresh-source-text', '--output-dir', tmp]))
            self.assertEqual(fetch.call_count, 1)
            self.assertEqual(len(result['errors']), 2)
            self.assertIn('Deferred', result['errors'][1])

    def test_truly_sparse_article_fails_before_loading_qwen(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = self.make_run(root)
            (root / 'generated_images').mkdir()
            Image.new('RGB', (512, 640), 'white').save(root / 'generated_images/Ada.png')
            replacement = {**original, **select_article_text('<p>Ada was a writer.</p>')}
            with patch('main.refresh_person_text', return_value=replacement), \
                 patch('Summarizer.Summarizer', side_effect=AssertionError('insufficient evidence')):
                result = run_pipeline(parse_args(['--refresh-source-text', '--repair-summaries',
                                                  '--output-dir', tmp, '--summary-min-words', '80',
                                                  '--summary-max-words', '110']))
            self.assertIsNone(result['book_path'])
            self.assertIn('Source has only 4 words', result['items'][0]['errors'][0])


if __name__ == '__main__':
    unittest.main()
