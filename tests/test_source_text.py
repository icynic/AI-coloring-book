import unittest
from unittest.mock import Mock, patch

from Fetcher import _fetch_article_text, _fetch_page_metadata
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


if __name__ == "__main__":
    unittest.main()
