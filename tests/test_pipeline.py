import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from main import get_names, parse_args, run_pipeline, slugify
from Summarizer import Summarizer


class PipelineHelpersTest(unittest.TestCase):
    def test_slugify_and_name_deduplication(self):
        args = parse_args(["--names", "Marie Curie", "marie curie", "Max Planck"])
        self.assertEqual(get_names(args), ["Marie Curie", "Max Planck"])
        self.assertEqual(slugify("Marie Curie"), "Marie_Curie")

    def test_grounded_summary_json_parser(self):
        summary, evidence = Summarizer._parse_json_response(
            'prefix {"summary": "A short biography.", '
            '"supporting_source_sentence_ids": [2, 1, 2]} suffix'
        )
        self.assertEqual(summary, "A short biography.")
        self.assertEqual(evidence, [1, 2])

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
                "source_text_sha256": "source-hash",
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
                ]
            )
            manifest = run_pipeline(args)

            self.assertTrue((run_dir / "coloring_book.pdf").exists())
            self.assertTrue((run_dir / "pages/Test_Person.pdf").exists())
            self.assertTrue((run_dir / "manifest.json").exists())
            self.assertEqual(manifest["items"][0]["errors"], [])


if __name__ == "__main__":
    unittest.main()
