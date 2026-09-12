import csv
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw

from evaluation.evaluate_images_auto import (
    compute_pixel_metrics,
    resolve_source_image,
    run,
)


class AutomaticImageEvaluationTest(unittest.TestCase):
    def test_blank_and_filled_images_have_expected_extremes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            white = root / "white.png"
            black = root / "black.png"
            Image.new("RGB", (200, 300), "white").save(white)
            Image.new("RGB", (200, 300), "black").save(black)

            white_metrics = compute_pixel_metrics(white, canvas_size=256)
            black_metrics = compute_pixel_metrics(black, canvas_size=256)
            self.assertEqual(white_metrics["white_space_ratio"], 1.0)
            self.assertEqual(white_metrics["ink_coverage_ratio"], 0.0)
            self.assertGreater(black_metrics["dark_fill_ratio"], 0.6)
            self.assertGreater(black_metrics["largest_dark_region_ratio"], 0.6)

    def test_simple_line_has_edges_without_color(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "line.png"
            image = Image.new("RGB", (256, 256), "white")
            ImageDraw.Draw(image).line((20, 20, 230, 230), fill="black", width=8)
            image.save(path)
            metrics = compute_pixel_metrics(path, canvas_size=256)
            self.assertGreater(metrics["edge_density"], 0)
            self.assertGreater(metrics["ink_coverage_ratio"], 0)
            self.assertEqual(metrics["color_pixel_ratio"], 0)

    def test_source_resolver_uses_saved_basename_from_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            image_dir = run_dir / "sources" / "images"
            image_dir.mkdir(parents=True)
            expected = image_dir / "Different_Name.jpg"
            Image.new("RGB", (10, 10), "white").save(expected)
            (run_dir / "sources" / "Person_One.json").write_text(
                json.dumps({"image_path": "/content/run/sources/images/Different_Name.jpg"}),
                encoding="utf-8",
            )
            self.assertEqual(resolve_source_image(run_dir, "Person_One"), expected)

    def test_end_to_end_without_embedding(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            subjects = root / "subjects.csv"
            subjects.write_text(
                "subject_id,name,slug\nM01,Person One,Person_One\nM02,Person Two,Person_Two\n",
                encoding="utf-8",
            )
            flux = root / "flux"
            baseline = root / "baseline"
            (flux / "sources" / "images").mkdir(parents=True)
            (flux / "generated_images").mkdir(parents=True)
            (baseline / "generated_images").mkdir(parents=True)
            for index, slug in enumerate(("Person_One", "Person_Two")):
                Image.new("RGB", (128, 128), (120 + index, 120, 120)).save(
                    flux / "sources" / "images" / f"{slug}.jpg"
                )
                flux_image = Image.new("RGB", (128, 128), "white")
                ImageDraw.Draw(flux_image).rectangle((30, 20, 90, 110), outline="black", width=4)
                flux_image.save(flux / "generated_images" / f"{slug}.png")
                baseline_image = Image.new("RGB", (128, 128), "white")
                rng = np.random.default_rng(index)
                pixels = np.asarray(baseline_image).copy()
                pixels[rng.random((128, 128)) < 0.05] = 0
                Image.fromarray(pixels).save(baseline / "generated_images" / f"{slug}.png")

            output = root / "results"
            args = SimpleNamespace(
                subjects=str(subjects),
                flux_run=str(flux),
                baseline_run=str(baseline),
                output_dir=str(output),
                canvas_size=256,
                skip_embedding=True,
                embedding_model="facebook/dinov2-small",
                embedding_batch_size=2,
                device="cpu",
                local_files_only=True,
                seed=7,
                bootstrap_repetitions=200,
            )
            result = run(args)
            self.assertEqual(len(result["summary"]), 9)
            self.assertTrue((output / "per_image_metrics.csv").exists())
            self.assertTrue((output / "paired_metric_summary.csv").exists())
            with (output / "per_image_metrics.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row["source_similarity_proxy"] == "" for row in rows))


if __name__ == "__main__":
    unittest.main()
