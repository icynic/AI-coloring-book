import csv
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from evaluation.analyze_evaluation import (
    analyze_image_ratings,
    exact_binomial_two_sided,
    exact_sign_flip_p,
)
from evaluation.prepare_blind_evaluation import prepare


class EvaluationTest(unittest.TestCase):
    def test_exact_tests(self):
        self.assertEqual(exact_sign_flip_p([1.0, 1.0]), 0.5)
        self.assertEqual(exact_sign_flip_p([0.0, 0.0]), 1.0)
        self.assertEqual(exact_binomial_two_sided(0, 3), 0.25)

    def test_blind_packet_is_complete_and_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            subjects = root / "subjects.csv"
            subjects.write_text(
                "subject_id,name,slug\nM01,Person One,Person_One\nM02,Person Two,Person_Two\n",
                encoding="utf-8",
            )
            for method in ("flux", "baseline"):
                image_dir = root / method / "generated_images"
                image_dir.mkdir(parents=True)
                for slug in ("Person_One", "Person_Two"):
                    (image_dir / f"{slug}.png").write_bytes(f"{method}-{slug}".encode())

            args = SimpleNamespace(
                subjects=str(subjects),
                flux_run=str(root / "flux"),
                baseline_run=str(root / "baseline"),
                output_dir=str(root / "packet"),
                raters=3,
                seed=7,
                force=False,
            )
            first_key = prepare(args)
            args.output_dir = str(root / "packet_again")
            second_key = prepare(args)
            self.assertEqual(
                [(row["A_method"], row["B_method"]) for row in first_key],
                [(row["A_method"], row["B_method"]) for row in second_key],
            )
            with (root / "packet" / "image_ratings.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 6)
            self.assertTrue((root / "packet" / "images" / "M01_A.png").exists())
            self.assertTrue((root / "packet" / "images" / "M02_B.png").exists())

    def test_image_analysis_uses_subject_level_preferences(self):
        key = [
            {
                "subject_id": "M01",
                "A_method": "flux_final",
                "B_method": "sd15_controlnet",
            },
            {
                "subject_id": "M02",
                "A_method": "sd15_controlnet",
                "B_method": "flux_final",
            },
        ]
        ratings = []
        for subject_id, preference in (("M01", "A"), ("M01", "A"), ("M02", "B"), ("M02", "A")):
            ratings.append(
                {
                    "subject_id": subject_id,
                    "identity_A": "5",
                    "identity_B": "3",
                    "line_cleanliness_A": "5",
                    "line_cleanliness_B": "3",
                    "coloring_suitability_A": "5",
                    "coloring_suitability_B": "3",
                    "overall_quality_A": "5",
                    "overall_quality_B": "3",
                    "preference": preference,
                }
            )
        metric_rows, subject_rows, preference = analyze_image_ratings(ratings, key)
        self.assertEqual(len(metric_rows), 4)
        self.assertEqual(len(subject_rows), 8)
        self.assertEqual(preference["flux_final"], 1)
        self.assertEqual(preference["sd15_controlnet"], 0)
        self.assertEqual(preference["Tie"], 1)


if __name__ == "__main__":
    unittest.main()
