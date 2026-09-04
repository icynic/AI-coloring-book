"""Randomize final and baseline images into a reproducible blinded packet."""

import argparse
import csv
from pathlib import Path
import random
import shutil


METHOD_FLUX = "flux_final"
METHOD_BASELINE = "sd15_controlnet"
IMAGE_FIELDS = [
    "rater_id",
    "subject_id",
    "image_A",
    "image_B",
    "identity_A",
    "identity_B",
    "line_cleanliness_A",
    "line_cleanliness_B",
    "coloring_suitability_A",
    "coloring_suitability_B",
    "overall_quality_A",
    "overall_quality_B",
    "preference",
    "notes",
]
TEXT_FIELDS = [
    "rater_id",
    "subject_id",
    "name",
    "summary_file",
    "claim_count",
    "supported_claim_count",
    "readability_1_5",
    "age_appropriateness_1_5",
    "evidence_ids_valid",
    "notes",
]


def read_subjects(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_image(run_dir, slug):
    image_dir = Path(run_dir) / "generated_images"
    candidates = [path for path in image_dir.glob(f"{slug}.*") if path.is_file()]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one generated image for {slug} in {image_dir}, found {len(candidates)}."
        )
    return candidates[0]


def prepare(args):
    subjects = read_subjects(args.subjects)
    output_dir = Path(args.output_dir).resolve()
    key_path = output_dir / "condition_key.csv"
    if key_path.exists() and not args.force:
        raise FileExistsError(
            f"{key_path} already exists. Reuse the packet or pass --force to overwrite it."
        )

    pairs = []
    missing = []
    for subject in subjects:
        try:
            flux = resolve_image(args.flux_run, subject["slug"])
            baseline = resolve_image(args.baseline_run, subject["slug"])
            pairs.append((subject, flux, baseline))
        except FileNotFoundError as exc:
            missing.append(str(exc))
    if missing:
        raise FileNotFoundError("Cannot build a complete blinded packet:\n" + "\n".join(missing))

    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    key_rows = []
    image_rows = []
    text_rows = []

    for subject, flux, baseline in pairs:
        methods = [(METHOD_FLUX, flux), (METHOD_BASELINE, baseline)]
        rng.shuffle(methods)
        assignment = dict(zip(("A", "B"), methods))
        copied = {}
        for label, (method, source) in assignment.items():
            destination = image_dir / f"{subject['subject_id']}_{label}{source.suffix.lower()}"
            shutil.copy2(source, destination)
            copied[label] = destination.relative_to(output_dir).as_posix()

        key_rows.append(
            {
                "subject_id": subject["subject_id"],
                "name": subject["name"],
                "slug": subject["slug"],
                "A_method": assignment["A"][0],
                "B_method": assignment["B"][0],
                "A_source_file": str(assignment["A"][1].resolve()),
                "B_source_file": str(assignment["B"][1].resolve()),
                "randomization_seed": args.seed,
            }
        )

        for rater_index in range(1, args.raters + 1):
            rater_id = f"R{rater_index:02d}"
            image_rows.append(
                {
                    "rater_id": rater_id,
                    "subject_id": subject["subject_id"],
                    "image_A": copied["A"],
                    "image_B": copied["B"],
                    "identity_A": "",
                    "identity_B": "",
                    "line_cleanliness_A": "",
                    "line_cleanliness_B": "",
                    "coloring_suitability_A": "",
                    "coloring_suitability_B": "",
                    "overall_quality_A": "",
                    "overall_quality_B": "",
                    "preference": "",
                    "notes": "",
                }
            )
            text_rows.append(
                {
                    "rater_id": rater_id,
                    "subject_id": subject["subject_id"],
                    "name": subject["name"],
                    "summary_file": f"summaries/{subject['slug']}.json",
                    "claim_count": "",
                    "supported_claim_count": "",
                    "readability_1_5": "",
                    "age_appropriateness_1_5": "",
                    "evidence_ids_valid": "",
                    "notes": "",
                }
            )

    write_csv(
        key_path,
        [
            "subject_id",
            "name",
            "slug",
            "A_method",
            "B_method",
            "A_source_file",
            "B_source_file",
            "randomization_seed",
        ],
        key_rows,
    )
    write_csv(output_dir / "image_ratings.csv", IMAGE_FIELDS, image_rows)
    write_csv(output_dir / "text_ratings.csv", TEXT_FIELDS, text_rows)
    print(f"Blinded packet: {output_dir}")
    print(f"Keep private: {key_path}")
    return key_rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flux-run", required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--output-dir", default="evaluation/blind_packet")
    parser.add_argument("--subjects", default=str(Path(__file__).with_name("subjects.csv")))
    parser.add_argument("--raters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.raters < 1:
        parser.error("--raters must be at least 1")
    return args


if __name__ == "__main__":
    prepare(parse_args())
