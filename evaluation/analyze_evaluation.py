"""Analyze blinded image ratings, text audits, and runtime metadata."""

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import random
import statistics


METHODS = ("flux_final", "sd15_controlnet")
METRICS = ("identity", "line_cleanliness", "coloring_suitability", "overall_quality")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fieldnames, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def score(value, field, row_number):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid {field} in ratings row {row_number}.") from exc
    if not 1 <= number <= 5:
        raise ValueError(f"{field} must be between 1 and 5 in ratings row {row_number}.")
    return number


def mean(values):
    return statistics.fmean(values) if values else None


def sample_sd(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def exact_sign_flip_p(differences):
    nonzero = [value for value in differences if value != 0]
    if not nonzero:
        return 1.0
    if len(nonzero) > 20:
        raise ValueError("Exact sign-flip test supports at most 20 non-zero pairs.")
    observed = abs(mean(nonzero))
    total = 2 ** len(nonzero)
    extreme = 0
    for signs in itertools.product((-1, 1), repeat=len(nonzero)):
        statistic = abs(mean([sign * value for sign, value in zip(signs, nonzero)]))
        if statistic >= observed - 1e-12:
            extreme += 1
    return extreme / total


def bootstrap_ci(differences, seed=42, repetitions=10000):
    if not differences:
        return [None, None]
    rng = random.Random(seed)
    estimates = []
    for _ in range(repetitions):
        sample = [rng.choice(differences) for _ in differences]
        estimates.append(mean(sample))
    estimates.sort()
    lower = estimates[int(0.025 * (repetitions - 1))]
    upper = estimates[int(0.975 * (repetitions - 1))]
    return [lower, upper]


def exact_binomial_two_sided(successes, trials):
    if trials == 0:
        return None
    tail = min(successes, trials - successes)
    lower_probability = sum(math.comb(trials, index) for index in range(tail + 1)) / (2**trials)
    return min(1.0, 2 * lower_probability)


def analyze_image_ratings(ratings, key_rows):
    key = {row["subject_id"]: row for row in key_rows}
    by_subject = {}
    preference_votes = {}
    for row_number, row in enumerate(ratings, start=2):
        subject_id = row["subject_id"]
        if subject_id not in key:
            raise ValueError(f"Unknown subject_id {subject_id} in ratings row {row_number}.")
        assignment = key[subject_id]
        bucket = by_subject.setdefault(
            subject_id,
            {method: {metric: [] for metric in METRICS} for method in METHODS},
        )
        for label in ("A", "B"):
            method = assignment[f"{label}_method"]
            if method not in METHODS:
                raise ValueError(f"Unknown method {method} for {subject_id}.")
            for metric in METRICS:
                bucket[method][metric].append(score(row[f"{metric}_{label}"], f"{metric}_{label}", row_number))

        preference = row.get("preference", "").strip()
        votes = preference_votes.setdefault(
            subject_id, {METHODS[0]: 0, METHODS[1]: 0}
        )
        if preference == "Tie":
            pass
        elif preference in ("A", "B"):
            votes[assignment[f"{preference}_method"]] += 1
        else:
            raise ValueError(f"preference must be A, B, or Tie in ratings row {row_number}.")

    result_rows = []
    subject_rows = []
    for metric in METRICS:
        final_values = []
        baseline_values = []
        for subject_id in sorted(by_subject):
            final_score = mean(by_subject[subject_id][METHODS[0]][metric])
            baseline_score = mean(by_subject[subject_id][METHODS[1]][metric])
            final_values.append(final_score)
            baseline_values.append(baseline_score)
            subject_rows.append(
                {
                    "subject_id": subject_id,
                    "metric": metric,
                    "flux_final_mean": round(final_score, 4),
                    "sd15_controlnet_mean": round(baseline_score, 4),
                    "paired_difference": round(final_score - baseline_score, 4),
                }
            )
        differences = [final - baseline for final, baseline in zip(final_values, baseline_values)]
        ci_low, ci_high = bootstrap_ci(differences)
        result_rows.append(
            {
                "metric": metric,
                "subjects": len(differences),
                "flux_final_mean": round(mean(final_values), 4),
                "flux_final_sd": round(sample_sd(final_values), 4),
                "sd15_controlnet_mean": round(mean(baseline_values), 4),
                "sd15_controlnet_sd": round(sample_sd(baseline_values), 4),
                "mean_paired_difference": round(mean(differences), 4),
                "bootstrap_95_ci_low": round(ci_low, 4),
                "bootstrap_95_ci_high": round(ci_high, 4),
                "exact_sign_flip_p": round(exact_sign_flip_p(differences), 6),
            }
        )

    preferences = {METHODS[0]: 0, METHODS[1]: 0, "Tie": 0}
    for votes in preference_votes.values():
        if votes[METHODS[0]] > votes[METHODS[1]]:
            preferences[METHODS[0]] += 1
        elif votes[METHODS[1]] > votes[METHODS[0]]:
            preferences[METHODS[1]] += 1
        else:
            preferences["Tie"] += 1
    non_ties = preferences[METHODS[0]] + preferences[METHODS[1]]
    preference_result = dict(preferences)
    preference_result["non_tie_trials"] = non_ties
    preference_result["exact_binomial_p"] = exact_binomial_two_sided(
        preferences[METHODS[0]], non_ties
    )
    return result_rows, subject_rows, preference_result


def analyze_text_ratings(rows):
    if not rows:
        return None
    claim_count = 0
    supported_count = 0
    readability = []
    age_appropriateness = []
    evidence_valid = 0
    for row_number, row in enumerate(rows, start=2):
        try:
            claims = int(row["claim_count"])
            supported = int(row["supported_claim_count"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid claim counts in text ratings row {row_number}.") from exc
        if claims < 0 or not 0 <= supported <= claims:
            raise ValueError(f"Unsupported claim counts in text ratings row {row_number}.")
        claim_count += claims
        supported_count += supported
        readability.append(score(row["readability_1_5"], "readability_1_5", row_number))
        age_appropriateness.append(
            score(row["age_appropriateness_1_5"], "age_appropriateness_1_5", row_number)
        )
        value = row["evidence_ids_valid"].strip().lower()
        if value not in ("yes", "no"):
            raise ValueError(f"evidence_ids_valid must be Yes or No in text ratings row {row_number}.")
        evidence_valid += value == "yes"
    return {
        "ratings": len(rows),
        "total_claims": claim_count,
        "supported_claims": supported_count,
        "supported_claim_rate": supported_count / claim_count if claim_count else None,
        "mean_readability": mean(readability),
        "mean_age_appropriateness": mean(age_appropriateness),
        "evidence_id_valid_rate": evidence_valid / len(rows),
    }


def operational_metrics(run_dir, method):
    if not run_dir:
        return None
    run_dir = Path(run_dir)
    metadata_paths = sorted((run_dir / "generation_metadata").glob("*.json"))
    elapsed = []
    peak_memory = []
    for path in metadata_paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("elapsed_seconds") is not None:
            elapsed.append(float(record["elapsed_seconds"]))
        if record.get("peak_cuda_memory_gb") is not None:
            peak_memory.append(float(record["peak_cuda_memory_gb"]))
    image_count = len(list((run_dir / "generated_images").glob("*.*")))
    return {
        "method": method,
        "generated_images": image_count,
        "metadata_records": len(metadata_paths),
        "mean_generation_seconds": mean(elapsed),
        "median_generation_seconds": statistics.median(elapsed) if elapsed else None,
        "max_peak_cuda_memory_gb": max(peak_memory) if peak_memory else None,
    }


def run(args):
    ratings = read_csv(args.image_ratings)
    key_rows = read_csv(args.condition_key)
    metric_rows, subject_rows, preference = analyze_image_ratings(ratings, key_rows)
    text_result = analyze_text_ratings(read_csv(args.text_ratings)) if args.text_ratings else None
    operations = [
        operational_metrics(args.flux_run, METHODS[0]),
        operational_metrics(args.baseline_run, METHODS[1]),
    ]
    operations = [row for row in operations if row is not None]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "image_metric_summary.csv", metric_rows[0].keys(), metric_rows)
    write_csv(output_dir / "image_subject_scores.csv", subject_rows[0].keys(), subject_rows)
    if operations:
        write_csv(output_dir / "operational_metrics.csv", operations[0].keys(), operations)
    summary = {
        "image_metrics": metric_rows,
        "preference": preference,
        "text_evaluation": text_result,
        "operational_metrics": operations,
    }
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-ratings", required=True)
    parser.add_argument("--condition-key", required=True)
    parser.add_argument("--text-ratings")
    parser.add_argument("--flux-run")
    parser.add_argument("--baseline-run")
    parser.add_argument("--output-dir", default="evaluation/results")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
