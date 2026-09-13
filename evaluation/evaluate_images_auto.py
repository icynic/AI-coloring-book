"""Compute reproducible proxy metrics for the paired image evaluation.

The pixel metrics describe properties associated with printable coloring pages.
An optional DINOv2 cosine similarity compares each generated image with the
saved source portrait.  Because photographs and line drawings are different
domains, that value is a similarity proxy, not a face-recognition score.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import itertools
import random
import statistics
from pathlib import Path
import platform
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

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


METHODS = ("flux_final", "sd15_controlnet")
METRIC_SPECS = {
    "source_similarity_proxy": {
        "direction": "higher",
        "description": (
            "DINOv2 cosine similarity between the saved source portrait and the "
            "generated image; a cross-domain structural/semantic proxy, not face recognition."
        ),
    },
    "white_space_ratio": {
        "direction": "higher",
        "description": "Fraction of pixels with grayscale value at least 245.",
    },
    "ink_coverage_ratio": {
        "direction": "descriptive",
        "description": "Fraction of pixels with grayscale value below 230.",
    },
    "dark_fill_ratio": {
        "direction": "lower",
        "description": "Fraction of pixels with grayscale value at most 64.",
    },
    "midtone_ratio": {
        "direction": "lower",
        "description": "Fraction of pixels with grayscale value from 65 through 229.",
    },
    "edge_density": {
        "direction": "lower",
        "description": "Fraction of pixels selected by Canny edge detection (100/200 thresholds).",
    },
    "small_components_per_megapixel": {
        "direction": "lower",
        "description": (
            "Connected ink components of 4-24 pixels per megapixel after normalization; "
            "a speckle/noise proxy."
        ),
    },
    "small_component_ink_ratio": {
        "direction": "lower",
        "description": "Fraction of thresholded ink belonging to 4-24 pixel components.",
    },
    "largest_dark_region_ratio": {
        "direction": "lower",
        "description": "Largest connected dark region as a fraction of the normalized canvas.",
    },
    "color_pixel_ratio": {
        "direction": "lower",
        "description": "Fraction of pixels whose maximum RGB channel difference exceeds 10.",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_generated_image(run_dir: Path, slug: str) -> Path:
    candidates = [
        path for path in (run_dir / "generated_images").glob(f"{slug}.*") if path.is_file()
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one generated image for {slug} in {run_dir}, found {len(candidates)}."
        )
    return candidates[0]


def resolve_source_image(flux_run: Path, slug: str) -> Path:
    image_dir = flux_run / "sources" / "images"
    record_path = flux_run / "sources" / f"{slug}.json"
    candidates = []
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        recorded_path = record.get("image_path")
        if recorded_path:
            local_copy = image_dir / Path(recorded_path).name
            if local_copy.is_file():
                candidates.append(local_copy)
            elif Path(recorded_path).is_file():
                candidates.append(Path(recorded_path))
    if not candidates:
        candidates = [path for path in image_dir.glob(f"{slug}.*") if path.is_file()]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one source portrait for {slug} in {image_dir}, found {len(candidates)}."
        )
    return candidates[0]


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        if opened.mode in ("RGBA", "LA") or "transparency" in opened.info:
            rgba = opened.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            return Image.alpha_composite(background, rgba).convert("RGB")
        return opened.convert("RGB")


def normalized_rgb(path: Path, canvas_size: int) -> np.ndarray:
    image = load_rgb(path)
    contained = ImageOps.contain(
        image,
        (canvas_size, canvas_size),
        method=Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (canvas_size, canvas_size), "white")
    offset = ((canvas_size - contained.width) // 2, (canvas_size - contained.height) // 2)
    canvas.paste(contained, offset)
    return np.asarray(canvas, dtype=np.uint8)


def largest_component_ratio(mask: np.ndarray) -> float:
    pixel_count = int(mask.size)
    if pixel_count == 0 or not mask.any():
        return 0.0
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return 0.0
    largest = int(stats[1:, cv2.CC_STAT_AREA].max())
    return largest / pixel_count


def compute_pixel_metrics(path: Path, canvas_size: int = 512) -> dict[str, float]:
    rgb = normalized_rgb(path, canvas_size)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    total_pixels = gray.size

    white = gray >= 245
    ink = gray < 230
    dark = gray <= 64
    midtone = (gray > 64) & (gray < 230)
    edges = cv2.Canny(gray, 100, 200, L2gradient=True) > 0
    color_pixels = (rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)) > 10

    component_mask = gray < 200
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        component_mask.astype(np.uint8), 8
    )
    component_areas = (
        stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
        if component_count > 1
        else np.array([], dtype=np.int64)
    )
    small_areas = component_areas[(component_areas >= 4) & (component_areas <= 24)]
    component_ink_pixels = int(component_mask.sum())

    return {
        "white_space_ratio": float(white.mean()),
        "ink_coverage_ratio": float(ink.mean()),
        "dark_fill_ratio": float(dark.mean()),
        "midtone_ratio": float(midtone.mean()),
        "edge_density": float(edges.mean()),
        "small_components_per_megapixel": float(
            len(small_areas) / (total_pixels / 1_000_000)
        ),
        "small_component_ink_ratio": float(
            small_areas.sum() / component_ink_pixels if component_ink_pixels else 0.0
        ),
        "largest_dark_region_ratio": float(largest_component_ratio(dark)),
        "color_pixel_ratio": float(color_pixels.mean()),
    }


class DinoSimilarity:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        batch_size: int = 4,
        local_files_only: bool = False,
    ) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.processor = AutoImageProcessor.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        ).to(self.device)
        self.model.eval()
        self.model_name = model_name
        self.revision = getattr(self.model.config, "_commit_hash", None)

    def encode(self, paths: list[Path]) -> dict[Path, np.ndarray]:
        unique_paths = list(dict.fromkeys(path.resolve() for path in paths))
        embeddings = {}
        for start in range(0, len(unique_paths), self.batch_size):
            batch_paths = unique_paths[start : start + self.batch_size]
            images = [load_rgb(path) for path in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.torch.inference_mode():
                output = self.model(**inputs)
            pooled = getattr(output, "pooler_output", None)
            if pooled is None:
                pooled = output.last_hidden_state[:, 0]
            pooled = self.torch.nn.functional.normalize(pooled.float(), dim=-1)
            values = pooled.detach().cpu().numpy()
            for path, value in zip(batch_paths, values):
                embeddings[path] = value
        return embeddings


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator == 0:
        return 0.0
    return float(np.dot(first, second) / denominator)


def rounded(value, digits=6):
    return None if value is None else round(float(value), digits)


def summarize_pairs(rows, seed: int, repetitions: int) -> list[dict]:
    by_metric = []
    for metric, spec in METRIC_SPECS.items():
        if all(row.get(metric) is None for row in rows):
            continue
        by_subject = {}
        for row in rows:
            by_subject.setdefault(row["subject_id"], {})[row["method"]] = row.get(metric)

        flux_values = []
        baseline_values = []
        for subject_id in sorted(by_subject):
            methods = by_subject[subject_id]
            if any(method not in methods or methods[method] is None for method in METHODS):
                raise ValueError(f"Incomplete {metric} pair for {subject_id}.")
            flux_values.append(float(methods[METHODS[0]]))
            baseline_values.append(float(methods[METHODS[1]]))

        differences = [
            flux - baseline for flux, baseline in zip(flux_values, baseline_values)
        ]
        ci_low, ci_high = bootstrap_ci(
            differences,
            seed=seed,
            repetitions=repetitions,
        )
        direction = spec["direction"]
        if direction == "higher":
            oriented = differences
        elif direction == "lower":
            oriented = [-difference for difference in differences]
        else:
            oriented = None
        difference_sd = sample_sd(differences)
        effect_size = (
            mean(oriented) / difference_sd
            if oriented is not None and difference_sd > 0
            else None
        )
        by_metric.append(
            {
                "metric": metric,
                "preferred_direction": direction,
                "subjects": len(differences),
                "flux_final_mean": rounded(mean(flux_values)),
                "flux_final_sd": rounded(sample_sd(flux_values)),
                "sd15_controlnet_mean": rounded(mean(baseline_values)),
                "sd15_controlnet_sd": rounded(sample_sd(baseline_values)),
                "mean_difference_flux_minus_baseline": rounded(mean(differences)),
                "bootstrap_95_ci_low": rounded(ci_low),
                "bootstrap_95_ci_high": rounded(ci_high),
                "exact_sign_flip_p_uncorrected": rounded(exact_sign_flip_p(differences)),
                "paired_effect_size_dz_oriented": rounded(effect_size),
                "flux_better_subjects": (
                    sum(value > 0 for value in oriented) if oriented is not None else None
                ),
                "baseline_better_subjects": (
                    sum(value < 0 for value in oriented) if oriented is not None else None
                ),
                "ties": sum(value == 0 for value in differences),
            }
        )
    return by_metric


def write_markdown(path: Path, summary_rows: list[dict], embedding_used: bool) -> None:
    lines = [
        "# Automatic image evaluation",
        "",
        "Measurements use the supplied subject list and paired FLUX–SD1.5 outputs. Consult the manifest for the exact run and inputs.",
        "Pixel metrics were calculated after aspect-preserving resize and white padding to 512×512.",
        "",
        "| Metric | Direction | FLUX mean | SD1.5 mean | Difference | 95% bootstrap CI | p | FLUX wins |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        interval = f"[{row['bootstrap_95_ci_low']:.4f}, {row['bootstrap_95_ci_high']:.4f}]"
        wins = (
            f"{row['flux_better_subjects']}/{row['subjects']}"
            if row["flux_better_subjects"] is not None
            else "n.a."
        )
        lines.append(
            "| {metric} | {direction} | {flux:.4f} | {baseline:.4f} | {difference:.4f} | "
            "{interval} | {p:.4f} | {wins} |".format(
                metric=row["metric"],
                direction=row["preferred_direction"],
                flux=row["flux_final_mean"],
                baseline=row["sd15_controlnet_mean"],
                difference=row["mean_difference_flux_minus_baseline"],
                interval=interval,
                p=row["exact_sign_flip_p_uncorrected"],
                wins=wins,
            )
        )
    lines.extend(
        [
            "",
            "The p-values are exploratory exact paired sign-flip tests and are not corrected for multiple comparisons.",
            "Automatic line-art measures are proxies: lower complexity is not always better, and white-space ratio can reward an overly empty image.",
        ]
    )
    if embedding_used:
        lines.append(
            "DINOv2 similarity crosses a photograph-to-line-art domain gap and must not be described as face-recognition accuracy."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args) -> dict:
    subjects_path = Path(args.subjects).resolve()
    flux_run = Path(args.flux_run).resolve()
    baseline_run = Path(args.baseline_run).resolve()
    output_dir = Path(args.output_dir).resolve()
    subjects = read_csv(subjects_path)
    if not subjects:
        raise ValueError("The subject list is empty.")

    records = []
    all_paths = []
    for subject in subjects:
        source = resolve_source_image(flux_run, subject["slug"])
        generated = {
            METHODS[0]: resolve_generated_image(flux_run, subject["slug"]),
            METHODS[1]: resolve_generated_image(baseline_run, subject["slug"]),
        }
        all_paths.append(source)
        all_paths.extend(generated.values())
        for method in METHODS:
            image = generated[method]
            with Image.open(image) as opened:
                width, height = opened.size
            record = {
                "subject_id": subject["subject_id"],
                "name": subject["name"],
                "slug": subject["slug"],
                "method": method,
                "source_image": str(source.resolve()),
                "generated_image": str(image.resolve()),
                "source_sha256": sha256(source),
                "generated_sha256": sha256(image),
                "original_width": width,
                "original_height": height,
                "source_similarity_proxy": None,
            }
            record.update(compute_pixel_metrics(image, args.canvas_size))
            records.append(record)

    embedding_metadata = {
        "enabled": not args.skip_embedding,
        "model": None,
        "revision": None,
        "device": None,
    }
    if not args.skip_embedding:
        encoder = DinoSimilarity(
            args.embedding_model,
            device=args.device,
            batch_size=args.embedding_batch_size,
            local_files_only=args.local_files_only,
        )
        embeddings = encoder.encode(all_paths)
        for record in records:
            source = Path(record["source_image"]).resolve()
            generated = Path(record["generated_image"]).resolve()
            record["source_similarity_proxy"] = cosine_similarity(
                embeddings[source], embeddings[generated]
            )
        embedding_metadata.update(
            {
                "model": encoder.model_name,
                "revision": encoder.revision,
                "device": str(encoder.device),
            }
        )

    summary_rows = summarize_pairs(records, args.seed, args.bootstrap_repetitions)
    output_dir.mkdir(parents=True, exist_ok=True)
    record_fields = list(records[0].keys())
    summary_fields = list(summary_rows[0].keys())
    serializable_records = [
        {key: rounded(value) if isinstance(value, (float, np.floating)) else value for key, value in row.items()}
        for row in records
    ]
    write_csv(output_dir / "per_image_metrics.csv", record_fields, serializable_records)
    write_csv(output_dir / "paired_metric_summary.csv", summary_fields, summary_rows)
    write_markdown(
        output_dir / "automatic_evaluation.md",
        summary_rows,
        embedding_used=not args.skip_embedding,
    )

    result = {
        "schema_version": 1,
        "created_at": utc_now(),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "configuration": {
            "subjects": str(subjects_path),
            "flux_run": str(flux_run),
            "baseline_run": str(baseline_run),
            "canvas_size": args.canvas_size,
            "bootstrap_seed": args.seed,
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "embedding": embedding_metadata,
        },
        "metric_definitions": METRIC_SPECS,
        "summary": summary_rows,
        "caveats": [
            "All automatic image-quality measurements are proxies.",
            "DINOv2 similarity is not face-recognition accuracy.",
            "Lower edge or ink density is not unconditionally better; an empty image can score well.",
            "Exact sign-flip p-values are exploratory and uncorrected for multiple comparisons.",
        ],
    }
    (output_dir / "automatic_evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Automatic evaluation: {output_dir}")
    print(f"Subjects: {len(subjects)}; images: {len(records)}")
    print(f"Embedding proxy: {'enabled' if not args.skip_embedding else 'skipped'}")
    return result


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flux-run", required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--output-dir", default="evaluation/automatic_results")
    parser.add_argument("--subjects", default=str(Path(__file__).with_name("subjects.csv")))
    parser.add_argument("--canvas-size", type=int, default=512)
    parser.add_argument("--embedding-model", default="facebook/dinov2-small")
    parser.add_argument("--embedding-batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    args = parser.parse_args(argv)
    if args.canvas_size < 128:
        parser.error("--canvas-size must be at least 128")
    if args.embedding_batch_size < 1:
        parser.error("--embedding-batch-size must be at least 1")
    if args.bootstrap_repetitions < 100:
        parser.error("--bootstrap-repetitions must be at least 100")
    return args


if __name__ == "__main__":
    run(parse_args())
