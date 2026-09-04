"""Generate the fixed SD1.5 + ControlNet evaluation baseline."""

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from datetime import datetime, timezone


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

DEFAULT_PROMPT = (
    "c0l0ringb00k, black and white coloring page, line art, "
    "white background, thick lines"
)
DEFAULT_NEGATIVE_PROMPT = (
    "shadow, shading, gradients, stippling, screentone, texture, background details, "
    "flowers, plants, stripes, grayscale, colored, 3d, realistic, photo, noise, blurry, "
    "deformed, filled, filled-in, filled-in lines, filled-in shapes, filled-in patterns, "
    "filled background"
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_subjects(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def find_source_image(source_run, slug):
    image_dir = Path(source_run) / "sources" / "images"
    source_record_path = Path(source_run) / "sources" / f"{slug}.json"
    matches = []
    if source_record_path.exists():
        source_record = json.loads(source_record_path.read_text(encoding="utf-8"))
        recorded_path = source_record.get("image_path")
        if recorded_path:
            copied_candidate = image_dir / Path(recorded_path).name
            if copied_candidate.is_file():
                matches.append(copied_candidate)
            elif Path(recorded_path).is_file():
                matches.append(Path(recorded_path))
    if not matches:
        matches = [path for path in image_dir.glob(f"{slug}.*") if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one source image for {slug} in {image_dir}, found {len(matches)}."
        )
    return matches[0]


def run(args):
    from Generator import ColoringPageGenerator

    subjects = read_subjects(args.subjects)
    source_run = Path(args.source_run).resolve()
    output_dir = Path(args.output_dir).resolve()
    image_dir = output_dir / "generated_images"
    control_dir = output_dir / "control_images"
    metadata_dir = output_dir / "generation_metadata"
    for directory in (image_dir, control_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    generator = ColoringPageGenerator()
    items = []
    for index, subject in enumerate(subjects):
        slug = subject["slug"]
        output_path = image_dir / f"{slug}.png"
        control_path = control_dir / f"{slug}.png"
        metadata_path = metadata_dir / f"{slug}.json"
        sample_seed = args.seed + index
        item = {
            "subject_id": subject["subject_id"],
            "name": subject["name"],
            "slug": slug,
            "seed": sample_seed,
            "output_image": str(output_path),
            "metadata_path": str(metadata_path),
            "errors": [],
        }

        if output_path.exists() and metadata_path.exists() and not args.force:
            print(f"[baseline] Reusing {output_path}")
            items.append(item)
            continue

        try:
            source_image = find_source_image(source_run, slug)
            started = time.perf_counter()
            print(f"[baseline] {subject['name']} (seed {sample_seed})")
            generator.process_image(
                image_path=str(source_image),
                output_path=str(output_path),
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                steps=args.steps,
                strength=args.controlnet_scale,
                guidance_scale=args.guidance_scale,
                control_image_path=str(control_path),
                seed=sample_seed,
            )
            metadata = {
                "method": "SD1.5 + Lineart ControlNet + AnimeLineartLoRA",
                "base_model": "runwayml/stable-diffusion-v1-5",
                "controlnet_model": "lllyasviel/control_v11p_sd15_lineart",
                "lineart_detector": "lllyasviel/Annotators",
                "lora_model": "beatless/AnimeLineartLoRA",
                "subject_id": subject["subject_id"],
                "name": subject["name"],
                "slug": slug,
                "source_image": str(source_image),
                "output_image": str(output_path),
                "control_image": str(control_path),
                "seed": sample_seed,
                "steps": args.steps,
                "controlnet_conditioning_scale": args.controlnet_scale,
                "guidance_scale": args.guidance_scale,
                "prompt": args.prompt,
                "negative_prompt": args.negative_prompt,
                "device": generator.device,
                "dtype": str(generator.dtype),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "created_at": utc_now(),
            }
            write_json(metadata_path, metadata)
        except Exception as exc:
            item["errors"].append(str(exc))
            print(f"[baseline] Failed for {subject['name']}: {exc}")
        items.append(item)

    manifest = {
        "schema_version": 1,
        "method": "baseline_sd15_controlnet",
        "source_run": str(source_run),
        "created_at": utc_now(),
        "configuration": {
            "seed": args.seed,
            "steps": args.steps,
            "controlnet_conditioning_scale": args.controlnet_scale,
            "guidance_scale": args.guidance_scale,
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
        },
        "items": items,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-dir", default="evaluation/baseline_run")
    parser.add_argument("--subjects", default=str(Path(__file__).with_name("subjects.csv")))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--controlnet-scale", type=float, default=0.6)
    parser.add_argument("--guidance-scale", type=float, default=10.0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
