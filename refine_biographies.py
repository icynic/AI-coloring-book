"""Refine a frozen run in a separate directory; never fetch or initialize FLUX."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from main import main as run_pipeline, read_json, slugify, utc_now, write_json
from source_text import text_sha256


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def plan_refinement(source_run, output_dir):
    source_run, output_dir = Path(source_run).resolve(), Path(output_dir).resolve()
    if (source_run == output_dir or source_run in output_dir.parents
            or output_dir in source_run.parents):
        raise ValueError("Source and output must be separate, non-nested directories.")
    manifest_path = source_run / "manifest.json"
    manifest = read_json(manifest_path)
    names = manifest["configuration"]["names"]
    if not names or len({slugify(n) for n in names}) != len(names):
        raise ValueError("Source run needs subjects with unique file slugs.")
    files = {}
    for name in names:
        slug = slugify(name)
        for relative in (f"sources/{slug}.json", f"summaries/{slug}.json",
                         f"generated_images/{slug}.png"):
            path = source_run / relative
            if not path.is_file():
                raise FileNotFoundError(f"Required saved artifact is missing: {path}")
            if not path.resolve().is_relative_to(source_run):
                raise ValueError("Input artifacts must resolve inside the source run.")
            files[relative] = file_sha256(path)
        source = read_json(source_run / f"sources/{slug}.json")
        text = source.get("summary")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Source text is missing for {name}.")
        if source.get("source_text_sha256") not in (None, text_sha256(text)):
            raise ValueError(f"Stored source hash is inconsistent for {name}.")
        optional = [source_run / f"generation_metadata/{slug}.json"]
        # Metadata paths can still point to the original Colab machine.
        image_name = Path((source.get("image_path") or "").replace("\\", "/")).name
        if image_name and (source_run / "sources/images" / image_name).is_file():
            optional.append(source_run / "sources/images" / image_name)
        for path in optional:
            if path.is_file():
                if not path.resolve().is_relative_to(source_run):
                    raise ValueError("Input artifacts must resolve inside the source run.")
                files[path.relative_to(source_run).as_posix()] = file_sha256(path)
    fingerprint = text_sha256(json.dumps(files, sort_keys=True))
    origin = {"version": 1, "source_run": str(source_run),
              "source_manifest_sha256": file_sha256(manifest_path),
              "input_fingerprint": fingerprint, "files": files}
    return manifest, origin


def prepare_refinement(source_run, output_dir, check_only=False):
    manifest, origin = plan_refinement(source_run, output_dir)
    source_run, output_dir = Path(source_run).resolve(), Path(output_dir).resolve()
    origin_path = output_dir / "refinement_origin.json"
    resumed = origin_path.exists()
    if resumed:
        stored = read_json(origin_path)
        if any(stored.get(key) != value for key, value in origin.items()):
            raise ValueError("Refinement inputs changed; use a new output directory.")
        # Only summaries, PDFs and manifests are mutable in the derived run.
        for relative, digest in origin["files"].items():
            path = output_dir / relative
            if not relative.startswith("summaries/") and path.exists() and file_sha256(path) != digest:
                raise ValueError(f"Derived immutable artifact changed: {relative}")
            original = output_dir / "original_summaries" / Path(relative).name
            if relative.startswith("summaries/") and original.exists() and file_sha256(original) != digest:
                raise ValueError(f"Saved original biography changed: {original}")
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory is not empty and is not a known refinement run.")
    print(f"[refine] Frozen source: {source_run}")
    print(f"[refine] Separate output: {output_dir}")
    print(f"[refine] {len(manifest['configuration']['names'])} subjects; reuse sources and FLUX images")
    if check_only:
        print("[refine] Read-only preflight passed. No models, network calls, or writes.")
        return origin

    output_dir.mkdir(parents=True, exist_ok=True)
    if not resumed:
        write_json(origin_path, {**origin, "created_at": utc_now()})
    for relative in origin["files"]:
        destination = output_dir / relative
        # Refuse links outside the derived directory before any write.
        if not destination.resolve().is_relative_to(output_dir):
            raise ValueError("Derived artifacts must resolve inside the output directory.")
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_run / relative, destination)
        if relative.startswith("summaries/"):
            original = output_dir / "original_summaries" / Path(relative).name
            if not original.resolve().is_relative_to(output_dir):
                raise ValueError("Original biography backup must be inside the output directory.")
            if not original.exists():
                original.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_run / relative, original)
    derived_manifest_path = output_dir / "manifest.json"
    if not derived_manifest_path.exists():
        derived = {**manifest, "book_path": None,
                   "refinement_origin": {key: value for key, value in origin.items() if key != "files"}}
        derived["items"] = [{**item, "pdf_path": None} for item in manifest.get("items", [])]
        write_json(derived_manifest_path, derived)
    return origin


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--summary-min-words", type=int)
    parser.add_argument("--summary-max-words", type=int)
    parser.add_argument("--max-review-revisions", type=int, choices=[0, 1, 2], default=2)
    args = parser.parse_args(argv)
    # Validate overrides before creating a derived directory.
    manifest = read_json(Path(args.source_run) / "manifest.json")
    low, high = manifest["configuration"].get("summary_word_range", [80, 110])
    low = args.summary_min_words if args.summary_min_words is not None else low
    high = args.summary_max_words if args.summary_max_words is not None else high
    if type(low) is not int or type(high) is not int or not 1 <= low <= high:
        parser.error("Word range must satisfy 1 <= minimum <= maximum.")
    origin = prepare_refinement(args.source_run, args.output_dir, args.check_only)
    if args.check_only:
        return origin
    command = ["--repair-summaries", "--verify-summaries", "--output-dir", args.output_dir,
               "--max-review-revisions", str(args.max_review_revisions)]
    for flag, value in (("--summary-min-words", args.summary_min_words),
                        ("--summary-max-words", args.summary_max_words)):
        if value is not None:
            command.extend([flag, str(value)])
    return run_pipeline(command)


if __name__ == "__main__":
    main()
