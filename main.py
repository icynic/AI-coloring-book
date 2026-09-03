"""End-to-end, resumable AI coloring-book pipeline.

The pipeline intentionally runs the text and image models in separate stages so
Qwen and FLUX never need to occupy GPU memory at the same time.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import sys

import torch

from Concatenator import Concatenator
from Fetcher import get_person_info


DEFAULT_QWEN_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_FLUX_MODEL = "black-forest-labs/FLUX.2-klein-4B"
DEFAULT_QWEN_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DEFAULT_FLUX_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def slugify(value):
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return value or "person"


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    os.replace(temporary_path, path)


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def get_names(args):
    names = list(args.names or [])
    if args.names_file:
        names.extend(
            line.strip()
            for line in Path(args.names_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    deduplicated = []
    seen = set()
    for name in names or ["Marie Curie"]:
        if name.casefold() not in seen:
            deduplicated.append(name)
            seen.add(name.casefold())
    return deduplicated


def runtime_info():
    cuda_device = None
    if torch.cuda.is_available():
        cuda_device = torch.cuda.get_device_name(0)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": cuda_device,
    }


def image_attribution(source):
    parts = []
    artist = source.get("image_artist")
    credit = source.get("image_credit")
    license_name = source.get("image_license")
    if artist:
        parts.append(artist)
    if credit and credit not in parts:
        parts.append(credit)
    if license_name:
        parts.append(license_name)
    return "; ".join(parts) if parts else "See source metadata for attribution"


def ensure_run_directories(run_dir):
    paths = {
        "run": run_dir,
        "source": run_dir / "sources",
        "source_images": run_dir / "sources" / "images",
        "summaries": run_dir / "summaries",
        "images": run_dir / "generated_images",
        "generation": run_dir / "generation_metadata",
        "pages": run_dir / "pages",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def fetch_stage(names, paths, force=False, fuzzy_search=True):
    records = []
    for query in names:
        slug = slugify(query)
        metadata_path = paths["source"] / f"{slug}.json"
        cached_source = read_json(metadata_path) if metadata_path.exists() else None
        cached_image = (cached_source or {}).get("image_path")
        cache_is_complete = bool(
            cached_source
            and cached_source.get("summary")
            and cached_source.get("source_text_sha256")
            and cached_image
            and Path(cached_image).exists()
            and cached_source.get("image_sha256")
        )
        if cache_is_complete and not force:
            source = cached_source
            print(f"[fetch] Reusing {metadata_path}")
        else:
            print(f"[fetch] {query}")
            source = get_person_info(
                query,
                fuzzy_search=fuzzy_search,
                save_folder=str(paths["source_images"]),
            )
            source["query"] = query
            write_json(metadata_path, source)

        record = {
            "query": query,
            "slug": slug,
            "source_metadata_path": str(metadata_path),
            "source": source,
            "errors": [],
        }
        if not source.get("summary"):
            record["errors"].append("No Wikipedia source text was retrieved.")
        if not source.get("image_path") or not Path(source["image_path"]).exists():
            record["errors"].append("No source image was retrieved.")
        records.append(record)
    return records


def summarization_stage(records, paths, args):
    pending = [
        record
        for record in records
        if record["source"].get("summary")
        and (args.force or not (paths["summaries"] / f"{record['slug']}.json").exists())
    ]
    if args.skip_summarization:
        print("[summarize] Skipped by command-line option.")
        pending = []

    summarizer = None
    if pending:
        from Summarizer import Summarizer

        summarizer = Summarizer(
            model_name=args.qwen_model,
            quantization=args.qwen_quantization,
            revision=args.qwen_revision,
        )
        try:
            for record in pending:
                print(f"[summarize] {record['source']['title']}")
                summary_path = paths["summaries"] / f"{record['slug']}.json"
                try:
                    summary = summarizer.summarize_with_evidence(
                        record["source"]["summary"],
                        target_age=args.target_age,
                        min_words=args.summary_min_words,
                        max_words=args.summary_max_words,
                    )
                    summary.update(
                        {
                            "query": record["query"],
                            "title": record["source"]["title"],
                            "source_revision_id": record["source"].get("revision_id"),
                            "created_at": utc_now(),
                        }
                    )
                    write_json(summary_path, summary)
                except Exception as exc:
                    record["errors"].append(f"Summarization failed: {exc}")
                    print(f"[summarize] Failed for {record['query']}: {exc}")
        finally:
            summarizer.cleanup()
            del summarizer

    for record in records:
        summary_path = paths["summaries"] / f"{record['slug']}.json"
        record["summary_path"] = str(summary_path)
        if summary_path.exists():
            record["summary"] = read_json(summary_path)
        elif not args.skip_summarization:
            record["errors"].append("No generated summary is available.")


def image_generation_stage(records, paths, args):
    pending = [
        record
        for record in records
        if record["source"].get("image_path")
        and Path(record["source"]["image_path"]).exists()
        and (args.force or not (paths["images"] / f"{record['slug']}.png").exists())
    ]
    if args.skip_image_generation:
        print("[image] Skipped by command-line option.")
        pending = []
    elif pending and not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "FLUX generation requires a CUDA runtime for the final prototype. "
            "Use Google Colab with an L4 GPU, or pass --allow-cpu for an impractical CPU run."
        )

    generator = None
    if pending:
        from GeneratorFlux2KleinL4Colab import Flux2KleinL4ColoringPageGenerator

        generator = Flux2KleinL4ColoringPageGenerator(
            model_id=args.flux_model,
            quantization=args.flux_quantization,
            offload=args.flux_offload,
            enable_vae_tiling=args.vae_tiling,
            revision=args.flux_revision,
        )
        try:
            for index, record in enumerate(records):
                if record not in pending:
                    continue
                output_path = paths["images"] / f"{record['slug']}.png"
                metadata_path = paths["generation"] / f"{record['slug']}.json"
                sample_seed = args.seed + index
                print(f"[image] {record['source']['title']} (seed {sample_seed})")
                try:
                    generator.process_image(
                        image_path=record["source"]["image_path"],
                        output_path=str(output_path),
                        steps=args.flux_steps,
                        guidance_scale=args.guidance_scale,
                        max_side=args.max_side,
                        max_sequence_length=args.max_sequence_length,
                        seed=sample_seed,
                    )
                    metadata = dict(generator.last_run_metadata or {})
                    metadata.update(
                        {
                            "query": record["query"],
                            "title": record["source"]["title"],
                            "source_image": record["source"]["image_path"],
                            "output_image": str(output_path),
                            "created_at": utc_now(),
                        }
                    )
                    write_json(metadata_path, metadata)
                except Exception as exc:
                    record["errors"].append(f"Image generation failed: {exc}")
                    print(f"[image] Failed for {record['query']}: {exc}")
        finally:
            generator.cleanup()
            del generator

    for record in records:
        image_path = paths["images"] / f"{record['slug']}.png"
        generation_path = paths["generation"] / f"{record['slug']}.json"
        record["generated_image_path"] = str(image_path)
        record["generation_metadata_path"] = str(generation_path)
        if generation_path.exists():
            record["generation"] = read_json(generation_path)
        if not image_path.exists() and not args.skip_image_generation:
            record["errors"].append("No generated image is available.")


def pdf_stage(records, paths, args):
    if args.skip_pdf:
        print("[pdf] Skipped by command-line option.")
        return None

    concatenator = Concatenator()
    book_pages = []
    for record in records:
        image_path = Path(record["generated_image_path"])
        summary = record.get("summary", {}).get("summary")
        if not image_path.exists() or not summary:
            continue

        source = record["source"]
        page = {
            "image_path": str(image_path),
            "text": summary,
            "title": source["title"],
            "attribution": image_attribution(source),
            "source_url": source.get("page_url"),
        }
        page_path = paths["pages"] / f"{record['slug']}.pdf"
        if args.force or not page_path.exists():
            print(f"[pdf] {source['title']}")
            if not concatenator.create_book([page], page_path):
                record["errors"].append("PDF page creation failed.")
                continue
        record["pdf_path"] = str(page_path)
        book_pages.append(page)

    if not book_pages:
        print("[pdf] No complete pages were available for the combined book.")
        return None

    book_path = paths["run"] / "coloring_book.pdf"
    if args.force or not book_path.exists():
        print(f"[pdf] Combined book with {len(book_pages)} page(s)")
        if not concatenator.create_book(book_pages, book_path):
            return None
    return book_path


def manifest_record(record):
    return {
        key: value
        for key, value in record.items()
        if key not in {"source", "summary", "generation"}
    } | {
        "title": record.get("source", {}).get("title"),
        "source_revision_id": record.get("source", {}).get("revision_id"),
    }


def run_pipeline(args):
    started_at = utc_now()
    names = get_names(args)
    run_dir = Path(args.output_dir).resolve()
    paths = ensure_run_directories(run_dir)
    print(f"Run directory: {run_dir}")
    print(f"People: {', '.join(names)}")

    records = fetch_stage(
        names,
        paths,
        force=args.force,
        fuzzy_search=not args.no_fuzzy_search,
    )
    summarization_stage(records, paths, args)
    image_generation_stage(records, paths, args)
    book_path = pdf_stage(records, paths, args)

    manifest = {
        "schema_version": 1,
        "started_at": started_at,
        "completed_at": utc_now(),
        "runtime": runtime_info(),
        "configuration": {
            "names": names,
            "qwen_model": args.qwen_model,
            "qwen_revision": args.qwen_revision,
            "qwen_quantization": args.qwen_quantization,
            "flux_model": args.flux_model,
            "flux_revision": args.flux_revision,
            "flux_quantization": args.flux_quantization,
            "flux_steps": args.flux_steps,
            "guidance_scale": args.guidance_scale,
            "max_side": args.max_side,
            "seed": args.seed,
            "target_age": args.target_age,
            "summary_word_range": [args.summary_min_words, args.summary_max_words],
            "t4_safe_mode": args.t4_safe_mode,
        },
        "book_path": str(book_path) if book_path else None,
        "items": [manifest_record(record) for record in records],
    }
    manifest_path = run_dir / "manifest.json"
    write_json(manifest_path, manifest)
    print(f"Manifest: {manifest_path}")
    if book_path:
        print(f"Final book: {book_path}")
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Create grounded biographical coloring-book pages from a list of names."
    )
    parser.add_argument("--names", nargs="+", default=None, help="Person names to process.")
    parser.add_argument("--names-file", default=None, help="UTF-8 file with one name per line.")
    parser.add_argument("--output-dir", default="output/final_run")
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--qwen-revision", default=DEFAULT_QWEN_REVISION)
    parser.add_argument("--qwen-quantization", choices=["none", "8bit", "4bit"], default="none")
    parser.add_argument("--flux-model", default=DEFAULT_FLUX_MODEL)
    parser.add_argument("--flux-revision", default=DEFAULT_FLUX_REVISION)
    parser.add_argument("--flux-quantization", choices=["none", "8bit", "4bit"], default="none")
    parser.add_argument("--flux-offload", action="store_true")
    parser.add_argument("--vae-tiling", action="store_true")
    parser.add_argument("--flux-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-age", default="10-14")
    parser.add_argument("--summary-min-words", type=int, default=80)
    parser.add_argument("--summary-max-words", type=int, default=110)
    parser.add_argument("--no-fuzzy-search", action="store_true")
    parser.add_argument("--force", action="store_true", help="Regenerate existing stage outputs.")
    parser.add_argument("--skip-summarization", action="store_true")
    parser.add_argument("--skip-image-generation", action="store_true")
    parser.add_argument("--skip-pdf", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow impractically slow FLUX CPU inference.")
    parser.add_argument(
        "--t4-safe-mode",
        action="store_true",
        help=(
            "Apply the free-Colab T4 preset: Qwen 4-bit, FLUX 8-bit, "
            "640px maximum side, 256 prompt tokens, and no CPU offload."
        ),
    )
    args = parser.parse_args(argv)
    if args.t4_safe_mode:
        args.qwen_quantization = "4bit"
        args.flux_quantization = "8bit"
        args.flux_offload = False
        args.max_side = 640
        args.max_sequence_length = 256
        args.flux_steps = 4
        args.guidance_scale = 1.0
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.summary_min_words > args.summary_max_words:
        raise SystemExit("--summary-min-words must not exceed --summary-max-words")
    run_pipeline(args)


if __name__ == "__main__":
    main()
