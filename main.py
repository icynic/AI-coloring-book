"""End-to-end, resumable AI coloring-book pipeline.

The pipeline intentionally runs the text and image models in separate stages so
Qwen and FLUX never need to occupy GPU memory at the same time.
"""

from __future__ import annotations

import argparse
import csv
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
from source_text import SOURCE_POLICY_VERSION, text_sha256
from summary_validation import validate_summary


DEFAULT_QWEN_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_FLUX_MODEL = "black-forest-labs/FLUX.2-klein-4B"
DEFAULT_QWEN_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DEFAULT_FLUX_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
DEFAULT_SUMMARY_TARGET_WORDS = 95


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
    if not names:
        with (Path(__file__).parent / "evaluation/subjects.csv").open(encoding="utf-8", newline="") as stream:
            names = [row["name"] for row in csv.DictReader(stream)]
    for name in names:
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


def ensure_run_directories(run_dir, create=True):
    paths = {
        "run": run_dir,
        "source": run_dir / "sources",
        "source_images": run_dir / "sources" / "images",
        "summaries": run_dir / "summaries",
        "images": run_dir / "generated_images",
        "generation": run_dir / "generation_metadata",
        "pages": run_dir / "pages",
    }
    if create:
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
    return paths


def load_valid_summary(record, paths, args):
    summary_path = paths["summaries"] / f"{record['slug']}.json"
    summary = read_json(summary_path)
    validate_summary(summary, record["source"].get("summary"),
                     args.summary_min_words, args.summary_max_words)
    revision = summary.get("source_revision_id")
    if revision is not None and revision != record["source"].get("revision_id"):
        raise ValueError("Summary refers to a different Wikipedia revision.")
    source_hash = text_sha256(record["source"].get("summary") or "")
    if summary.get("source_text_sha256") or record["source"].get("source_policy_version") == SOURCE_POLICY_VERSION:
        if summary.get("source_text_sha256") != source_hash:
            raise ValueError("Source text changed (or the old summary has no source hash); regenerate its evidence IDs.")
    return summary


def resolve_summary_word_range(args, saved_range=(80, 110)):
    """Validate the biography length configured for this run."""
    lower = args.summary_min_words if args.summary_min_words is not None else saved_range[0]
    upper = args.summary_max_words if args.summary_max_words is not None else saved_range[1]
    if type(lower) is not int or type(upper) is not int or not 1 <= lower <= upper:
        raise ValueError("Summary word range must satisfy 1 <= minimum <= maximum.")
    if type(args.summary_target_words) is not int or not lower <= args.summary_target_words <= upper:
        raise ValueError("Summary target words must be an integer within the accepted word range.")
    args.summary_min_words, args.summary_max_words = lower, upper
    return [lower, upper]


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
            and cached_source.get("source_policy_version") == SOURCE_POLICY_VERSION
            and cached_source.get("source_text_sha256") == text_sha256(cached_source["summary"])
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
    pending = []
    for record in records:
        record.pop("summary", None)
        source_words = len((record["source"].get("summary") or "").split())
        if record["source"].get("source_policy_version") == SOURCE_POLICY_VERSION and source_words < args.summary_min_words:
            record["errors"].append(
                f"Source has only {source_words} words for a {args.summary_min_words}-word minimum biography. "
                "Provide more verified source material or explicitly revise the length policy; generation was skipped."
            )
            continue
        summary_path = paths["summaries"] / f"{record['slug']}.json"
        record["summary_path"] = str(summary_path)
        invalid_reason = None
        try:
            record["summary"] = load_valid_summary(record, paths, args)
        except (OSError, ValueError) as exc:
            invalid_reason = str(exc)
            if summary_path.exists():
                print(f"[summarize] Rejecting cache for {record['query']}: {exc}")
        if (invalid_reason or args.force) and not args.skip_summarization and record["source"].get("summary"):
            record.pop("summary", None)
            pending.append(record)
        elif invalid_reason:
            record["errors"].append(f"No valid summary: {invalid_reason}")

    # A known impossible/disabled biography should stop the stage before Qwen is loaded.
    if any(record.get("summary") is None and record["errors"] for record in records):
        return

    summarizer = None
    if pending:
        from Summarizer import Summarizer

        summarizer = Summarizer(
            model_name=args.qwen_model,
            quantization=args.qwen_quantization,
            revision=args.qwen_revision,
        )
        try:
            for pending_index, record in enumerate(pending):
                print(f"[summarize] {record['source']['title']}", flush=True)
                summary_path = paths["summaries"] / f"{record['slug']}.json"
                try:
                    summary = summarizer.summarize_with_evidence(
                        record["source"]["summary"],
                        target_age=args.target_age,
                        min_words=args.summary_min_words,
                        max_words=args.summary_max_words,
                        target_words=args.summary_target_words,
                    )
                    summary.update(
                        {
                            "query": record["query"],
                            "title": record["source"]["title"],
                            "source_revision_id": record["source"].get("revision_id"),
                            "source_text_sha256": text_sha256(record["source"]["summary"]),
                            "source_policy_version": record["source"].get("source_policy_version", 1),
                            "created_at": utc_now(),
                        }
                    )
                    validate_summary(summary, record["source"]["summary"],
                                     args.summary_min_words, args.summary_max_words)
                    write_json(summary_path, summary)
                    record["summary"] = summary
                except Exception as exc:
                    record["errors"].append(f"Summarization failed: {exc}")
                    print(f"[summarize] Failed for {record['query']}: {exc}")
                    failure_path = paths["run"] / "summary_failures" / f"{record['slug']}.json"
                    write_json(failure_path, {
                        "created_at": utc_now(), "error": str(exc),
                        "attempts": getattr(exc, "attempts", []),
                        "context": {"query": record["query"],
                                    "source_revision_id": record["source"].get("revision_id"),
                                    "source_text_sha256": text_sha256(record["source"]["summary"]),
                                    "model_id": args.qwen_model, "model_revision": args.qwen_revision,
                                    "quantization": args.qwen_quantization, "target_age": args.target_age,
                                    "requested_word_range": [args.summary_min_words, args.summary_max_words],
                                    "target_words": args.summary_target_words},
                    })
                    for remaining in pending[pending_index + 1:]:
                        remaining["errors"].append(
                            "Summarization not attempted because an earlier biography failed."
                        )
                    break
        finally:
            summarizer.cleanup()
            del summarizer


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
            "Use Google Colab with a T4 or L4 GPU, or pass --allow-cpu for an impractical CPU run."
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

    # Never publish an old or incomplete book as a successful new result.
    incomplete = False
    for record in records:
        record.pop("pdf_path", None)
        try:
            if record.get("summary") is None:
                raise ValueError("No validated biography is available; see the summarization error above.")
            validate_summary(record.get("summary"), record["source"].get("summary"),
                             args.summary_min_words, args.summary_max_words)
            if not Path(record["generated_image_path"]).is_file():
                raise ValueError("Generated image is missing.")
        except ValueError as exc:
            record["errors"].append(f"PDF not rebuilt: {exc}")
            incomplete = True
    if incomplete:
        print("[pdf] Book not rebuilt: some pages lack a valid biography or image. Existing PDFs are unchanged.")
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
        # PDFs are cheap to rebuild, and depend on both current text and images.
        # Existence alone cannot detect stale PDFs after changed text.
        print(f"[pdf] {source['title']}")
        if not concatenator.create_book([page], page_path):
            record["errors"].append("PDF page creation failed.")
            return None
        record["pdf_path"] = str(page_path)
        book_pages.append(page)

    if not book_pages:
        print("[pdf] No complete pages were available for the combined book.")
        return None

    book_path = paths["run"] / "coloring_book.pdf"
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
        "source_text_sha256": record.get("source", {}).get("source_text_sha256"),
        "source_policy_version": record.get("source", {}).get("source_policy_version", 1),
    }


def build_manifest(started_at, configuration, records, book_path=None):
    """Build a resumable checkpoint for both complete and failed runs."""
    return {
        "schema_version": 2,
        "started_at": started_at,
        "completed_at": utc_now(),
        "runtime": runtime_info(),
        "configuration": configuration,
        "book_path": str(book_path) if book_path else None,
        "items": [manifest_record(record) for record in records],
    }


def stop_before_stage(stage, next_stage, failed_records, manifest_path,
                      started_at, configuration, records):
    """Checkpoint an incomplete stage and stop before expensive downstream work."""
    if not failed_records:
        return

    manifest = build_manifest(started_at, configuration, records)
    write_json(manifest_path, manifest)
    details = []
    for record in failed_records:
        reasons = record.get("errors") or [f"Required {stage} output is missing."]
        reason_text = " | ".join(str(reason).rstrip(".") for reason in reasons)
        details.append(f"{record['query']}: {reason_text}")
    message = (
        f"[{stage}] Stage incomplete; stopping before {next_stage}. "
        f"{'; '.join(details)} Rerun the same command to resume."
    )
    print(message, flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    raise RuntimeError(message)


def run_configuration(args, names):
    keys = ("qwen_model", "qwen_revision", "qwen_quantization", "flux_model",
            "flux_revision", "flux_quantization", "flux_steps", "guidance_scale",
            "max_side", "max_sequence_length", "flux_offload", "vae_tiling",
            "seed", "target_age", "summary_target_words", "t4_safe_mode", "no_fuzzy_search")
    return {"names": names, **{key: getattr(args, key) for key in keys},
            "summary_word_range": [args.summary_min_words, args.summary_max_words]}


def run_pipeline(args):
    started_at = utc_now()
    names = get_names(args)
    run_dir = Path(args.output_dir).resolve()
    configuration = run_configuration(args, names)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        previous = read_json(manifest_path)
        if previous.get("schema_version") != 2 or previous.get("configuration") != configuration:
            raise ValueError("Existing run uses older code or different settings. Choose a new output directory.")
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("Nonempty output has no current run manifest. Choose a new output directory.")
    paths = ensure_run_directories(run_dir)
    # Save the configuration before downloads/models so interrupted runs can resume.
    write_json(manifest_path, {"schema_version": 2, "started_at": started_at,
                               "configuration": configuration, "book_path": None, "items": []})
    print(f"Run directory: {run_dir}")
    print(f"People: {', '.join(names)}")
    print(f"Summary word range: {args.summary_min_words}-{args.summary_max_words}")
    print(f"Summary soft target: {args.summary_target_words} words")

    records = fetch_stage(
        names,
        paths,
        force=args.force,
        fuzzy_search=not args.no_fuzzy_search,
    )
    source_text_required = not args.skip_summarization or not args.skip_pdf
    source_image_required = not args.skip_image_generation
    incomplete_sources = [
        record
        for record in records
        if (
            source_text_required
            and not record["source"].get("summary")
        ) or (
            source_image_required
            and (
                not record["source"].get("image_path")
                or not Path(record["source"]["image_path"]).is_file()
            )
        )
    ]
    stop_before_stage(
        "fetch", "loading Qwen", incomplete_sources, manifest_path,
        started_at, configuration, records,
    )

    summarization_stage(records, paths, args)
    summaries_required = not args.skip_summarization or not args.skip_pdf
    incomplete_summaries = [
        record for record in records
        if summaries_required and record.get("summary") is None
    ]
    stop_before_stage(
        "summarize", "loading FLUX", incomplete_summaries, manifest_path,
        started_at, configuration, records,
    )

    image_generation_stage(records, paths, args)
    generated_images_required = not args.skip_image_generation or not args.skip_pdf
    incomplete_images = [
        record for record in records
        if generated_images_required
        and not Path(record.get("generated_image_path") or "").is_file()
    ]
    stop_before_stage(
        "image", "building the PDF", incomplete_images, manifest_path,
        started_at, configuration, records,
    )

    book_path = pdf_stage(records, paths, args)

    manifest = build_manifest(started_at, configuration, records, book_path)
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
    parser.add_argument("--output-dir", default="output/final_run_v2")
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
    parser.add_argument("--summary-min-words", type=int, default=None,
                        help="Minimum biography words (default: 80).")
    parser.add_argument("--summary-max-words", type=int, default=None,
                        help="Maximum biography words (default: 110).")
    parser.add_argument("--summary-target-words", type=int, default=DEFAULT_SUMMARY_TARGET_WORDS,
                        help="Soft biography target within the accepted range (default: 95).")
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
    try:
        resolve_summary_word_range(args)
    except ValueError as exc:
        parser.error(str(exc))
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
    result = run_pipeline(args)
    if not args.skip_pdf and not result.get("book_path"):
        raise SystemExit("Run incomplete. Inspect manifest.json; rerun the same command to resume.")
    return result


if __name__ == "__main__":
    main()
