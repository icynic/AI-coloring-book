"""Recover the final logged model answer only when its source provenance matches."""

from datetime import datetime, timezone
import json
from pathlib import Path

from source_text import text_sha256
from summary_validation import fit_summary_length, parse_response, split_source_sentences, validate_summary


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def recover_failed_summary(record, run_dir, min_words, max_words):
    run_dir = Path(run_dir)
    failure_path = run_dir / "summary_failures" / f"{record['slug']}.json"
    failure_raw = failure_path.read_text(encoding="utf-8")
    failure = json.loads(failure_raw)
    source = record["source"]
    source_hash = text_sha256(source["summary"])
    context = failure.get("context")
    provenance_method = "failure_context"
    if context is None:
        # Older logs lack explicit context. Accept them only if the last repair
        # record binds this same failure, time interval, source hash and revision.
        repair = _read(run_dir / "repair_manifest.json")
        item = next((item for item in repair["items"] if item.get("slug") == record["slug"]), None)
        if not item or not failure.get("error") or not any(failure["error"] in error for error in item.get("errors", [])):
            raise ValueError("Failure log is not associated with the saved repair run.")
        dates = [datetime.fromisoformat(value) for value in
                 (repair["started_at"], failure["created_at"], repair["completed_at"])]
        if any(date.tzinfo is None for date in dates) or not dates[0] <= dates[1] <= dates[2]:
            raise ValueError("Failure log is outside the saved repair time interval.")
        config = _read(run_dir / "manifest.json")["configuration"]
        context = {"query": item["query"], "source_revision_id": item["source_revision_id"],
                   "source_text_sha256": item["source_text_sha256"],
                   "model_id": config["qwen_model"], "model_revision": config["qwen_revision"],
                   "quantization": config["qwen_quantization"], "target_age": config["target_age"],
                   "requested_word_range": repair["summary_word_range"]}
        provenance_method = "repair_manifest_interval"
    if (context.get("query") != record["query"]
            or context.get("source_text_sha256") != source_hash
            or context.get("source_revision_id") != source.get("revision_id")):
        raise ValueError("The failed model answer belongs to a different source or subject.")
    attempts = failure.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("No failed model answer was saved.")
    raw = attempts[-1]["raw_model_response"]
    summary, evidence = parse_response(raw)
    result = fit_summary_length({"summary": summary, "supporting_source_sentence_ids": evidence},
                                source["summary"], min_words, max_words)
    sentences = split_source_sentences(source["summary"])
    result.update({
        "query": record["query"], "title": source["title"],
        "supporting_source_sentences": [sentences[index - 1] for index in evidence],
        "word_count": len(result["summary"].split()),
        "source_revision_id": source.get("revision_id"), "source_text_sha256": source_hash,
        "source_policy_version": source.get("source_policy_version", 1),
        "model_id": context["model_id"], "model_revision": context["model_revision"],
        "quantization": context["quantization"], "target_age": context["target_age"],
        "requested_word_range": context["requested_word_range"],
        "raw_model_response": raw, "generation_attempts": attempts,
        "created_at": failure["created_at"], "validation_version": 1, "postprocessing_version": 1,
        "recovery": {"method": "last_failed_answer", "provenance_method": provenance_method,
                     "failure_log": f"summary_failures/{record['slug']}.json",
                     "failure_log_sha256": text_sha256(failure_raw),
                     "recovered_at": datetime.now(timezone.utc).isoformat()},
    })
    validate_summary(result, source["summary"], min_words, max_words)
    return result
