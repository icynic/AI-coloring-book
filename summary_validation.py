"""Model-free validation shared by generation, cache loading, and recovery."""

import json
import re


def split_source_sentences(text):
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", cleaned)
        if sentence.strip()
    ]


def validate_fields(record):
    if not isinstance(record, dict):
        raise ValueError("Expected a JSON object.")
    summary = record.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("summary must be a non-empty string.")
    summary = summary.strip()
    if not any(char.isalpha() for char in summary) or summary.casefold() in {
        "null", "none", "n/a", "tbd", "summary", "placeholder",
    }:
        raise ValueError("summary is a placeholder, not a biography.")
    if "<think>" in summary or "</think>" in summary or summary.startswith("Thinking Process:"):
        raise ValueError("summary contains reasoning instead of a final biography.")
    evidence = record.get("supporting_source_sentence_ids")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("supporting_source_sentence_ids must be a non-empty list.")
    if any(type(item) is not int or item < 1 for item in evidence):
        raise ValueError("Evidence IDs must be positive integers.")
    return summary, sorted(set(evidence))


def parse_response(raw_text):
    """Parse the whole final answer, never search arbitrary prose for braces."""
    if not isinstance(raw_text, str):
        raise ValueError("Expected a text response.")
    answer = raw_text.strip()
    if answer.startswith("<think>"):
        if "</think>" not in answer:
            raise ValueError("Reasoning was truncated before a final answer.")
        answer = answer.split("</think>", 1)[1].strip()
    if answer.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", answer, flags=re.DOTALL)
        if not match:
            raise ValueError("The JSON code fence is incomplete.")
        answer = match.group(1).strip()
    try:
        parsed = json.loads(answer)
    except json.JSONDecodeError as exc:
        raise ValueError("Expected only a complete final JSON object; prose or truncation found.") from exc
    return validate_fields(parsed)


def validate_summary(record, source_text, min_words=80, max_words=110):
    summary, evidence = validate_fields(record)
    word_count = len(summary.split())
    if not min_words <= word_count <= max_words:
        raise ValueError(f"Biography has {word_count} words; required {min_words}-{max_words}.")
    sentences = split_source_sentences(source_text)
    if any(index > len(sentences) for index in evidence):
        raise ValueError(f"Evidence ID exceeds the {len(sentences)} source sentences.")
    if "word_count" in record and record["word_count"] != word_count:
        raise ValueError("Stored word_count does not match the biography.")
    if "supporting_source_sentences" in record:
        expected = [sentences[index - 1] for index in evidence]
        if record["supporting_source_sentences"] != expected:
            raise ValueError("Stored supporting sentences do not match the saved source.")
    return word_count
