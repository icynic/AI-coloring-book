"""Audit saved biographies without Qwen inference or invented human ratings.

Deterministic integrity/readability checks are separate from optional, explicit
source-grounded claim annotations.  Semantic support is never inferred merely
from the presence of an evidence ID.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import statistics
import sys
import unicodedata


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from source_text import text_sha256
from summary_validation import split_source_sentences, validate_summary


CLAIM_STATUSES = {"supported", "partial", "unsupported", "source_inconsistent"}
WORD_PATTERN = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)


def estimated_syllables(word):
    """English vowel-group heuristic; proper names can make it inaccurate."""
    normalized = unicodedata.normalize("NFKD", word.lower())
    word = re.sub(r"[^a-z]", "", normalized)
    if not word:
        return 0
    groups = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and not word.endswith(("le", "ye")) and groups > 1:
        groups -= 1
    return max(1, groups)


def readability_metrics(text):
    sentences = split_source_sentences(text)
    words = WORD_PATTERN.findall(text)
    sentence_lengths = [len(WORD_PATTERN.findall(sentence)) for sentence in sentences]
    syllables = sum(estimated_syllables(word) for word in words)
    words_per_sentence = len(words) / len(sentences) if sentences else 0.0
    syllables_per_word = syllables / len(words) if words else 0.0
    return {
        "lexical_word_count": len(words),
        "sentence_count": len(sentences),
        "mean_sentence_words": round(words_per_sentence, 3),
        "max_sentence_words": max(sentence_lengths, default=0),
        "sentences_over_25_words": sum(length > 25 for length in sentence_lengths),
        "estimated_syllable_count": syllables,
        "estimated_flesch_reading_ease": round(
            206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word, 3
        ) if words else None,
        "estimated_flesch_kincaid_grade": round(
            0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59, 3
        ) if words else None,
        "syllable_method": "english_vowel_group_heuristic_v1",
        "contains_marburg": bool(re.search(r"\bMarburg\b", text, re.I)),
    }


def check_integrity(summary, source, word_range):
    errors = []
    actual_hash = text_sha256(source["summary"])
    checks = {
        "source_metadata_hash_matches_text": source.get("source_text_sha256") == actual_hash,
        "summary_source_hash_matches_text": summary.get("source_text_sha256") == actual_hash,
        "source_revision_matches": summary.get("source_revision_id") == source.get("revision_id"),
    }
    try:
        validate_summary(summary, source["summary"], *word_range)
        checks["schema_length_and_evidence_match"] = True
    except ValueError as exc:
        checks["schema_length_and_evidence_match"] = False
        errors.append(str(exc))
    errors.extend(key for key, valid in checks.items() if not valid and key != "schema_length_and_evidence_match")
    return {"valid": all(checks.values()), "checks": checks, "errors": errors}


def validate_annotation(annotation, summary, source):
    if annotation["summary_text_sha256"] != text_sha256(summary["summary"]):
        raise ValueError(f"Stale claim audit for {annotation['slug']}: summary text changed.")
    if annotation["source_text_sha256"] != text_sha256(source["summary"]):
        raise ValueError(f"Stale claim audit for {annotation['slug']}: source changed.")
    source_sentences = split_source_sentences(source["summary"])
    if not annotation.get("claims"):
        raise ValueError(f"No reviewed claims for {annotation['slug']}.")
    claims = []
    for index, claim in enumerate(annotation["claims"], 1):
        status = claim["status"]
        if status not in CLAIM_STATUSES:
            raise ValueError(f"Invalid claim status {status}.")
        ids = claim.get("evidence_ids", [])
        if any(type(item) is not int or not 1 <= item <= len(source_sentences) for item in ids):
            raise ValueError(f"Invalid reviewed evidence IDs for {annotation['slug']} claim {index}.")
        if status in {"supported", "partial", "source_inconsistent"} and not ids:
            raise ValueError(f"Missing reviewed evidence for claim {index}.")
        claims.append({
            **claim,
            "claim_id": index,
            "evidence_sentences": [source_sentences[item - 1] for item in ids],
            "evidence_all_in_generated_citations": bool(ids) and set(ids).issubset(
                summary["supporting_source_sentence_ids"]
            ),
        })
    counts = {status: sum(claim["status"] == status for claim in claims) for status in sorted(CLAIM_STATUSES)}
    supported = [claim for claim in claims if claim["status"] == "supported"]
    used_ids = {item for claim in claims for item in claim.get("evidence_ids", [])}
    return {
        "claim_count": len(claims),
        "status_counts": counts,
        "strict_supported_claim_rate": counts["supported"] / len(claims) if claims else None,
        "supported_claims_with_complete_generated_citations": sum(
            claim["evidence_all_in_generated_citations"] for claim in supported
        ),
        "generated_citation_ids_unused_by_reviewed_claims": sorted(
            set(summary["supporting_source_sentence_ids"]) - used_ids
        ),
        "claims": claims,
        "content_notes": annotation.get("content_notes", []),
    }


def run(args):
    run_dir = Path(args.flux_run).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    word_range = manifest["configuration"]["summary_word_range"]
    annotation_document = (
        json.loads(Path(args.annotations).read_text(encoding="utf-8"))
        if args.annotations else None
    )
    annotations = {
        record["slug"]: record for record in annotation_document["records"]
    } if annotation_document else {}
    if annotation_document and len(annotations) != len(annotation_document["records"]):
        raise ValueError("Duplicate subject in claim annotations.")
    items = []
    review_lines = ["# Biography review packet", "", "Source IDs use the generation-time sentence splitter.", ""]
    for item in manifest["items"]:
        slug = item["slug"]
        summary = json.loads((run_dir / "summaries" / f"{slug}.json").read_text(encoding="utf-8"))
        source = json.loads((run_dir / "sources" / f"{slug}.json").read_text(encoding="utf-8"))
        source_sentences = split_source_sentences(source["summary"])
        result = {
            "slug": slug,
            "name": item.get("query", slug),
            "summary": summary["summary"],
            "summary_text_sha256": text_sha256(summary["summary"]),
            "source_text_sha256": text_sha256(source["summary"]),
            "source_revision_id": source["revision_id"],
            "source_url": source.get("text_source_url", source.get("page_url")),
            "word_count": len(summary["summary"].split()),
            "word_range": word_range,
            "integrity": check_integrity(summary, source, word_range),
            "readability": readability_metrics(summary["summary"]),
            "generation_attempts": len(summary.get("generation_attempts", [])),
            "length_adjusted": "length_adjustment" in summary,
            "source_fragment_sentence_ids": [
                index for index, sentence in enumerate(source_sentences, 1)
                if re.fullmatch(r"[A-Z]\.", sentence)
            ],
            "claim_audit": validate_annotation(annotations[slug], summary, source) if slug in annotations else None,
        }
        items.append(result)
        review_lines.extend([
            f"## {result['name']}", "", summary["summary"], "",
            f"Generated evidence IDs: {summary['supporting_source_sentence_ids']}", "",
            f"Source hash: {result['source_text_sha256']}", "",
        ])
        review_lines.extend(f"{index}. {sentence}" for index, sentence in enumerate(source_sentences, 1))
        review_lines.append("")
    if annotation_document and set(annotations) != {item["slug"] for item in items}:
        raise ValueError("Claim annotations must cover exactly the saved subjects.")
    total_claims = sum(item["claim_audit"]["claim_count"] for item in items if item["claim_audit"])
    claim_counts = Counter()
    for item in items:
        if item["claim_audit"]:
            claim_counts.update(item["claim_audit"]["status_counts"])
    overview = {
        "biographies": len(items),
        "integrity_passed": sum(item["integrity"]["valid"] for item in items),
        "min_word_count": min(item["word_count"] for item in items),
        "max_word_count": max(item["word_count"] for item in items),
        "mean_word_count": statistics.fmean(item["word_count"] for item in items),
        "marburg_mentioned": sum(item["readability"]["contains_marburg"] for item in items),
        "mean_estimated_grade": statistics.fmean(item["readability"]["estimated_flesch_kincaid_grade"] for item in items),
        "reviewed_claims": total_claims,
        "claim_status_counts": dict(claim_counts),
        "strict_supported_claim_rate": claim_counts["supported"] / total_claims if total_claims else None,
    }
    document = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "flux_run": str(run_dir),
        "claim_review_provenance": annotation_document.get("provenance") if annotation_document else None,
        "overview": overview,
        "limitations": [
            "Rule checks validate stored provenance and structure, not semantic entailment.",
            "Claim labels are explicit agent-assisted annotations, not an independent human evaluation or an automatic NLI metric.",
            "Support is relative to saved Wikipedia input, which can contain factual conflicts.",
            "Flesch metrics use estimated English syllables and are approximate, especially for proper names and specialist vocabulary.",
            "Readability formulas do not establish suitability for ages 10-14.",
        ],
        "readability_references": [
            {"citation": "Flesch, R. (1948). A new readability yardstick.", "url": "https://doi.org/10.1037/h0057532"},
            {"citation": "Kincaid, J. P., Fishburne, R. P., Rogers, R. L., and Chissom, B. S. (1975). Derivation of new readability formulas for Navy enlisted personnel.", "url": "https://stars.library.ucf.edu/istlibrary/56/"},
        ],
        "readability_method": {
            "word_count_for_length": "Whitespace-separated tokens; identical to generation validation.",
            "words_for_readability": "Unicode letter sequences; dates are excluded and hyphenated terms are separated.",
            "reading_ease_formula": "206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word",
            "grade_formula": "0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59",
            "syllables": "English vowel-group heuristic with silent-e adjustment; not a pronunciation dictionary.",
        },
        "items": items,
    }
    lines = ["# Biography audit", "", "No biographies or PDFs were modified.", "",
             "Integrity checks are deterministic. Claim labels come from one explicit agent-assisted source review, with zero independent human raters.",
             "Supported means entailed by the saved input. Partial means an unsupported qualifier or causal relation. Source-inconsistent means saved passages conflict.", "",
             "| Biography | Words | Integrity | Mean sentence words | Estimated grade | Marburg | Strictly supported claims |",
             "| --- | ---: | --- | ---: | ---: | --- | ---: |"]
    for item in items:
        audit = item["claim_audit"]
        support = f"{audit['status_counts']['supported']}/{audit['claim_count']}" if audit else "not reviewed"
        read = item["readability"]
        lines.append(f"| {item['name']} | {item['word_count']} | {'pass' if item['integrity']['valid'] else 'fail'} | {read['mean_sentence_words']:.1f} | {read['estimated_flesch_kincaid_grade']:.1f} | {'yes' if read['contains_marburg'] else 'no'} | {support} |")
    lines.extend(["", "## Overview", "", f"```json\n{json.dumps(overview, ensure_ascii=False, indent=2)}\n```", "", "## Findings", ""])
    for item in items:
        audit = item["claim_audit"]
        if audit:
            for claim in audit["claims"]:
                if claim["status"] != "supported":
                    lines.append(f"- {item['name']} ({claim['status']}): {claim['claim']} — {claim.get('note', '')}")
            lines.extend(f"- {item['name']}: {note}" for note in audit["content_notes"])
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in document["limitations"])
    lines.extend(["", "## Readability method and references", "",
                  "Length uses whitespace tokens. Readability uses Unicode letter words, excluding numeric dates.",
                  "Syllables are estimated by a documented heuristic, not a pronunciation dictionary.", ""])
    lines.extend(f"- [{reference['citation']}]({reference['url']})" for reference in document["readability_references"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "biography_audit.json").write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "biography_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "review_packet.md").write_text("\n".join(review_lines) + "\n", encoding="utf-8")
    audit_method = (
        " and a qualitative, agent-assisted source-grounding audit"
        if total_claims else ""
    )
    marburg_result = (
        f"Marburg was explicitly mentioned in {overview['marburg_mentioned']}/{overview['biographies']} biographies."
    )
    if overview['marburg_mentioned'] < overview['biographies']:
        marburg_result += " This reveals incomplete preservation of the collection's local educational theme."
    paper_lines = ["# Draft text-evaluation section", "",
                   f"We evaluated {overview['biographies']} saved biographies using model-free consistency checks{audit_method}. "
                   "The consistency checks verified the accepted word range, stored source hashes, Wikipedia revision identifiers, evidence-ID bounds, and exact agreement between stored supporting sentences and the source. "
                   f"{overview['integrity_passed']}/{overview['biographies']} biographies passed these mechanical checks, with {overview['min_word_count']}-{overview['max_word_count']} words (mean {overview['mean_word_count']:.1f}). "
                   + marburg_result, ""]
    if total_claims:
        paper_lines.append(
            f"A single Codex-assisted review decomposed the outputs into {total_claims} checkable propositions and attached evidence from the saved input. "
            f"Of these, {claim_counts['supported']} were supported, {claim_counts['partial']} partially supported, {claim_counts['unsupported']} unsupported, and {claim_counts['source_inconsistent']} conflicted with another saved source passage. "
            "These labels are presented as a qualitative audit rather than independent human factuality judgments or automatic entailment scores. "
            "Support is relative to the saved input, not independently verified historical truth; specific findings and evidence appear in biography_audit.md."
        )
        paper_lines.append("")
    else:
        paper_lines.extend([
            "No claim annotations were supplied; these results do not measure factual support.", "",
        ])
    paper_lines.append(
        "We additionally recorded sentence lengths and approximate Flesch readability measures. Syllables were estimated with a documented English heuristic, so proper names and foreign titles can distort the scores. "
        "Neither these formulas nor structural evidence checks establish suitability for ages 10-14. No independent human readability study was conducted, and no significance test was applied to the qualitative claim labels."
    )
    (output_dir / "report_text_evaluation.md").write_text("\n".join(paper_lines) + "\n", encoding="utf-8")
    print(json.dumps(overview, indent=2))
    print(f"Biography audit: {output_dir}")
    return document


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flux-run", required=True)
    parser.add_argument("--output-dir", default="evaluation/biography_results")
    parser.add_argument("--annotations")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
