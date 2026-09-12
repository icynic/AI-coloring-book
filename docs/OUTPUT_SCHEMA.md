# Output and provenance schema

Each run uses one output directory. Files are reusable checkpoints as well as
the provenance record for the final evaluation.

## `sources/<person>.json`

Contains the original query, resolved Wikipedia title, selected source prose,
Wikipedia page and revision IDs, revision timestamp, retrieval timestamp,
portrait URL and local path, Wikimedia file title/revision information, artist,
credit, license, license URL, and SHA-256 hashes of the source text and image.

Source policy 2 stores the actual Qwen input in the legacy `summary` field and
keeps the full introductory text separately as `lead_summary`. It adds
`source_text_kind`, `source_policy_version`, `source_word_count`,
`source_max_words` (800), and `source_passages` (section paths, paragraph indices,
selected text). `text_source_url` identifies the requested article revision;
`text_retrieved_at` records the new text retrieval, separately from portrait
retrieval. The selected text and its hash are the frozen evidence snapshot;
rendered templates can change even when the article revision is unchanged.

`source_refresh_manifest.json` and `manifest.json.source_refreshes` record
previous/new source hashes, word counts, revisions, selected sections, and errors.
Source refresh preserves image-related fields and files and invalidates the
current book pointer before replacing any source. Backups retain the earlier run.

## `summaries/<person>.json`

Contains the generated biography, supporting source sentence IDs and sentence
text, word count, target age, requested length, Qwen model ID and revision,
quantization and model-load time,
raw model response, source revision, and creation time.

New summaries also record `source_text_sha256` and `source_policy_version`.
Policy-2 source caches require a matching summary source hash; a matching page
revision alone is insufficient after expanding the selected text. Legacy
summaries without hashes are regenerated when their source is upgraded.

New records include `validation_version`, `generation_settings` (thinking
disabled, deterministic decoding, output-token budget), and
`generation_attempts` (including any rejected output and validation error).
Validation checks content type, placeholder text, word length and evidence IDs;
it does not replace human factual review.

`postprocessing_version: 1` supports modest overlength answers by retaining a
complete-sentence prefix. When applied, `length_adjustment` records the original
biography, original/final word counts, removed tail, method and accepted range.
The raw model response is unchanged. Human review must check that the shorter
biography retains the important facts.

Recovered failed answers include `recovery`: the failure-log path/hash,
recovery time and provenance method. New failure logs carry a `context` with
source hash/revision, subject, model settings and requested word range. Legacy
logs may be bound to the matching repair's source metadata and time interval.
The model's original creation time is preserved, and no model latency is
invented for offline recovery.

`summary_failures/<person>.json` records exhausted retries. Old evaluation
artifacts are kept in `backups/<timestamp>/` when replacements are written.
`repair_manifest.json` records summary-only recovery; `manifest.json` retains
the original image-run runtime and adds `summary_repairs`.
Each new repair record includes `previous_summary_word_range` and
`summary_word_range`. Explicit CLI length overrides update the manifest's
`configuration.summary_word_range` for subsequent repair resumes. The earlier
configuration is retained in backups and repair history; model/image settings
are not changed. Cached biographies keep their original requested length, so
inspect per-summary settings when reporting prompt consistency.

The evidence is an audit aid, not a correctness guarantee. The final evaluation
must still verify each atomic claim against the saved source.

## Optional source-grounded model review

With `--verify-summaries`, a usable biography also requires `summary_review`:

- `version: 2`, `status: model_verified`, exact final summary/source SHA-256 hashes.
- `policy`: target age, effective whole-biography word range, source-Marburg keyword
  trigger, and `editorial_issues: warnings_only`.
- `reviewer`: model/revision/quantization, same-model and fresh-chat flags.
- `initial_draft`: the complete input biography record before refinement.
- `events`: sequential verification/revision requests, raw model outputs, parsed
  responses, validation errors, acceptance problems, editorial warnings, token
  ceilings, elapsed time and optional `output_text_tokens` (decoded text retokenized
  without special tokens, not the original generation IDs).
- `content_revisions`, `max_revisions`, deterministic generation settings.
- `final_review`: compact ordered verdicts per biography sentence with `sentence_id`,
  `status`, `source_sentence_ids`, and a short `reason`, plus `issues`.
- `resolved_evidence`: each biography sentence paired with full verbatim source
  sentences retrieved by the program, not quotations composed by the model.
- `editorial_warnings`: final `age_style` and `unnecessary_detail` issues.

Each verdict checks all factual details in its biography sentence. The program
checks coverage, ID bounds, and retrieved-evidence consistency, not semantic
entailment. Any `partial`, `unsupported`, or `source_conflict` verdict blocks
acceptance; `marburg_missing` issues also block. Style and unnecessary-detail issues
are warnings only: they do not trigger revisions or block PDF publication. Conflict
verdicts need at least two distinct source sentence IDs. Final supporting source
IDs/sentences are rebuilt from the accepted review, replacing the old citation
superset. Source omissions are allowed unless the written claim is unsupported.
The model is not asked to assess word counts: the program checks the full biography.
A changed summary, source or editorial policy invalidates that review. The source
sentence segmentation remains unchanged.

Defaults are one content revision, review ceilings of 1024/2048 tokens, revision
ceilings of 512/1024, and two format attempts per stage. The CLI can explicitly
request two revisions. Successful version-1 caches retain their version and are
reused if their original strict no-issue review, hashes, policy, and verbatim quotes
still validate. A new manifest's `summary_review_version: 2` records the latest
repair implementation, not necessarily every reused record's version. Failed old
reviews remain diagnostic logs and are not converted into passed reviews.

Top-level `raw_model_response` and `generation_attempts` retain initial-generation
provenance. Revised text and raw editor responses are tracked in `summary_review`;
do not treat the initial raw response as the final biography. Prior length edits
remain in `initial_draft`; a new length adjustment, if used, belongs to the revision.

`refine_biographies.py` derives a separate run without copying old PDFs. It adds
`original_summaries/` and `refinement_origin.json` (source path, manifest hash,
input fingerprint, per-input file hashes and preparation time). The original run
is not edited. The derived manifest retains original image runtime/provenance,
adds `refinement_origin`, and records refinement runtime in `summary_repairs`.
Enabled review/version/revision-limit settings are persisted in `configuration`.

The status deliberately says **model_verified**, not factually correct: this is
same-model feedback, not an independent accuracy measurement or human review.

## `generation_metadata/<person>.json`

Contains the FLUX model ID and revision, precision and quantization, model-load
time, complete prompt, seed,
steps, guidance scale, output dimensions, elapsed generation time, peak
allocated CUDA memory, input path, output path, and creation time.

## `manifest.json`

Contains the final run configuration, Python/PyTorch/GPU runtime information,
combined book path, and a compact record for every requested person. Errors are
retained in the per-person `errors` list rather than dropping failed samples.

The manifest schema is versioned with `schema_version`. New fields may be added
without changing the meaning of existing fields.
