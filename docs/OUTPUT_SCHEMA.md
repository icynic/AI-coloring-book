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
