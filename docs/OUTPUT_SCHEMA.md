# Output and provenance schema

Each complete pipeline run has one output directory and a version-2
`manifest.json`. Its configuration is saved before downloads/models begin.
Resuming requires the same configuration and schema; older runs are kept
read-only by using a new directory.

## Sources

`sources/<slug>.json` stores the query, resolved title, page/revision IDs,
timestamps, source URL, selected article prose in `summary`, source text hash
and downloaded portrait metadata. `sources/images/` holds the input portraits.

Source policy 2 selects at most 800 words of article paragraphs, prioritizing
the lead, biographical sections and Marburg passages. `lead_summary`,
`source_passages`, `source_word_count`, `source_policy_version` and
`text_source_url` record the actual input and selection provenance. A revision
ID does not freeze external templates; the saved text/hash is the evidence
snapshot.

Portrait fields include the Wikimedia file title, URL, artist, credit, license,
license URL, timestamp and hashes. Keep attribution and source licensing
requirements with the submitted software/data.

## Biographies

`summaries/<slug>.json` stores the generated text, word count, requested range,
target age, supporting sentence IDs and their resolved source text, source
revision/hash, model ID/revision/quantization and generation/load timing.

`raw_model_response`, `generation_settings` and `generation_attempts` retain
the original output and any validation retry. Thinking is disabled. At most
two generation attempts are used for JSON, placeholder, length and evidence
validation; there is no model factuality reviewer.

A modest overlength answer can retain an unchanged complete-sentence prefix.
`length_adjustment` records the original text, word counts and removed tail.
This is deterministic postprocessing, not a factual correction.

Exhausted attempts go to `summary_failures/<slug>.json` with their source/model
context. These are diagnostic logs, never accepted caches or an offline
recovery path. Mechanical validation does not establish factual accuracy.

## Images and PDFs

`generated_images/<slug>.png` stores FLUX outputs.
`generation_metadata/<slug>.json` records model/revision, precision,
quantization, prompt, seed, steps, dimensions, elapsed time and peak GPU memory.

`pages/<slug>.pdf` contains one A4 page;
`coloring_book.pdf` contains the complete book. PDFs are rebuilt from validated
biographies and available images. Incomplete runs have a null manifest
`book_path`; any older PDF on disk is not advertised as the current result.

## Manifest

`manifest.json` contains configuration, runtime, current book path and every
requested subject with its paths, source provenance and errors. No failed
subject is silently dropped. It does not create repair/refinement histories or
per-rerun backup directories.

Existing original experiment files and legacy backup histories are retained,
but the cleaned pipeline does not edit or migrate them.
