# Output and provenance schema

This guide describes the current core pipeline's saved files. See the
[main README](../README.md) for entry points, [Colab guide](COLAB.md) for
generation/recovery, and [evaluation guide](../evaluation/README.md) for
post-generation analysis. All paths below are relative to a run directory.

## File layout and portability

```text
<run-directory>/
  manifest.json
  coloring_book.pdf
  sources/<slug>.json
  sources/images/<portrait-file>
  summaries/<slug>.json
  generated_images/<slug>.png
  generation_metadata/<slug>.json
  pages/<slug>.pdf
  summary_failures/<slug>.json   # diagnostic record if summary generation fails
```

Slugs derive from the requested name, replacing unsupported filename characters
with underscores; redirects can yield a different article title. For example,
the query `K. Ferdinand Braun` uses `K._Ferdinand_Braun` in filenames. Failure logs
can remain after a successful resumed run; inspect the current manifest and
validated summaries, not the mere presence of that directory.

Download this entire tree without editing its JSON or images. The delivered
`output/final_run_v2` retains original `/content/drive/...` paths in metadata.
For local inspection use the corresponding files in the downloaded tree, not
those historical absolute paths. Evaluation scripts resolve their inputs from
the selected local run directory. Core `main.py` source-cache reuse checks the
stored image path, so a folder relocated to Windows is **not** guaranteed to
resume offline. Do not rewrite historical metadata to make it look like a new run.

## Manifest and acceptance checklist

The version-2 `manifest.json` is written before downloads/model loading, then
replaced at the end of a normally completed pipeline invocation. An exception
before that point can leave the initial manifest with no items; it is not a
completion record.

| Field | Meaning |
| --- | --- |
| `schema_version` | Current manifest schema: `2`. |
| `configuration` | Ordered names, model IDs/revisions, quantization, image settings, seed, target age, accepted word range, soft word target, and preset/search settings. |
| `runtime` | Python, platform, PyTorch, CUDA availability and device, when the invocation reaches its final manifest write. |
| `started_at`, `completed_at` | UTC timestamps for that invocation, not all earlier attempts in a resumed experiment. |
| `book_path` | Current combined PDF path, or `null` when no complete book was built. |
| `items` | One record per requested subject, with query, slug, resolved title, artifact paths, source provenance, and `errors`. |

Accept the default eight-person prototype only after checking:

1. The complete pipeline process exits with code 0, without skip-stage flags.
2. The manifest lists all eight intended subjects in order and every `errors`
   list is empty. For a custom or smoke-test run, check its actual requested count.
3. `book_path` is non-null, and the corresponding local `coloring_book.pdf` exists.
4. The combined PDF has one page per requested subject, each with the expected
   title, drawing, and nonempty biography; the default collection has eight pages.
5. Individual PDFs and the source, summary, and image files are present for all
   subjects. The mechanical biography audit reports valid source/summary integrity.

These establish operational completion, not factual correctness, portrait
identity, age suitability, or completeness of attribution. A file left from an
older invocation is not sufficient evidence of success.

Resumption requires an identical schema and run configuration, including the
ordered names and summary soft target. Old schemas, changed settings, or unknown nonempty directories
are rejected before the new manifest is written. Use a new empty directory for
a changed experiment. Configuration comparison does not freeze prompt versions
or identify every historical code revision.

## Sources

`sources/<slug>.json` stores the query, resolved title, page/revision IDs,
timestamps, source URL, selected article prose in `summary`, source text hash
and downloaded portrait metadata. `sources/images/` holds the input portraits.
Here `summary` is the selected Wikipedia input, **not** the generated biography;
the generated text is in `summaries/<slug>.json`.

Source policy 2 selects at most 800 words of article paragraphs, prioritizing
the lead, biographical sections and Marburg passages. `lead_summary`,
`source_passages`, `source_word_count`, `source_policy_version` and
`text_source_url` record the actual input and selection provenance. A revision
ID does not freeze external templates; the saved text/hash is the evidence
snapshot. Selection can omit major achievements and can include conflicting
statements; provenance does not establish historical correctness.

Portrait fields include the Wikimedia file title, URL, artist, credit, license,
license URL, timestamp and hashes, where available. Keep these resource records
with the submitted software/data; missing fields are not inferred to be complete.
Source-cache reuse requires a valid saved text hash, current source policy, an
existing stored portrait path, and a recorded image hash. It does not recompute
the image's byte hash during core cache reuse or assess portrait suitability.

## Biographies

`summaries/<slug>.json` stores the generated text, word count, requested range,
target age, supporting sentence IDs and their resolved source text, source
revision/hash, model ID/revision/quantization and generation/load timing.

Evidence IDs are one-based indexes into the shared sentence splitter's output
for the exact saved source text. They are not Wikipedia citation numbers.
`supporting_source_sentences` must match the source sentences selected by those
IDs. Validation also checks a non-placeholder biography, integer IDs within
bounds, the requested whitespace word range, and stored word counts when present.
Cache acceptance additionally checks the source text hash and any recorded
revision. None of these checks determines whether a sentence entails a claim.

`raw_model_response`, `generation_settings` and `generation_attempts` retain
the original output and any validation retry. Thinking is disabled. At most
three generation attempts are used for JSON, placeholder, length and evidence
validation in the current implementation; there is no model factuality reviewer.
Recorded attempts do not necessarily recover all earlier invocations of a
resumed historical run.

New summaries record `prompt_version: 3` and `prompt_constraints` (target words,
suggested sentence count and sentence length). The soft target is independently
configured and defaults to 95 words in five roughly 18–20 word sentences. The
CLI's default accepted range remains 80–110; the current Colab notebook explicitly
uses 60–110 with the same 95-word target.
All generation overrides use one independent `GenerationConfig`, with no
competing `max_length` and with a padding token set explicitly when available.
A malformed answer gets at most twice the original token budget on retry;
a structurally valid draft uses the original budget for length correction. If
the same invalid output is returned twice, the final attempt keeps the complete
original source but removes the rejected assistant draft and requests a fresh
rewrite; `repeated_invalid_output` records this transition.
Valid older summaries remain reusable and retain their original metadata;
prompt versions can therefore differ within a resumed run.
In the submitted collection, Bunsen records version 2; the other seven accepted
cached biographies have no prompt-version field. Missing fields remain unknown.

A modest overlength answer can retain an unchanged complete-sentence prefix.
`length_adjustment` records the original text, word counts and removed tail.
This is allowed only up to 1.25 times the upper word bound and is deterministic
postprocessing, not a factual correction. Short answers are not padded and
evidence IDs are not reselected after shortening.

Exhausted attempts go to `summary_failures/<slug>.json` with their source/model
context. These are diagnostic logs, never accepted caches or an offline
recovery path. Normal resumption retries missing or invalid summaries; do not
copy raw failure responses into the accepted summary directory.

## Images and PDFs

`generated_images/<slug>.png` stores FLUX outputs.
`generation_metadata/<slug>.json` records model/revision, precision,
quantization, prompt, seed, steps, dimensions, elapsed time and peak GPU memory.
The floating-point key is `dtype`; the saved current collection records
`torch.bfloat16`. The generator selects dtype at runtime. Quantization applies
to selected components and is not proof of every component's arithmetic dtype.
`peak_cuda_memory_gb` is PyTorch peak allocated memory divided by `1024**3`
(GiB), despite its field name; it is not total device memory consumption.
Load timing is separate from per-image generation timing.

The current image seed is the run seed plus the zero-based subject index.
Output dimensions can vary by portrait. The T4 preset caps the longest input
side at 640, without guaranteeing every output is a 640-square image. Existing
drawings are reused by file existence, not image-quality or identity validation.

`pages/<slug>.pdf` contains one A4 page;
`coloring_book.pdf` contains the complete book. PDFs are rebuilt from validated
biographies and available images. Incomplete runs have a null manifest
`book_path`; any older PDF on disk is not advertised as the current result.
The missing-biography/image check prevents rebuilding the book, preserving old
PDFs. A later rendering error can leave some newly rebuilt individual pages
without a new complete book. Always use the final manifest and acceptance
checklist rather than assuming every PDF has the same invocation timestamp.

## Analysis artifacts are separate

Baseline images, pixel/embedding metrics, and optional source-audit annotations
live under the evaluation directories, not this core run tree. Recalculation
must write to a new analysis directory, leaving the submitted results unchanged.
Archived semantic annotations are bound to source and biography hashes; changed
text requires a new annotation pass, not reuse of old judgments.

Preserve the distinction between recorded inputs, generated outputs, mechanical
checks, and qualitative reviewer labels. The core pipeline neither migrates
legacy histories nor uses post-generation evaluation to correct its outputs.
