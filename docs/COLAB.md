# Colab runbook

## Runtime

The notebook defaults to the free NVIDIA T4 preset. `T4_SAFE_MODE=True` applies
Qwen 4-bit, FLUX 8-bit, a 640px maximum image side, a 256-token prompt sequence,
four FLUX steps, FP16 compute, and no CPU offload. An L4 is faster but is not
required. The tested notebook is `colab/AIColoringBook.ipynb`.

If the dependency check cannot import Pillow's core `Image` and `ImageOps`
modules,
choose **Runtime > Restart session** and run all cells again. A normal restart
keeps the installed packages and Drive checkpoints. Do not factory-reset the
runtime. The requirements accept Colab's compatible Pillow version instead of
forcing an in-place replacement. They also retain Colab-compatible
`requests==2.32.4` and `protobuf==5.29.5`; do not upgrade either package in a
later notebook cell.

Before the experiment run, verify the GPU name with `nvidia-smi`. Record the
Colab subscription tier and approximate compute-unit cost separately; Colab
hardware availability and pricing are not controlled by this project.

## Execution order

The final entry point executes four stages:

1. Download Wikipedia text, portrait, revision, and license metadata.
2. Load Qwen3.5-4B, summarize all valid records, and release the model.
3. Load FLUX.2 [klein] 4B, generate all line drawings, and release the model.
4. Render individual pages and the combined PDF.

Every completed item is saved before the next item starts. Re-run the same
command and output directory to resume. `--force` intentionally overwrites all
stage outputs and should not be used during an interrupted evaluation run.

## Repair a book whose summaries contain `...`

After syncing the updated repository, keep the existing run in Drive and set
`REPAIR_SUMMARIES_ONLY = True` in the notebook configuration, then run the
pipeline cell. Alternatively, from the repository directory in Colab run:

```python
import subprocess, sys
subprocess.run([
    sys.executable, '-u', 'main.py', '--repair-summaries',
    '--output-dir', '/content/drive/MyDrive/AIColoringBook/evaluation_flux_t4',
], check=True)
```

Recovery loads names, source records and Qwen settings from the existing run.
It never fetches Wikipedia or loads FLUX. Windows/Colab paths in old source
records do not affect recovery: it resolves saved output images relative to
the run directory. Only missing or invalid summaries are regenerated. Valid
biographies must contain 80-110 words (or the range in the saved manifest),
non-empty integer evidence IDs within the source range, and no placeholder.
These checks do not establish factual correctness; the human audit is still
required.

Qwen's thinking mode is disabled explicitly. Generation uses a 1024-token
budget and at most one validation retry, with 2048 tokens and corrective
feedback. The prompt targets the middle of the requested word range. When a
complete JSON draft fails validation, the retry receives that draft and its
measured word count so it can revise the text; malformed JSON and reasoning
are never reused as drafts. Failed outputs and their reasons are recorded in `summary_failures/`.
They are not treated as usable summaries.

All individual PDFs and the combined PDF are rebuilt from validated text.
Original summaries, PDFs and manifests are copied to `backups/<timestamp>/`
before replacement. Original image metadata, sources and image-run runtime
remain unchanged. `repair_manifest.json` records the repair runtime and errors.
If any summary/image is missing or invalid, no new final PDF is published and
`manifest.json` has a null `book_path`; old PDFs remain available as old results.

For a model-free, read-only check of the downloaded local run:

```bash
python main.py --repair-summaries --check-only --output-dir output/evaluation_flux_t4
```

Recovery needs the project dependencies and a runtime able to load Qwen. It
does not require FLUX inference. A regular notebook rerun also validates cached
summaries now, but recovery is preferable when the source downloads are fixed.

### Fix lead-only sources before retrying summaries

Old runs used only the Wikipedia introduction (`exintro=1`). This can be just
one sentence. The new source policy reads article HTML at the saved revision,
extracts prose paragraphs (not tables, navigation, references or lists), and
selects at most 800 words. Paragraphs mentioning Marburg, lead text, achievements
and biographical sections are prioritized deterministically. Each paragraph is
capped at 200 words and the lead at 180; complete sentences are selected and
returned in original article order. Unclassified sections are used only while
fewer than 200 words have been selected. These heuristics need human review.

For an existing evaluation run, sync **all updated Python files**, including the
new `source_text.py`, and open the **updated notebook**. Pulling repository files
does not change cells in an already-open Colab notebook. Set:

```python
REPAIR_SUMMARIES_ONLY = True
REFRESH_SOURCE_TEXT = True
SUMMARY_MIN_WORDS = 80
SUMMARY_MAX_WORDS = 110
FORCE_REGENERATE = False
```

Keep the existing Drive `OUTPUT_DIR`, then rerun configuration and pipeline cells.
The command printed above generation must contain `--refresh-source-text` and
the chosen `--summary-min-words`/`--summary-max-words` flags. You may explicitly
choose 60 as the minimum instead; there is no silent relaxation.

A self-contained alternative cell, run from the updated repository directory:

```python
import subprocess, sys
subprocess.run([
    sys.executable, '-u', 'main.py', '--refresh-source-text', '--repair-summaries',
    '--output-dir', '/content/drive/MyDrive/AIColoringBook/evaluation_flux_t4',
    '--summary-min-words', '80', '--summary-max-words', '110',
], check=True)
```

This upgrades every legacy subject, not just the last failed names. The article
page/revision IDs, portrait files, portrait attribution, FLUX images, generation
metadata and original GPU runtime are preserved. Old source JSON is backed up.
Changing the selected text invalidates cached summaries by their text hash,
even when the revision number is unchanged; evidence sentence IDs are rebuilt
by Qwen. Previously successful summaries can therefore also need regeneration.
Old PDFs stay on disk but are not advertised as current books after a refresh.

`--refresh-source-text` alone performs **only text recovery**, on local CPU or
Colab, without loading either model. For example, locally:

```bash
python main.py --refresh-source-text --output-dir output/evaluation_flux_t4
```

If recovering locally, synchronize the updated `sources/*.json` back to the same
Drive run before running `--repair-summaries` in Colab; keep backups and
`source_refresh_manifest.json` for provenance. A refresh against already-current
sources performs no further requests. Add `--check-only` for a read-only,
network-free inspection. With both recovery flags, check-only checks the source
upgrade plan; run a separate `--repair-summaries --check-only` to inspect summaries.

Failures preserve the previous source and stop a combined run before Qwen is
loaded. HTTP 429 aborts subsequent source requests in that run; retry later.
Do not use `--force`. If even the full selected source is shorter than the
requested biography minimum, generation is skipped with an actionable error
instead of padding the biography with guessed facts. Adequate source length
does not guarantee factuality or successful generation; audit the new summaries.

### Complete biographies rejected only for length

The default 80-110-word range is a project design choice, not a model constraint.
`max_new_tokens` is an output ceiling: increasing it does not force the model
to write a longer biography. Do not pad a short source with unsupported facts.

If a 60-110-word range is acceptable for the final prototype, select it
explicitly in the updated notebook:

```python
REPAIR_SUMMARIES_ONLY = True
SUMMARY_MIN_WORDS = 60
SUMMARY_MAX_WORDS = 110
```

Or use the updated CLI from the repository directory:

```python
import subprocess, sys
subprocess.run([
    sys.executable, '-u', 'main.py', '--repair-summaries',
    '--output-dir', '/content/drive/MyDrive/AIColoringBook/evaluation_flux_t4',
    '--summary-min-words', '60', '--summary-max-words', '110',
], check=True)
```

Explicit bounds override the saved range; omitted bounds inherit it. A repair
records the previous and effective ranges in `repair_manifest.json` and
`manifest.json.summary_repairs`, and persists the effective range in
`manifest.json.configuration` for the next repair resume. `--check-only` never
persists changes. Without an override, the original stricter range is retained.

Valid cached biographies are reused. Failed attempts in `summary_failures/`
are diagnostic logs, not validated caches; those people are regenerated.
Existing source text and all FLUX images are preserved. Evidence, JSON,
placeholder and word-count checks still apply, and generation can still fail.

This is an explicit change to the length criterion after inspecting failures.
Document it in the report, use the chosen final acceptance range consistently,
and retain the original failure logs. Existing cached biographies retain their
original `requested_word_range`; newly generated biographies record the new
range. Do not describe all samples as generated with an identical prompt or
as satisfying the original 80-word minimum unless that is actually true.

### Recover a complete answer that slightly exceeds the maximum

Generation now has a deterministic postprocessing step for moderate overshoots
(at most 25% over the maximum). It keeps the longest unchanged prefix ending at
a conservative English sentence boundary, provided the result still meets the
minimum. It never cuts at an arbitrary word count, adds facts, pads short text,
or relaxes JSON/evidence validation. Ambiguous boundaries and answers that
cannot fit this way still require a model revision or fail explicitly.

The original model response and removed tail are preserved in the summary's
`length_adjustment`. Deleting tail sentences can omit useful information; review
the resulting biography, and report this postprocessing in the evaluation.
Evidence IDs are retained as a conservative superset, not remapped to invented
per-sentence citations. Wikipedia source sentence segmentation is unchanged.

Repair also tries the **last** logged failed answer before loading Qwen. Its
subject, revision and source text hash must match. Older logs without explicit
context additionally require a matching error and timestamp within the saved
repair run. Malformed/truncated JSON is never promoted. Recovery records the
failure-log hash and provenance check used; the original failure file is kept.

To require recovery with no network or model inference, locally or in Colab:

```bash
python main.py --repair-summaries --offline-repair --output-dir output/evaluation_flux_t4
```

Use the Drive output path in Colab. This mode fails rather than loading Qwen if
no usable saved answer exists. Without `--offline-repair`, normal repair can
regenerate any unresolved biographies. Do not combine offline repair with source
refresh. Already-valid summaries and all FLUX images remain unchanged.

## Source-grounded checking and revision in a separate prototype

After syncing `Summarizer.py`, `summary_validation.py`, `summary_review.py`,
`main.py` and `refine_biographies.py`, run this cell **from the repository
directory**, with the existing dependencies and Drive mounted:

```python
import subprocess, sys
subprocess.run([
    sys.executable, '-u', 'refine_biographies.py',
    '--source-run', '/content/drive/MyDrive/AIColoringBook/evaluation_flux_t4',
    '--output-dir', '/content/drive/MyDrive/AIColoringBook/final_refined_t4',
], check=True)
```

The source run is read-only. The derived directory must be empty on its first
run and must not contain, or be inside, the source run. This command copies
selected sources, biographies, images and available generation metadata, not
old PDFs or backups. `original_summaries/` preserves the input biographies;
`refinement_origin.json` binds the source manifest and copied inputs by hash.
Qwen settings, target age and word range are inherited from the source manifest
(60-110 words for the current eight-person run). It does not fetch Wikipedia,
load FLUX, or regenerate images. Source metadata and generation metadata retain
their original paths as provenance; repair resolves generated images relative
to the derived directory.

Qwen is loaded once. Each review/editor request uses a fresh chat, thinking
disabled, with the full numbered saved source. The checker covers each biography
sentence and checks all factual details in it, quoting source excerpts. The four
verdicts are `supported`, `partial`, `unsupported`, and `source_conflict`.
Feedback also checks essential technical language and unnecessary personal detail.
The program rejects missing sentence coverage, invalid IDs, quotes not found in
the referenced sentence, and conflicts lacking two different source IDs. A
Marburg keyword in the source additionally requires Marburg in the biography;
the model checks the actual relation. This keyword rule is not a relation extractor.

Only an all-supported review with no editorial issues passes. There are at most
two content revisions, each followed by a new review; disputed source facts should
be omitted rather than resolved from model memory. Review JSON and revision JSON
each allow two format/validation attempts. Review token budgets are 2048/4096;
revision budgets are 1024/2048. Moderate length overshoots can still use the existing
complete-sentence-prefix policy, but the resulting text is reviewed again.

The derived summary's `summary_review` stores the initial draft, raw verification
and revision outputs, quotes, verdicts, reasons, timing, policy, and final text/source
hashes. Final citations are rebuilt from the accepted review. Failed attempts stay
in `summary_failures/`; no combined PDF is published if any person remains unresolved.
Inspect `repair_manifest.json` for per-person failures. A rerun of the same command
reuses accepted reviews and retries unresolved people; changed frozen inputs require
a new output directory. This is not a guarantee that a retry will succeed.

For a local read-only input preflight, without loading either model:

```bash
python refine_biographies.py --source-run output/evaluation_flux_t4 --output-dir output/final_refined_t4 --check-only
```

The main CLI also accepts `--verify-summaries` for new runs or in-place summary
repair, and `--max-review-revisions 0`, `1`, or `2`. **Use the separate-directory
entry point above for the frozen evaluation run.** Model verification cannot be
combined with `--offline-repair`. Normal repair inherits an enabled review policy
from its derived manifest, so it cannot silently fall back to unreviewed text.

`model_verified` means the model's verdict passed mechanical checks, not that
facts or age appropriateness are independently established. Report this as a
system revision stage, not an independent LLM evaluation. Do not reuse the original
claim annotations after changing biographies; their hashes will be stale. Re-audit
the revised output, and keep the original experiment results separate.

## Memory fallbacks

Use `T4_SAFE_MODE=True` first. If FLUX still runs out of memory, try these
changes in order:

1. Disable the preset and retain Qwen 4-bit and FLUX 8-bit while reducing
   `--max-side` from 640 to 512.
2. Enable `--vae-tiling`.
3. Change FLUX from 8-bit to 4-bit.
4. Enable CPU offload only as a final fallback.

Do not change memory settings halfway through a formal evaluation run; use a
new output directory and regenerate every sample with the same configuration.

## Clean-room verification

Before release:

1. Start a fresh Colab runtime with an empty `/content` directory.
2. Open the notebook from GitHub and run all cells without manual code edits.
3. Process at least two names.
4. Confirm that every stage directory, `manifest.json`, individual pages, and
   `coloring_book.pdf` were produced.
5. Restart the runtime and rerun against the same Drive directory to confirm
   that completed stages are reused.
6. Open the PDF and manually verify portrait attribution, page layout, and text.

## Experiment discipline

- Freeze the name list, prompts, model revisions, seed, and image size first.
- Preserve failed samples in the manifest.
- Do not choose the best-looking seed for the main comparison.
- Copy the final manifest and evaluation CSV into the report artifact.
- Check every portrait's recorded license before redistributing sample books.
