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
feedback. Failed outputs and their reasons are recorded in `summary_failures/`.
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
