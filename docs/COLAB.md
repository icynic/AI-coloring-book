# Colab runbook

## Runtime

The notebook defaults to the free NVIDIA T4 preset. `T4_SAFE_MODE=True` applies
Qwen 4-bit, FLUX 8-bit, a 640px maximum image side, a 256-token prompt sequence,
four FLUX steps, FP16 compute, and no CPU offload. An L4 is faster but is not
required. The tested notebook is `colab/AIColoringBook.ipynb`.

If the first dependency installation reports a Pillow/PIL import mismatch,
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
