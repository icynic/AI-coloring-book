# Colab runbook

## One notebook, one full run

Use `colab/AIColoringBook.ipynb` after syncing the current repository to GitHub.
Select a GPU runtime, verify the GPU with the first cell and run all cells.

The default T4 preset uses Qwen 4-bit, FLUX 8-bit, FP16 compute on T4, 640px
maximum side, a 256-token image prompt sequence and four image steps. Models
load sequentially; an L4 is not required.

The notebook reads all eight names from `evaluation/subjects.csv` and writes
the new complete book to:

```text
/content/drive/MyDrive/AIColoringBook/final_run_v2
```

Do not point the new notebook at `evaluation_flux_t4`, `final_refined_t4` or an
older exported demonstration. Old-format runs and configuration changes are
rejected before outputs are changed. To start another experiment, change the
output directory to a new empty directory.

## Configuration

```python
T4_SAFE_MODE = True
FORCE_REGENERATE = False
FUZZY_SEARCH = False
SEED = 42
SUMMARY_MIN_WORDS, SUMMARY_MAX_WORDS = 80, 110
```

Each figure receives seed `SEED + index`, with zero-based indexing. Keep the same
settings and directory to resume an interrupted run; do not use `--force` unless
deliberately regenerating all stages.

A normal run retrieves revision-bound article prose and portraits, generates
validated Qwen biographies, releases Qwen, generates FLUX images, releases FLUX,
then builds the PDFs. A new directory regenerates all eight subjects rather
than merging older pages.

## Dependencies and progress

Install only `requirements-colab.txt`. It retains compatible Pillow rather than
replacing an already-loaded PIL unnecessarily, pins requests to 2.32.4 and
protobuf to 5.29.5, and leaves Colab's CUDA-enabled PyTorch installation in place.

If Pillow's core modules fail to import, reinstall the requirements, choose
Runtime > Restart session, then rerun. A normal restart preserves installed
packages and Drive checkpoints; a factory reset does not.

The pipeline cell uses a fresh unbuffered Python process, so progress appears as
stages execute. First-time model downloads and model loading are expensive.
Wikimedia can rate-limit requests; use the same normal command later to retry
failed source retrieval instead of launching concurrent download jobs.

## Completion check

The last cell displays runtime, every subject's errors and the combined PDF
link when complete. No valid book means the run is incomplete: inspect
`manifest.json` and any diagnostic `summary_failures/`, then rerun the same
pipeline. There are no separate repair or model self-review cells.

Inspect all eight PDF pages, biographies and portrait credits before submission.
Download the entire `final_run_v2` folder, not only its PDF, to preserve sources
and reproducibility metadata.

If the T4 runs out of memory, start a separate run with custom lower-resolution
or 4-bit FLUX settings; do not change settings midway through an evaluation.
CPU offload remains an explicit last-resort option in the CLI.
