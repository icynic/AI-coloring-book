# Colab runbook

Use [the notebook](../colab/AIColoringBook.ipynb) to run the complete name → Wikipedia source → Qwen biography → FLUX image → PDF pipeline. GPU inference runs in Colab; viewing downloaded outputs and running saved-artifact evaluation can be done locally. See the [project overview](../README.md), [output schema](OUTPUT_SCHEMA.md), and [evaluation instructions](../evaluation/README.md).

This guide describes the submitted source. A new run reproduces the workflow, not necessarily identical archived outputs: Wikipedia, Colab, and some dependency versions can change. The archived run has no recorded historical application Git commit; the report appendix identifies the documented source snapshot separately.

## 1. Select a GPU and choose the source copy

Open the notebook in Colab. Under **Runtime → Change runtime type**, select a GPU runtime, then run the first code cell (`!nvidia-smi`). Confirm that a GPU was actually assigned. The default preset targets a T4; an L4 is not required. Availability depends on Colab, so selecting a GPU is not sufficient by itself.

There are two ways to obtain the source:

- **Submitted software:** upload the source ZIP through Colab's Files sidebar. For the example below, name it `AIColoringBook.zip` and ensure its root contains `main.py`, `requirements-colab.txt`, and `evaluation/subjects.csv` (not an extra enclosing folder). Replace the notebook's clone/pull cell with:

  ```python
  from pathlib import Path
  import shutil

  PROJECT_DIR = Path('/content/submitted')
  shutil.unpack_archive('/content/AIColoringBook.zip', PROJECT_DIR)
  %cd /content/submitted
  ```

  Use an empty extraction directory on first setup. After a session restart, if this copy still exists, use the following instead of extracting again; do not extract over modified files or run the remote clone/pull cell:

  ```python
  from pathlib import Path
  PROJECT_DIR = Path('/content/submitted')
  %cd /content/submitted
  ```

- **Live repository:** leave the notebook's setup cell unchanged. It clones `https://github.com/icynic/AI-coloring-book.git`, or runs `git pull --ff-only` when the clone already exists, and prints its current short commit hash. This pulls remote HEAD, not a pinned historical submission. Record the printed hash and avoid rerunning this cell during a resume unless you intend to update the code.

Both routes must leave the working directory at the project root. Do not run the source-copy setup and remote setup as if they were sequential steps.

## 2. Install and check the environment

Run the dependency cell and the compatibility-check cell:

```python
%pip install -q -r requirements-colab.txt
```

Use this requirements file rather than the broad upgrade command in the standalone generator's header. It leaves Colab's CUDA-enabled PyTorch in place, pins the project Diffusers revision and core packages, requires `requests==2.32.4` and `protobuf==5.29.5`, and accepts compatible Pillow versions without unnecessarily replacing a loaded PIL module. Some requirements remain ranges; this is not a full historical environment lock.

The next cell checks `PIL.Image`, `PIL.ImageOps`, requests, and protobuf. If it fails after installation, choose **Runtime → Restart session**, return to the same project root, and rerun the checks and configuration. A normal restart on the same runtime generally retains installed packages; if it does not, reinstall this requirements file. Do not disconnect/delete the runtime to fix a stale import. Drive outputs persist independently of the runtime.

Model weights are downloaded on first use. Authentication to Hugging Face is optional for these checkpoints; an unauthenticated-rate-limit warning is not itself a pipeline failure. If using an `HF_TOKEN`, keep it in Colab Secrets rather than in notebook source or shared logs.

## 3. Configure a new run

Run the notebook's configuration cell. It loads `NAMES` from `evaluation/subjects.csv` in this order:

```text
Otto Hahn; Robert Bunsen; Emil von Behring; Jacob Grimm;
Hannah Arendt; K. Ferdinand Braun; Alfred Wegener; Boris Pasternak
```

Its other defaults are:

```python
USE_GOOGLE_DRIVE = True
T4_SAFE_MODE = True
FORCE_REGENERATE = False
FUZZY_SEARCH = False
SEED = 42
SUMMARY_MIN_WORDS, SUMMARY_MAX_WORDS = 80, 110
```

Authorize Drive mounting when prompted. **The notebook currently assigns a directory named `final_run_v2`; that name alone does not make it fresh. Preserve any archived directory with that name.** Immediately after the configuration cell and before the pipeline cell, add and run a small override cell for a new experiment:

```python
OUTPUT_DIR = '/content/drive/MyDrive/AIColoringBook/reproduction_t4'
FORCE_REGENERATE = False
print('People:', NAMES)
print('Output directory:', OUTPUT_DIR)
```

Choose a directory that is absent or empty. If `reproduction_t4` already contains another experiment, choose another name. Keep all eight subjects for the full reproduction.

For a smaller two-person smoke test, use this override **instead**, at the same point:

```python
NAMES = ['Otto Hahn', 'Robert Bunsen']
OUTPUT_DIR = '/content/drive/MyDrive/AIColoringBook/smoke_t4'
FORCE_REGENERATE = False
print('People:', NAMES)
print('Output directory:', OUTPUT_DIR)
```

The configuration cell reloads `NAMES` and `OUTPUT_DIR` whenever it runs. Rerun your override after it, including after a restart. Once overrides are added, do not use “Run all” without checking that the override precedes the pipeline cell. A smoke test still incurs first-time model downloads and loading; it only reduces the number of generated pages.

With `USE_GOOGLE_DRIVE=False`, no Drive is mounted and the notebook assigns a local `/content/...` output directory. Override it with a fresh local path such as `/content/AIColoringBook/smoke_t4`. These files are ephemeral: download them before the runtime is deleted or replaced.

### T4 preset versus custom settings

`T4_SAFE_MODE=True` passes `--t4-safe-mode`. The CLI then overrides the following settings, even if different values were also supplied:

| Setting | T4 preset |
| --- | --- |
| Qwen / FLUX quantization | 4-bit / 8-bit |
| Maximum image side / prompt sequence length | 640 px / 256 tokens |
| Image steps / guidance | 4 / 1.0 |
| FLUX CPU offload | Disabled |

Qwen and FLUX load sequentially and are released between stages. The maximum side is a bound, not a fixed square image size. Image seeds are `SEED + index`, using zero-based subject order.

Quantization describes weight storage, not the compute dtype. Both wrappers choose their main dtype from the runtime's CUDA/BF16 support check, falling back to FP16 on CUDA or FP32 on CPU. The saved `final_run_v2` FLUX metadata records `torch.bfloat16` on its T4 runtime; do not describe all T4 runs as FP16. Consult `generation_metadata/*.json` for the actual image dtype.

To customize, set `T4_SAFE_MODE=False`. The notebook exposes `QWEN_QUANTIZATION`, `FLUX_QUANTIZATION`, and `FLUX_OFFLOAD`; its defaults for the two quantization variables are `'none'`, so set them explicitly before running on a T4. Resolution, sequence length, steps, guidance, and VAE tiling are **CLI options**, not configuration-cell constants. For example, this deliberately separate, lower-resolution/offloaded experiment can replace the notebook pipeline cell:

```bash
!python -u main.py --names "Otto Hahn" "Robert Bunsen" \
  --output-dir /content/drive/MyDrive/AIColoringBook/custom_t4 \
  --seed 42 --summary-min-words 80 --summary-max-words 110 \
  --no-fuzzy-search --qwen-quantization 4bit --flux-quantization 8bit \
  --max-side 512 --max-sequence-length 256 --flux-steps 4 \
  --guidance-scale 1.0 --flux-offload --vae-tiling
```

Do not add `--t4-safe-mode` to this command: it would restore 640 px and disable CPU offload. This custom command does not use the notebook variables; set `OUTPUT_DIR` to its actual directory before running the inspection cell. Offload can slow execution and still requires available host memory. Treat custom settings as a new experiment, not as a way to alter the archived evaluation midway.

## 4. Run, inspect, and resume

Run the pipeline cell. It prints the command and starts a fresh unbuffered Python process, merging stderr into stdout and printing each line as it arrives. Stage messages include `[fetch]`, `[summarize]`, `[image]`, and `[pdf]`. Model downloads/loading can dominate the first run, and there is no fixed completion-time guarantee. There are no separate repair or model self-review stages.

Then run the last inspection cell. A successful full run requires:

- The pipeline process exits with code 0.
- All eight subjects print `OK` and have empty `errors` in `manifest.json`.
- The manifest has a non-null `book_path` and the PDF exists.
- `coloring_book.pdf` has eight pages, each containing its portrait, biography, and source/portrait credits. A smoke test should have two pages instead.

Inspect the PDF rather than assuming that successful generation establishes factual accuracy or visual quality. Validation checks structure, length, and source/evidence bookkeeping; it does not prove that every factual claim follows from its cited sentences.

To resume, use the **same code copy, directory, ordered names, and settings**, keep `FORCE_REGENERATE=False`, and rerun the pipeline cell. Complete source caches, validated summaries, and existing generated images can be reused; PDFs are rebuilt from the current validated text and images. Downloaded archives contain Colab absolute paths, so inspecting them locally is not the same as making them portable caches for a new machine.

The CLI rejects a nonempty directory without a current manifest, an old manifest schema, or a different recorded configuration before changing its outputs. These checks do not pin all source-file contents. Do not update code midway if you need an unchanged experiment. Use a new directory for different people, seeds, models, quantization, resolution, or word policy. `--force`/`FORCE_REGENERATE=True` regenerates cached stages and is not the normal resume mechanism.

If the run fails, use the actual stage error above the final notebook exception. A null `book_path` means no complete new book was produced; existing PDFs may remain unchanged and must not be presented as a successful rerun. Inspect `manifest.json` and any `summary_failures/*.json`. If the process was killed before its final manifest write, that manifest may contain only the initial configuration, with no runtime or populated items; read the streamed log first.

## 5. Download the complete result

For Drive-backed runs, open **Google Drive → My Drive → AIColoringBook**, find the experiment folder, and download the **folder**, not only `coloring_book.pdf`. Keep sources, summaries, generation metadata, page PDFs, and `manifest.json` together. Extract the downloaded ZIP locally and use the [output schema](OUTPUT_SCHEMA.md) to check its contents. Do not replace the archived `output/final_run_v2` with a new run.

For local `/content` outputs, the Colab Files sidebar can download individual files. To preserve the complete folder, create a ZIP after generation in a notebook cell:

```python
import shutil
from google.colab import files

archive_path = shutil.make_archive('/content/coloring_book_result', 'zip', OUTPUT_DIR)
files.download(archive_path)
```

The ZIP includes the directory contents, not just the PDF. Download promptly; runtime-local files are not durable storage. This export does not modify the run itself.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| No GPU in `nvidia-smi`, or CUDA unavailable | Check the runtime selection and actual assignment. Do not use `--allow-cpu` as the normal reproduction path: FLUX CPU inference is impractical for this prototype. |
| Out of GPU memory | Stop other GPU jobs or restart the same runtime, restore the configuration/override, and resume. If the T4 preset still fails, use a new directory for a custom lower-resolution, 4-bit FLUX, or offloaded experiment. Do not silently mix settings in one evaluation. |
| Pillow import error, including `_Ink` or missing `ImageText` | Rerun the requirements install, restart the session, return to the project root, and run the current core-module check. That check uses `Image` and `ImageOps`, not `ImageText`; do not add an `ImageText` check. |
| requests/protobuf conflicts | Install the project Colab requirements and restart before checking again. Unrelated preinstalled Colab packages can also emit resolver warnings; use the actual compatibility-check result and failure traceback rather than a broad upgrade. |
| Wikimedia HTTP 429 or slow `[fetch]` | The downloader retries with backoff and respects `Retry-After`, so a request can remain quiet while waiting. Avoid parallel download jobs. Let retries finish or stop and resume later with the same configuration; complete cached sources will be reused. No wait time guarantees that the shared Colab IP is unblocked. |
| No apparent output | Confirm the current pipeline cell contains `subprocess.Popen` and its line-printing loop, and that `Running:` appears. It only prints when the child emits a line; loading and retry waits can be quiet. The cell's final exception is not the root cause: retain the preceding stdout/stderr. |
| Notebook `RuntimeError` / process exit code 1 | Read the preceding stage failure, then the manifest if it reached the final write. An invalid biography prevents rebuilding the complete book. Rerun normally to retry missing/invalid outputs; do not use old PDFs as evidence of success. |
| “Existing run uses older code or different settings” | Restore the original configuration for a genuine resume, or choose a new empty directory for a new experiment. Do not delete the manifest to bypass the safeguard. |
