# AI Coloring Book

AI Coloring Book is a reproducible pipeline that turns a list of historical
figures into printable biographical coloring-book pages. It retrieves grounded
source material from English Wikipedia, writes a child-oriented biography with
Qwen3.5-4B, edits the source portrait into line art with FLUX.2 [klein] 4B, and
renders individual pages plus a combined A4 PDF book.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/icynic/AI-coloring-book/blob/main/colab/AIColoringBook.ipynb)

## Final system

```text
Names
  -> Wikipedia lead, exact revision, portrait, and license metadata
  -> Qwen/Qwen3.5-4B grounded biography and supporting sentence IDs
  -> release Qwen GPU memory
  -> black-forest-labs/FLUX.2-klein-4B portrait-to-line-art editing
  -> individual A4 pages, combined PDF, and reproducibility manifest
```

The text and image models are deliberately loaded in separate stages. They do
not need to fit in GPU memory at the same time.

## Recommended: Google Colab

The complete final pipeline is tested for a Colab L4 GPU with a High-RAM
runtime. Open the notebook using the badge above, select an L4 GPU, edit the
`NAMES` list, and run all cells. Google Drive output is enabled by default so a
disconnected runtime can resume from completed stages.

The first run downloads the model weights and takes substantially longer than
subsequent runs. If memory is tight, the notebook exposes Qwen 4-bit, FLUX
8-bit, and FLUX CPU-offload options.

## Command-line use

Colab already provides a CUDA-enabled PyTorch installation. Install the pinned
project environment with:

```bash
pip install -r requirements-colab.txt
```

Run two people end to end:

```bash
python main.py \
  --names "Marie Curie" "Albert Einstein" \
  --output-dir output/final_run \
  --seed 42
```

Use a UTF-8 file with one name per line for larger runs:

```bash
python main.py --names-file people.txt --output-dir output/final_run
```

Existing source, summary, image, and page files are reused automatically. Pass
`--force` only when they should be regenerated. Run `python main.py --help` for
quantization, offload, image-size, and stage-skipping options.

## Outputs

Each run directory contains:

```text
sources/                 Wikipedia records and downloaded portraits
summaries/               generated biographies and supporting source sentences
generated_images/        FLUX coloring-page images
generation_metadata/     prompts, seeds, latency, dimensions, and peak VRAM
pages/                    one PDF per person
coloring_book.pdf         combined book
manifest.json             runtime, configuration, file paths, and failures
```

The exact Wikipedia revision and the portrait's Wikimedia license, artist, and
credit are retained in every source record. Generated pages are labelled as
AI-generated and include a source link. Publication outside the research
prototype still requires a manual license and factual review.

## Reproducibility

- Dependency versions and the exact Qwen/FLUX Hugging Face revisions are pinned.
- The default seed is `42`; person `n` receives seed `42 + n`.
- Failed samples are recorded rather than silently removed.
- Intermediate results are written atomically and form resumable checkpoints.
- `manifest.json` records the GPU, PyTorch version, model IDs, run configuration,
  source revisions, and output paths.

For a clean rerun, use a new output directory. For the final evaluation, freeze
the subject list and configuration before generating outputs; do not select the
best result from multiple seeds.

## Repository map

- `main.py` — complete staged pipeline and CLI.
- `Fetcher.py` — Wikipedia text, portrait, revision, and license retrieval.
- `Summarizer.py` — grounded Qwen3.5 biography generation.
- `GeneratorFlux2KleinL4Colab.py` — final FLUX image editor.
- `Concatenator.py` — individual and multi-page A4 PDF rendering.
- `Generator.py` — SD1.5 + ControlNet evaluation baseline.
- `colab/AIColoringBook.ipynb` — final Colab entry point.
- `tests/` — model-free control-flow and PDF smoke tests.

## Tests

The default tests do not download or load either large model:

```bash
python -m unittest discover -s tests -v
```

An actual end-to-end GPU run is intentionally performed in Colab because image
generation is hardware-dependent.

## Known limitations

- FLUX can simplify or alter identity-relevant facial and clothing details.
- Supporting sentence IDs make the biography auditable but do not guarantee
  factual correctness; final pages require human verification.
- Wikipedia lead images have heterogeneous quality and licenses.
- The final system is designed for historical figures with a clear lead portrait.
- A Colab GPU and network access are required for the complete uncached run.

## Models and licenses

- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) — Apache 2.0.
- [FLUX.2 klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) — Apache 2.0.
- Wikipedia text and Wikimedia images retain their own attribution and license
  requirements; inspect the saved metadata before redistribution.
