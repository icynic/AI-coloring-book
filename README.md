# AI Coloring Book

This prototype turns a list of historical figures into a printable A4 coloring
book. It retrieves Wikipedia text and portraits, asks Qwen3.5-4B for short
biographies, uses FLUX.2 [klein] 4B to make line drawings, and assembles the
pages into a PDF. The pipeline saves intermediate results so interrupted runs
can resume. No model is trained or fine-tuned here.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/icynic/AI-coloring-book/blob/main/colab/AIColoringBook.ipynb)

## Run it in Colab

The notebook is the simplest way to verify the complete pipeline. You do not
need a local Python setup or a local GPU. Colab installs the project dependencies
and runs the models on an assigned GPU; the demonstrated preset works on a free
T4 when one is available.

1. [Open the notebook](https://colab.research.google.com/github/icynic/AI-coloring-book/blob/main/colab/AIColoringBook.ipynb) and select **Runtime → Change runtime type → T4 GPU**.
2. Run the setup cells, then choose a **new output directory** in the configuration cell. The default list of eight people comes from [`evaluation/subjects.csv`](evaluation/subjects.csv).
   The generation notebook and main CLI read only its `name` column; the other
   fields support evaluation pairing or record why subjects were selected, not
   model inputs or ground-truth labels.
3. Run the pipeline and final inspection cells. A complete run downloads its
   `coloring_book.pdf` and records eight items without errors in `manifest.json`.

The current notebook accepts 50–110 biography words with a separate 95-word
writing target. The archived evaluated run used 80–110 words. For a faster
two-person smoke test, resuming a run, or troubleshooting Colab, follow the
[Colab runbook](docs/COLAB.md). Even the smoke test downloads and loads both
models on first use; GPU availability and network waits affect its duration.
The live notebook pulls the current GitHub code, which may differ from the code
that produced the archived results.

## Submitted artifacts

- [Demo video](https://github.com/icynic/AI-coloring-book/releases/download/Output/ai_coloring_book_demo.mp4)
- [Eight-page coloring book](https://github.com/icynic/AI-coloring-book/releases/download/Output/coloring_book.pdf)
- [ACL-style report](https://github.com/icynic/AI-coloring-book/releases/download/Output/ai_coloring_book_report.pdf)
- [Complete saved run](https://github.com/icynic/AI-coloring-book/releases/download/Output/final_run_v2.zip)

The saved-run ZIP includes source records, biographies, generated drawings,
metadata, individual pages, the final book, and its manifest. `output/` and
model weights are not tracked in Git. To inspect the archived run locally,
extract `final_run_v2/` under `output/`.

## How it works

```text
Names ──> Wikipedia text ──> Qwen biography ───┐
     └──> Wikipedia portrait ──> FLUX drawing ─┴──> A4 pages ──> PDF book
```

The program checks each stage before loading the next model. A missing source
portrait stops before Qwen; an invalid biography stops before FLUX. Repeating
the same command with the same settings reuses valid saved outputs. A successful
run requires a non-null `book_path` and an empty `errors` list for every person;
an old PDF in a failed run directory does not establish success.

## Documentation and verification

| Guide | What it covers |
| --- | --- |
| [Colab runbook](docs/COLAB.md) | Setup, configuration, smoke test, resume, and troubleshooting |
| [Output schema](docs/OUTPUT_SCHEMA.md) | Files, manifest fields, provenance, and completion checks |
| [Evaluation guide](evaluation/README.md) | Baselines, image measurements, and biography audit |
| [Report guide](report/README.md) | Paper source, appendices, and compilation |
| [Development record](docs/DEVELOPMENT_RECORD.md) | Dated milestones and verification evidence |

For local development in an environment with the project dependencies installed,
run the CPU-compatible regression suite:

```bash
python -m unittest discover -s tests -v
```

The local suite passed 34 tests on 2026-09-29. These tests use mocks and small
random models; they do not download Qwen or FLUX weights or replace a Colab GPU
run.

## Scope and limitations

The eight Marburg-associated people form a purposively selected case study, not
a held-out benchmark. Automatic format checks and image proxies do not establish
historical correctness, portrait identity, coloring usability, or suitability
for children. Keep the recorded Wikipedia/Wikimedia attribution and the notices
for [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) and
[FLUX.2 klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
with redistributed outputs.
