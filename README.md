# AI Coloring Book

A single end-to-end pipeline turns historical figures into an A4 coloring book:
Wikipedia article prose and portrait → Qwen3.5-4B biography → FLUX.2 [klein] 4B
line art → individual PDFs, combined PDF and reproducibility manifest.

[Open the notebook in Colab](https://colab.research.google.com/github/icynic/AI-coloring-book/blob/main/colab/AIColoringBook.ipynb)

## Run the complete book in Colab

Sync the current code to GitHub first, then open `colab/AIColoringBook.ipynb`,
choose a GPU runtime and run all cells. The default preset targets a free T4:
Qwen 4-bit, FLUX 8-bit, FP16 compute on T4, 640px maximum image side and four
image-generation steps. Qwen and FLUX load sequentially.

The only shipped subject list is `evaluation/subjects.csv`. Its current eight
names are Otto Hahn, Robert Bunsen, Emil von Behring, Jacob Grimm, Hannah Arendt,
Ferdinand Braun, Alfred Wegener and Boris Pasternak. Braun's exact Wikipedia
query is `K. Ferdinand Braun`. The notebook reads this list directly.

The new complete run writes to
`/content/drive/MyDrive/AIColoringBook/final_run_v2`. It does not merge pages from
older runs. First-time model downloads can take substantial time.

If Pillow imports fail after installation, use Runtime > Restart session,
then rerun the notebook. See [the Colab runbook](docs/COLAB.md).

## Command line

Install the environment on a CUDA-capable machine or Colab:

```bash
python -m pip install -r requirements-colab.txt
python main.py --output-dir output/final_run_v2 --t4-safe-mode --no-fuzzy-search
```

With no names specified, `main.py` uses the same current eight-person CSV.
Use `--names "Otto Hahn" "K. Ferdinand Braun"` to override it, or
`--names-file people.txt` for a user-provided UTF-8 list.

Rerun the same command and directory to resume. Completed sources, valid
summaries and image outputs are reused. The run configuration is saved before
downloads begin. Old-format runs, changed settings and unknown nonempty
directories are rejected: choose a new directory instead. `--force` explicitly
regenerates all stages in a compatible run.

There is no model reviewer, summary-only repair, failure-log recovery, source
refresh, separate demonstration exporter or replacement notebook. The normal
summarizer retains up to two attempts for deterministic JSON/length/evidence
validation. A modest length overshoot may be shortened at a complete sentence
boundary, with the original answer and removed tail retained.

## Outputs

```text
sources/                 selected article text, revisions, portraits, attribution
summaries/               biographies, raw answers and supporting source sentences
generated_images/        FLUX line drawings
generation_metadata/     prompts, seeds, dimensions, runtime and VRAM
pages/                   one PDF per figure
coloring_book.pdf         combined A4 book
manifest.json            configuration, runtime, paths and per-person errors
summary_failures/        diagnostic logs only, if generation fails
```

An incomplete book is never advertised as successful. Failed items remain in
the manifest. Rerun the normal pipeline to retry them; logs are not promoted
into accepted summaries. See [output metadata](docs/OUTPUT_SCHEMA.md).

## Core files

- `main.py`: stages, CLI, checkpoints and manifest.
- `Fetcher.py`, `source_text.py`: revision-bound Wikipedia source and portrait.
- `Summarizer.py`, `summary_validation.py`: grounded generation and validation.
- `GeneratorFlux2KleinL4Colab.py`: FLUX generator, also supporting the T4 preset.
- `Concatenator.py`: A4 PDF rendering.
- `colab/AIColoringBook.ipynb`: the sole full-run notebook.
- `tests/`: model-free regression tests.

## Evaluation and existing material

Baseline and automatic evaluation remain because the final report requires
evaluation: see [evaluation/README.md](evaluation/README.md).
`Generator.py` is used by the SD1.5 + ControlNet baseline, not the final pipeline.

Existing outputs, model weights, source pictures and presentation/research
material are retained. The completed original eight-person evaluation is
unchanged; its scores and claim annotations do not describe the new subject
list. Two demonstration subjects were changed after inspecting results;
disclose that selection and do not present it as an untouched preregistered
evaluation.

## Tests and limitations

```bash
python -m unittest discover -s tests -v
```

Tests do not download or load the large models. Full GPU inference still runs
in Colab. Evidence IDs and word counts do not prove factual accuracy or
suitability for children; check the final text and each portrait's recorded
license/attribution before redistribution. Automatic image measurements are
proxies, not independent human preference or face-recognition accuracy.

The pinned [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) and
[FLUX.2 klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
models have their own licenses. Wikipedia text and portraits retain their
source licensing requirements.
