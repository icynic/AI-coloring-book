# AI Coloring Book

An automatic, resumable prototype turns a list of historical figures into an
A4 coloring book. Selected Wikipedia prose goes to Qwen3.5-4B for a short
biography; the saved portrait goes separately to FLUX.2 [klein] 4B for line art.
ReportLab combines the results into individual pages and a complete PDF.
No model is trained or fine-tuned for this project.

## Start here

- **Inspect the submitted prototype:** open the
  [completed coloring book](https://github.com/icynic/AI-coloring-book/releases/download/Output/coloring_book.pdf)
  and the
  [final report](https://github.com/icynic/AI-coloring-book/releases/download/Output/ai_coloring_book_report.pdf).
  No GPU or model download is needed.
- **Download the complete saved run:** use
  [final_run_v2.zip](https://github.com/icynic/AI-coloring-book/releases/download/Output/final_run_v2.zip)
  for the manifest, sources, biographies, generated images, metadata, and
  individual page PDFs used in the report.
- **Generate a new book:** follow the [Colab runbook](docs/COLAB.md), including
  its new-directory and two-person smoke-test instructions.
- **Inspect or recalculate evaluation:** follow
  [the evaluation guide](evaluation/README.md). The current eight-person
  baseline and evaluation are already complete.
- **Understand intermediate files and completion checks:** see
  [the output schema](docs/OUTPUT_SCHEMA.md).
- **Read the paper source and reproduction appendix:** see
  [the report guide](report/README.md).

`output/` and model weights are not tracked in Git. The Release links above are
the public artifact downloads for a source-only checkout. To run the documented
local evaluation commands, extract the archive and place its `final_run_v2/`
directory at `output/final_run_v2/`. Opening the notebook does not download the
saved run or model weights.

## Where each task runs

| Task | Environment | Model inference required? |
| --- | --- | --- |
| Read saved PDFs, metadata, or reports | Local computer | No |
| Generate the complete Qwen + FLUX prototype | Colab CUDA GPU; tested on T4 | Yes, sequentially |
| Generate the optional SD1.5 baseline | Prepared CUDA environment; originally local RTX 3050 | Yes |
| Recalculate pixel image proxies | Local Python with Pillow, NumPy, OpenCV | No |
| Recalculate DINOv2 similarity | Local CPU or GPU with PyTorch and Transformers | Yes, embedding model only |
| Check saved biographies mechanically | Local Python, standard library | No |
| Run regression tests | Prepared Python environment; CPU is sufficient | No pretrained-weight downloads |

The final Colab manifest records Python 3.13.15, PyTorch 2.11.0+cu128, and Tesla
T4. Image evaluation records Windows/Python 3.11.3. These are observed runtimes,
not a guarantee for every future Colab image or Python/package combination.

## Generate in Colab

Use the supplied [notebook](colab/AIColoringBook.ipynb) for the submitted source
version, or [open the live GitHub notebook in Colab](https://colab.research.google.com/github/icynic/AI-coloring-book/blob/main/colab/AIColoringBook.ipynb).
The live setup clones/pulls the remote repository; it is not pinned to the code
that produced the archived outputs. The runbook explains this distinction.

The T4 preset uses Qwen 4-bit loading, FLUX 8-bit loading for selected
components, a maximum image side of 640, four image steps, and a maximum image
prompt sequence length of 256. The models load sequentially. Floating-point
dtype is selected at runtime and recorded in image metadata; the archived
FLUX outputs record `torch.bfloat16`. Quantization bits do not specify compute
dtype. An L4 is not required for the demonstrated configuration.

The CLI below is for Colab with CUDA-enabled PyTorch already supplied. It uses
a **new** directory, not the archived `final_run_v2`:

```bash
python -m pip install -r requirements-colab.txt
python -u main.py --output-dir /content/reproduction_t4 --seed 42 --summary-min-words 60 --summary-max-words 110 --summary-target-words 95 --t4-safe-mode --no-fuzzy-search
```

The current notebook accepts 60--110 words while keeping a separate 95-word
soft target. Lowering the accepted minimum therefore no longer asks Qwen to
write a shorter biography. The archived evaluated `final_run_v2` used an
80--110-word policy and is not modified by this current-run recommendation.

`/content` is temporary; use the notebook's mounted Drive path to retain a run
across runtime replacement. Neither requirements file installs PyTorch.
Do not install Colab-specific pins into a shared system Python environment;
local optional evaluation has separate setup instructions in its guide.

With no names supplied, the CLI reads [evaluation/subjects.csv](evaluation/subjects.csv):
Otto Hahn, Robert Bunsen, Emil von Behring, Jacob Grimm, Hannah Arendt,
K. Ferdinand Braun, Alfred Wegener, and Boris Pasternak. `K. Ferdinand Braun`
is the exact Wikipedia query. Override with
`--names "Otto Hahn" "K. Ferdinand Braun"`, or `--names-file people.txt` for
a UTF-8 file containing one name per line.

To resume, keep the same directory, ordered names, and settings, including the
accepted word range and soft target. Valid sources,
biographies, and existing drawings are reused. Changed configuration, old
manifest schemas, and unknown nonempty directories are rejected; choose a new
empty directory instead. `--force` deliberately regenerates stages in a
compatible directory; it is not the normal retry command. Downloaded source
metadata retains Colab absolute paths, so moving a result folder to Windows
does not make `main.py` an offline replay tool.

Normal full runs stop at stage boundaries instead of discovering missing work
only during PDF assembly. Missing Wikipedia text or a local reference portrait
writes a checkpoint manifest and stops before Qwen is loaded. An exhausted
biography retry writes `summary_failures/<slug>.json`, releases Qwen, and stops
before FLUX is loaded. A missing generated drawing stops before PDF assembly.
Rerun the same command to reuse valid stage outputs and retry the missing item.

## Completion and outputs

A successful default run exits with code 0, records eight items without errors,
has a non-null manifest `book_path`, and produces an eight-page book with a
biography and drawing on every page. A two-person smoke test should instead
produce two complete pages. Confirm these conditions rather than treating an
old PDF on disk as a new success.

```text
<run-directory>/
  sources/                 selected prose, revisions, portraits, attribution
  summaries/               biographies, raw answers, supporting sentences
  generated_images/        FLUX drawings
  generation_metadata/     prompts, seeds, dimensions, dtype, timings, VRAM
  pages/                   individual A4 PDFs
  coloring_book.pdf        complete A4 book
  manifest.json           settings, runtime, current book path, per-person errors
  summary_failures/        diagnostic logs when generation fails
```

Incomplete runs retain per-item errors in a checkpoint manifest with a null
`book_path` and do not advertise an old PDF as the current book. Inspect the
streamed log and manifest, then resume normally. Model
downloads/loading and network retries can be expensive; the saved resumed
manifest duration is not a fresh end-to-end generation time. Download the
**whole run directory**, not just the PDF, as explained in the runbook.

## Software structure

- `main.py`: CLI, stage orchestration, checkpoints, manifest.
- `Fetcher.py`, `source_text.py`: Wikipedia retrieval and bounded prose selection.
- `Summarizer.py`, `summary_validation.py`: Qwen prompts, JSON/length/evidence
  validation, rule-guided retries and recorded sentence-boundary shortening.
- `GeneratorFlux2KleinL4Colab.py`: final FLUX generator; the filename retains
  an earlier L4 name but the current preset also supports the demonstrated T4 run.
- `Concatenator.py`: A4 document rendering.
- `Generator.py`, `evaluation/`: optional SD1.5 baseline and post-generation checks.
- `colab/AIColoringBook.ipynb`: full-run notebook.
- `tests/`: regression tests using mocks, temporary data, and small random models.

## Tests

From the repository root, inside a prepared environment:

```bash
python -m unittest discover -s tests -v
```

The full suite requires the imported project dependencies, including PyTorch,
Transformers, Requests, ReportLab, Pillow, NumPy, and OpenCV. It does not require
CUDA or download Qwen/FLUX/SD1.5 weights. Missing imports indicate an incomplete
test environment, not necessarily a pipeline regression. A standard-library-only
check of the biography-audit logic is available without those dependencies:

```bash
python -m unittest discover -s tests -p test_biography_audit.py -v
```

Full GPU generation must be checked separately using the Colab smoke test.

## Evaluation boundaries and resource attribution

The completed current collection is in `output/final_run_v2`, with paired
baseline, image evaluation, and biography audit in `evaluation/*_v2` folders
listed in the evaluation guide. Philip I of Hesse and Gertrud von Le Fort were
replaced after inspecting earlier image outputs; the earlier experiments remain
separate. The current set is not random or held out. Completion involved retries
and resumption, and accepted cached biographies do not all record one prompt
version. A fresh live-source run need not reproduce these historical artifacts.

Evidence IDs, word ranges, and fluent prose do not establish factual correctness
or suitability for children. The pipeline uses deterministic checks, not a model
factuality self-review. The separate Codex-assisted source audit diagnoses the
unchanged outputs; it is not independent human evaluation or a generation gate.
Image measures are proxies, not verified portrait identity or coloring usability.

Model resources are [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) and
[FLUX.2 klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B).
Keep their resource notices and the recorded Wikipedia/Wikimedia source,
license, and portrait-attribution metadata with any submitted result bundle.
Available metadata is not assumed complete. The prototype outputs are
AI-generated examples, not independently verified educational publications.
