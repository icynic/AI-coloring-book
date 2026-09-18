# Baseline and evaluation

Evaluation is separate from the [core pipeline](../README.md). You can inspect
the completed results without installing or loading any models. Running the
commands below is optional; all examples write to new directories so the saved
submission artifacts remain unchanged. Run commands from the repository root.

For file meanings and success checks, see the
[output schema](../docs/OUTPUT_SCHEMA.md). The recorded Windows Python 3.11.3
and Colab Python 3.13.15 runtimes are observed run records, not a universal
minimum-version or compatibility guarantee.

## Prepare only the environment you need

Reading result files and running the mechanical biography check need no model
installation. For optional local image analysis, use a separate environment.
From the repository root, this Windows PowerShell example creates one under
the ignored `output/` tree (choose a fresh environment path):

```powershell
python -m venv output/evaluation_env
.\output\evaluation_env\Scripts\python.exe -m pip install "Pillow>=10,<13" "numpy>=2,<3" "opencv-python==4.13.0.92"
```

For the analysis commands below, replace `python` with
`.\output\evaluation_env\Scripts\python.exe` to use that environment. On
Linux/macOS the equivalent executable is `output/evaluation_env/bin/python`.
No activation is required when the executable path is used explicitly.
These packages are sufficient for the nine pixel proxies, not model inference.

For DINOv2, install an appropriate PyTorch build using the
[official installation selector](https://pytorch.org/get-started/locally/),
choosing CPU or the compute platform suited to the machine. Run its install
command with the environment's Python (`-m pip`), then install the project's
embedding-library version:

```powershell
.\output\evaluation_env\Scripts\python.exe -m pip install "transformers==5.5.0"
```

For the optional CUDA baseline or full regression suite, prepare PyTorch in
that isolated environment, then install the full project requirements:

```powershell
.\output\evaluation_env\Scripts\python.exe -m pip install -r requirements.txt
.\output\evaluation_env\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Baseline GPU reproduction requires the CUDA check to print `True`; CPU is
sufficient for regression tests and DINOv2. Neither requirements file installs
Torch. `requirements.txt` includes the Colab pins, so do not install it into
shared global Windows Python. Installation specs do not reconstruct an
unrecorded historical environment; retain any installation failure traceback
rather than broadly upgrading dependencies. Actual Colab setup is covered in
[the Colab runbook](../docs/COLAB.md).

## Completed results: start here

The final prototype is `output/final_run_v2`. Its paired evaluation was completed
on 2026-09-13 for all eight subjects in [subjects.csv](subjects.csv): Otto Hahn,
Robert Bunsen, Emil von Behring, Jacob Grimm, Hannah Arendt, K. Ferdinand Braun,
Alfred Wegener and Boris Pasternak. This CSV is also the default subject list for
the core CLI, Colab notebook, baseline and image evaluation.

| Saved artifact | What to inspect |
| --- | --- |
| [Baseline manifest](baseline_run_v2/manifest.json) | Eight SD1.5 + ControlNet outputs, per-person seeds and errors |
| [Image evaluation](automatic_results_v2/automatic_evaluation.md) | All ten metrics, paired differences, bootstrap intervals and exploratory p-values for 16 images |
| [Image evaluation provenance](automatic_results_v2/automatic_evaluation.json) | Metric definitions, runtime, canvas settings and resolved DINOv2 revision |
| [Biography audit](biography_results_v2/biography_audit.md) | Integrity, approximate readability and source-review findings |
| [Claim annotations](biography_results_v2/claim_annotations.json) | Archived qualitative labels, evidence IDs, text hashes and reviewer provenance |
| [Informative Drawings comparator](informative_drawings_v2/README.md) | Eight appendix-only outputs plus code, checkpoint, environment and hash provenance |

The baseline used the exact downloaded source portraits and seeds 42-49,
matching the FLUX metadata. Source hashes and seed pairing were checked; the
original Colab outputs were verified byte-for-byte unchanged. No images were
chosen from multiple candidate seeds.

After the quantitative analysis was frozen, the released Informative Drawings
`anime_style` checkpoint was run once on all eight same source portraits. These
images provide all-case qualitative context in the report appendix. They are
not part of `automatic_results_v2`, any significance test, or model selection;
no quantitative result below includes them.

Key image results are mean DINOv2 source similarity 0.4634 for FLUX versus
0.3869 for SD1.5 (7/8 higher; exact paired p = 0.0625), white space 89.57%
versus 77.97%, and midtone fill 5.47% versus 10.77% (both 8/8 in the indicated
direction; p = 0.0078125). The linked report contains all ten metrics. Largest
connected dark regions do not favor FLUX on average, so these findings do not
establish uniform superiority.

All eight biographies pass mechanical checks, contain 80-105 words (mean
93.875) and mention Marburg. The separate qualitative audit labels 143
propositions: 137 supported, one partial, three unsupported and two
source-inconsistent. These labels are from one Codex-assisted pass, not the Qwen
generator, independent human raters or an automatic entailment metric. They
describe support relative to the saved Wikipedia input, not independently
verified historical truth.

## 1. Optional: generate a new paired baseline

This step performs image inference, unlike the recalculations below. Download
the **complete** Colab run folder to `output/final_run_v2`, including
`sources/images/`; downloading only the PDF is insufficient. Use a suitable
CUDA machine and the full baseline dependencies described above.
The wrapper supports CPU fallback, but full baseline inference is not a
recommended CPU workflow and can be very slow. Network access is needed to
download uncached model weights.

```bash
python -u evaluation/run_baseline.py --source-run output/final_run_v2 --output-dir evaluation/baseline_reproduction --seed 42 --steps 15 --controlnet-scale 0.6 --guidance-scale 10.0
```

This uses `Generator.py`: SD1.5, line-art ControlNet, a line-art detector and
AnimeLineartLoRA. The model is loaded once. The script reads the default CSV in
order and assigns `seed + subject_index`; keep this order to preserve pairing.
For another cohort, pass the same `--subjects path/to/subjects.csv` to both
baseline generation and image evaluation, and ensure all corresponding source
portraits and FLUX outputs exist.

Outputs are `generated_images/`, `control_images/`, `generation_metadata/` and
`manifest.json`. Check that the manifest has eight items with empty `errors`
lists and that every expected image and metadata file exists. The baseline
wrapper can finish with exit code 0 despite recorded per-person failures, so
the exit code alone is not a success check.

Existing image/metadata pairs are reused unless `--force` is passed. To resume,
keep the same directory, subjects and settings. For changed settings, use
another new directory: this wrapper does not validate cached outputs against
changed configuration. Do not regenerate into `baseline_run_v2` or select the
best-looking seed after inspecting results. A fresh baseline run is a new
experiment, not a byte-identical recovery of the archived baseline: historical
checkpoint revisions were not pinned in its wrapper.

## 2. Recalculate automatic image metrics

This step reads saved images; it does not run FLUX or SD1.5. The commands here
use the archived pair so generation changes are not mixed into recalculation.
To evaluate a newly generated baseline instead, change `--baseline-run` to
`evaluation/baseline_reproduction` and use a distinct output directory.

### Pixel-only: offline, no model inference

With Pillow, NumPy and OpenCV installed, these nine pixel metrics need no GPU,
Torch, Transformers or network access:

```bash
python evaluation/evaluate_images_auto.py --flux-run output/final_run_v2 --baseline-run evaluation/baseline_run_v2 --output-dir evaluation/automatic_pixel_recalculation --skip-embedding --canvas-size 512 --seed 20260911 --bootstrap-repetitions 10000
```

They cover white-space/ink, dark-fill, midtone, edge density,
connected-component and unexpected-color ratios. Pixel images are resized
without changing aspect ratio and white-padded to a common 512-square canvas.
The source-similarity result is unavailable in this mode; do not present it as
a zero score or a complete ten-metric evaluation.

### All ten metrics: include DINOv2

This adds Torch, Transformers and a DINOv2-small model download (unless cached).
CPU is supported; GPU is optional. `--device auto` selects CUDA when available,
otherwise CPU. This example explicitly uses CPU:

```bash
python evaluation/evaluate_images_auto.py --flux-run output/final_run_v2 --baseline-run evaluation/baseline_run_v2 --output-dir evaluation/automatic_recalculation --device cpu --canvas-size 512 --seed 20260911 --bootstrap-repetitions 10000
```

For a compatible CUDA environment, use `--device cuda`; an unavailable CUDA
device raises an error. Add `--local-files-only` only if the model and processor
are already cached. Missing cached weights cause an error, not an automatic
switch to pixel-only mode. Reduce `--embedding-batch-size` from its default 4
if necessary.

The saved evaluation used `facebook/dinov2-small`, resolved revision
`ed25f3a31f01632728cabb09d1542f84ab7b0056`. The current loader records the
resolved revision but does not pin it. A later download or another runtime can
therefore differ; inspect the new JSON provenance before comparing scores.
Embeddings use the DINO image processor, not the padded pixel-metric canvas.

Both modes produce:

- `per_image_metrics.csv`: one row per generated image, input/output hashes and
  original dimensions (16 rows for this cohort).
- `paired_metric_summary.csv`: method means, paired differences, exact
  subject-level sign-flip tests and 95% bootstrap intervals.
- `automatic_evaluation.md`: readable summary and caveats.
- `automatic_evaluation.json`: settings, definitions, resolved embedding
  revision and runtime provenance.

Check the log reports `Subjects: 8; images: 16` and the expected embedding mode.
An absent input image raises an error; do not silently omit failed subjects.
All image measures are proxies, not human preference, coloring usability or
face-recognition accuracy. Lower complexity is not always better, and white
space can reward an overly empty image. P-values are exploratory and uncorrected
for multiple comparisons.

## 3. Check saved biographies without Qwen

These checks use the Python standard library and repository validation helpers.
They load no language model, do not regenerate text, and do not modify source
files, summaries or PDFs.

### Mechanical checks only

```bash
python evaluation/audit_biographies.py --flux-run output/final_run_v2 --output-dir evaluation/biography_checks_recalculation
```

Without `--annotations`, the script checks JSON structure, the manifest's word
range, source text hashes/revisions and stored evidence IDs/sentences. It also
estimates readability and produces a numbered source review packet. Evidence
IDs being valid does **not** mean every generated claim is semantically
supported. The script does not infer new claim labels or run model self-review.

### Reaggregate the archived qualitative audit

```bash
python evaluation/audit_biographies.py --flux-run output/final_run_v2 --output-dir evaluation/biography_recalculation --annotations evaluation/biography_results_v2/claim_annotations.json
```

This recalculates aggregates from existing explicit labels; it does not perform
a fresh factuality review. Each annotation is bound to the saved source and
summary text by SHA256. Changed text raises a stale-annotation error and needs
new, transparently documented annotations if claim-support results are to be
reported. Do not reuse original-cohort annotations for replaced subjects or
relabeled/edited text. Codex was the separate single-pass reviewer on
2026-09-13; there were zero independent human raters and no Qwen self-review
loop.

Both commands produce `biography_audit.json`, `biography_audit.md`,
`review_packet.md` and `report_text_evaluation.md`. Check eight biographies are
present and inspect every integrity result. With archived annotations, verify
the 143-claim denominator and status counts above. Approximate Flesch measures
use an English vowel-group syllable heuristic; proper names and technical terms
can distort them, and the scores do not establish suitability for ages 10-14.

## Provenance and interpretation limits

The final prototype was resumed, not generated as a clean single-prompt
benchmark: only Bunsen's saved biography records prompt version 2; the seven
reused biographies predate that metadata. Current prompt templates and
dependency specifications are not complete historical prompt/environment logs.
Report saved metadata where available and state what was not recorded.

FLUX's saved dtype is `torch.bfloat16`; do not describe that saved run as FP16
or claim independent verification of every internal compute dtype. Its prompt
and output resolutions differ from the baseline. Grimm's FLUX image is only
256 by 288 pixels. The shared pixel canvas does not make the underlying
generation settings identical.

Recorded per-person image times sum to 83.026 seconds for the local baseline
and 95.617 seconds for Colab FLUX, excluding model loading. Different GPUs and
resolutions make these unsuitable for a controlled speed comparison. The
manifest's short resumed duration is not the cost of generating a full book
from scratch.

The current cohort replaced Philip I and Gertrud von Le Fort after image-quality
inspection. The original `output/evaluation_flux_t4`, `baseline_run`,
`automatic_results`, `biography_results`, A/B images and hash-bound
`biography_claim_annotations.json` describe the original cohort, not this run;
they are not submission artifacts and remain available in Git history. They
must not be substituted for or relabeled as the current cohort. Report post-hoc
replacements, length-policy changes, resource constraints, every intended
subject and any failures. The final prototype and evaluated text remain
unchanged; manual text polishing is not part of these evaluation commands.
