# Marburg evaluation protocol

This directory defines the frozen evaluation for the final report. The eight
subjects all have a documented connection to Philipps-Universität Marburg. Do
not replace a subject because one method produces an unattractive result.

## 1. Freeze and run the final system in Colab

Commit and push the final code before starting the experiment. Record the
printed Git commit and GPU name. In the Colab notebook, replace `NAMES` with the
contents of `subjects.txt` and use:

```python
OUTPUT_DIR = "/content/drive/MyDrive/AIColoringBook/evaluation_flux_t4"
SEED = 42
T4_SAFE_MODE = True
FORCE_REGENERATE = False
```

Keep the completed run directory unchanged. Download or synchronize the whole
directory, including `manifest.json`, `sources`, `summaries`,
`generated_images`, and `generation_metadata`.

## 2. Generate the paired baseline locally

Use the exact source portraits saved by the final run. From the repository
root, run:

```bash
python evaluation/run_baseline.py \
  --source-run path/to/evaluation_flux_t4 \
  --output-dir evaluation/baseline_run \
  --seed 42
```

The wrapper loads SD1.5 + ControlNet once and generates one baseline per
subject. Existing outputs are reused. Never select the best of multiple seeds.

## 3. Create the blinded image packet

```bash
python evaluation/prepare_blind_evaluation.py \
  --flux-run path/to/evaluation_flux_t4 \
  --baseline-run evaluation/baseline_run \
  --output-dir evaluation/blind_packet \
  --raters 3 \
  --seed 20260904
```

Give evaluators only the `images` directory and `image_ratings.csv`. Keep
`condition_key.csv` private until all ratings are returned. Copy the completed
ratings into the matching sheets of `marburg_evaluation.xlsx` if spreadsheet
entry is preferred.

Each evaluator independently scores A and B from 1 (poor) to 5 (excellent):

- `identity`: resemblance to the source historical figure.
- `line_cleanliness`: clear contours with little noise or unwanted shading.
- `coloring_suitability`: open white regions and lines suitable for coloring.
- `overall_quality`: usefulness as a children's coloring-book illustration.
- `preference`: A, B, or Tie.

Evaluators may view the original source portrait while rating identity. They
must not see method names or the condition key.

## 4. Evaluate generated biographies

For every summary, split the text into atomic factual claims. Record the total
claim count and the number supported by the saved Wikipedia source. Also score
readability and age appropriateness from 1 to 5 and verify that every saved
supporting sentence ID is within the source sentence range.

This is an audit of the final Qwen output, not a comparison against an image
baseline. Do not use the generator's self-check verdicts as independent factual
accuracy measurements. Same-model feedback may be a system refinement stage,
but its outputs still need a separate source audit.

## 5. Analyze the completed forms

```bash
python evaluation/analyze_evaluation.py \
  --image-ratings evaluation/blind_packet/image_ratings.csv \
  --condition-key evaluation/blind_packet/condition_key.csv \
  --text-ratings evaluation/blind_packet/text_ratings.csv \
  --flux-run path/to/evaluation_flux_t4 \
  --baseline-run evaluation/baseline_run \
  --output-dir evaluation/results
```

The analysis uses subject-level paired means, an exact sign-flip permutation
test, and a paired bootstrap confidence interval. Preference uses an exact
two-sided binomial test after excluding ties. This avoids treating multiple
ratings of the same subject as independent samples.

## 6. Automatic image proxies

If no perceptual study is conducted, run the reproducible automatic image
analysis instead of leaving the Evaluation section empty:

```bash
python evaluation/evaluate_images_auto.py \
  --flux-run path/to/evaluation_flux_t4 \
  --baseline-run evaluation/baseline_run \
  --output-dir evaluation/automatic_results
```

The script normalizes every generated image to a 512 by 512 white canvas and
reports white-space, ink, dark-fill, midtone, edge, small-component, largest
dark-region, and unexpected-color ratios. By default it also downloads
`facebook/dinov2-small` once and records a source-portrait cosine similarity.
Use `--skip-embedding` for a fully offline run.

The output contains per-image measurements, paired method summaries, an exact
subject-level sign-flip test, a paired bootstrap interval, and the exact model
revision used. Treat all measurements as proxies. In particular, DINOv2
similarity is not face-recognition accuracy, low edge density can reward an
overly empty drawing, and the exploratory p-values are not corrected for
multiple comparisons.

## 7. Biography integrity and source-grounded audit

```bash
python evaluation/audit_biographies.py \
  --flux-run path/to/evaluation_flux_t4 \
  --annotations evaluation/biography_claim_annotations.json \
  --output-dir evaluation/biography_results
```

Without `--annotations`, the script runs only deterministic length, provenance,
evidence-sentence matching, and approximate readability checks and creates a
numbered source review packet. It does not automatically label factual claims
as supported. The saved annotations are a single **agent-assisted** qualitative
review, not a human evaluation or an automatic entailment benchmark. Every
annotation is bound to the exact summary and source text hashes; changing either
requires a new review.

The four claim labels are supported, partial, unsupported, and source-inconsistent.
The latter separates Wikipedia input conflicts from unsupported additions by
the summarizer. Approximate Flesch scores use an English syllable heuristic and
do not prove age appropriateness. The script never modifies the frozen summaries
or PDFs. Use any suggested edits in a separate final-prototype revision and retain
the evaluated version for reproducibility.

## Reporting rules

The Qwen self-review/refinement stage has been removed; automatic image metrics
and the frozen source-grounded biography audit remain available. Keep later
prototype edits separate from the evaluated biographies. Existing annotations
are hash-bound and must be reviewed anew for changed text. A before/after
comparison must use fresh source audits, not the model's own pass rate.

- Report all eight intended subjects and every failure.
- Report the number of evaluators and missing ratings.
- Treat automatic line/colorability measurements as proxies, not proof of
  perceptual quality.
- Do not claim statistical significance when ratings are incomplete.
- Keep the randomization seed, model parameters, manifests, raw ratings, and
  condition key with the final submission artifacts.
