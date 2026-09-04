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
baseline. Do not use an LLM to judge its own output.

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

## Reporting rules

- Report all eight intended subjects and every failure.
- Report the number of evaluators and missing ratings.
- Treat automatic line/colorability measurements as proxies, not proof of
  perceptual quality.
- Do not claim statistical significance when ratings are incomplete.
- Keep the randomization seed, model parameters, manifests, raw ratings, and
  condition key with the final submission artifacts.
