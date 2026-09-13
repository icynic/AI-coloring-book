# Baseline and automatic evaluation

These tools remain separate from the core generation pipeline because the
final report needs evaluation. The canceled human-rating packet, reviewer
collage and rating workbook are removed; paired statistics are contained in
`evaluate_images_auto.py`.

## Current subjects and original results

`subjects.csv` is the sole current eight-person list used by the notebook,
default CLI, baseline and image evaluation. It replaces Philip I with Otto Hahn
and Gertrud von Le Fort with Ferdinand Braun for the new complete prototype.

This selection followed inspection of image quality. The completed original
`output/evaluation_flux_t4`, `baseline_run`, `automatic_results`,
`biography_results` and existing A/B images are preserved. Their scores and
hash-bound `biography_claim_annotations.json` describe the original subjects and
text, not the new run. The former subject table is available in Git history.
Do not overwrite or relabel the original results.

## Generate a paired baseline locally

After downloading the complete Colab `final_run_v2` folder to `output/`, run:

```bash
python evaluation/run_baseline.py --source-run output/final_run_v2 --output-dir evaluation/baseline_run_v2 --seed 42
```

The baseline uses `Generator.py` (SD1.5 + ControlNet) and the exact downloaded
source portraits. It loads the model once. Use one predetermined seed per
subject; do not choose the best-looking output.

## Automatic image evaluation

```bash
python evaluation/evaluate_images_auto.py --flux-run output/final_run_v2 --baseline-run evaluation/baseline_run_v2 --output-dir evaluation/automatic_results_v2
```

Measurements include white-space/ink, dark-fill, midtone, edge density,
connected-component and unexpected-color ratios. DINOv2 source-portrait cosine
similarity is included by default; use `--skip-embedding` for offline pixel
metrics only.

Results contain per-image metrics, paired summaries, exact subject-level
sign-flip tests, bootstrap intervals and provenance. They are image proxies,
not human preference, coloring usability or face-recognition accuracy.
Exploratory p-values are uncorrected for multiple comparisons. Rerun the
analysis for the new eight figures; old scores cannot be transferred.

## Biography checks

```bash
python evaluation/audit_biographies.py --flux-run output/final_run_v2 --output-dir evaluation/biography_results_v2
```

Without annotations, this performs only mechanical integrity/provenance and
approximate readability checks and produces a numbered source review packet.
It does not automatically decide whether claims are factually supported.

Old annotations are a single agent-assisted qualitative audit, not independent
human evaluation. Changed sources or text require fresh, hash-bound annotations
if claim-support results are to be reported. Keep final prototype edits and
original evaluated text distinct; there is no same-model self-review loop.

Report every intended subject and failure, actual GPU/model settings and any
post-hoc subject/length-policy changes. Keep the less successful original
examples when discussing limitations.
