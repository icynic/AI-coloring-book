# ACL report

The compiled submission PDF is available from the
[GitHub Release](https://github.com/icynic/AI-coloring-book/releases/download/Output/ai_coloring_book_report.pdf).
The instructions below rebuild it from source.

The report contains Abstract, Introduction, Related Work, Data and Resources,
Method, Evaluation, Results, Discussion, Conclusion, and two informative appendices.
It is a systems-prototype case study, not a newly trained model or a verified
educational intervention. Author information is still empty and must be supplied
before submission; confirm the working title at the same time.

## Files

- `main.tex`: article, result tables, and appendix inclusion.
- `appendix_reproducibility.tex`: saved configuration, current prompt templates,
  installation specification, commands, artifact locations, current source-file
  SHA-256 hashes, and provenance limits.
- `appendix_examples.tex`: all eight current image pairs and the unchanged first
  generated book page.
- `figures/`: exact copies of the evaluated drawings plus a rendering of the
  first generated page. These assets make report compilation self-contained.
- `references.bib`: cited papers, model cards, and resource documentation.
- `acl-template/`: unchanged official ACL files.

The verified PDF has 15 pages: content ends on page 8, references occupy pages
8-9, reproducibility details pages 10-12, and unchanged output examples pages
13-15. The course limit is eight two-column content pages; references and
informative appendices are outside that limit. All pages were rendered and
visually checked, with no overflow or unresolved citations/references. Confirm
pagination after every later change, especially after adding author information.

## Compile locally

Run from `report/` with a TeX distribution and `latexmk` available:

```powershell
$aclStyleSearch = (Join-Path (Get-Location).Path 'acl-template').Replace('\', '/')
$aclCaptionCache = (Join-Path (Get-Location).Path '../output/.latex-deps/caption').Replace('\', '/')
$env:TEXINPUTS = "$aclStyleSearch;$aclCaptionCache;"
$env:BSTINPUTS = "$aclStyleSearch;"
$env:BIBINPUTS = "$((Get-Location).Path.Replace('\', '/'));"
latexmk '-pdf' '-interaction=nonstopmode' '-halt-on-error' '-jobname=ai_coloring_book_report' '-outdir=../output/pdf/acl_report' 'main.tex'
```

The PDF is `../output/pdf/acl_report/ai_coloring_book_report.pdf`.
Build products stay under the project's ignored `output/` directory.

For Overleaf, upload `main.tex`, both appendix `.tex` files,
`references.bib`, and the entire `figures/` directory, preserving that directory
name. Also upload `acl-template/acl.sty` and
`acl-template/acl_natbib.bst` to the project root. Select pdfLaTeX.
The report does not need generation outputs outside these copied assets to compile.

## Evidence and reproducibility

Reported results use only the current collection:

- `../output/final_run_v2/`: saved Wikipedia inputs, summaries, FLUX drawings,
  generation metadata, individual pages, and the eight-page book.
- `../evaluation/baseline_run_v2/`: SD1.5-based baseline outputs and metadata.
- `../evaluation/automatic_results_v2/`: ten paired image proxies and statistics.
- `../evaluation/biography_results_v2/`: integrity/readability checks,
  hash-bound claim annotations, review inputs, and the qualitative source audit.

No evaluated biography, drawing, annotation, or metric is revised for presentation.
The examples contain all eight subjects, not a favorable subset. The displayed
complete page is the first book page (Otto Hahn), chosen by the saved ordering.
Graphic scaling and PDF-to-image rendering are presentation operations only.

Completion includes retries and resumption. Only Bunsen's cached summary records
prompt version 2; the other seven do not record a prompt version. The appendix
labels the full Qwen prompt as the current source template, not a recovered prompt
for every historical summary. A fresh run retrieves current Wikipedia material
and need not recreate those cached outputs. Recalculation of reported results
instead uses the saved artifacts and annotations.

Codex supplied claim decomposition, selected evidence, and semantic labels.
Python checked artifact integrity and evidence-ID inclusion and aggregated the
annotations; it did not perform semantic entailment evaluation. No independent
human raters, historical verification, or Qwen self-review were used.
Missing reviewer-model information and unrecorded historical software/code
versions remain explicitly unknown rather than being inferred from this machine.

## Official style and local dependencies

Official template source: <https://github.com/acl-org/acl-style-files/>.
Downloaded on 2026-09-14 at upstream revision
`d5adc823ff0f80f98c80405ca0ab66c68e684409`.
No official style definitions, font sizes, or margins are modified.
The report uses `preprint` mode for course-report page numbers. Optional
`microtype` and `inconsolata` are omitted; required Times text remains unchanged.

A standard TeX installation supplies ACL's required `caption` package.
This machine's minimal TinyTeX lacks it and its package manager requires a
self-update. Instead of upgrading the environment, unchanged `caption.sty` and
`caption3.sty` were cached in `../output/.latex-deps/caption/` from
<https://tlnet.yihui.org/archive/caption.tar.xz>.
The local search path above uses that cache if present. A complete TeX distribution
or Overleaf supplies the package normally.
