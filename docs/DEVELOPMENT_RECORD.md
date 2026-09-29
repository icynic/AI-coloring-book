# Development record and verification

This is a retrospective record of dated Git commits and saved artifacts. Commit
dates show when work was recorded, not how long a task took or a pre-registered
schedule. The [report](../report/README.md) describes the evaluated system and
its limits; the author-contribution statement is in `report/main.tex`.

| Recorded date | Milestone and deliverable | Evidence |
| --- | --- | --- |
| 2026-03-09 | Recorded model exploration, generated examples, and project notes. | Commit `e5b15d3` |
| 2026-04-08–09 | Added the first fetch, summary, image, and orchestration scripts, then batch processing. | Commits `b14f3b8`, `d52b5b0` |
| 2026-06-05 | Added a FLUX experiment and presentation assets. | Commit `da1b379` |
| 2026-09-03 | Integrated Qwen3.5-4B and FLUX.2 [klein] in Colab and added a T4 preset. | Commits `69c2e50`, `1092d78` |
| 2026-09-04–13 | Prepared evaluation, added image/text checks, finalized the eight-person cohort, and saved the completed T4 run. | Commits `4cbf8af`, `e268daf`, `644b781`; `output/final_run_v2/manifest.json` |
| 2026-09-16–18 | Added the ACL report, software documentation, demo, and post-hoc Informative Drawings comparison. | Commits `ec33578`, `0c8abcc`, `4202f86` |

## Verification recorded on 2026-09-29

From the repository root in the prepared local `.venv`:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The suite reported **34 tests, OK**. It covers pipeline helpers, source
selection, summary validation and retries, output gates, automatic image
evaluation, biography-audit integrity, and the distributed notebook. It uses
mocks and small random models; passing it does not exercise Qwen/FLUX weights or
prove a fresh end-to-end GPU run.

The archived `final_run_v2/manifest.json` records a completed invocation on
2026-09-13 with a Tesla T4, eight requested people, eight empty per-person
`errors` lists, and a non-null `book_path`. Because the run used resumption,
its last invocation timestamps are not the full generation time. The saved
eight-page PDF and generated assets are separate evidence of the completed
prototype. A new live-source run may produce different text and images.
