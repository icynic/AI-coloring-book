# Biography audit

No biographies or PDFs were modified.

Integrity checks are deterministic. Claim labels come from one explicit agent-assisted source review, with zero independent human raters.
Supported means entailed by the saved input. Partial means an unsupported qualifier or causal relation. Source-inconsistent means saved passages conflict.

| Biography | Words | Integrity | Mean sentence words | Estimated grade | Marburg | Strictly supported claims |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| Otto Hahn | 89 | pass | 17.0 | 15.3 | yes | 17/17 |
| Robert Bunsen | 80 | pass | 13.2 | 11.8 | yes | 15/18 |
| Emil von Behring | 105 | pass | 20.6 | 13.3 | yes | 18/18 |
| Jacob Grimm | 90 | pass | 14.5 | 11.9 | yes | 20/20 |
| Hannah Arendt | 103 | pass | 25.2 | 15.5 | yes | 16/17 |
| K. Ferdinand Braun | 91 | pass | 18.6 | 13.0 | yes | 17/17 |
| Alfred Wegener | 105 | pass | 20.2 | 13.4 | yes | 17/18 |
| Boris Pasternak | 88 | pass | 17.0 | 12.3 | yes | 17/18 |

## Overview

```json
{
  "biographies": 8,
  "integrity_passed": 8,
  "min_word_count": 80,
  "max_word_count": 105,
  "mean_word_count": 93.875,
  "marburg_mentioned": 8,
  "mean_estimated_grade": 13.318375,
  "reviewed_claims": 143,
  "claim_status_counts": {
    "partial": 1,
    "source_inconsistent": 2,
    "supported": 137,
    "unsupported": 3
  },
  "strict_supported_claim_rate": 0.958041958041958
}
```

## Findings

- Otto Hahn: The return to Germany in 1906 is supported by source sentence 4, which is absent from the generated evidence list. This is a citation-coverage problem, not an unsupported source claim.
- Otto Hahn: The text emphasizes radiochemistry and Marburg but omits nuclear fission and a Nobel Prize; those facts are not present in the selected input, so omission is not scored as fabrication.
- Robert Bunsen (unsupported): Bunsen studied with Liebig. — The saved input says that he met Liebig during his travels, not that he studied under him.
- Robert Bunsen (unsupported): The 1855 burner development occurred at Marburg, as implied by There after the Marburg sentence. — The source moves him to Breslau in 1851 and Heidelberg in 1852; the burner appears in the Heidelberg account. The summary merges different career stages.
- Robert Bunsen (partial): His emission-spectrum work led to the creation of the Bunsen-Kirchhoff Award. — The award is named after Bunsen and Kirchhoff, but the saved source does not establish the stated causal account of its creation.
- Robert Bunsen: Correct the Liebig student relationship and the Marburg burner location before releasing a polished educational text; remove the unsupported award-creation causal wording.
- Robert Bunsen: The source contains both the Marburg association and the later Heidelberg burner work, so this is a summarization relation error, not insufficient source length.
- Emil von Behring: The principal factual statements are supported relative to the saved input. This does not independently verify the historical accuracy of the input.
- Emil von Behring: The tetanus-therapy claim requires sentences 13-15, which the generator did not cite; this illustrates incomplete citation coverage despite mechanically valid evidence IDs.
- Emil von Behring: For a children's text, explain serum therapy and diphtheria in simpler language; approximate readability alone is not an age-suitability judgment.
- Jacob Grimm: The factual content is supported by the saved source. The dictionary was unfinished by the brothers and completed by later scholars; wrote should not be read as claiming they completed all volumes.
- Jacob Grimm: The line-art output is only 256 by 288 pixels before placement on A4, unlike the larger portraits. Report this input/output-resolution limitation rather than selecting another image after evaluation.
- Hannah Arendt (source_inconsistent): Arendt obtained her doctorate in 1929. — The lead says 1929; the later saved passage says that she completed her dissertation in 1928 and received her Ph.D. that year.
- Hannah Arendt: The doctorate year cannot be resolved consistently from this saved input. This is a source conflict, not a demonstrated invention by the generator.
- Hannah Arendt: The romantic relationship is source-supported but unnecessary for the intended coloring-book audience; suitability is a qualitative editorial concern, not a factual-support error.
- Hannah Arendt: Long topic lists and specialist terms make this less accessible than its word count alone suggests.
- K. Ferdinand Braun: These claims, including the first-invention descriptions, are supported relative to the saved source; priority claims were not independently checked against the wider historical literature.
- K. Ferdinand Braun: The summary includes his student connection with Marburg but omits his later professorship; that is an omission, not a source-grounding error.
- Alfred Wegener (unsupported): Wegener died in 1931, as stated in the final sentence. — The source gives death in November 1930; 12 May 1931 is the discovery date. The biography also contradicts its own opening life dates.
- Alfred Wegener: Correct the death/discovery-year confusion before publication. The saved source distinguishes the two dates; this is not an internally conflicting-source case.
- Alfred Wegener: The account of the fatal journey takes substantial space that could instead introduce continental drift simply for the intended age group.
- Boris Pasternak (source_inconsistent): My Sister, Life was Pasternak's first book of poems. — Sentence 3 calls the 1922 book his first, but sentence 38 says that he published his first book four years after 1910; sentence 28 also describes earlier first and second books.
- Boris Pasternak: Resolve or omit the first-book qualifier before publication; the saved source gives incompatible first-publication accounts.
- Boris Pasternak: The age-thirteen composition claim is supported by source sentence 36, which the generated evidence list omits.
- Boris Pasternak: His Nobel Prize and Doctor Zhivago are present in the input but absent from this biography; achievement coverage is an editorial limitation, not fabrication.

## Limitations

- Rule checks validate stored provenance and structure, not semantic entailment.
- Claim labels are explicit agent-assisted annotations, not an independent human evaluation or an automatic NLI metric.
- Support is relative to saved Wikipedia input, which can contain factual conflicts.
- Flesch metrics use estimated English syllables and are approximate, especially for proper names and specialist vocabulary.
- Readability formulas do not establish suitability for ages 10-14.

## Readability method and references

Length uses whitespace tokens. Readability uses Unicode letter words, excluding numeric dates.
Syllables are estimated by a documented heuristic, not a pronunciation dictionary.

- [Flesch, R. (1948). A new readability yardstick.](https://doi.org/10.1037/h0057532)
- [Kincaid, J. P., Fishburne, R. P., Rogers, R. L., and Chissom, B. S. (1975). Derivation of new readability formulas for Navy enlisted personnel.](https://stars.library.ucf.edu/istlibrary/56/)
