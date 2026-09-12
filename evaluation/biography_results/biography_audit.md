# Biography audit

No biographies or PDFs were modified.

Integrity checks are deterministic. Claim labels come from one explicit agent-assisted source review, with zero independent human raters.
Supported means entailed by the saved input. Partial means an unsupported qualifier or causal relation. Source-inconsistent means saved passages conflict.

| Biography | Words | Integrity | Mean sentence words | Estimated grade | Marburg | Strictly supported claims |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| Philip I, Landgrave of Hesse | 108 | pass | 17.3 | 14.2 | yes | 15/16 |
| Robert Bunsen | 102 | pass | 14.1 | 11.5 | no | 17/19 |
| Emil von Behring | 110 | pass | 17.7 | 13.3 | yes | 17/18 |
| Jacob Grimm | 85 | pass | 12.1 | 11.9 | no | 19/20 |
| Hannah Arendt | 90 | pass | 17.4 | 13.8 | yes | 18/19 |
| Gertrud von Le Fort | 98 | pass | 15.7 | 10.4 | yes | 19/19 |
| Alfred Wegener | 85 | pass | 11.6 | 10.8 | no | 15/18 |
| Boris Pasternak | 95 | pass | 12.7 | 10.3 | no | 19/20 |

## Overview

```json
{
  "biographies": 8,
  "integrity_passed": 8,
  "min_word_count": 85,
  "max_word_count": 110,
  "mean_word_count": 96.625,
  "marburg_mentioned": 4,
  "mean_estimated_grade": 12.03125,
  "reviewed_claims": 149,
  "claim_status_counts": {
    "partial": 5,
    "source_inconsistent": 2,
    "supported": 139,
    "unsupported": 3
  },
  "strict_supported_claim_rate": 0.9328859060402684
}
```

## Findings

- Philip I, Landgrave of Hesse (partial): The church reorganization led to the university's founding. — The source places church reform and scholastic expenditure together, but does not explicitly establish this causal relationship.
- Philip I, Landgrave of Hesse: Dense political and religious terminology needs explanation for younger readers.
- Philip I, Landgrave of Hesse: The birth/death evidence is in source sentence 1, which the generated citation list omits.
- Robert Bunsen (unsupported): Bunsen studied with Liebig before becoming a professor. — The saved source says he met Liebig during his travels; it does not say he studied with him.
- Robert Bunsen (partial): Bunsen's work led to effective antidotes for poisoning. — The source supports one specific antidote against arsenic poisoning, not plural antidotes against poisoning in general.
- Robert Bunsen: No Marburg connection is mentioned, although source sentences 15-17 describe it.
- Robert Bunsen: Photochemistry, spectroscope and organic arsenic chemistry are not explained.
- Robert Bunsen: Several factual claims are supported by the full saved source but their required evidence IDs are omitted from the generated citation list.
- Emil von Behring (partial): Receiving the Nobel Prize earned Behring the saviour of children nickname. — The source supports the award and the nickname separately, but not an award-to-nickname causal relationship.
- Emil von Behring: Physiologist, serum therapy, diphtheria, tetanus and antitoxin are dense unexplained medical terms.
- Emil von Behring: Revolutionized medicine is treated as evaluative rhetoric rather than a separately scored factual claim.
- Emil von Behring: The generated citations omit sentence 1 (identity and birth), sentence 8 (Berlin) and sentence 12 (Koch).
- Jacob Grimm (unsupported): Grimm's linguistic discoveries influenced the western world. — The saved source describes linguistic discoveries but does not establish their western-world impact; sentence 25 supports the fairy-tale claim only.
- Jacob Grimm: No Marburg connection is mentioned, although source sentences 6-7 describe it.
- Jacob Grimm: Linguist, folklorist, Grimm's law, Deutsches Wörterbuch and Weisthümer may need brief explanations.
- Jacob Grimm: The source selection contains bibliographic prose and abbreviation-induced sentence fragments near IDs 32-38.
- Hannah Arendt (source_inconsistent): Arendt obtained her doctorate in 1929. — Sentence 8 gives 1929, whereas sentences 32-33 say she completed the dissertation and received the Ph.D. in 1928. The saved input does not resolve this conflict.
- Hannah Arendt: The romantic affair is source-supported, but is not necessary for an educational achievement-focused biography for ages 10-14.
- Hannah Arendt: Totalitarianism, antisemitism and Gestapo are unexplained and need careful contextualization.
- Hannah Arendt: The saved source contains stand-alone N. and C. fragments; do not interpret those as substantive evidence.
- Gertrud von Le Fort: All reviewed claims are supported by the saved source, but this is not independent verification of Wikipedia's statements.
- Gertrud von Le Fort: The opera adaptation chain and foreign titles may be difficult for younger readers.
- Gertrud von Le Fort: The saved summary was recovered by retaining an unchanged complete-sentence prefix; the frozen experiment should retain that length-adjustment metadata.
- Alfred Wegener (partial): Wegener pioneered meteorology. — The source calls him a pioneer of polar research and credits achievements in meteorology; it does not make the same pioneering claim for meteorology.
- Alfred Wegener (unsupported): They pioneered weather balloon use in Greenland. — The source locates the brothers' balloon work at the Lindenberg observatory near Beeskow, not Greenland. The summary conflates separate episodes.
- Alfred Wegener (partial): Wegener published nearly 20 papers before his death. — The source says almost 20 additional papers by the end of the war. Omitting additional and the period can suggest an incorrect lifetime total.
- Alfred Wegener: No Marburg connection is mentioned, although source sentences 11-15 describe it.
- Alfred Wegener: All 36 source sentence IDs were cited, including irrelevant family details; valid IDs alone do not demonstrate precise grounding.
- Alfred Wegener: Meteorology, continental drift and plate tectonics need accessible definitions.
- Boris Pasternak (source_inconsistent): My Sister, Life was Pasternak's first book. — The lead calls this his first book of poems, but sentences 28 and 38 describe earlier books and a first publication in 1914. The saved source is inconsistent.
- Boris Pasternak: No Marburg connection is named, although source sentences 12 and 37 describe it.
- Boris Pasternak: Composing at age 13 refers to music in the saved source context; readers could wrongly interpret it as composing poems.
- Boris Pasternak: Futurist and USSR are unexplained.
- Boris Pasternak: Source sentence 1 is an abbreviation-induced fragment, and the saved Marburg chronology requires care.

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
