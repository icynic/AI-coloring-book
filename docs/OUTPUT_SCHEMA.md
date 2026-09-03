# Output and provenance schema

Each run uses one output directory. Files are reusable checkpoints as well as
the provenance record for the final evaluation.

## `sources/<person>.json`

Contains the original query, resolved Wikipedia title, plain-text lead,
Wikipedia page and revision IDs, revision timestamp, retrieval timestamp,
portrait URL and local path, Wikimedia file title/revision information, artist,
credit, license, license URL, and SHA-256 hashes of the source text and image.

## `summaries/<person>.json`

Contains the generated biography, supporting source sentence IDs and sentence
text, word count, target age, requested length, Qwen model ID and revision,
quantization and model-load time,
raw model response, source revision, and creation time.

The evidence is an audit aid, not a correctness guarantee. The final evaluation
must still verify each atomic claim against the saved source.

## `generation_metadata/<person>.json`

Contains the FLUX model ID and revision, precision and quantization, model-load
time, complete prompt, seed,
steps, guidance scale, output dimensions, elapsed generation time, peak
allocated CUDA memory, input path, output path, and creation time.

## `manifest.json`

Contains the final run configuration, Python/PyTorch/GPU runtime information,
combined book path, and a compact record for every requested person. Errors are
retained in the per-person `errors` list rather than dropping failed samples.

The manifest schema is versioned with `schema_version`. New fields may be added
without changing the meaning of existing fields.
