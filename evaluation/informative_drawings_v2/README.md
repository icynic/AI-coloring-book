# Informative Drawings qualitative comparator

This directory preserves one complete, non-selected run of the released
Informative Drawings `anime_style` checkpoint over the eight frozen
`final_run_v2` source portraits.

The outputs are an appendix-only, post-hoc qualitative comparator. They were
not included in `automatic_results_v2`, statistical testing, model selection,
or any claim of quantitative superiority. Every subject is displayed in the
report appendix. No generated image was edited, reranked, or regenerated.

## Contents

- `generated_images/`: the eight exact generated PNG files.
- `manifest.json`: code and checkpoint identity, exact command, runtime,
  preprocessing, filename mapping, dimensions, and SHA-256 hashes.

The report copies these PNG files byte-for-byte into `report/figures/` using
the prefix `informative_`.

## Protocol

The upstream repository was checked out at commit
`2349aee4daf7cb01d8de645b0bbb4f4392fd1395`. The checkpoint was fixed to the
upstream README example, `anime_style`, before inspecting the eight formal
outputs. The formal batch ran once. A separate upstream-example smoke test was
performed first and is not part of these artifacts.

The eight source portraits came from
`output/final_run_v2/sources/images/`. They were copied byte-for-byte to an
ignored staging directory. `K._Ferdinand_Braun.jpg` was staged as
`K_Ferdinand_Braun.jpg` because the upstream loader truncates basenames at the
first period; the canonical output name was restored afterward without pixel
changes.

From the Informative Drawings checkout, the formal command was:

```powershell
conda run --no-capture-output -n drawings python test.py `
  --name anime_style `
  --dataroot "D:\Projects\AIColoringBook\output\informative_drawings_work\inputs_sanitized" `
  --results_dir "D:\Projects\AIColoringBook\output\informative_drawings_work\raw"
```

The released test path resizes the shorter image edge to 256 pixels with
bicubic interpolation while preserving aspect ratio and then applies one
feed-forward generator pass. It uses neither a diffusion prompt nor a random
sampling seed. See `manifest.json` for the recorded modern environment; it is
not a reconstruction of the original paper's PyTorch 1.7.1 environment.
