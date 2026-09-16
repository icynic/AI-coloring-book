# Automatic image evaluation

Measurements use the supplied subject list and paired FLUX–SD1.5 outputs. Consult the manifest for the exact run and inputs.
Pixel metrics were calculated after aspect-preserving resize and white padding to 512×512.

| Metric | Direction | FLUX mean | SD1.5 mean | Difference | 95% bootstrap CI | p | FLUX wins |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| source_similarity_proxy | higher | 0.4634 | 0.3869 | 0.0765 | [0.0141, 0.1364] | 0.0625 | 7/8 |
| white_space_ratio | higher | 0.8957 | 0.7797 | 0.1159 | [0.0550, 0.1904] | 0.0078 | 8/8 |
| ink_coverage_ratio | descriptive | 0.0947 | 0.1675 | -0.0729 | [-0.1137, -0.0412] | 0.0078 | n.a. |
| dark_fill_ratio | lower | 0.0400 | 0.0599 | -0.0199 | [-0.0453, 0.0006] | 0.1406 | 6/8 |
| midtone_ratio | lower | 0.0547 | 0.1077 | -0.0530 | [-0.0749, -0.0312] | 0.0078 | 8/8 |
| edge_density | lower | 0.0607 | 0.1009 | -0.0403 | [-0.0525, -0.0267] | 0.0078 | 8/8 |
| small_components_per_megapixel | lower | 160.2173 | 468.7309 | -308.5136 | [-765.3236, 9.5367] | 0.1953 | 6/8 |
| small_component_ink_ratio | lower | 0.0227 | 0.0392 | -0.0165 | [-0.0466, 0.0065] | 0.4453 | 4/8 |
| largest_dark_region_ratio | lower | 0.0249 | 0.0222 | 0.0027 | [-0.0108, 0.0130] | 0.6953 | 2/8 |
| color_pixel_ratio | lower | 0.0014 | 0.0262 | -0.0248 | [-0.0430, -0.0131] | 0.0078 | 8/8 |

The p-values are exploratory exact paired sign-flip tests and are not corrected for multiple comparisons.
Automatic line-art measures are proxies: lower complexity is not always better, and white-space ratio can reward an overly empty image.
DINOv2 similarity crosses a photograph-to-line-art domain gap and must not be described as face-recognition accuracy.
