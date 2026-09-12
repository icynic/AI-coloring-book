# Automatic image evaluation

All measurements use the eight frozen Marburg subjects and paired FLUX–SD1.5 outputs.
Pixel metrics were calculated after aspect-preserving resize and white padding to 512×512.

| Metric | Direction | FLUX mean | SD1.5 mean | Difference | 95% bootstrap CI | p | FLUX wins |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| source_similarity_proxy | higher | 0.4669 | 0.3777 | 0.0892 | [0.0353, 0.1430] | 0.0312 | 7/8 |
| white_space_ratio | higher | 0.8830 | 0.7952 | 0.0878 | [0.0511, 0.1313] | 0.0078 | 8/8 |
| ink_coverage_ratio | descriptive | 0.1058 | 0.1836 | -0.0778 | [-0.1174, -0.0453] | 0.0078 | n.a. |
| dark_fill_ratio | lower | 0.0416 | 0.0728 | -0.0312 | [-0.0525, -0.0150] | 0.0156 | 7/8 |
| midtone_ratio | lower | 0.0642 | 0.1108 | -0.0466 | [-0.0653, -0.0283] | 0.0078 | 8/8 |
| edge_density | lower | 0.0649 | 0.1071 | -0.0422 | [-0.0543, -0.0283] | 0.0078 | 8/8 |
| small_components_per_megapixel | lower | 104.9042 | 232.2197 | -127.3155 | [-286.5791, 3.3379] | 0.1719 | 6/8 |
| small_component_ink_ratio | lower | 0.0137 | 0.0204 | -0.0067 | [-0.0204, 0.0032] | 0.4141 | 5/8 |
| largest_dark_region_ratio | lower | 0.0255 | 0.0259 | -0.0004 | [-0.0133, 0.0109] | 0.9609 | 3/8 |
| color_pixel_ratio | lower | 0.0028 | 0.0280 | -0.0252 | [-0.0433, -0.0136] | 0.0078 | 8/8 |

The p-values are exploratory exact paired sign-flip tests and are not corrected for multiple comparisons.
Automatic line-art measures are proxies: lower complexity is not always better, and white-space ratio can reward an overly empty image.
DINOv2 similarity crosses a photograph-to-line-art domain gap and must not be described as face-recognition accuracy.
