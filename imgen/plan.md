Found a project called Informative-drawing, implementing the papaer "Learning To Generate Line Drawings That Convey Geometry and Semantics"
Which is more close to edge detecting

Traditional edge detection (like Canny or HED) gives you exact structure but overwhelming noise. Stable Diffusion gives you beautiful, clean lines, but its generative nature compels it to hallucinate textures, extra shapes, and shading that don't exist in the original photo. Even with a Lineart ControlNet, SD is fundamentally a pixel-generating engine that wants to fill empty space.

Tried Nano banana and ChatGPT 
multimodal vision-language models (VLMs) like them work much better
because they possess "common sense." They don't look at a photo of a furry dog and see a million tiny shadow gradients; they just see a "dog" and know a dog has a single outer silhouette.

Actually, there are two ways:
1. The LLM writes a detailed text description of the source image, appends the modification request, and then sends the text prompt to the Diffusion model.
2. The LLM passes The Text Prompt and The Noisy Latents (Mathematical representation of the source image) to the Diffusion model.

looked into Differentiable Vector Rendering.
Instead of asking a model to generate a grid of pixels that look like lines, these models are forced to output mathematical SVG coordinates (Bézier curves). Because they can only draw continuous strokes, it is physically impossible for them to generate pixel noise or complex hallucinated textures.
SwiftSketch outputs distorted shapes, DiffSketcher does not take an image as input and it outputs a draft rather than coloring page, CLIPasso & CLIPascene are better but they still have messy lines.

Although I loaded three lora models, only the last one is actually effective. However, if I use the keyword "<lora:animeoutlineV3-000008:0.5>" in the prompt, it will produce pure anime faces.
Not adding this keyword is better.
Somehow, using the prompt "c0l0ringb00k" for the second lora is better than "coloringbook" or removing it.

Tried base model Lykon/DreamShaper. The result looks more artistic than realistic.
Tried base model xyn-ai/anything-v4.0. The result is too anime-like

Tried vision language models. They can't output images.