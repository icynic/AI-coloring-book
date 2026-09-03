"""
Colab T4/L4-friendly FLUX.2 [klein] coloring page generator.

Colab setup cell:
    !pip install -q --upgrade git+https://github.com/huggingface/diffusers.git transformers accelerate safetensors sentencepiece protobuf pillow bitsandbytes

Recommended Colab hardware:
    - Free tier: NVIDIA T4 16GB, 8-bit quantization, max side 640
    - Faster tier: NVIDIA L4, about 22.5GB VRAM

Example:
    python GeneratorFlux2KleinL4Colab.py --input images/Marie_Curie.jpg

T4 example:
    python GeneratorFlux2KleinL4Colab.py --input images/Marie_Curie.jpg --quantization 8bit --max-side 640

If you still hit OOM:
    python GeneratorFlux2KleinL4Colab.py --input images/Marie_Curie.jpg --offload
    python GeneratorFlux2KleinL4Colab.py --input images/Marie_Curie.jpg --quantization 8bit
"""

import argparse
import gc
import os
from pathlib import Path
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image, ImageOps

try:
    from diffusers import Flux2KleinPipeline
except ImportError as exc:
    raise ImportError(
        "Flux2KleinPipeline is not available in your installed diffusers. "
        "In Colab, run:\n"
        "!pip install -q --upgrade git+https://github.com/huggingface/diffusers.git "
        "transformers accelerate safetensors sentencepiece protobuf pillow bitsandbytes"
    ) from exc

try:
    from diffusers.quantizers import PipelineQuantizationConfig
except ImportError:
    PipelineQuantizationConfig = None


DEFAULT_MODEL_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"


class Flux2KleinL4ColoringPageGenerator:
    def __init__(
        self,
        model_id="black-forest-labs/FLUX.2-klein-4B",
        dtype=None,
        quantization="none",
        offload=False,
        enable_vae_tiling=False,
        revision=DEFAULT_MODEL_REVISION,
    ):
        print("Loading FLUX.2 [klein] for Colab L4...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            if self.device == "cuda" and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            elif self.device == "cuda":
                dtype = torch.float16
            else:
                dtype = torch.float32
        self.dtype = dtype
        self.model_id = model_id
        self.revision = revision
        self.quantization = quantization
        self.last_run_metadata = None

        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        print(f"Running on {self.device.upper()} with {self.dtype}")
        print(f"Model: {model_id}")
        print(f"Quantization: {quantization}")
        print(f"CPU offload: {offload}")

        if self.device != "cuda":
            print("Warning: FLUX.2 [klein] is not practical on CPU.")

        load_kwargs = {
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
        }
        if revision:
            load_kwargs["revision"] = revision

        quantization_config = self._build_quantization_config(quantization)
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config

        load_started_at = time.perf_counter()
        self.pipe = Flux2KleinPipeline.from_pretrained(model_id, **load_kwargs)
        self._place_pipeline(offload=offload)
        self._enable_memory_helpers(enable_vae_tiling=enable_vae_tiling)
        self.model_load_seconds = round(time.perf_counter() - load_started_at, 3)

        print("Model loaded successfully.")

    @staticmethod
    def _build_quantization_config(quantization):
        if quantization in (None, "none", "fp16"):
            return None

        if PipelineQuantizationConfig is None:
            raise ImportError(
                "PipelineQuantizationConfig is not available. Reinstall diffusers from GitHub:\n"
                "!pip install -q --upgrade git+https://github.com/huggingface/diffusers.git"
            )

        if quantization == "8bit":
            return PipelineQuantizationConfig(
                quant_backend="bitsandbytes_8bit",
                quant_kwargs={"load_in_8bit": True},
                components_to_quantize=["transformer", "text_encoder"],
            )

        if quantization == "4bit":
            return PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch.float16,
                },
                components_to_quantize=["transformer", "text_encoder"],
            )

        raise ValueError("quantization must be one of: none, 8bit, 4bit")

    def _place_pipeline(self, offload):
        if self.device != "cuda":
            self.pipe.to("cpu")
            return

        if offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to("cuda")

    def _enable_memory_helpers(self, enable_vae_tiling):
        if getattr(self.pipe, "enable_vae_slicing", None) is not None:
            self.pipe.enable_vae_slicing()

        if enable_vae_tiling and getattr(self.pipe, "enable_vae_tiling", None) is not None:
            self.pipe.enable_vae_tiling()

        transformer = getattr(self.pipe, "transformer", None)
        if transformer is not None:
            try:
                transformer.to(memory_format=torch.channels_last)
            except (RuntimeError, ValueError, TypeError):
                pass

    def cleanup(self):
        print("Releasing FLUX.2 [klein] pipeline from memory...")
        if hasattr(self, "pipe"):
            del self.pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    @staticmethod
    def load_image(image_path):
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")

    @staticmethod
    def resize_to_flux_bounds(image, max_side=768, multiple=16):
        width, height = image.size
        scale = min(max_side / max(width, height), 1.0)
        new_width = max(multiple, int(width * scale))
        new_height = max(multiple, int(height * scale))
        new_width = (new_width // multiple) * multiple
        new_height = (new_height // multiple) * multiple
        return image.resize((new_width, new_height), Image.LANCZOS)

    @staticmethod
    def build_coloring_prompt(prompt, negative_prompt):
        instruction = (
            "Transform the reference image into a clean black-and-white coloring book page. "
            "Preserve the subject identity, facial features, pose, and composition. "
            "Use bold crisp outlines, simple readable contour lines, open white areas for coloring, "
            "plain white background, no color, no gray fill, no shadows, no gradients, no hatching, "
            "no text, no logo, no watermark."
        )

        if prompt:
            instruction = f"{instruction} {prompt}"

        if negative_prompt:
            instruction = f"{instruction} Avoid: {negative_prompt}."

        return instruction

    def process_image(
        self,
        image_path,
        output_path,
        prompt="portrait coloring page, clean line art, white background, thick lines",
        negative_prompt=(
            "shadow, shading, gradients, stippling, screentone, texture, grayscale, colored, "
            "realistic photo, blurry, deformed, filled-in shapes, busy background"
        ),
        steps=4,
        guidance_scale=1.0,
        max_side=768,
        max_sequence_length=512,
        seed=None,
        reference_image_path=None,
    ):
        print(f"Processing {image_path}...")

        started_at = time.perf_counter()
        image = self.load_image(image_path)
        image = self.resize_to_flux_bounds(image, max_side=max_side)
        width, height = image.size
        print(f"Resized input to {width}x{height}")

        if reference_image_path:
            image.save(reference_image_path)
            print(f"Saved resized reference image to {reference_image_path}")

        if seed is None:
            seed = torch.randint(0, 1_000_000, (1,)).item()

        generator_device = "cuda" if self.device == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        final_prompt = self.build_coloring_prompt(prompt, negative_prompt)

        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        with torch.inference_mode():
            result = self.pipe(
                image=image,
                prompt=final_prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                max_sequence_length=max_sequence_length,
            ).images[0]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)

        print(f"Done! Saved to {output_path}")
        print(f"Seed: {seed}")
        peak_gb = None
        if self.device == "cuda":
            peak_gb = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak CUDA memory allocated: {peak_gb:.2f} GB")

        self.last_run_metadata = {
            "model_id": self.model_id,
            "model_revision": self.revision,
            "quantization": self.quantization,
            "dtype": str(self.dtype),
            "device": self.device,
            "seed": seed,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "max_sequence_length": max_sequence_length,
            "elapsed_seconds": round(time.perf_counter() - started_at, 3),
            "peak_cuda_memory_gb": round(peak_gb, 3) if peak_gb is not None else None,
            "prompt": final_prompt,
            "model_load_seconds": self.model_load_seconds,
        }

        return result

    def generate_batch(
        self,
        image_path,
        output_path,
        count=4,
        batch_size=2,
        prompt="portrait coloring page, clean line art, white background, thick lines",
        negative_prompt=(
            "shadow, shading, gradients, stippling, screentone, texture, grayscale, colored, "
            "realistic photo, blurry, deformed, filled-in shapes, busy background"
        ),
        steps=4,
        guidance_scale=1.0,
        max_side=768,
        max_sequence_length=512,
        seed=None,
        reference_image_path=None,
    ):
        if count < 1:
            raise ValueError("count must be at least 1")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        print(f"Generating {count} images with batch size {batch_size}...")

        image = self.load_image(image_path)
        image = self.resize_to_flux_bounds(image, max_side=max_side)
        width, height = image.size
        print(f"Resized input to {width}x{height}")

        if reference_image_path:
            image.save(reference_image_path)
            print(f"Saved resized reference image to {reference_image_path}")

        if seed is None:
            seed = torch.randint(0, 1_000_000, (1,)).item()
        seeds = [seed + i for i in range(count)]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_prompt = self.build_coloring_prompt(prompt, negative_prompt)
        generator_device = "cuda" if self.device == "cuda" else "cpu"
        saved_paths = []

        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for start in range(0, count, batch_size):
            end = min(start + batch_size, count)
            current_seeds = seeds[start:end]
            current_batch_size = len(current_seeds)
            generators = [
                torch.Generator(device=generator_device).manual_seed(current_seed)
                for current_seed in current_seeds
            ]

            print(f"Batch {start // batch_size + 1}: images {start + 1}-{end}, seeds {current_seeds}")

            with torch.inference_mode():
                results = self.pipe(
                    image=image,
                    prompt=final_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generators,
                    num_images_per_prompt=current_batch_size,
                    max_sequence_length=max_sequence_length,
                ).images

            for offset, result in enumerate(results):
                index = start + offset + 1
                image_seed = current_seeds[offset]
                current_output = self._batch_output_path(output_path, index, image_seed)
                result.save(current_output)
                saved_paths.append(current_output)
                print(f"Saved {current_output}")

            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

        print(f"Done! Generated {len(saved_paths)} images.")
        print(f"Seeds: {seeds}")
        if self.device == "cuda":
            peak_gb = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak CUDA memory allocated: {peak_gb:.2f} GB")

        return saved_paths

    @staticmethod
    def _batch_output_path(output_path, index, seed):
        stem = output_path.stem
        suffix = output_path.suffix or ".png"
        return output_path.with_name(f"{stem}_{index:02d}_seed_{seed}{suffix}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="images/Marie_Curie.jpg")
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--quantization", choices=["none", "8bit", "4bit"], default="none")
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--vae-tiling", action="store_true")
    parser.add_argument("--save-reference", default=None)
    parser.add_argument(
        "--keep-model-loaded",
        action="store_true",
        help="Keep the model in memory after generation. Useful only inside a notebook if you reuse the same generator object.",
    )

    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print(f"Ignoring notebook/runtime arguments: {unknown_args}")
    return args


if __name__ == "__main__":
    args = parse_args()
    input_path = Path(args.input)
    output_path = args.output or str(input_path.with_name(input_path.stem + "_flux2_klein_l4.png"))

    generator = None
    try:
        generator = Flux2KleinL4ColoringPageGenerator(
            model_id=args.model,
            quantization=args.quantization,
            offload=args.offload,
            enable_vae_tiling=args.vae_tiling,
            revision=args.revision,
        )
        if args.count == 1:
            generator.process_image(
                image_path=args.input,
                output_path=output_path,
                steps=args.steps,
                guidance_scale=args.guidance_scale,
                max_side=args.max_side,
                max_sequence_length=args.max_sequence_length,
                seed=args.seed,
                reference_image_path=args.save_reference,
            )
        else:
            generator.generate_batch(
                image_path=args.input,
                output_path=output_path,
                count=args.count,
                batch_size=args.batch_size,
                steps=args.steps,
                guidance_scale=args.guidance_scale,
                max_side=args.max_side,
                max_sequence_length=args.max_sequence_length,
                seed=args.seed,
                reference_image_path=args.save_reference,
            )
    finally:
        if generator is not None and not args.keep_model_loaded:
            generator.cleanup()
