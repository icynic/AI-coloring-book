import torch
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from controlnet_aux import LineartDetector, CannyDetector, PidiNetDetector
import os


class ColoringPageGenerator:
    def __init__(self):
        print("Loading models...")
        # 0. Check Device & precision
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"Running on {self.device.upper()} with {self.dtype}")

        # 1. Load ControlNet
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_lineart",
            # "lllyasviel/control_v11p_sd15_canny",
            # "lllyasviel/control_v11p_sd15_softedge",
            torch_dtype=self.dtype,
        )
        print("ControlNet loaded successfully.")

        # 2. Load Lineart Detector
        self.processor = LineartDetector.from_pretrained("lllyasviel/Annotators")
        # self.processor = CannyDetector()
        # self.processor = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
        print("Soft Edge Detector loaded successfully.")

        # 3. Load Stable Diffusion Pipeline
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            # "Lykon/DreamShaper",
            # "xyn-ai/anything-v4.0",
            controlnet=self.controlnet,
            torch_dtype=self.dtype,
            safety_checker=None,
        )
        
        ##################################################################### Load local model
        # self.pipe = StableDiffusionControlNetPipeline.from_single_file(
        #     "models/coloringPage_v10.safetensors",
        #     controlnet=self.controlnet,
        #     torch_dtype=self.dtype,
        #     safety_checker=None,
        # )
        #####################################################################

        # Load LoRA
        print("Loading LoRA...")
        self.pipe.load_lora_weights(
            "beatless/AnimeLineartLoRA"
            # "ColoringBookRedmond15V-LiberteRedmond-ColoringBookAF.safetensors"
            # "animeoutlineV4_16.safetensors"
            # "models/EroticLineart.safetensors"
            # "models/linebaby.safetensors"
        ) # <lora:animeoutlineV3-000008:0.5>

        # 4. Optimize
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
            # self.pipe.enable_xformers_memory_efficient_attention()
        else:
            self.pipe.to("cpu")

        print("Models loaded successfully.")

    def load_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return image

    def process_image(
        self,
        image_path,
        output_path,
        prompt="c0l0ringb00k, black and white coloring page, line art, white background, thick lines",
        negative_prompt="shadow, shading, gradients, stippling, screentone, texture, background details, flowers, plants, stripes, grayscale, colored, 3d, realistic, photo, noise, blurry, deformed, filled, filled-in, filled-in lines, filled-in shapes, filled-in patterns, filled background",
        steps=15,
        strength=0.6,
        guidance_scale=10,
        control_image_path=None,
        seed=None,
    ):

        print(f"Processing {image_path}...")

        # Prepare Input Image
        image = self.load_image(image_path)

        # Resize logic
        width, height = image.size
        aspect_ratio = height / width
        new_width = 512
        new_height = int(new_width * aspect_ratio)
        new_height = new_height - (new_height % 8)
        image = image.resize((new_width, new_height))

        print(f"Resized input to {new_width}x{new_height}")

        # Extract Lineart (Control Image)
        control_image = self.processor(image, coarse=True)
        if control_image_path:
            control_image.save(control_image_path)  # Optional: save debug

        # Generate
        if seed is None:
            seed = torch.randint(0, 1000000, (1,)).item()
        generator = torch.manual_seed(seed)

        result = self.pipe(
            prompt,
            image=control_image,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            generator=generator,
            controlnet_conditioning_scale=strength,
            guidance_scale=guidance_scale,
        ).images[0]

        result.save(output_path)
        print(f"Done! Saved to {output_path}")
        print(f"Seed: {seed}")


    def generate_batch_from_control_image(
        self,
        control_image_path,
        output_prefix,
        count=5,
        prompt="c0l0ringb00k, black and white coloring page, line art, white background, thick lines",
        negative_prompt="shadow, shading, gradients, stippling, screentone, texture, background details, flowers, plants, stripes, grayscale, colored, 3d, realistic, photo, noise, blurry, deformed, filled, filled-in, filled-in lines, filled-in shapes, filled-in patterns, filled background",
        steps=15,
        strength=0.6,
        guidance_scale=10,
    ):
        print(f"Loading existing control image: {control_image_path}")
        control_image = self.load_image(control_image_path)

        for i in range(count):
            seed = torch.randint(0, 1000000, (1,)).item()
            generator = torch.manual_seed(seed)
            
            output_path = f"{output_prefix}_{i+1}_seed_{seed}.png"
            print(f"Generating image {i+1}/{count} with seed {seed}...")

            result = self.pipe(
                prompt,
                image=control_image,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                generator=generator,
                controlnet_conditioning_scale=strength,
                guidance_scale=guidance_scale,
            ).images[0]

            result.save(output_path)
            print(f"Done! Saved to {output_path} (Seed: {seed})")


if __name__ == "__main__":
    # Example usage
    generator = ColoringPageGenerator()

    # input_image_path = "images/Albert_Einstein.jpg"
    # input_image_path = "images/Max_Planck.jpg"
    input_image_path = "images/Marie_Curie.jpg"

    output_image_path = input_image_path.rsplit(".", 1)[0] + "_output.png"
    control_image_path = input_image_path.rsplit(".", 1)[0] + "_debug_control_image.png"

    # prompt = "c0l0ringb00k, portrait, coloring page, black and white, line art, high contrast, clean lines, white background, masterpiece, best quality, monochrome, flat, low detail, cartoon style, distinct outlines"
    # prompt = "c0l0ringb00k, black and white coloring page, line art, white background, thick lines, portrait"
    prompt="c0l0ringb00k, Coloring page of a portrait, thick lines"
    negative_prompt = "realistic, shadow, shading, gradients, stippling, screentone, texture, background details, grayscale, colored, 3d, photo, noise, blurry, deformed, filled, filled-in, filled-in lines, filled-in shapes, filled-in patterns, filled background, bad, bad art, lowres, ugly, poorly drawn, text, logo, colors, "
    # negative_prompt=""

    # The number of iterations the AI uses to "clean up" (denoise) the image.
    steps = 15
    # How strictly the AI must obey the ControlNet line art extracted from the original photo.
    strength = 0.6
    # How strongly the AI should obey your text prompt
    guidance_scale = 7.5

    seed = 24
    # seed = None

    # generator.process_image(
    #     input_image_path,  # Required
    #     output_image_path,  # Required
    #     prompt,  # Optional
    #     negative_prompt,  # Optional
    #     steps,  # Optional
    #     strength,  # Optional
    #     guidance_scale,  # Optional
    #     control_image_path,  # Optional: save debug control image
    #     seed,  # Optional
    # )

    # 假设你已经有一张提取好的边缘图
    # existing_control_image = "images/Max_Planck_debug_control_image.png"
    existing_control_image = "images/Marie_Curie_debug_control_image.png"
    
    # 批量生成 5 张新图，前缀为 Max_Planck_batch
    generator.generate_batch_from_control_image(
        control_image_path=existing_control_image,
        output_prefix="images/Marie_Curie_batch",
        count=5,               # 生成的数量
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        strength=strength,
        guidance_scale=guidance_scale
    )