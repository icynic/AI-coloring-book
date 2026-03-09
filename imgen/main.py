import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import LineartDetector
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
            torch_dtype=self.dtype
        )

        # 2. Load Lineart Detector
        self.processor = LineartDetector.from_pretrained("lllyasviel/Annotators")

        # 3. Load Stable Diffusion Pipeline
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            # "runwayml/stable-diffusion-v1-5",
            # "Lykon/DreamShaper",
            "xyn-ai/anything-v4.0",
            controlnet=self.controlnet,
            torch_dtype=self.dtype,
            safety_checker=None
        )
        
        # Load LoRa
        print("Loading LoRa...")
        self.pipe.load_lora_weights(
            "beatless/AnimeLineartLoRA"
        ) # <lora:animeoutlineV3-000008:0.5>
        
        # 4. Optimize
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        
        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
            # self.pipe.enable_xformers_memory_efficient_attention()
        else:
            self.pipe.to("cpu")

        print("Models loaded successfully.")

    def load_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return image

    def process_image(self, image_path, output_path, prompt, negative_prompt, steps=15, strength=0.6, guidance_scale=10, control_image_path=None):
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
            control_image.save(control_image_path) # Optional: save debug

        # Generate
        generator = torch.manual_seed(42)
        
        result = self.pipe(
            prompt,
            image=control_image,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            generator=generator,
            controlnet_conditioning_scale=strength,
            guidance_scale=guidance_scale
        ).images[0]

        result.save(output_path)
        print(f"Done! Saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    generator = ColoringPageGenerator()
    
    input_image_path = os.path.join(os.path.dirname(__file__), "Marie_Curie_c._1920s.jpg")
    output_image_path = os.path.join(os.path.dirname(__file__), "marie_curie_output.png")
    control_image_path = os.path.join(os.path.dirname(__file__), "debug_control_image.png")
    
    # prompt = "c0l0ringb00k, portrait, coloring page, black and white, line art, high contrast, clean lines, white background, masterpiece, best quality, monochrome, flat, low detail, cartoon style, distinct outlines"
    prompt = "c0l0ringb00k, black and white coloring page, line art, white background, thick lines"
    negative_prompt = "shadow, shading, gradients, stippling, screentone, texture, background details, flowers, plants, stripes, grayscale, colored, 3d, realistic, photo, noise, blurry, deformed, filled, filled-in, filled-in lines, filled-in shapes, filled-in patterns, filled background"


    steps = 15
    strength = 0.6
    guidance_scale = 10
    

    generator.process_image(input_image_path, output_image_path, prompt, negative_prompt, steps, strength, guidance_scale, control_image_path)