import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler

class TextToImageGenerator:
    def __init__(self):
        print("Loading models...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"Running on {self.device.upper()} with {self.dtype}")

        # Load Stable Diffusion Pipeline without ControlNet
        self.pipe = StableDiffusionPipeline.from_single_file(
            "models/coloringPage_v10.safetensors",
            torch_dtype=self.dtype,
            safety_checker=None,
        )

        # Optimize
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to("cpu")

        print("Models loaded successfully.")

    def generate_image(
        self,
        output_path,
        prompt="c0l0ringb00k, black and white coloring page, line art, white background, thick lines",
        negative_prompt="shadow, shading, gradients, stippling, screentone, texture, background details, grayscale, colored, 3d, realistic, photo, noise, blurry, deformed",
        steps=20,
        guidance_scale=7.5,
        width=512,
        height=512,
        seed=None,
    ):
        print("Generating image...")
        
        if seed is None:
            seed = torch.randint(0, 1000000, (1,)).item()
        generator = torch.manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        ).images[0]

        result.save(output_path)
        print(f"Done! Saved to {output_path}")
        print(f"Seed: {seed}")

    def generate_batch(
        self,
        output_prefix,
        count=5,
        prompt="c0l0ringb00k, black and white coloring page, line art, white background, thick lines",
        negative_prompt="shadow, shading, gradients, stippling, screentone, texture, background details, grayscale, colored, 3d, realistic, photo, noise, blurry, deformed",
        steps=20,
        guidance_scale=7.5,
        width=512,
        height=512,
    ):
        print(f"Generating {count} images in batch...")
        
        for i in range(count):
            seed = torch.randint(0, 1000000, (1,)).item()
            generator = torch.manual_seed(seed)
            output_path = f"{output_prefix}_{i+1}_seed_{seed}.png"
            
            print(f"Generating image {i+1}/{count} with seed {seed}...")

            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator,
            ).images[0]

            result.save(output_path)
            print(f"Done! Saved to {output_path}")

if __name__ == "__main__":
    generator = TextToImageGenerator()

    # prompt = "Coloring page of a portrait, line art, black and white, white background, thick lines"
    prompt = "Coloring page of a portrait"
    # prompt="coloring page of a cat with necktie"
    negative_prompt = "bad, bad art, lowres, ugly, poorly drawn, text, logo, colors, shadow, shading, gradients, stippling, screentone, texture, background details, grayscale, colored, 3d, realistic, photo, noise, blurry, deformed, filled, filled-in"
    # negative_prompt = ""

    # 测试生成单张图片
    # generator.generate_image(
    #     output_path="images/text_to_image_test.png",
    #     prompt=prompt,
    #     negative_prompt=negative_prompt,
    #     steps=20,
    #     guidance_scale=7.5,
    #     width=512,
    #     height=512,
    #     seed=None,
    # )

    # 批量生成 5 张图片
    generator.generate_batch(
        output_prefix="images/text_to_image_batch",
        count=5,
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=20,
        guidance_scale=7.5,
        width=512,
        height=512,
    )
