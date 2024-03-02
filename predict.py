# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import CLIPImageProcessor
from diffusers import DPMSolverMultistepScheduler
from SDXL.diff_pipe import StableDiffusionXLDiffImg2ImgPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)

device = "cuda"
MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
REFINER_NAME = "stabilityai/stable-diffusion-xl-refiner-1.0"
MODEL_CACHE = "checkpoints"
REFINER_CACHE = "refiner-cache"
SAFETY_CACHE = "safety-cache"
FEATURE_EXTRACTOR = "feature-extractor"
SDXL_URL = "https://weights.replicate.delivery/default/sdxl/sdxl-1.0.tar"
REFINER_URL = "https://weights.replicate.delivery/default/sdxl/refiner-1.0.tar"
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"

def preprocess_image(image):
    image = image.convert("RGB")
    image = transforms.CenterCrop((image.size[1] // 64 * 64, image.size[0] // 64 * 64))(image)
    image = transforms.ToTensor()(image)
    image = image * 2 - 1
    image = image.unsqueeze(0).to(device)
    return image

def preprocess_map(map):
    map = map.convert("L")
    map = transforms.CenterCrop((map.size[1] // 64 * 64, map.size[0] // 64 * 64))(map)
    map = transforms.ToTensor()(map)
    map = map.to(device)
    return map

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        print("Download base model")
        if not os.path.exists(MODEL_CACHE):
            download_weights(SDXL_URL, MODEL_CACHE)
        print("Download refiner model")
        if not os.path.exists(REFINER_CACHE):
            download_weights(REFINER_URL, REFINER_CACHE)
        print("Loading safety checker...")
        if not os.path.exists(SAFETY_CACHE):
            download_weights(SAFETY_URL, SAFETY_CACHE)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_CACHE, torch_dtype=torch.float16
        ).to("cuda")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(FEATURE_EXTRACTOR)
        self.base = StableDiffusionXLDiffImg2ImgPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            cache_dir=MODEL_CACHE
        )
        self.refiner = StableDiffusionXLDiffImg2ImgPipeline.from_pretrained(
            REFINER_NAME,
            text_encoder_2=self.base.text_encoder_2,
            vae=self.base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            cache_dir=REFINER_CACHE
        )

    def run_safety_checker(self, image):
        safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(
            "cuda"
        )
        np_image = [np.array(val) for val in image]
        image, has_nsfw_concept = self.safety_checker(
            images=np_image,
            clip_input=safety_checker_input.pixel_values.to(torch.float16),
        )
        return image, has_nsfw_concept
    
    def predict(
        self,
        image: Path = Input(description="Input image"),
        depth_img: Path = Input(description="Change map"),
        prompt: str = Input(
            description="Prompt to guide the edit",
            default="Tree of life under the sea, ethereal, glittering, lens flares, cinematic lighting, artwork by Anna Dittmann & Carne Griffiths, 8k, unreal engine 5, hightly detailed, intricate detailed"
        ),
        negative_prompt: str = Input(
            description="Negative prompt to guide the edit",
            default="bad anatomy, poorly drawn face, out of frame, gibberish, lowres, duplicate, morbid, darkness, maniacal, creepy, fused, blurry background, crosseyed, extra limbs, mutilated, dehydrated, surprised, poor quality, uneven, off-centered, bird illustration, painting, cartoons"
        ),
        num_inference_steps: int = Input(description="Number of inference steps", ge=1, le=100, default=50),
        guidance_scale: float = Input(description="Factor to scale image by", ge=0, le=50, default=17.5),
        denoising_strength: float = Input(description="Denoising strength", ge=0, le=1, default=0.8),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        disable_safety_checker: bool = Input(
            description="Disable safety checker for generated images. This feature is only available through the API. See https://replicate.com/docs/how-does-replicate-work#safety",
            default=False
        )
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        self.base.scheduler = DPMSolverMultistepScheduler.from_config(self.base.scheduler.config)
        self.refiner.scheduler = DPMSolverMultistepScheduler.from_config(self.base.scheduler.config)

        pil_image = Image.open(image)
        image = preprocess_image(pil_image)
        pil_map = Image.open(depth_img)
        map = preprocess_map(pil_map)

        base_cuda = self.base.to(device)
        edited_images = base_cuda(
            prompt=prompt, original_image=image, image=image, strength=1, guidance_scale=guidance_scale,
            num_images_per_prompt=1,
            negative_prompt=negative_prompt,
            map=map,
            num_inference_steps=num_inference_steps,
            denoising_end=denoising_strength,
            output_type="latent",
            generator=generator
        ).images
        base_cuda=None

        refiner_cuda = self.refiner.to(device)
        edited_images = refiner_cuda(
            prompt=prompt, original_image=image, image=edited_images, strength=1, guidance_scale=7.5,
            num_images_per_prompt=1,
            negative_prompt=negative_prompt,
            map=map,
            num_inference_steps=num_inference_steps,
            denoising_start=denoising_strength,
            generator=generator
        ).images
        refiner_cuda=None

        _, has_nsfw_content = self.run_safety_checker(edited_images)
        if not disable_safety_checker:
            if has_nsfw_content[0]:
                raise Exception(f"NSFW content detected. Try running it again, or try a different prompt.")
        
        output_path = "/tmp/output.png"
        edited_images[0].save(output_path)
        return Path(output_path)