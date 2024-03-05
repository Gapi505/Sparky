import torch
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
prompt = "nice cars on speed"
num_images = 1
image = pipe(prompt).images[0]
image.save(f'image.png')