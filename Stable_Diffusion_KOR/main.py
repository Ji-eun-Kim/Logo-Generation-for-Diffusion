import torch
from diffusers import StableDiffusionPipeline

model_path = "sd-logo-model"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

image = pipe(prompt="모던한 느낌의 카페 로고").images[0]
image.save("logo_sample.png")