## pipeline.py

import torch
from diffusers import StableDiffusionPipeline
from model import get_models
from huggingface_hub import login

# 허깅페이스 허브에 로그인
login(token="your_token_here")  # 실제 토큰을 여기에 넣으세요.

## 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## 모델 로드
unet, vae, text_encoder, _ = get_models()

## Stable Diffusion 파이프라인 구성
pipeline = StableDiffusionPipeline.from_pretrained(
    "Bingsu/my-korean-stable-diffusion-v1-5",
    text_encoder=text_encoder,
    vae=vae,
    unet=unet,
)

pipeline.to(device)

## 허깅페이스에 푸시
pipeline.push_to_hub("stable-diffusion-v1-5-finetune-Logo")
