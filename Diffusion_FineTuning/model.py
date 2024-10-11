## model.py

import torch
from transformers import CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler

## 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## 사전 학습된 모델 이름
PRE_TRAINED_MODEL_NAME = "Bingsu/my-korean-stable-diffusion-v1-5"

## 스케줄러 설정
noise_scheduler = DDPMScheduler.from_pretrained(PRE_TRAINED_MODEL_NAME, subfolder="scheduler")

## 텍스트 인코더, VAE, UNet 모델 로드
text_encoder = CLIPTextModel.from_pretrained(PRE_TRAINED_MODEL_NAME, subfolder="text_encoder").to(device)
vae = AutoencoderKL.from_pretrained(PRE_TRAINED_MODEL_NAME, subfolder="vae").to(device)
unet = UNet2DConditionModel.from_pretrained(PRE_TRAINED_MODEL_NAME, subfolder="unet").to(device)

## 모델 파라미터 설정 (VAE와 텍스트 인코더는 학습하지 않음)
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.requires_grad_(True)

## 모델 가져오기 함수
def get_models():
    return unet, vae, text_encoder, noise_scheduler
