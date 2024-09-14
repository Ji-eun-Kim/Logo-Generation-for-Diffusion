## train.py

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from diffusers.optimization import get_scheduler
from tqdm import tqdm
from model import get_models
from data import create_dataloader

## 하이퍼파라미터 설정
BATCH_SIZE = 4
TOTAL_STEPS = 25000
LEARNING_RATE = 2e-05
WARMUP_STEPS = 500
LORA_RANK = 4
use_amp = True

## 데이터 로더 생성
train_dataloader = create_dataloader(batch_size=BATCH_SIZE)

## 모델 로드
unet, vae, text_encoder, noise_scheduler = get_models()

## Optimizer 설정
optimizer = AdamW(unet.parameters(), lr=LEARNING_RATE)

## 스케줄러 설정
lr_scheduler = get_scheduler(
    'linear', optimizer=optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=TOTAL_STEPS
)

## Mixed Precision 설정
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

## 학습 루프
def train():
    current_step = 0
    train_losses = []
    
    while current_step < TOTAL_STEPS:
        for batch in tqdm(train_dataloader):
            with torch.cuda.amp.autocast(enabled=use_amp):
                latents = vae.encode(batch["pixel_values"].to(device)).latent_dist.sample() * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0]
                target = noise if noise_scheduler.config.prediction_type == "epsilon" else noise_scheduler.get_velocity(latents, noise, timesteps)
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_losses.append(loss.item())
            current_step += 1

            if current_step % 100 == 0:
                avg_loss = sum(train_losses[-100:]) / 100
                print(f"스텝 {current_step}/{TOTAL_STEPS}, 평균 손실: {avg_loss:.5f}")

            if current_step >= TOTAL_STEPS:
                break

    if train_losses:
        avg_loss = sum(train_losses[-100:]) / 100
        print(f"트레이닝 완료. 최종 평균 손실: {avg_loss:.5f}")

## 학습 시작
if __name__ == "__main__":
    train()
