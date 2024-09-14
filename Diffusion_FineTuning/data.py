## data.py

import torch
import random
import numpy as np
from torchvision import transforms
from datasets import load_dataset
from transformers import CLIPTokenizer
from PIL import Image

## 모델과 토크나이저 설정
PRE_TRAINED_MODEL_NAME = "Bingsu/my-korean-stable-diffusion-v1-5"
tokenizer = CLIPTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, subfolder="tokenizer")

## 데이터셋 로드
dataset = load_dataset("Junhoee/Logo-Dataset-Korean")['train']

## 텍스트 전처리 함수
def tokenize_captions(examples, caption_column='text', is_train=True):
    captions = []
    for caption in examples[caption_column]:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(f"캡션 열에는 `{caption_column}` 문자열, 문자열 목록이 포함되어야 합니다.")
    inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
    return inputs.input_ids

## 이미지 전처리 설정
IMG_SIZE = 256
CENTER_CROP = True
RANDOM_FLIP = True
train_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(IMG_SIZE) if CENTER_CROP else transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip() if RANDOM_FLIP else transforms.Lambda(lambda x: x),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]), # (0,1) -> (-1,1)
])

## 데이터셋 전처리 함수
def preprocess_train(examples, image_column='image'):
    images = [image.convert("RGB") for image in examples[image_column]]
    examples["pixel_values"] = [train_transforms(image) for image in images]
    examples["input_ids"] = tokenize_captions(examples)
    return examples

## 데이터 로더 함수
def create_dataloader(batch_size=4, num_workers=8):
    train_dataset = dataset.with_transform(preprocess_train)
    return torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn
    )

# 데이터 로더 생성
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}
