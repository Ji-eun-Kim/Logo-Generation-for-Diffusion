import torch
import torchvision.transforms as transforms
from PIL import Image

# 이미지 로드
img_path = "/home/work/rvc/logo/input/contents/cropped_maskedlogo.jpg"  # 이미지 파일 경로
img = Image.open(img_path)

# 텐서로 변환
transform = transforms.ToTensor()
tensor_img = transform(img)

# 흑백 반전 (완전 반전)
inverted_img = 1 - tensor_img

# 텐서를 이미지로 변환
inverse_transform = transforms.ToPILImage()
inverted_image = inverse_transform(inverted_img)

# 결과 이미지 저장
inverted_image.save("inverted_image.jpg")