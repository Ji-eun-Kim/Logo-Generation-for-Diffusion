## setup.py

import os

# 패키지 제거
os.system("pip uninstall -y torch torchvision torchaudio torchtext")

# 필수 패키지 설치
os.system("pip install torch==2.0.0 torchvision datasets")

# Diffusers와 PEFT 라이브러리 설치
os.system("git clone https://github.com/huggingface/diffusers.git")
os.system("cd diffusers && pip install . && cd ..")

os.system("git clone https://github.com/huggingface/peft.git")
os.system("cd peft && pip install . && cd ..")
