# 🧠 VQ-VAE Fine-Tuning with CGA (Codebook Genetic Algorithm)

이 프로젝트는 **VQ-VAE** 모델의 코드북(codebook)을 **유전 알고리즘(CGA)**을 통해 파인튜닝하는 연구/실험용 코드입니다.  
학습된 VQ-VAE 모델의 코드북을 효율적으로 업데이트하여, 보다 다양하고 효과적인 토큰 활용을 유도합니다.

---

## 🚀 실행 환경

다음 환경에서 개발 및 테스트되었습니다:

- Python 3.12
- PyTorch
- CUDA (GPU 가속을 원할 경우 필수)
- matplotlib
- numpy
- torchvision

> ⚠️ CUDA가 설치된 환경에서 실행하면 학습/재구성 속도가 크게 향상됩니다.

---

> fine-tuning.ipynb에서 GA-VAE 실험가능, VQ-VAE 실험이 필요한 경우 main.py 실행.

