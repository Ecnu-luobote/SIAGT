# LIIF

This repository is implementation for SIAGT

**"Scale-Invariant Adversarial Attack against Arbitrary-scale Super-resolution"**

![image](https://github.com/user-attachments/assets/999f6817-404e-4831-8534-a301d549acf6)


### Environment
- Python 3
- Pytorch 1.6.0
- TensorboardX
- basicsr
- yaml, numpy, tqdm, imageio

## Quick Start

1. Download a DIV2K pre-trained model.

Model|File size|Download
:-:|:-:|:-:
EDSR-baseline-LIIF|18M|[Dropbox](https://www.dropbox.com/s/6f402wcn4v83w2v/edsr-baseline-liif.pth?dl=0) &#124; [Google Drive](https://drive.google.com/file/d/1wBHSrgPLOHL_QVhPAIAcDC30KSJLf67x/view?usp=sharing)
RDN-LIIF|256M|[Dropbox](https://www.dropbox.com/s/mzha6ll9kb9bwy0/rdn-liif.pth?dl=0) &#124; [Google Drive](https://drive.google.com/file/d/1xaAx6lBVVw_PJ3YVp02h3k4HuOAXcUkt/view?usp=sharing)

2. Download benchmark datasets: `cd` into `load/`. Download and `tar -xf` the [benchmark datasets](https://cv.snu.ac.kr/research/EDSR/benchmark.tar) (provided by [this repo](https://github.com/thstkdgus35/EDSR-PyTorch)), get a `load/benchmark` folder with sub-folders `Set5/, Set14/, B100/, Urban100/`.

## Reproducing Experiments

**1. Running the Attack code**

```
python SIAGT.py --config configs/test_attack/test-liif-attack.yaml --model "your pre-model"
```

**2. Tuning parameters**

Modify the "attack_setting" parameter in test-liif-attack.yaml

**3. Demo**

- attack_setting:

```
attack_setting:
  alpha: 0.031372 #8/255
  beta: 2
  delta_g: 0.005
  num_iters: 50
  query_number: 4
  query_block: 1
  pred_scales: [2,4,8]
```

```
python SIAGT.py --config configs/test_attack/test-liif-attack.yaml --model pre-models/edsr-baseline-liif.pth
```

- result
  
Due to the randomness of the attack, there may be slight fluctuations in the results.

  ```
  LR: 
  LR_PSNR: 38.3284
  LR_SSIM: 0.9673
  LR_LPIPS: 0.0371
  scale: 2
  SR_PSNR: 17.8614
  SR_SSIM: 0.3356
  SR_LPIPS: 0.5652
  scale: 4
  SR_PSNR: 16.6216
  SR_SSIM: 0.2429
  SR_LPIPS: 0.5780
  scale: 8
  SR_PSNR: 17.0773
  SR_SSIM: 0.2748
  SR_LPIPS: 0.5967
  ```

- vis_result

![image](https://github.com/user-attachments/assets/b3dfef97-28f9-4604-8a39-2cf6d797e3d3)



  
