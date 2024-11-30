# Scale-Invariant Adversarial Attack against Arbitrary-scale Super-resolution

This repository is implementation for SIAGT,

> **Scale-Invariant Adversarial Attack against Arbitrary-scale Super-resolution**<br>
> Yihao Huang, Xin Luo, Qing Guo, Felix Juefei-Xu,Xiaojun Jia,Weikai Miao,Geguang Pu, Yang Liu <br>

>**Abstract**: <br>
> The advent of local continuous image function (LIIF) has garnered significant attention for arbitrary-scale super-resolution (SR) techniques. However, while the vulnerabilities of fixed-scale SR have been assessed, the robustness of continuous representation-based arbitrary-scale SR against adversarial attacks remains an area warranting further exploration. The elaborately designed adversarial attacks for fixed-scale SR are scale-dependent, which will cause time-consuming and memory-consuming problems when applied to arbitrary-scale SR. To address this concern, we propose a simple yet effective ``scale-invariant'' SR adversarial attack method with good transferability, termed SIAGT. Specifically, we propose to construct resource-saving attacks by exploiting finite discrete points of continuous representation. In addition, we formulate a coordinate-dependent loss to enhance the cross-model transferability of the attack. The attack can significantly deteriorate the SR images while introducing imperceptible distortion to the targeted low-resolution (LR) images. Experiments carried out on three popular LIIF-based SR approaches and four classical SR datasets show remarkable attack performance and transferability of SIAGT.


### Environment
- Python 3
- Pytorch 1.6.0
- TensorboardX
- basicsr
- yaml, numpy, tqdm, imageio

## Quick Start

1. Prepare pre-trained weights in the pre-model folder. Currently, LIIF, LTE, A-LIIF, and LMF models are supported.

2. Download benchmark datasets: `cd` into `load/`. Download and `tar -xf` the [benchmark datasets](https://cv.snu.ac.kr/research/EDSR/benchmark.tar) (provided by [this repo](https://github.com/thstkdgus35/EDSR-PyTorch)), get a `load/benchmark` folder with sub-folders `Set5/, Set14/, B100/, Urban100/`.

## Demo

**1. Modify the Attack parameter**

Modify the "attack_setting" parameter in configuration file,such as:

```
attack_setting:
  alpha: 0.031372 #8/255
  beta: 2
  delta_g: 0.005
  num_iters: 50
  query_number: 4
  query_block: 1
  pred_scales: [2,4]
  source_model: liif #[liif,lte,aliif,lmf]
  target_model: liif
  save: True
  trans: True
```

**2. Run**

```
python SIAGY.py --config configs/test_attack/test-B100-attack.yaml --model "your pre-model"
```

# Acknowledge
The code is built on [LIIF](https://github.com/yinboc/liif)、[LTE](https://github.com/jaewon-lee-b/lte)、[A-LIIF](https://github.com/LeeHW-THU/A-LIIF)、[LMF](https://github.com/HeZongyao/LMF). We thank the authors for sharing the codes.



  
