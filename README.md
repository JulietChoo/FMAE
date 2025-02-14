# A Lottery Ticket Hypothesis Approach with Sparse Fine-tuning and MAE for Image Forgery Detection and Localization (AAAI 2025)

## Installation

This repository is built in Python 3.8 and PyTorch 1.12.
Follow these intructions:

1. Install dependencies
```
pip install timm==1.0.11 rich albumentations==1.3.0 jpegio opencv-python fvcore ttach grad_cam einops -i https://pypi.mirrors.ustc.edu.cn/simple/
```
2. Train
   
**Detection**
```
cd /code/FMAE && sh /code/FMAE/runs/FMAE_det.sh
```

**Localization**
```
cd /code/FMAE && sh /code/FMAE/runs/FMAE_loc.sh
```


## Contact
If you have any question, please contact zhujy53@mail.ustc.edu.cn

**Acknowledgment:** This code is based on the [IMDLBenco](https://github.com/scu-zjz/IMDLBenCo) toolbox