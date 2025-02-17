# A Lottery Ticket Hypothesis Approach with Sparse Fine-tuning and MAE for Image Forgery Detection and Localization (AAAI 2025)

## Installation

This repository is built in Python 3.8 and PyTorch 1.12.
Follow these intructions:

### Install dependencies
```
pip install timm==1.0.11 rich albumentations==1.3.0 jpegio opencv-python fvcore ttach grad_cam einops -i https://pypi.mirrors.ustc.edu.cn/simple/
```

### Pretrained weight download link
   
[mae_pretrain_vit_base.pth](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth)


### Train

Localization
```
cd FMAE && sh runs/FMAE_loc.sh
```

Detection
```
cd FMAE && sh runs/FMAE_det.sh
```

### Test
```
cd FMAE && sh runs/test_FMAE_det.sh
cd FMAE && sh runs/test_FMAE_loc.sh
```

### Inference
```
python inference.py --chkpt /path/to/your-checkpoint
```

## Contact
If you have any question, please contact zhujy53@mail.ustc.edu.cn

**Acknowledgment:** This code is based on the [IMDLBenco](https://github.com/scu-zjz/IMDLBenCo) toolbox

## Citation
If this repository is helpful to your research, you can cite it like this:

