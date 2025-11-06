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

### Localization weight download link
[checkpoint-25.pth](https://drive.google.com/file/d/1PCjmmtY40ORuypOYGLf2rqhcVEbkSWkY/view?usp=drive_link)

### Detection weight download link
[checkpoint-47.pth](https://drive.google.com/file/d/1nsEq9xWaxlM3_T3DV-bSJSqxKoa3rNWf/view?usp=drive_link)



### Train

Localization
```
cd FMAE && sh runs/FMAE_loc.sh
```

Detection
```
cd FMAE && sh runs/FMAE_det.sh
```

💡 **Tip:** For convenience, you can use FMAE/tools/sample_data.py to generate a validation set for training.

Through experiments, we found that better results can be achieved by adjusting the ratio of Casiav2 and FantasticReality, see IMDLBenCo.datasets.UnBancedDataset for details.

### Test
```
cd FMAE && sh runs/test_FMAE_det.sh   # checkpoint-47.pth
cd FMAE && sh runs/test_FMAE_loc.sh   # checkpoint-25.pth
```

### Visualization
```
cd FMAE && python inference.py --chkpt /path/to/loc-checkpoint/checkpoint-25.pth
```

## Contact
If you have any question or need any dataset, please contact zhujy53@mail.ustc.edu.cn.

**Acknowledgment:** This code is based on the [IMDLBenco](https://github.com/scu-zjz/IMDLBenCo) toolbox.

<!-- ## Citation
If this repository is helpful to your research, you can cite it like this: -->

