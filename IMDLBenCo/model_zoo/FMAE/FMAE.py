# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm.models.vision_transformer
from einops import rearrange
from .MoE import MoENE

# from thop import profile

import sys

sys.path.append('./modules')

from IMDLBenCo.registry import MODELS

def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]   #768
        num_patches = model.patch_embed.num_patches   #32*32=1024
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches #1
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5) #14
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5) #32
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]  #1, 1, 768
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:] #1, 196, 768
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)  # 1, 768, 14, 14
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False) # 1, 768, 32, 32
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2) #1,1024,768
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class Decoder2D(nn.Module):
    def __init__(self, in_channels, out_channels, features=[512, 256, 128, 64]):
        super().__init__()
        self.decoder_1 = nn.Sequential(
                    nn.Conv2d(in_channels, features[0], 3, padding=1),
                    nn.BatchNorm2d(features[0]),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )
        self.decoder_2 = nn.Sequential(
                    nn.Conv2d(features[0], features[1], 3, padding=1),
                    nn.BatchNorm2d(features[1]),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )
        self.decoder_3 = nn.Sequential(
            nn.Conv2d(features[1], features[2], 3, padding=1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
        )
        self.decoder_4 = nn.Sequential(
            nn.Conv2d(features[2], features[3], 3, padding=1),
            nn.BatchNorm2d(features[3]),
            nn.ReLU(inplace=True),
        )

        self.final_out = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(features[-1], out_channels, 3, padding=1)
        )

    def forward(self, x):
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        x = self.decoder_4(x)
        x = self.final_out(x)
        return x
    

class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm=3, output_size=(1, 1), eps=1e-6, *args, **kwargs):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'
    
    
class ConvGEM(nn.Module):
    def __init__(self,lambda_=0.2):
        super(ConvGEM, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(1,32,3,1,1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,1,1,1,0)
        )
        
        self.lambda_ = lambda_
    def forward(self, x):
        gem = GeneralizedMeanPooling(norm=10)
        score = self.lambda_*gem(x) + (1-self.lambda_)*(gem(torch.sigmoid(self.conv(x))))
        
        return score
    
@MODELS.register_module()  
class FMAE(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, fusion_layers=[8,9,10,11], add_noise=True, vit_pretrain_path='/model/zhujy/IMDL/mae_pretrain_vit_base.pth', det_resume_ckpt=None, num_classes=1, img_size=512, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(FMAE, self).__init__(img_size=512, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))

        self.vit_pretrain_path = vit_pretrain_path
        self.decoder = Decoder2D(embed_dim, num_classes)

        self.patch_size = patch_size

        self.img_size = img_size

        self.fusion_layers = fusion_layers
        self.noise_adapter = MoENE(img_size=self.img_size, layers=len(self.fusion_layers))
        self.zero_layer = nn.ModuleList([nn.Linear(384, 768, bias=False) for _ in range(len(self.fusion_layers))])
        self.linear_weights_init(self.zero_layer)

        self.add_noise = add_noise
        if self.add_noise:
            print('Using Noise Adapter!')
            print('fusion layers:',self.fusion_layers)
            
        self.BCE_loss = nn.BCEWithLogitsLoss()
        
        self.convgem = nn.Sequential(
            nn.Conv2d(1,32,3,1,1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,1,1,1,0)
        )
        
        self._mae_init_weights()

        self.det_resume_ckpt = det_resume_ckpt
        if self.det_resume_ckpt != None:
            self._load_det_weights()

    def forward_features(self, x):
        noise_weight = torch.tensor([0,0,0,0])
        # noise分支
        if self.add_noise:
            x_ = x
            noise_output, moe_feature, noise_weight = self.noise_adapter(x_)

        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        if self.add_noise:
            j = 0
            for i, blk in enumerate(self.blocks):
                if i in self.fusion_layers:
                    noise_feature = self.zero_layer[j](noise_output[j])
                    x = x + noise_feature
                    j += 1
                x = blk(x)
        else:
            for i, blk in enumerate(self.blocks):
                x = blk(x)

        x = self.norm(x)
        x_ = x[:, 1:, :]
        h, w = self.img_size // self.patch_size, self.img_size // self.patch_size

        outcome = rearrange(x_, 'b (h w) c -> b c h w', h=h, w=w)

        return outcome, noise_weight



    def forward(self, image, mask, edge_mask, label, **kwargs):
        x_, noise_weight = self.forward_features(image)
        mask_logits = self.decoder(x_)
        
        mask_pred = F.interpolate(mask_logits, size = (self.img_size, self.img_size), mode='bilinear', align_corners=False)
        bce_loss = kwargs['alpha']*self.BCE_loss(mask_pred, mask)
        dice_loss = kwargs['beta']*self.dice_loss(mask_pred, mask)
        predict_loss = bce_loss + dice_loss
        mask_pred = torch.sigmoid(mask_pred)
        
        gem = GeneralizedMeanPooling(norm=10)
        epoch = kwargs['epoch']
        lambda_ = 0.9975**(epoch*epoch)
        img_pred = lambda_*gem(mask_pred) + (1-lambda_)*(gem(torch.sigmoid(self.convgem(mask_pred))))
        img_pred = img_pred.squeeze(-1).squeeze(-1).squeeze(-1)        
        loss_cls = kwargs['gamma']*self.weighted_cross_entropy_loss(img_pred, label.float())
        predict_loss += loss_cls
        output_dict = {
            # loss for backward
            "backward_loss": predict_loss,
            # predicted mask, will calculate for metrics automatically
            "pred_mask": mask_pred,
            "pred_label": img_pred,           
            
            # ----values below is for visualization----
            # automatically visualize with the key-value pairs
            "visual_loss": {
                "bce_loss": bce_loss,
                "dice_loss": dice_loss,
                "label_loss": loss_cls,
                "combined_loss": predict_loss
            },

            "visual_image": {
                "pred_mask": mask_pred
            }
            # -----------------------------------------
        }

        return output_dict


    def linear_weights_init(self, m):
        for mm in m:
            if isinstance(mm, nn.Linear):
                mm.weight.data.fill_(0)
                
    def _mae_init_weights(self):
        if self.vit_pretrain_path != None:
            ckpt = torch.load(self.vit_pretrain_path, map_location='cpu')['model']
            interpolate_pos_embed(self, ckpt)
            msg = self.load_state_dict(
                ckpt, # BEIT MAE
                strict=False
            )
            print('load pretrained weights from \'{}\'.'.format(self.vit_pretrain_path))
            # print(msg)
    
    def _load_det_weights(self):
        if self.det_resume_ckpt != None:
            ckpt = torch.load(self.det_resume_ckpt, map_location='cpu')['model']
            msg = self.load_state_dict(
                ckpt,
                strict=False
            )
            print('load det pretrained weights from \'{}\'.'.format(self.det_resume_ckpt))
            
    
    def dice_loss(self, logits, targets):
        num = targets.size(0)
        smooth = 1e-8

        probs = torch.sigmoid(logits)
        intersection = (probs * targets)
        score = (2*torch.sum(intersection,dim=(2,3)) + smooth)/(torch.sum(probs*probs, dim=(2,3))+torch.sum(targets*targets, dim=(2,3)) + smooth)
        loss = 1 - score.sum()/num

        return loss
     
    def weighted_cross_entropy_loss(self, prediction, target, gamma_0=1, gamma_1=1):
        loss = - (gamma_0 * (1 - target) * torch.log(1 - prediction) +
                  gamma_1 * target * torch.log(prediction))
        return loss.mean()
    
            

def fmae_base_patch16(**kwargs):
    model = FMAE(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def fmae_large_patch16(**kwargs):
    model = FMAE(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def fmae_huge_patch14(**kwargs):
    model = FMAE(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


if __name__ == '__main__':
    x = torch.randn([1, 768, 64, 64])

    
    