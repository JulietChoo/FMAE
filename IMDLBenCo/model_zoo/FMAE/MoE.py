import torch
import torch.nn as nn
import torch.nn.functional as F
from .DnCNN import make_net
import timm
from functools import partial
from einops import rearrange

def rgb2gray(rgb):
    b, g, r = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
    gray = 0.2989*r + 0.5870*g + 0.1140*b
    gray = torch.unsqueeze(gray, 1)
    return gray


class BayarConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.minus1 = (torch.ones(self.in_channels, self.out_channels, 1) * -1.000)

        super(BayarConv2d, self).__init__()
        # only (kernel_size ** 2 - 1) trainable params as the center element is always -1
        self.kernel = nn.Parameter(torch.rand(self.in_channels, self.out_channels, kernel_size ** 2 - 1),
                                   requires_grad=True)


    def bayarConstraint(self):
        self.kernel.data = self.kernel.permute(2, 0, 1)
        self.kernel.data = torch.div(self.kernel.data, self.kernel.data.sum(0))
        self.kernel.data = self.kernel.permute(1, 2, 0)
        ctr = self.kernel_size ** 2 // 2
        real_kernel = torch.cat((self.kernel[:, :, :ctr], self.minus1.to(self.kernel.device), self.kernel[:, :, ctr:]), dim=2)
        real_kernel = real_kernel.reshape((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        return real_kernel

    def forward(self, x):
        output = F.conv2d(x, self.bayarConstraint(), stride=self.stride, padding=self.padding)
        return output


class SRMConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=2):
        super(SRMConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.SRMWeights = nn.Parameter(
            self._get_srm_list(), requires_grad=False)

    def _get_srm_list(self):
        # srm kernel 1
        srm1 = [[0,  0, 0,  0, 0],
                [0, -1, 2, -1, 0],
                [0,  2, -4, 2, 0],
                [0, -1, 2, -1, 0],
                [0,  0, 0,  0, 0]]
        srm1 = torch.tensor(srm1, dtype=torch.float32) / 4.

        # srm kernel 2
        srm2 = [[-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1]]
        srm2 = torch.tensor(srm2, dtype=torch.float32) / 12.

        # srm kernel 3
        srm3 = [[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, -2, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]]
        srm3 = torch.tensor(srm3, dtype=torch.float32) / 2.

        return torch.stack([torch.stack([srm1, srm1, srm1], dim=0), torch.stack([srm2, srm2, srm2], dim=0), torch.stack([srm3, srm3, srm3], dim=0)], dim=0)

    def forward(self, X):
        # X1 =
        return F.conv2d(X, self.SRMWeights, stride=self.stride, padding=self.padding)

def get_noiseprint(path):
    num_levels = 17
    out_channel = 1
    state_dict = torch.load(path)
    noiseprint = make_net(3, kernels=[3, ] * num_levels,
                                   features=[64, ] * (num_levels - 1) + [out_channel],
                                   bns=[False, ] + [True, ] * (num_levels - 2) + [False, ],
                                   acts=['relu', ] * (num_levels - 1) + ['linear', ],
                                   dilats=[1, ] * num_levels,
                                   bn_momentum=0.1, padding=1)

    msg = noiseprint.load_state_dict(state_dict)
    print('Loading the weights of noiseprint:', msg)
    noiseprint.requires_grad_(False)

    return noiseprint


class noise_output(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save(self, idx: int, clip_feat: torch.Tensor):
        self[idx] = clip_feat  # 包括cls_token


class noise_vit_tiny(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, **kwargs):
        super(noise_vit_tiny, self).__init__(**kwargs)

    def forward(self, x):
        outputs = noise_output()

        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks, start=0):
            x = blk(x)
            outputs.save(i, x)

        return outputs


class MoENE(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, layers=4,**kwargs):
        super(MoENE, self).__init__()
        # SRM
        self.srm = SRMConv2D(in_channels, out_channels)
        # Bayar
        self.bayar = BayarConv2d(in_channels=1, out_channels=out_channels)
        # Noiseprint
        self.noiseprint = get_noiseprint('IMDLBenCo/model_zoo/FMAE/noiseprint.pth')

        self.layers = layers
        self.vit = noise_vit_tiny(patch_size=16, embed_dim=384, depth=self.layers, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        self.pool = nn.Sequential(
            nn.Conv2d(3,3,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(3, 3, 1, 1, 0),
            nn.AdaptiveAvgPool2d(1),
        )
        self.linear = nn.Linear(1,9)
        self.conv_1 = nn.Conv2d(9, 3, kernel_size=1,stride=1,padding=0)
        self.relu = nn.ReLU()
        self.W2 = nn.Parameter(torch.empty(9, 9))
        nn.init.normal_(self.W2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ = x.clone()
        W1 = self.pool(x_).view(B,C,-1)
        W1 = self.linear(W1).permute(0,2,1)
        Tc = x_.mean(dim=(2, 3), keepdim=True).squeeze(-1)
        We = self.relu(torch.matmul(W1, Tc))
        We = self.softmax(torch.matmul(self.W2, We))
        We = We.unsqueeze(-1).repeat(1,1,H,W)

        # SRM
        srm_feature = self.srm(x)

        # Bayar
        x_gray = rgb2gray(x)
        bayar_feature = self.bayar(x_gray)

        # noiseprint
        noiseprint_feature = self.noiseprint(x).repeat(1,3,1,1)

        # 返回noise_feature
        noise_feature = torch.cat([srm_feature, bayar_feature, noiseprint_feature], dim=1)
        noise_feature = We*noise_feature

        Gf = self.conv_1(noise_feature)
        outputs = self.vit(Gf)
        return outputs, Gf, We
    


if __name__ == '__main__':
    x = torch.randn([2, 3, 224, 224])
    model = MoENE()
    y = model(x)
    print(y)