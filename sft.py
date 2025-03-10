import copy
from torch.nn.utils import prune
import torch
from IMDLBenCo.model_zoo import FMAE

def extract_mask(model_dict):

    new_dict = {}

    for key in model_dict.keys():
        if '_mask' in key:

            if 'module' in key:
                new_key = key[len('module.'):-5]
            else:
                new_key = key[:-5]

            new_dict[new_key] = model_dict[key]

    return new_dict


def get_gradmask(path, mask_ratio):
    checkpoint = torch.load(path, map_location='cpu')
    checkpoint.pop("pos_embed", None)
    print("Load the gradmask checkpoint from: %s" % path)
    model = FMAE()
    msg = model.load_state_dict(checkpoint, strict=False)

    model.to('cuda')

    parameters_to_prune = []
    for i in range(12):
        parameters_to_prune.append((model.blocks[i].norm1, 'weight'))
        parameters_to_prune.append((model.blocks[i].norm1, 'bias'))
        parameters_to_prune.append((model.blocks[i].attn.qkv, 'weight'))
        parameters_to_prune.append((model.blocks[i].attn.qkv, 'bias'))
        parameters_to_prune.append((model.blocks[i].attn.proj, 'weight'))
        parameters_to_prune.append((model.blocks[i].attn.proj, 'bias'))
        parameters_to_prune.append((model.blocks[i].norm2, 'weight'))
        parameters_to_prune.append((model.blocks[i].norm2, 'bias'))
        parameters_to_prune.append((model.blocks[i].mlp.fc1, 'weight'))
        parameters_to_prune.append((model.blocks[i].mlp.fc1, 'bias'))
        parameters_to_prune.append((model.blocks[i].mlp.fc2, 'weight'))
        parameters_to_prune.append((model.blocks[i].mlp.fc2, 'bias'))
    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=mask_ratio)

    new_dict = extract_mask(model.state_dict())

    return new_dict


if __name__ == '__main__':
    get_gradmask('/model/zhujy/IMDL/gradmask.pth', 0.3)