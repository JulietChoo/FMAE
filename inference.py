import os
import json
import time
import types
import inspect
import argparse
import datetime
import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

import IMDLBenCo.training_scripts.utils.misc as misc

from IMDLBenCo.registry import MODELS, POSTFUNCS
from IMDLBenCo.transforms import get_albu_transforms
from IMDLBenCo.datasets import ManiDataset
from IMDLBenCo.model_zoo import *
from tqdm import tqdm
import cv2
import numpy as np


def get_args_parser():
    parser = argparse.ArgumentParser('IMDLBench testing launch!', add_help=True)
    # Model name
    parser.add_argument('--model', default='FMAE', type=str,
                        help='The name of applied model')
    
    # 可以接受label的模型是否接受label输入，并启用相关的loss。
    parser.add_argument('--if_predict_label', default=True,
                        help='Does the model that can accept labels actually take label input and enable the corresponding loss function?')
    # ----Dataset parameters 数据集相关的参数----
    parser.add_argument('--image_size', default=512, type=int,
                        help='image size of the images in datasets')
    
    parser.add_argument('--if_padding', default=False,
                        help='padding all images to same resolution.')
    
    parser.add_argument('--if_resizing', default=True, 
                        help='resize all images to same resolution.')
    # If edge mask activated
    parser.add_argument('--edge_mask_width', default=7, type=int,
                        help='Edge broaden size (in pixels) for edge maks generator.')
    parser.add_argument('--test_data_json', default='./test_datasets_loc.json', type=str,
                        help='test dataset json, should be a json file contains many datasets. Details are in readme.md')
    # ------------------------------------
    # Testing 相关的参数
    parser.add_argument('--chkpt', default = '/model/zhujy/IMDL/FMAE_loc/checkpoint-25.pth', type=str, help='path to the dir where saving checkpoints')
    parser.add_argument('--test_batch_size', default=1, type=int,
                        help="batch size for testing")

    # ----输出的日志相关的参数-----------
    parser.add_argument('--output_dir', default='/output',
                        help='path where to save, empty for no saving')
    # -----------------------
    
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    parser.add_argument('--if_detect', default=False, 
                        help='detect')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='weight loss')
    parser.add_argument('--beta', type=float, default=0.7,
                        help='weight loss')
    parser.add_argument('--gamma', type=float, default=0,
                        help='weight loss')
    args, remaining_args = parser.parse_known_args()
    # 获取对应的模型类
    model_class = MODELS.get(args.model)

    # 根据模型类动态创建参数解析器
    model_parser = misc.create_argparser(model_class)
    model_args = model_parser.parse_args(remaining_args)

    return args, model_args


def main(args, model_args):
    # init parameters for distributed training
    misc.init_distributed_mode(args)
    # import torch.multiprocessing
    # torch.multiprocessing.set_sharing_strategy('file_system')
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("=====args:=====")
    print("{}".format(args).replace(', ', ',\n'))
    print("=====Model args:=====")
    print("{}".format(model_args).replace(', ', ',\n'))
    device = torch.device(args.device)
    
    test_transform = get_albu_transforms('test')

    with open(args.test_data_json, "r") as f:
        test_dataset_json = json.load(f)
    
    
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
    else:
        global_rank = 0
    
    # ========define the model directly==========
    # model = FMAE(
    #     det_resume_ckpt=args.checkpoint_path
    # )
    
    # --------------- or -------------------------
    # Init model with registry
    model = MODELS.get(args.model)
    
    # Filt usefull args
    if isinstance(model,(types.FunctionType, types.MethodType)):
        model_init_params = inspect.signature(model).parameters
    else:
        model_init_params = inspect.signature(model.__init__).parameters
        
    combined_args = {k: v for k, v in vars(args).items() if k in model_init_params}
    combined_args.update({k: v for k, v in vars(model_args).items() if k in model_init_params})
    model = model(**combined_args)
    
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.to(device)
    model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    
    start_time = time.time()
    # get post function (if have)
    post_function_name = f"{args.model}_post_func".lower()
    print(f"Post function check: {post_function_name}")
    print(POSTFUNCS)
    if POSTFUNCS.has(post_function_name):
        post_function = POSTFUNCS.get(post_function_name)
    else:
        post_function = None
    
    # Start go through each datasets:
    for dataset_name, dataset_path in test_dataset_json.items():
        dataset_test = ManiDataset(
                dataset_path,
                is_padding=args.if_padding,
                is_resizing=args.if_resizing,
                output_size=(args.image_size, args.image_size),
                common_transforms=test_transform,
                edge_width=args.edge_mask_width,
                post_funcs=post_function
            )
        # ------------------------------------
        print(dataset_test)
        print("len(dataset_test)", len(dataset_test))

        # Sampler
        if args.distributed:
            sampler_test = torch.utils.data.DistributedSampler(
                    dataset_test, 
                    num_replicas=num_tasks, 
                    rank=global_rank, 
                    shuffle=False,
                    drop_last=True
            )
            print("Sampler_test = %s" % str(sampler_test))
        else:
            sampler_test = torch.utils.data.RandomSampler(dataset_test)

        data_loader_test = torch.utils.data.DataLoader(
                dataset_test, 
                sampler=sampler_test,
                batch_size=args.test_batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=True,
            )

        print(f"Start testing on {dataset_name}! ")
        ckpt = torch.load(args.chkpt, map_location='cuda')
        if args.distributed:
            model.module.load_state_dict(ckpt['model']) 
        else:
            model.load_state_dict(ckpt['model'])
        epoch = ckpt['epoch']

        model.eval()
        with torch.no_grad():
            for data_dict in tqdm(data_loader_test):
                # move to device
                for key in data_dict.keys():
                    if isinstance(data_dict[key], torch.Tensor):
                        data_dict[key] = data_dict[key].to(device)
                output_dict = model(**data_dict, alpha=args.alpha, beta=args.beta, gamma=args.gamma, epoch=epoch)
                mask_pred = output_dict['pred_mask']
                img_shape = data_dict['shape']
                mask_pred = mask_pred[0][0].detach().cpu().numpy()
                save_dir = os.path.join(args.output_dir, dataset_name)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, data_dict['name'][0][:-4]+'.png')
                output_img = (mask_pred* 255).astype(np.uint8)
                output_img = cv2.resize(output_img, (int(img_shape[0][1]), int(img_shape[0][0])))
                cv2.imwrite(save_path, output_img)
            
            
            


if __name__ == '__main__':
    args, model_args = get_args_parser()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, model_args)        