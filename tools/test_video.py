
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------
import math
import os
import pprint
import _init_paths
from lib.core.Counter import Counter
from lib.utils.utils import create_logger, random_seed_setting
from lib.utils.modelsummary import get_model_summary
from lib.core.cc_function import test_cc
import datasets
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
import numpy as np
import timeit
import time
import logging
import argparse
from lib.models.build_counter import Baseline_Counter
from lib.utils.dist_utils import (
    get_dist_info,
    init_dist)
from mmcv import Config, DictAction

import cv2
import tqdm
import matplotlib.pyplot as plt
import pickle

import decord


def pad_image( image, h, w, size, padvalue):
    pad_image = image.copy()
    pad_h = max(size[0] - h, 0)
    pad_w = max(size[1] - w, 0)
    if pad_h > 0 or pad_w > 0:
        pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0,
            pad_w, cv2.BORDER_CONSTANT,
            value=padvalue)
    
    return pad_image

def check_img(image,divisor ):
    h, w = image.shape[:2]
    if h % divisor != 0:
        real_h = h+(divisor - h % divisor)
    else:
        real_h = h
    if w % divisor != 0:
        real_w = w+(divisor - w % divisor)
    else:
        real_w = 0
    image = pad_image(image, size=(real_h, real_w), h=h,w=w, padvalue= (0.0, 0.0, 0.0))
    return image

def input_transform(image):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    return image

def parse_args():
    parser = argparse.ArgumentParser(description='Test crowd counting network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--checkpoint',
                    help='experiment configure file name',
                    required=True,
                    type=str)
    parser.add_argument('--video',
                        required=True,
                        type=str,
                        help='video path')
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi','torchrun'],
                        default='none',
                        help='job launcher')
    
        
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    config = Config.fromfile(args.cfg)

    logger, final_output_dir = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))
    random_seed_setting(config)

    # build model
    device = torch.device('cuda:{}'.format(args.local_rank))
    model = Baseline_Counter(config.network,config.dataset.den_factor,config.train.route_size,device)
    if args.checkpoint:
        model_state_file = args.checkpoint
    elif config.test.model_file:
        model_state_file = config.test.model_file
    else:
        model_state_file = "/home/cho092871/Desktop/Networks/STEERER/pretrained/SHHB_mae_5.8_mse_8.5.pth"
    logger.info('=> loading model from {}'.format(model_state_file))
    pretrained_dict = torch.load(model_state_file)
    model.load_state_dict(pretrained_dict,strict=False)

    vr = decord.VideoReader(args.video)

    time_res=[]
    pred_results = []
    model = model.cuda()
    model.eval()
    
    
    height, width, _= check_img(vr[0].asnumpy(), 32).shape
    print(height, width)
    video_name = os.path.basename(args.video)
    writer = cv2.VideoWriter(video_name, fourcc=cv2.VideoWriter_fourcc(*"MP4V"), fps=30, frameSize=(width, height))
    
    for idx, frame in enumerate(tqdm.tqdm(vr)):
        if idx > 30 * 10:
            break
        # frame = frame.asnumpy()
        frame = vr[idx + 30 * 381]
        frame = frame.asnumpy()
        with torch.no_grad():
            cv_image = check_img(frame, 32)
            input_image = input_transform(cv_image)
            input_image = input_image.transpose((2, 0, 1))
            
            input_image = torch.from_numpy(input_image)
            input_image = input_image.unsqueeze(0).cuda()
            # print(input_image.shape)
            start_time = time.time()
            result = model(input_image, mode='test')
            time_res.append(time.time() - start_time)
            pred = result[0]
            pred_cnt = torch.sum(pred).item() 
            pred = pred / 100
            # pred = torch.where(torch.isinf(pred), torch.tensor(0.0), pred)
            # print(pred.shape)
            
            pred_results.append(pred_cnt)

            
            pred = pred.squeeze().detach().cpu().numpy()
            max_pred = np.max(pred)
            min_pred = np.min(pred)
            
            pred = (pred - min_pred) / (max_pred - min_pred)
            pred = pred * 255
            pred = pred.astype(np.uint8)
            
            pred = cv2.applyColorMap(pred, cv2.COLORMAP_JET)

            res = cv2.addWeighted(cv_image, 0.6, pred, 0.4, 0)
            text = f"pred : {int(pred_cnt)}"
            org = (10, 40)  # Bottom-left corner of the text string in the image
            font = cv2.FONT_HERSHEY_SIMPLEX
            res = cv2.putText(res, text, org, font, 1, (0, 0, 255), 2)
            writer.write(res)
            
    writer.release()
    
    plt.plot(pred_results)
    plt.ylim(bottom=0)
    plt.xlabel("frame")
    plt.ylabel("count")
    plt.savefig(f"{video_name}.png")
    

if __name__ == '__main__':
    main()
