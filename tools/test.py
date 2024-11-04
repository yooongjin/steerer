
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------
from PIL import Image
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
import einops
import cv2
import tqdm
import matplotlib.pyplot as plt
import pickle

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

    # logger.info('GPU idx:'+os.environ['CUDA_VISIBLE_DEVICES'])
    # gpus = config.gpus
    # distributed = torch.cuda.device_count() > 1
    # if distributed:
    #     torch.cuda.set_device(args.local_rank)
    #     # torch.distributed.init_process_group(
    #     #     backend="nccl", init_method="env://",
    #     # )
    #     init_dist(args.launcher)
    #     # torch.cuda.set_device(args.local_rank)
    #     if args.launcher == 'pytorch':
    #         args.local_rank = int(os.environ["LOCAL_RANK"])
    #     else:
    #         rank, world_size = get_dist_info()
    #         args.local_rank = rank
    
    # cudnn related setting
    random_seed_setting(config)

    # build model
    device = torch.device('cuda:{}'.format(args.local_rank))

    model = Baseline_Counter(config.network,config.dataset.den_factor,config.train.route_size,device)

    # dump_input = torch.rand(
    #     (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    # )
    # logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    if args.checkpoint:
        model_state_file = args.checkpoint
    elif config.test.model_file:
        model_state_file = config.test.model_file
    else:
        model_state_file = "/home/cho092871/Desktop/Networks/STEERER/exp/ASSEMBLE/MocHRBackbone_hrnet48/ASSEMBLE_final_2024-09-20-17-09/Ep_441_mae_59.899008520254604_mse_227.68671281631933.pth"
    
    logger.info('=> loading model from {}'.format(model_state_file))

    pretrained_dict = torch.load(model_state_file)
    model.load_state_dict(pretrained_dict, strict=False)
    
    import glob
    root = "/home/cho092871/Desktop/Datasets/Itaewon_ver2.0"
    imgs = glob.glob(os.path.join(root, "imgs", "*", "*.jpeg"))
    save_path = "Itaewon"
    os.makedirs(save_path, exist_ok=True)
    mae = []
    mse = []
    time_res=[]
    pred_results = []
    gt_restults = []
    model = model.cuda()
    model.eval()

    with torch.no_grad():
        for img_path in tqdm.tqdm(imgs):
            gt_file = os.path.basename(img_path).replace(".jpeg", ".npy")
            gt_path = os.path.join(os.path.join(root, "annotations", gt_file))
            points = np.load(gt_path)
            gt_cnt = points.shape[0]

            cv_image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            
            # pil_image = Image.open(img_path)
            # width, height = pil_image.size
            # pil_image = pil_image.resize((width//2, height//2))
            height, width, _ = cv_image.shape
            cv_image = cv2.resize(cv_image, dsize=(width//4, height//4))
            print(cv_image.shape)
            # cv_image = np.array(pil_image)
            cv_image = check_img(cv_image, 32)

            input_image = input_transform(cv_image)
            input_image = input_image.transpose((2, 0, 1))
            
            input_image = torch.from_numpy(input_image)
            input_image = input_image.unsqueeze(0).cuda()
            # input_image = einops.rearrange(input_image, "B C (H TH) (W TW) -> (B TH TW) C H W", 
            #                                H=input_image.shape[2]//2, 
            #                                W=input_image.shape[3]//2,
            #                                TH=2,
            #                                TW=2)
            input_height, input_width = input_image.shape[2:]
            # print(input_image.shape)
            start_time = time.time()
            result = model(input_image, mode='test')
            time_res.append(time.time() - start_time)
            pred = result
            # print(pred.shape)
            pred_cnt = torch.sum(pred).item() / 100
            # pred = einops.rearrange(pred, "(B TH TW) C H W -> B C (H TH) (W TW)", B=1, TH=2, TW=2)
            
            pred_results.append(pred_cnt)
            gt_restults.append(gt_cnt)
            mae.append(abs(gt_cnt - pred_cnt))
            mse.append((gt_cnt - pred_cnt)**2)
            
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

            # Put the text on the image
            if gt_cnt is not None:
                text = f"gt : {int(gt_cnt)} pred : {int(pred_cnt)}"
            cv2.putText(res, text, org, font, 1, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(save_path, gt_file.replace(".npy", ".png")), res)
            
    sorted_indices = sorted(range(len(gt_restults)), key=lambda k: gt_restults[k])
    gt_restults = [gt_restults[i] for i in sorted_indices]
    pred_results = [pred_results[i] for i in sorted_indices]

    plt.plot(gt_restults, label='GT')
    plt.plot(pred_results, label='Pred')
    plt.legend()
    plt.title("Predict and GT")
    plt.savefig(os.path.join(save_path, "predict_gt.png"))
    plt.show()

    print("MAE : {}".format(sum(mae) / len(mae)))
    print("MSE : {}".format(math.sqrt(sum(mse) / len(mse))))
    print("Inference time : {}".format(sum(time_res) / len(time_res)))

    with open("STEERER_pred.pickle", 'wb') as fw:
        pickle.dump(pred_results, fw)
    
    with open("gt.pickle", 'wb') as fw:
        pickle.dump(gt_restults, fw)

if __name__ == '__main__':
    main()
