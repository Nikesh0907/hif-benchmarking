# coding: utf-8
# Script for testing TSFN
#
# Reference: 
# Hyperspectral Image Super-Resolution via Deep Prior Regularization with Parameter Estimation
# Xiuheng Wang, Jie Chen, Qi Wei, Cédric Richard
#
# 2019/05
# Implemented by
# Xiuheng Wang
# xiuheng.wang@mail.nwpu.edu.cn

from __future__ import division
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from model import Net
from dataset import Dataset_cave_test
from save_image import save_image
import time
import argparse

import  os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# ===========================================================
# Test settings
# ===========================================================
parser = argparse.ArgumentParser(description='CAVE database superresolution:')
# model configuration
parser.add_argument('--HSI_num_residuals', type=int, default=6, help="Set numbers of ResBlock of HSI branch")
parser.add_argument('--RGB_num_residuals', type=int, default=6, help="Set numbers of ResBlock of RGB branch")
# model storage path
# If --weights is not provided, we will load <model_path>/ssfsr_9layers_epoch500.pkl
parser.add_argument('--model_path', type=str, default='./models/', help='Directory that contains TSFN pretrained weights')
parser.add_argument('--weights', type=str, default=None, help='Explicit path to weights .pkl (overrides --model_path default)')
# test data path
parser.add_argument('--img_path', type=str, default='./data/test', help='Directory with test .tif files')
# degradation scenario
parser.add_argument('--degradation_mode', type=int, default=0, help='Degradation scenario: 0,1,2,3')
# results storage path
parser.add_argument('--results_path', type=str, default='./results/', help='Set results storage path')

args = parser.parse_args()

if args.weights is not None:
    model_path = args.weights
else:
    model_path = os.path.join(args.model_path, 'ssfsr_9layers_epoch500.pkl')

img_path = args.img_path
degradation_mode = args.degradation_mode  # choose different degradation scenarios: 0, 1, 2, 3.

save_point = torch.load(model_path)
model_param = save_point['state_dict']
model = Net(HSI_num_residuals=args.HSI_num_residuals, RGB_num_residuals=args.RGB_num_residuals)
model = nn.DataParallel(model)
model.load_state_dict(model_param)


device="cuda:0"
model=model.to(device=device, dtype=torch.float)
model.eval()


test_data = Dataset_cave_test(img_path, degradation_mode)
num = test_data.__len__()
print('the total number of test images is:', num)

results_path = args.results_path
if not os.path.exists(results_path):
    os.makedirs(results_path)

test_loader = DataLoader(dataset=test_data,
                        num_workers=0, 
                        batch_size=1,
                        shuffle=False,
                        pin_memory=True)

for i, (data_hsi, data_rgb, label) in enumerate(test_loader):
    
    data_hsi = data_hsi.to(device=device, dtype=torch.float)
    data_rgb = data_rgb.to(device=device, dtype=torch.float)
    data_label = label.to(device=device, dtype=torch.float)

    start = time.time()
    # compute output
    output = model(data_hsi, data_rgb) 
    end = time.time()
    print('The No', str(i), 'image costs', end - start, 'seconds')

    output = output.to('cpu')
    output = output.detach().numpy()
    output = np.squeeze(output) 

    label = label.to('cpu')
    label = label.detach().numpy()
    label = np.squeeze(label)

    save_image(output, label, i, results_path)

