#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

import os
import sys
import random
import argparse

import torch
import numpy as np
import torch.backends.cudnn as cudnn
from models.training.trainer import Trainer

__author__ = "ThanhHoang <hoangducthanh283@gmail.com>"
__status__ = "Modules"




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """ Data params """
    parser.add_argument('--experiment_name', default='Dewarp', help='Where to store logs and models')
    parser.add_argument('--train_path', default='./assets/Dewarp_Toyota4_data', help='name of train label file')
    parser.add_argument('--val_path', default='./assets/Invoice_Toyota4_CameraData_20191224', help='name of val label')
    parser.add_argument('--save_path', type=str, default='./weights', help='Path to save logs and models')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained models')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--num_workers', type=int, help='number of data loading workers', default=4)

    """ Training params """
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=1000, help='number of epochs to train for')
    parser.add_argument('--step_interval', type=int, default=50, help='Interval to print result each step')
    parser.add_argument('--loss', type=str, default='l1', help='Choose the type of loss, l1|l2|l1_smooth|laplace')
    parser.add_argument('--optimizer', type=str, default='Radam', help='Choose the type of optimizers, SGD|Adadelta|Adam|Radam|Adamw|PlainRAdam')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--warmup', type=float, default=0, help='warmup steps for adam')
    parser.add_argument('--lambdas', type=list, default=[1, 1, 1, 1, 1, 1], help='weight parameter for calculate final loss')

    """ Data processing """
    parser.add_argument('--img_height', type=int, default=380, help='the height of the input image')
    parser.add_argument('--img_width', type=int, default=380, help='the width of the input image, before=800')
    parser.add_argument('--padding', action='store_false', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--augment', action='store_true', help='whether to augment data or not')
    parser.add_argument('--sigma', type=int, default=7, help='sigma value')
    parser.add_argument('--paf_thickness', type=int, default=1, help='the thickness of paf')
    parser.add_argument('--stride', type=int, default=8, help='value of stride')

    """ Model Architecture """
    parser.add_argument('--backbone', type=str, default='shufllenetv2', help='FeatureExtraction stage, shufllenetv2|mobilenet')
    parser.add_argument('--input_channel', type=int, default=3, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=256, help='the number of output channel of Feature extractor')
    parser.add_argument('--dropout_rate', type=int, default=0.1, help='Dropout rate for dropout')
    parser.add_argument('--num_refinement_stages', type=int, default=2)
    parser.add_argument('--num_heatmaps', type=int, default=6, help="number of heat-maps, --> number keypoint = num_heatmaps - 1, as 1 spare for background")
    parser.add_argument('--num_pafs', type=int, default=16)
    opt = parser.parse_known_args()[0]

    if not opt.experiment_name:
        opt.experiment_name = f'{opt.FeatureExtraction}'
        opt.experiment_name += f'-Seed{opt.manualSeed}'
        print(opt.experiment_name)

    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    """ Seed and GPU setting """
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    print('device count', opt.num_gpu)
    
    """ Start training """
    trainer = Trainer(opt)
    trainer.train()
