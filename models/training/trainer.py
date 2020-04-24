#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

""" Training scheme for sequence to sequence modeling. """

import os
import sys
import time
import logging
import argparse
import numpy as np
import collections

import cv2
import torch
import torch.nn.init as init
import torch.optim as optim
from torch.nn import DataParallel

from datasets.data_loader import GetDataLoader
from models.keypoint_net import KeyPointNet
from utils.tools import Averager, Saver
from models.training.losses import (
    l2_loss, l1_loss, laplace_loss, margin_loss, l1_smooth_loss
)
from models.training.optimizers import (
    RAdam, AdamW, PlainRAdam
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(level=logging.DEBUG)

__author__ = "ThanhHoang"
__status__ = "Module"




class Trainer(object):
    """ Trainer """
    def __init__(self, opt):
        self.opt = opt
        self.saver = Saver()

        """ model configuration """
        if self.opt.pretrained_path != None:
            model_state, prev_opt = self.load_model_config()

        """ dataset preparation """
        self.dataset_loader = GetDataLoader(self.opt)
        self.train_loader = self.dataset_loader.train_loader
        self.val_loader = self.dataset_loader.val_loader

        """ set criterion """
        self.criterion = self.get_criterion()
        logging.debug("Current Model config parameters: {0}".format(self.opt))

        """ initialize model if not loaded by pretrained """
        if self.opt.pretrained_path == None:
            self.net = KeyPointNet(self.opt)
            self.model = self.initialize_model(self.net)
        else:
            self.model = self.load_model_state(model_state, prev_opt)

        # model = torch.nn.DataParallel(model).to(DEVICE)
        self.model = self.model.to(DEVICE)
        logging.debug("Model total parameters: {0}".format(\
            self.count_parameters(self.model)))

        """ setup optimizer """
        self.optimizer = self.get_optimizer()
        logging.debug("Optimizer: {0}".format(self.opt.optimizer))

        """ train and val static averager """
        self.train_loss_stat = Averager()
        self.val_loss_stat = Averager()

        """ define save folder """
        self.save_folder = "{0}/{1}".format(self.opt.save_path, self.opt.experiment_name)
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)

        """ logging options """
        with open(f'{self.opt.save_path}/{self.opt.experiment_name}/opt.txt', 'a') as opt_file:
            opt_log = '------------ Options -------------\n'
            args = vars(self.opt)
            for k, v in args.items():
                opt_log += f'{str(k)}: {str(v)}\n'
            opt_log += '---------------------------------------\n'
            logging.debug(opt_log)
            opt_file.write(opt_log)

    def initialize_model(self, model):
        """ Weight initialization """
        for name, param in model.named_parameters():
            try:
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    #init.kaiming_normal_(param)
                    init.xavier_uniform_(param)
            except Exception as e:  # for batchnorm.
                # logging.debug(e)
                if 'weight' in name:
                    param.data.fill_(1)
                continue
        return model

    def load_model_config(self):
        logging.debug("Loading model configures ...")
        checkpoint_dict = self.saver.load_checkpoint(self.opt.pretrained_path)
        model_state = checkpoint_dict.get("state_dict", None)
        prev_opt = checkpoint_dict.get("configs", None)

        logging.debug("Previous training Epoch: {0} Train loss: {1} Val loss: {2}".format(\
        checkpoint_dict.get("epoch"), checkpoint_dict.get("train_loss"), checkpoint_dict.get("val_loss")))
        logging.debug("Old configuration: {0}".format(checkpoint_dict.get("configs", None)))
        return (model_state, prev_opt)

    def load_model_state(self, source_state, prev_opt):
        logging.debug("Loading pretrained model ...")
        net = KeyPointNet(self.opt)
        target_state = net.state_dict()
        new_target_state = collections.OrderedDict()
        
        for target_key, target_value in target_state.items():
            if (target_key in source_state and \
                source_state[target_key].size() == target_state[target_key].size()):
                new_target_state[target_key] = source_state[target_key]
            else:
                new_target_state[target_key] = target_state[target_key]
                print('[WARNING] Not found pre-trained parameters for {}'.format(target_key))

        try:
            net.load_state_dict(new_target_state) #strict=False
        except RuntimeError:
            logging.debug("Could not load_state_dict by the normal way, \
                retrying with DataParallel loading mode...")
            net = torch.nn.DataParallel(net)
            net.load_state_dict(new_target_state) #strict=False
            logging.debug("Loading Success!")
        return net

    def get_optimizer(self):
        if self.opt.optimizer.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), \
                lr=self.opt.lr, momentum=self.opt.momentum, weight_decay=self.opt.weight_decay)
        elif self.opt.optimizer.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), \
                lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
        elif self.opt.optimizer.lower() == 'adadelta':
            optimizer = optim.Adadelta(self.model.parameters(), \
                lr=self.opt.lr, rho=self.opt.rho, eps=self.opt.eps)
        elif self.opt.optimizer.lower() == 'radam':
            optimizer = RAdam(self.model.parameters(), \
                lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2), weight_decay=self.opt.weight_decay)
        elif self.opt.optimizer.lower() == 'plainradam':
            optimizer = PlainRAdam(self.model.parameters(), \
                lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2), weight_decay=self.opt.weight_decay)
        elif self.opt.optimizer.lower() == 'adamw':
            optimizer = AdamW(self.model.parameters(), \
                lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2), weight_decay=self.opt.weight_decay, warmup=self.opt.warmup)
        else:
            raise ValueError("Invalid argumetns optimizer !!!")
        return optimizer

    def get_criterion(self):
        """ setup loss """
        if 'l2' in self.opt.loss:
            criterion = l2_loss
        elif 'l1_smooth' in self.opt.loss:
            criterion = l1_smooth_loss
        elif 'l1' in self.opt.loss:
            criterion = l1_loss
        elif 'laplace' in self.opt.loss:
            criterion = laplace_loss
        elif 'margin' in self.opt.loss:
            criterion = margin_loss
        else:
            raise ValueError("Invalid arguments loss !!!")
        return criterion

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def count_parameters(self, model):
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return f'{num_params:,}'

    def train_batch(self):
        """ train for each epoch """
        self.model.train()
        for p in self.model.parameters():
                p.requires_grad = True
        self.train_loss_stat.reset()
        self.optimizer.zero_grad()

        for idx, batch_samples in enumerate(self.train_loader):
            # Input images and label masks 
            images = batch_samples['image'].to(DEVICE)
            keypoint_masks = batch_samples['keypoint_mask'].to(DEVICE)
            paf_masks = batch_samples['paf_mask'].to(DEVICE)
            keypoint_maps = batch_samples['keypoint_maps'].to(DEVICE)
            paf_maps = batch_samples['paf_maps'].to(DEVICE)
            
            # output masks
            stages_output = self.model(images)

            # heatmaps loss, paf loss per stage
            losses = []
            total_losses = [0, 0] * (self.opt.num_refinement_stages + 1)

            # Calculate heat-map and paf losses
            for loss_idx in range(len(total_losses) // 2):
                losses.append(self.criterion(stages_output[loss_idx * 2], \
                    keypoint_maps, keypoint_masks, images.shape[0]))
                    
                losses.append(self.criterion(stages_output[loss_idx * 2 + 1], \
                    paf_maps, paf_masks, images.shape[0]))
                total_losses[loss_idx * 2] += losses[-2].item()
                total_losses[loss_idx * 2 + 1] += losses[-1].item()

            loss_values = [lam * l for lam, l in \
                zip(self.opt.lambdas, losses) if l is not None]
            loss = sum(loss_values)
            
            self.model.zero_grad()
            loss.backward()

            # gradient clipping with 5 (Default)
            torch.nn.utils.clip_grad_norm_(\
                self.model.parameters(), self.opt.grad_clip)
            self.optimizer.step()
            self.train_loss_stat.add(loss.data)

            if self.num_step % self.opt.step_interval == 0:
                logging.debug("Train Step: {0} Loss: {1} Head losses: {2} learning rate: {3} Elapsed time: {4}".format(\
                    self.num_step, loss.data, total_losses, self.get_lr(self.optimizer), time.time() - self.stime))
            self.num_step += 1
        return self.train_loss_stat.average().data

    def val_batch(self):
        """ validation or evaluation """
        self.model.eval()
        self.val_loss_stat.reset()
        forward_time = time.time()

        for p in self.model.parameters():
            p.requires_grad = False

        for idx, batch_samples in enumerate(self.val_loader):
            # Run infer the stage outputs 
            with torch.no_grad():
                images = batch_samples['image'].to(DEVICE)
                keypoint_masks = batch_samples['keypoint_mask'].to(DEVICE)
                paf_masks = batch_samples['paf_mask'].to(DEVICE)
                keypoint_maps = batch_samples['keypoint_maps'].to(DEVICE)
                paf_maps = batch_samples['paf_maps'].to(DEVICE)
                stages_output = self.model(images)

            # heatmaps loss, paf loss per stage
            losses = []
            total_losses = [0, 0] * (self.opt.num_refinement_stages + 1)

            # Calculate heat-map and paf losses
            for loss_idx in range(len(total_losses) // 2):
                losses.append(self.criterion(stages_output[loss_idx * 2], \
                    keypoint_maps, keypoint_masks, images.shape[0]))
                    
                losses.append(self.criterion(stages_output[loss_idx * 2 + 1], \
                    paf_maps, paf_masks, images.shape[0]))
                total_losses[loss_idx * 2] += losses[-2].item()
                total_losses[loss_idx * 2 + 1] += losses[-1].item()

            loss_values = [lam * l for lam, l in \
                zip(self.opt.lambdas, losses) if l is not None]
            loss = sum(loss_values)

            # Update all loss static
            loss = sum(losses)
            self.val_loss_stat.add(loss.data)

        logging.debug("Val: Loss: {0} learning rate: {1} Elapsed time: {2}".format(\
            self.val_loss_stat.average().data, self.get_lr(self.optimizer), time.time() - forward_time))
        return self.val_loss_stat.average().data

    def train(self):
        self.num_step = 0
        self.stime = time.time()
        best_loss = 1000.0

        # drop_after_epoch = [100, 200, 260]
        # scheduler = optim.lr_scheduler.MultiStepLR(\
        #     self.optimizer, milestones=drop_after_epoch, gamma=0.333)

        for epoch in range(self.opt.num_epoch):
            # scheduler.step()
            train_loss = self.train_batch()
            val_loss = self.val_batch()

            if val_loss < best_loss:
                best_loss = val_loss
                self.saver.save_checkpoint({
                    "state_dict": self.model.state_dict(),
                    "configs": self.opt,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss
                }, f'{self.save_folder}/best_loss.pt')
        self.saver.save_checkpoint({
            "state_dict": self.model.state_dict(),
            "configs": self.opt,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        }, os.path.join(self.save_folder, "model_epoch_{0}.pt".format(epoch)))
