# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import copy
from PIL import Image
import time
import os
#from reid_sampler import StratifiedSampler
from model import ft_net, ft_net_dense, PCB
from random_erasing import RandomErasing
from tripletfolder import TripletFolder
import json
from shutil import copyfile
from scipy import spatial as spatial
import random
from functools import partial
from torch.utils.data import DataLoader

version =  torch.__version__

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--data_dir',default='../Market/pytorch',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--poolsize', default=128, type=int, help='poolsize')
parser.add_argument('--margin', default=0.3, type=float, help='margin')
parser.add_argument('--lr', default=0.01, type=float, help='margin')
parser.add_argument('--alpha', default=0.0, type=float, help='regularization, push to -1')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--PCB', action='store_true', help='use PCB+ResNet50' )

parser.add_argument('--pretrained_dir',default='',type=str, help='pretrained dir path')
parser.add_argument('--prune_rate',default=0.9,type=float, help='prune rate for SFP')
parser.add_argument('--Epochs',default=100,type=int, help='maximum epochs for SFP')
parser.add_argument('--save_epoch',default=100,type=int, help='start epoch to save model for SFP')
parser.add_argument('--network_arch',default='resnet50',type=str, help='network for the training and pruning')


opt = parser.parse_args()

data_dir = opt.data_dir
name = opt.name

pretrained_dir = opt.pretrained_dir
prune_rate = opt.prune_rate
Epochs = opt.Epochs
start_save_epoch = opt.save_epoch

prune_arch = ['resnet']
prune_skip_downsample = 1
resnet_arch = 'resnet50'
layer_begin =0
layer_end = 156
layer_inter = 1
saving_dir = '../results/ft_resnet50'

if 'resnet50' in opt.network_arch:
    resnet_arch = 'resnet50'
    layer_begin =0
    layer_end = 156
    layer_inter = 1
    saving_dir = '../results/ft_resnet50'
elif 'resnet18' in opt.network_arch:
    resnet_arch = 'resnet18'
    layer_begin =0
    layer_end = 57
    layer_inter = 1
    saving_dir = '../results/ft_resnet18'
elif 'resnet34' in opt.network_arch:
    resnet_arch = 'resnet34'
    layer_begin =0
    layer_end = 105
    layer_inter = 1
    saving_dir = '../results/ft_resnet34'
else:
    pass


str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])

train_all = ''
if opt.train_all:
     train_all = '_all'


batch = {}
use_gpu = torch.cuda.is_available()



def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    last_margin = 0.0

    for i, (name, param) in enumerate(model.named_parameters()):
        if 'layer2.0.conv2.weight'  in name:
            np.save('SFP_before_pruning_layer2.0conv2_0.5_'+opt.network_arch,param.cpu().detach().numpy())

    m = Mask(model)
    m.init_length()
    print("-" * 10 + "one epoch begin" + "-" * 10)
    print("the compression rate now is {:}".format(prune_rate))

    
    for epoch in range(1):
        # Each epoch has a training and validation phase
        model.load_state_dict(best_model_wts)
        m.model = model
        m.if_zero()
        m.init_mask(prune_rate,decay_rate = 0)
        m.do_mask()
        m.if_zero()
        model = m.model
        model = model.cuda()
       
    for i, (name, param) in enumerate(model.named_parameters()):
            if 'layer2.0.conv2.weight'  in name:
                np.save('SFP_after_pruning_layer2.0conv2_0.5_'+opt.network_arch,param.cpu().detach().numpy())

    
    return model



class Mask:
    def __init__(self, model):
        self.model_size = {}
        self.model_length = {}
        self.compress_rate = {}
        self.mat = {}
        self.model = model
        self.mask_index = []
        self.decay_rate = 0.2
    def pair_ind_to_dist_ind(self,d, i, j):
        index = d*(d-1)/2 - (d-i)*(d-i-1)/2 + j - i - 1
        return index
    
    def dist_ind_to_pair_ind(self,d, i):
        b = 1 - 2 * d
        x = np.floor((-b - np.sqrt(b**2 - 8*i))/2).astype(int)
        y = (i + x * (b + x + 2) / 2 + 1).astype(int)
        return (x,y)    

    def get_codebook(self, weight_torch, compress_rate, length):
        weight_vec = weight_torch.view(length)
        weight_np = weight_vec.cpu().numpy()

        weight_abs = np.abs(weight_np)
        weight_sort = np.sort(weight_abs)

        threshold = weight_sort[int(length * (1 - compress_rate))]
        weight_np[weight_np <= -threshold] = 1
        weight_np[weight_np >= threshold] = 1
        weight_np[weight_np != 1] = 0

        print("codebook done")
        return weight_np

    def get_filter_codebook(self, weight_torch, compress_rate, length):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            norm2 = torch.norm(weight_vec,2,1)
            norm2_np = norm2.cpu().numpy()
            filter_index = norm2_np.argsort()[:filter_pruned_num]
            
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(filter_index)):
                codebook[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = self.decay_rate
            #print('filter_index:',filter_index)
            print("filter codebook done")
        else:
            pass
        return codebook

    def convert2tensor(self, x):
        x = torch.FloatTensor(x)
        return x

    def init_length(self):
        for index, item in enumerate(self.model.parameters()):
            self.model_size[index] = item.size()

        for index1 in self.model_size:
            for index2 in range(0, len(self.model_size[index1])):
                if index2 == 0:
                    self.model_length[index1] = self.model_size[index1][0]
                else:
                    self.model_length[index1] *= self.model_size[index1][index2]

    def init_rate(self, layer_rate):
        if 'vgg' in prune_arch:
            cfg_5x = [24, 22, 41, 51, 108, 89, 111, 184, 276, 228, 512, 512, 512]
            cfg_official = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
            # cfg = [32, 64, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256]
            cfg_index = 0
            pre_cfg = True
            for index, item in enumerate(self.model.named_parameters()):
                self.compress_rate[index] = 1
                if len(item[1].size()) >= 4: 
                    #print(item[1].size())
                    if not pre_cfg:
                        self.compress_rate[index] = layer_rate
                        self.mask_index.append(index)
                        print(item[0], "self.mask_index", self.mask_index)
                    else:
                        self.compress_rate[index] =  1 - cfg_5x[cfg_index] / item[1].size()[0]
                        self.mask_index.append(index)
                        print(item[0], "self.mask_index", self.mask_index, cfg_index, cfg_5x[cfg_index], item[1].size()[0],
                               )
                        cfg_index += 1
        elif "resnet" in prune_arch:
            for index, item in enumerate(self.model.parameters()):
                self.compress_rate[index] = 1
            for key in range(layer_begin, layer_end + 1, layer_inter):
                self.compress_rate[key] = layer_rate
            if resnet_arch== 'resnet18':
                # last index include last fc layer
                last_index = 60
                skip_list = [0,21, 36, 51]
            elif resnet_arch == 'resnet34':
                last_index = 108
                skip_list = [0,27, 54, 93]
            elif resnet_arch == 'resnet50':
                last_index = 159
                skip_list = [12, 42, 81, 138]
            elif resnet_arch == 'resnet101':
                last_index = 312
                skip_list = [12, 42, 81, 291]
            elif resnet_arch == 'resnet152':
                last_index = 465
                skip_list = [12, 42, 117, 444]
            self.mask_index = [x for x in range(0, last_index, 1)]
            # skip downsample layer
            if prune_skip_downsample == 1:
                for x in skip_list:
                    self.compress_rate[x] = 1
                    self.mask_index.remove(x)
                    
                    #remove corresponding bn from list
                    self.compress_rate[x+1] = 1
                    self.compress_rate[x+2] = 1
                    self.mask_index.remove(x+1)
                    self.mask_index.remove(x+2)
                    #print(self.mask_index)
            else:
                pass

    def init_mask(self, layer_rate,decay_rate = 0.2):
        self.init_rate(layer_rate)
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index) and len(item.data.size()) == 4: #conv
                self.decay_rate = decay_rate
                self.mat[index] = self.get_filter_codebook(item.data, self.compress_rate[index],
                                                           self.model_length[index])
                self.mat[index] = self.convert2tensor(self.mat[index])
                
                self.mat[index] = self.mat[index].cuda()
                
                #set bn.weight and bias 
                codebook = np.zeros(item.data.size()[0])
                tmp = self.mat[index].view(item.data.size()[0],-1).sum(axis=1)
                nonzero_idx = np.nonzero(tmp.cpu().numpy())[0].tolist()
                #print(nonzero_idx)
                codebook[nonzero_idx] = 1
                
                self.mat[index+1] = codebook
                self.mat[index+2] = codebook
                
                self.mat[index+1] = self.convert2tensor(self.mat[index+1])
                self.mat[index+2] = self.convert2tensor(self.mat[index+2])
                
                self.mat[index+1] = self.mat[index+1].cuda()
                self.mat[index+2] = self.mat[index+2].cuda()
                
                #print('nonzero_idx.size',len(nonzero_idx))
                #print('index:',index)
        print("mask Ready")

    def do_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index):
                a = item.data.view(self.model_length[index])
                b = a * self.mat[index]
                item.data = b.view(self.model_size[index])
        print("mask Done")

    def if_zero(self):
        for index, item in enumerate(self.model.parameters()):
            #            if(index in self.mask_index):
            if index in [x for x in range(layer_begin, layer_end + 1, layer_inter)]:
                a = item.data.view(self.model_length[index])
                b = a.cpu().numpy()

                print("layer: %d, number of nonzero weight is %d, zero is %d" % (
                    index, np.count_nonzero(b), len(b) - np.count_nonzero(b)))


# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#

class_num = 751
 


model = ft_net(class_num,opt.network_arch)

    
print(model)

if use_gpu:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()

if not opt.PCB:
    ignored_params = list(map(id, model.model.fc.parameters() )) + list(map(id, model.classifier.parameters() ))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
             {'params': base_params, 'lr': 0.1*opt.lr},
             {'params': model.model.fc.parameters(), 'lr': opt.lr},
             {'params': model.classifier.parameters(), 'lr': opt.lr}
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)
else:
    ignored_params = list(map(id, model.model.fc.parameters() ))
    ignored_params += (list(map(id, model.classifier0.parameters() )) 
                     +list(map(id, model.classifier1.parameters() ))
                     +list(map(id, model.classifier2.parameters() ))
                     +list(map(id, model.classifier3.parameters() ))
                     +list(map(id, model.classifier4.parameters() ))
                     +list(map(id, model.classifier5.parameters() ))
                     #+list(map(id, model.classifier6.parameters() ))
                     #+list(map(id, model.classifier7.parameters() ))
                      )
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
             {'params': base_params, 'lr': 0.001},
             {'params': model.model.fc.parameters(), 'lr': 0.01},
             {'params': model.classifier0.parameters(), 'lr': 0.01},
             {'params': model.classifier1.parameters(), 'lr': 0.01},
             {'params': model.classifier2.parameters(), 'lr': 0.01},
             {'params': model.classifier3.parameters(), 'lr': 0.01},
             {'params': model.classifier4.parameters(), 'lr': 0.01},
             {'params': model.classifier5.parameters(), 'lr': 0.01},
             #{'params': model.classifier6.parameters(), 'lr': 0.01},
             #{'params': model.classifier7.parameters(), 'lr': 0.01}
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)

exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[40,60], gamma=0.1)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU. 
#

model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=Epochs)

