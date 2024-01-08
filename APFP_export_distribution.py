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
from tripletfolder_market import TripletFolder
import json
from shutil import copyfile
from scipy import spatial as spatial
import random

from scipy import spatial as spatial
from sklearn.neighbors import NearestNeighbors
from heapq import nsmallest
from operator import itemgetter

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
parser.add_argument('--knn',default=2,type=int, help='k for PFP')
parser.add_argument('--prune_rate',default=0.9,type=float, help='prune rate for PFP')
parser.add_argument('--max_lrate',default=0.9,type=float, help='max layer prune rate for APFP')
parser.add_argument('--decay_rate',default=1e-2,type=float, help='decay rate for PFP')
parser.add_argument('--Epochs',default=100,type=int, help='maximum epochs for PFP')
parser.add_argument('--save_epoch',default=100,type=int, help='start epoch to save model for PFP')
parser.add_argument('--network_arch',default='resnet50',type=str, help='network for the training and pruning')
parser.add_argument('--efficient_pruning', action='store_true', help='pruning in an efficient way' )

opt = parser.parse_args()

pretrained_dir = opt.pretrained_dir
knn_k = opt.knn
prune_rate = opt.prune_rate
decay_rate = opt.decay_rate
Epochs = opt.Epochs
start_save_epoch = opt.save_epoch
opt_max_rate = opt.max_lrate

prune_arch = ['resnet']
prune_skip_downsample = 1
resnet_arch = 'resnet50'
layer_begin =0
layer_end = 156
layer_inter = 1
saving_dir = '../results/ft_resnet50'

rank_with_kernel_1x1 = False #ranking without 1x1 conv kernel, for resnet18, set it true to average ranking in same layers

if 'resnet50' in opt.network_arch:
    resnet_arch = 'resnet50'
    layer_begin =0
    layer_end = 156
    layer_inter = 1
    rank_with_kernel_1x1 = False
    saving_dir = '../results/ft_resnet50'
elif 'resnet18' in opt.network_arch:
    resnet_arch = 'resnet18'
    layer_begin =0
    layer_end = 57
    layer_inter = 1
    rank_with_kernel_1x1 = True
    saving_dir = '../results/ft_resnet18'
elif 'resnet34' in opt.network_arch:
    resnet_arch = 'resnet34'
    layer_begin =0
    layer_end = 105
    layer_inter = 1
    rank_with_kernel_1x1 = True
    saving_dir = '../results/ft_resnet34'
else:
    pass


print('knn={},prune_rate={},decay_rate={},pretrained_dir={},Epochs={},Arch={}'.format(knn_k,prune_rate,decay_rate,pretrained_dir,Epochs,opt.network_arch))
print('start_save_epoch=',start_save_epoch)
data_dir = opt.data_dir
name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
#print(gpu_ids[0])

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

train_all = ''
if opt.train_all:
     train_all = '_all'

batch = {}

use_gpu = torch.cuda.is_available()

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    global prune_rate
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    last_margin = 0.0

    for i, (name, param) in enumerate(model.named_parameters()):
        if 'layer2.0.conv2.weight'  in name:
            np.save('APFP_before_pruning_layer2.0conv2_0.5_'+opt.network_arch,param.cpu().detach().numpy())
            
    m = Mask(model)
    m.init_length()

    auto_prune_rate = m.get_geometry_global_ranking(prune_rate,k=knn_k,rank_with_kernel_1x1 = rank_with_kernel_1x1,max_rate=opt_max_rate)
    print("Auto prune rate is {}".format(auto_prune_rate))
    prune_rate = -1 #indicate the non_structured pruning
    
    for epoch in range(1):
        model.load_state_dict(best_model_wts)
        old_model = copy.deepcopy(model)
        m.model = old_model
        m.if_zero()
        m.init_mask(prune_rate,decay_rate = 0,k = knn_k)
        m.do_mask()
        m.if_zero()
        model = m.model
            
    for i, (name, param) in enumerate(model.named_parameters()):
            if 'layer2.0.conv2.weight'  in name:
                np.save('APFP_after_pruning_layer2.0conv2_0.5_'+opt.network_arch,param.cpu().detach().numpy())

    return model


class Mask:
    def __init__(self, model):
        self.model_size = {}
        self.model_length = {}
        self.compress_rate = {}
        self.max_removed_filters = {}
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
        
    def get_group_compress_rate(self,s_dict, compress_rate,mask_index,rank_with_kernel_1x1):
        
        g_idx = -1
        idx = 0
        g_dict = {}
        init_key = '$'
        g_dict[0] = 0
        compress_rate_group = {}
        #notice: here the idx is used to synchronize the index in resnet.parameters()
        for key,value in s_dict.items():
            if(value.numel() > 0) and 'running' not in key and 'tracked' not in key: #we only calculate conv
                if len(value.size()) == 4 and idx > 0: #skip the first conv1
                    if init_key not in key:
                        init_key = key[:14] #used to determine if there is a new block coming,should be adjusted accordning to different networks
                        g_idx = g_idx + 1
                        g_dict[g_idx] = []
                        
                        compress_rate_group[g_idx] = compress_rate[idx]
                    else:
                        compress_rate_group[g_idx] = compress_rate_group[g_idx] + compress_rate[idx] #accumulate the rate in same group

                    if idx in mask_index: #jump the skipped layer
                        g_dict[g_idx].append(idx)
                                            
                idx  = idx +1   
        
        for key in g_dict.keys():
            for j in range(len(g_dict[key])):
                if rank_with_kernel_1x1:
                    compress_rate[g_dict[key][j]] = compress_rate_group[key]/len(g_dict[key]) #update compress rate in same group
                else:
                    compress_rate[g_dict[key][j]] = compress_rate_group[key] #only the 3x3 one's compress_rate is not zero
        
        return compress_rate
        
    def get_group_compress_rate_by_mask(self,s_dict, compress_rate,mask_index,group_mask):
        
        compress_rate_group= dict.fromkeys(group_mask.keys(), 0) 
        for key in group_mask.keys():
            for j in range(len(group_mask[key])):
                compress_rate_group[key]  = compress_rate_group[key] + compress_rate[group_mask[key][j]] #accumulate the rate in same group
        
        for key in group_mask.keys():
            for j in range(len(group_mask[key])):        
                compress_rate[group_mask[key][j]] = compress_rate_group[key]/len(group_mask[key]) #update compress rate in same group
        
        return compress_rate

        
    def get_geometry_global_ranking(self,compress_rate,k,rank_with_kernel_1x1,max_rate=0.9):
        #be compatible with old version
        self.init_rate(compress_rate,max_rate)
        global_geometry_ranking = []
        total_filters = 0
        for index, item in enumerate(self.model.parameters()):
            if not rank_with_kernel_1x1: #only rank with 3x3 kernel or others
                temp_data = item.data.squeeze()
            else:
                temp_data = item.data
                
            if (index in self.mask_index) and len(temp_data.size()) == 4: #conv:
                v = self.get_local_channel_ranking_iterative(temp_data,index,k)
                #print(index,v[:50])
                #store them in [(layer,channel_index,normal_index),...]
                for i in range(len(v)):
                    global_geometry_ranking.append((index,i,v[i]))
                
                total_filters = total_filters + len(v)
                #print('total_filters:',total_filters,'len(v):',len(v))
        #ranking
        #sorted_filters = nsmallest(int(total_filters*compress_rate), global_geometry_ranking, itemgetter(2)) 
        sorted_filters = nsmallest(total_filters, global_geometry_ranking, itemgetter(2)) 
        #init compress_rate for each layer
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index):
                self.compress_rate[index] = 0
        
        
        current_pruned = 0
        #calculate pruning rate of each layer
        for i in range(len(sorted_filters)):
            if current_pruned >= int(total_filters*compress_rate):
                break;
            if self.compress_rate[sorted_filters[i][0]] < self.max_removed_filters[sorted_filters[i][0]]:
                self.compress_rate[sorted_filters[i][0]] =  self.compress_rate[sorted_filters[i][0]] + 1
                current_pruned = current_pruned + 1
        
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index) and len(item.data.size()) == 4: #conv::
                self.compress_rate[index] = self.compress_rate[index]/item.size()[0] #its ok if not rank_with_kernel_1x1:
                
        '''        
        #self.compress_rate = self.get_group_compress_rate(self.model.state_dict(),self.compress_rate,self.mask_index,rank_with_kernel_1x1)
        if 'resnet18' in opt.network_arch:
            #filling goup_mask, we only calculating 3x3 convolutionals
            #resnet18 group mask
            group_mask = {0:[0,3,6,9,12],1:[15,18,24,27],2:[30,33,39,42],3:[45,48,54,57]}
            #for resnet18, resnet34 only
            self.compress_rate = self.get_group_compress_rate_by_mask(self.model.state_dict(),self.compress_rate,self.mask_index,group_mask)    
        elif 'resnet34' in opt.network_arch:
            #filling goup_mask, we only calculating 3x3 convolutionals
            #resnet34 group mask
            group_mask = {0:[0,3,6,9,12,15,18],1:[21,24,30,33,36,39,42,45],2:[48,51,57,60,63,66,69,72,75,78,81,84],3:[87,90,96,99,102,105]}
            #for resnet18, resnet34 only
            self.compress_rate = self.get_group_compress_rate_by_mask(self.model.state_dict(),self.compress_rate,self.mask_index,group_mask)
        else:
        '''
        self.compress_rate = self.get_group_compress_rate(self.model.state_dict(),self.compress_rate,self.mask_index,rank_with_kernel_1x1)
            
        return  self.compress_rate       
        
    def get_local_channel_ranking(self,weight_torch,layer_index,k):
        
        normalized_local_ranking = []
        if len(weight_torch.size()) == 4:
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            
            X=weight_vec.cpu()

            #for indexing knn, after deleting points
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X)
            nb_dst, nb_idx = nbrs.kneighbors(X)

            #search points with smallest local power i and j
            nb_dst_sumK = nb_dst.sum(axis=1)

            #normalize
            v = np.abs(nb_dst_sumK)
            v = v / np.sqrt(np.sum(v * v))
            
            normalized_local_ranking = v
        else:
            pass
        
        return normalized_local_ranking

    def get_local_channel_ranking_iterative(self,weight_torch,layer_index,k):
        normalized_local_ranking = []
        if len(weight_torch.size()) == 4:
            
            filter_pruned_num = weight_torch.size()[0]
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            filter_index = []
                        
            similarities = spatial.distance.pdist(weight_vec.cpu())#similar vector
            similarities_matrix = spatial.distance.squareform(similarities) 
            small_sim_matrix = similarities_matrix.copy()

            #for indexing knn, after deleting points
            ori_index = np.array(list(range(len(similarities_matrix))))
            pruned_list = []
            for i in range(filter_pruned_num):
                if len(small_sim_matrix) > k:
                    #find k smallest neighbors for 
                    nb_idx = []
                    nb_dst = []
                    #notice the deleted filters + k < n
                    for j in range(len(small_sim_matrix)):
                        #print(np.argpartition(small_sim_matrix[j], k))
                        nb_idx.append( np.argpartition(small_sim_matrix[j], k)[:k] )
                        nb_dst.append( small_sim_matrix[j][nb_idx[j]])
                    
                    #4.2 search points with smallest local power i and j
                    nb_dst_sumK = np.array(nb_dst).sum(axis=1)
                    fi = np.argmin(nb_dst_sumK)
                    #print(nb_dst.sum(axis=1))
                    min_edge_list = list(np.where(np.isclose(nb_dst_sumK, nb_dst_sumK[fi],1e-5, 1e-8))[0])
                    #print(fi,min_edge_list, nb_idx[fi][1])
                    
                    global_power_list = []
                    for j in range(len(min_edge_list)):
                        idx_ori = ori_index[min_edge_list[j]]
                        global_power_list.append( np.average(similarities_matrix[idx_ori,:]))
                    
                    #print(min_edge_list[np.argmin(global_power_list)],global_power_list)

                    delete_idx = min_edge_list[np.argmin(global_power_list)]
                    
                    #store the dst for global ranking
                    normalized_local_ranking.append(nb_dst_sumK[delete_idx])
                    
                    delete_idx_ori = ori_index[delete_idx]
                    pruned_list.append(delete_idx_ori)
                    
                    similarities_matrix[delete_idx_ori,:] = 0
                    similarities_matrix[:,delete_idx_ori] = 0

                    #delete points
                    small_sim_matrix = np.delete(small_sim_matrix,delete_idx,0)
                    small_sim_matrix = np.delete(small_sim_matrix,delete_idx,1)
                    #adjust indexs
                    ori_index[delete_idx:len(ori_index)-1] = ori_index[delete_idx+1:len(ori_index)]
                else:
                    sum_vec_similarity = np.sum(small_sim_matrix,axis=0)
                    for j in range(len(sum_vec_similarity)):
                        #pruned_list.append(ori_index[sorted_similarity[i]])
                        normalized_local_ranking.append(sum_vec_similarity[j])
                    break
            #normalize
            v = np.abs(normalized_local_ranking)
            v = v / np.sqrt(np.sum(v * v))
            
            normalized_local_ranking = v
        else:
            pass
        
        return normalized_local_ranking
                

    def get_filter_codebook(self, weight_torch, compress_rate, length,k=2):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            filter_index = []
      
            similarities = spatial.distance.pdist(weight_vec.cpu(),'seuclidean')#similar vector
            #similarities = spatial.distance.pdist(weight_vec.cpu(),'cosine')#similar vector
            similarities_matrix = spatial.distance.squareform(similarities)
            #if k < weight_torch.size()[0] - filter_pruned_num, perform iterative prune
            if k < (weight_torch.size()[0] - filter_pruned_num):
                
                #pruning rate is larger than 0.5, deleting large local power filters first
                if compress_rate > 0.5 and opt.efficient_pruning:
                    small_sim_matrix = similarities_matrix.copy()
                    
                    filter_keeped_num = weight_torch.size()[0] - filter_pruned_num
                    #for indexing knn, after deleting points
                    ori_index = np.array(list(range(len(similarities_matrix))))
                    pruned_list = []
                    for i in range(filter_keeped_num):
                        #find k smallest neighbors for 
                        nb_idx = []
                        nb_dst = []
                        #notice the deleted filters + k < n
                        for j in range(len(small_sim_matrix)):
                            nb_idx.append( np.argpartition(small_sim_matrix[j], k)[:k] )
                            nb_dst.append( small_sim_matrix[j][nb_idx[j]])

                        #4.2 search points with largest local power i and j
                        nb_dst_sumK = np.array(nb_dst).sum(axis=1)
                        fi = np.argmax(nb_dst_sumK) #deleting large local power
                        #print(nb_dst.sum(axis=1))
                        min_edge_list = list(np.where(np.isclose(nb_dst_sumK, nb_dst_sumK[fi],1e-5, 1e-8))[0])
                        #print(fi,min_edge_list, nb_idx[fi][1])

                        global_power_list = []
                        for i in range(len(min_edge_list)):
                            idx_ori = ori_index[min_edge_list[i]]
                            global_power_list.append( np.average(similarities_matrix[idx_ori,:]))

                        #print(min_edge_list[np.argmin(global_power_list)],global_power_list)

                        delete_idx = min_edge_list[np.argmax(global_power_list)] #deleting large local power
                        delete_idx_ori = ori_index[delete_idx]
                        pruned_list.append(delete_idx_ori)

                        similarities_matrix[delete_idx_ori,:] = 0
                        similarities_matrix[:,delete_idx_ori] = 0

                        #delete points
                        small_sim_matrix = np.delete(small_sim_matrix,delete_idx,0)
                        small_sim_matrix = np.delete(small_sim_matrix,delete_idx,1)
                        #adjust indexs
                        ori_index[delete_idx:len(ori_index)-1] = ori_index[delete_idx+1:len(ori_index)]

                    filter_index = list(set([i for i in range(weight_torch.size()[0])]) - set(pruned_list))
                #pruning rate is smaller than 0.5, deleting small local power filters first
                else:
                    small_sim_matrix = similarities_matrix.copy()
                    #for indexing knn, after deleting points
                    ori_index = np.array(list(range(len(similarities_matrix))))
                    pruned_list = []
                    for i in range(filter_pruned_num):
                        #find k smallest neighbors for 
                        nb_idx = []
                        nb_dst = []
                        #notice the deleted filters + k < n
                        for j in range(len(small_sim_matrix)):
                            nb_idx.append( np.argpartition(small_sim_matrix[j], k)[:k] )
                            nb_dst.append( small_sim_matrix[j][nb_idx[j]])

                        #4.2 search points with smallest local power i and j
                        nb_dst_sumK = np.array(nb_dst).sum(axis=1)
                        fi = np.argmin(nb_dst_sumK)
                        #print(nb_dst.sum(axis=1))
                        min_edge_list = list(np.where(np.isclose(nb_dst_sumK, nb_dst_sumK[fi],1e-5, 1e-8))[0])
                        #print(fi,min_edge_list, nb_idx[fi][1])

                        global_power_list = []
                        for i in range(len(min_edge_list)):
                            idx_ori = ori_index[min_edge_list[i]]
                            global_power_list.append( np.average(similarities_matrix[idx_ori,:]))

                        #print(min_edge_list[np.argmin(global_power_list)],global_power_list)

                        delete_idx = min_edge_list[np.argmin(global_power_list)]
                        delete_idx_ori = ori_index[delete_idx]
                        pruned_list.append(delete_idx_ori)

                        similarities_matrix[delete_idx_ori,:] = 0
                        similarities_matrix[:,delete_idx_ori] = 0

                        #delete points
                        small_sim_matrix = np.delete(small_sim_matrix,delete_idx,0)
                        small_sim_matrix = np.delete(small_sim_matrix,delete_idx,1)
                        #adjust indexs
                        ori_index[delete_idx:len(ori_index)-1] = ori_index[delete_idx+1:len(ori_index)]

                        filter_index = pruned_list
           
            else:#global similarity
                print('global prune')
                sum_vec_similairty = np.argsort(np.sum(similarities_matrix,axis=0))
                pruned_list = sum_vec_similairty[:filter_pruned_num]
                print(pruned_list)
                filter_index = pruned_list
           
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(filter_index)):
                codebook[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = self.decay_rate
                
            print("filter codebook done")
        else:
            pass
        return codebook,filter_index

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

    def init_rate(self, layer_rate, max_rate= 0.9):
        if 'vgg' in prune_arch:
            cfg_5x = [24, 22, 41, 51, 108, 89, 111, 184, 276, 228, 512, 512, 512]
            cfg_official = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
            # cfg = [32, 64, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256]
            cfg_index = 0
            pre_cfg = True
            for index, item in enumerate(self.model.named_parameters()):
                #self.compress_rate[index] = 1
                if len(item[1].size()) == 4:
                    print(item[1].size())
                    if not pre_cfg:
                        if layer_rate > 0:
                            self.compress_rate[index] = layer_rate
                        self.max_removed_filters[index] = int(max_rate*item[1].size()[0])
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
                #self.compress_rate[index] = 1
                self.max_removed_filters[index] = int(max_rate*item.data.size()[0])
            if layer_rate > 0:    
                for key in range(layer_begin, layer_end + 1, layer_inter):
                    self.compress_rate[key] = layer_rate
                
            if resnet_arch== 'resnet18':
                # last index include last fc layer
                last_index = 60
                skip_list = [0,21, 36, 51] #the first block didn't have downsample, so we skip the first conv
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
                    self.compress_rate[x] = 0
                    self.mask_index.remove(x)
                    #print(self.mask_index)
                    
                    #remove corresponding bn from list
                    self.compress_rate[x+1] = 0
                    self.compress_rate[x+2] = 0
                    self.mask_index.remove(x+1)
                    self.mask_index.remove(x+2)
                    #print(self.mask_index)
            else:
                pass

    def init_mask(self, layer_rate,decay_rate = 0.2,k = 2):
        self.init_rate(layer_rate)
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index) and len(item.data.size()) == 4: #conv:
                self.decay_rate = decay_rate
                self.mat[index],pruned_list = self.get_filter_codebook(item.data, self.compress_rate[index],
                                                           self.model_length[index],k)
                self.mat[index] = self.convert2tensor(self.mat[index])
                
                self.mat[index] = self.mat[index].cuda()
                
                #set bn.weight and bias 
                codebook = np.ones(item.data.size()[0])
               
                #print(nonzero_idx)
                codebook[pruned_list] = decay_rate
                
                self.mat[index+1] = codebook
                self.mat[index+2] = codebook
                
                self.mat[index+1] = self.convert2tensor(self.mat[index+1])
                self.mat[index+2] = self.convert2tensor(self.mat[index+2])
                
                self.mat[index+1] = self.mat[index+1].cuda()
                self.mat[index+2] = self.mat[index+2].cuda()
                
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

class_num = 751

model = ft_net(class_num,arch=opt.network_arch)

    
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
    print('*'*20,opt.lr)
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
             {'params': base_params, 'lr': lr},
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

model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=Epochs)
