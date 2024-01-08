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
import tripletfolder_market
import tripletfolder_duke
import tripletfolder_msmt
import json
from shutil import copyfile
from scipy import spatial as spatial
import random
from scipy.spatial import distance
from functools import partial
from torch.utils.data import DataLoader

version =  torch.__version__

def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1 # count the image number in every class
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

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
parser.add_argument('--Epochs',default=100,type=int, help='maximum epochs for SFP')
parser.add_argument('--save_epoch',default=100,type=int, help='start epoch to save model for SFP')
parser.add_argument('--network_arch',default='resnet50',type=str, help='network for the training and pruning')
parser.add_argument('--prune_rate',default=0.1,type=float, help='prune rate for FPGM')

opt = parser.parse_args()

data_dir = opt.data_dir
name = opt.name
Epochs = opt.Epochs
start_save_epoch = opt.save_epoch


prune_arch = ['resnet']
prune_skip_downsample = 1
resnet_arch = 'resnet50'
layer_begin =0
layer_end = 156
layer_inter = 1
saving_dir = './results/ft_resnet50'

prune_rate = 0
dis_rate = opt.prune_rate

if 'resnet50' in opt.network_arch:
    resnet_arch = 'resnet50'
    layer_begin =0
    layer_end = 156
    layer_inter = 1
    saving_dir = './results/ft_resnet50'
elif 'resnet18' in opt.network_arch:
    resnet_arch = 'resnet18'
    layer_begin =0
    layer_end = 57
    layer_inter = 1
    saving_dir = './results/ft_resnet18'
elif 'resnet34' in opt.network_arch:
    resnet_arch = 'resnet34'
    layer_begin =0
    layer_end = 105
    layer_inter = 1
    saving_dir = './results/ft_resnet34'
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
#print(gpu_ids[0])

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
######################################################################
# Load Data
# ---------
#

transform_train_list = [
        #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((256,128), interpolation=3),
        #transforms.RandomCrop((256,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

transform_val_list = [
        transforms.Resize(size=(256,128),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

if opt.PCB:
    transform_train_list = [
        transforms.Resize((384,192), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    transform_val_list = [
        transforms.Resize(size=(384,192),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

if opt.erasing_p>0:
    transform_train_list = transform_train_list +  [RandomErasing(probability = opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose( transform_train_list ),
    'val': transforms.Compose(transform_val_list),
}


train_all = ''
if opt.train_all:
     train_all = '_all'

triplet_folder_name = ''
image_datasets = {}
if 'Market' in opt.data_dir:
    image_datasets['train'] = tripletfolder_market.TripletFolder(os.path.join(data_dir, 'train_all'),
                                            data_transforms['train'])
    image_datasets['val'] = tripletfolder_market.TripletFolder(os.path.join(data_dir, 'val'),
                                          data_transforms['val'])
    triplet_folder_name += 'tripletfolder_market.py'
elif 'Duke' in opt.data_dir:
    image_datasets['train'] = tripletfolder_duke.TripletFolder(os.path.join(data_dir, 'train_all'),
                                                                 data_transforms['train'])
    image_datasets['val'] = tripletfolder_duke.TripletFolder(os.path.join(data_dir, 'val'),
                                                             data_transforms['val'])
    triplet_folder_name += 'tripletfolder_duke.py'
elif 'MSMT' in opt.data_dir:
    image_datasets['train'] = tripletfolder_msmt.TripletFolder(os.path.join(data_dir, 'train_all'),
                                                                 data_transforms['train'])
    image_datasets['val'] = tripletfolder_msmt.TripletFolder(os.path.join(data_dir, 'val'),
                                                               data_transforms['val'])
    triplet_folder_name += 'tripletfolder_msmt.py'

weights = make_weights_for_balanced_classes(image_datasets['train'].imgs, len(image_datasets['train'].classes))
weights = torch.DoubleTensor(weights)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
CustomDataLoader = partial(
    DataLoader,
    num_workers=1,
    batch_size=opt.batchsize,
    sampler = sampler,
    pin_memory=True,
    drop_last=True)

batch = {}

class_names = image_datasets['train'].classes
class_vector = [s[1] for s in image_datasets['train'].samples]
#dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize, 
#                                             shuffle=True, num_workers=8)
#              for x in ['train', 'val']}

dataloaders = {x: CustomDataLoader(image_datasets[x], batch_size=opt.batchsize, 
                                             shuffle=False, num_workers=8)
              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

use_gpu = torch.cuda.is_available()

since = time.time()
#inputs, classes, pos, pos_classes = next(iter(dataloaders['train']))
print(time.time()-since)


######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []



def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    last_margin = 0.0


    m = Mask(model)
    m.init_length()
    print("-" * 10 + "one epoch begin" + "-" * 10)
    print("the compression rate now is {:}".format(prune_rate))

    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        model,best_model_wts,last_margin = train_epoch(model,criterion,scheduler,optimizer,epoch,best_model_wts, last_margin,'train')
        # load best model weights
        model.load_state_dict(best_model_wts)
        if epoch < num_epochs -1:
            m.model = model
            m.if_zero()
            m.init_mask(prune_rate,dis_rate,decay_rate = 0)
            m.do_similar_mask()
            m.if_zero()
            model = m.model
            model = model.cuda()
        else:
            m.model = model
            m.if_zero()
            m.init_mask(prune_rate,dis_rate,decay_rate = 0)
            m.do_similar_mask()
            m.if_zero()
            model = m.model
            model = model.cuda()
        print('-'*10 +'after pruning' + '-'*10)  
        #if epoch%2 == 1:
        #    save_network(model, epoch)
       
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))

    
    save_network(model, 'last')
    return model

def train_epoch(model,criterion,scheduler,optimizer,epoch,last_model_wts,last_margin,phase):
    if phase == 'train':
        scheduler.step()
        model.train(True)  # Set model to training mode
    else:
        model.train(False)  # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0.0
    running_margin = 0.0
    running_reg = 0.0
    # Iterate over data.
    for data in dataloaders[phase]:
        # get the inputs
        inputs, labels, pos, pos_labels = data
        now_batch_size,c,h,w = inputs.shape

        if now_batch_size<opt.batchsize: # next epoch
            continue
        pos = pos.view(4*opt.batchsize,c,h,w)
        #copy pos 4times
        pos_labels = pos_labels.repeat(4).reshape(4,opt.batchsize)
        pos_labels = pos_labels.transpose(0,1).reshape(4*opt.batchsize)

        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
            pos = Variable(pos.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        #model_eval = copy.deepcopy(model)
        #model_eval = model_eval.eval()
        outputs, f = model(inputs)
        _, pf = model(pos)
        #pf = Variable( pf, requires_grad=True)
        neg_labels = pos_labels
        # hard-neg
        # ----------------------------------
        nf_data = pf # 128*512
        # 128 is too much, we use pool size = 64
        rand = np.random.permutation(4*opt.batchsize)[0:opt.poolsize]
        nf_data = nf_data[rand,:]
        neg_labels = neg_labels[rand]
        nf_t = nf_data.transpose(0,1) # 512*128
        score = torch.mm(f.data, nf_t) # cosine 32*128 
        score, rank = score.sort(dim=1, descending = True) # score high == hard
        labels_cpu = labels.cpu()
        nf_hard = torch.zeros(f.shape).cuda()
        for k in range(now_batch_size):
            hard = rank[k,:]
            for kk in hard:
                now_label = neg_labels[kk] 
                anchor_label = labels_cpu[k]
                if now_label != anchor_label:
                    nf_hard[k,:] = nf_data[kk,:]
                    break

        # hard-pos
        # ----------------------------------
        pf_hard = torch.zeros(f.shape).cuda() # 32*512
        for k in range(now_batch_size):
            pf_data = pf[4*k:4*k+4,:]
            pf_t = pf_data.transpose(0,1) # 512*4
            ff = f.data[k,:].reshape(1,-1) # 1*512
            score = torch.mm(ff, pf_t) #cosine
            score, rank = score.sort(dim=1, descending = False) #score low == hard
            pf_hard[k,:] = pf_data[rank[0][0],:]

        # loss
        # ---------------------------------
        criterion_triplet = nn.MarginRankingLoss(margin=opt.margin)                
        pscore = torch.sum( f * pf_hard, dim=1) 
        nscore = torch.sum( f * nf_hard, dim=1)
        y = torch.ones(now_batch_size)
        y = Variable(y.cuda())

        if not opt.PCB:
            _, preds = torch.max(outputs.data, 1)
            #loss = criterion(outputs, labels)
            #loss_triplet = criterion_triplet(f, pf, nf)
            reg = torch.sum((1+nscore)**2) + torch.sum((-1+pscore)**2)
            loss = torch.sum(torch.nn.functional.relu(nscore + opt.margin - pscore))  #Here I use sum
            loss_triplet = loss + opt.alpha*reg
        else:
            part = {}
            sm = nn.Softmax(dim=1)
            num_part = 6
            for i in range(num_part):
                part[i] = outputs[i]

            score = sm(part[0]) + sm(part[1]) +sm(part[2]) + sm(part[3]) +sm(part[4]) +sm(part[5])
            _, preds = torch.max(score.data, 1)

            loss = criterion(part[0], labels)
            for i in range(num_part-1):
                loss += criterion(part[i+1], labels)

        # backward + optimize only if in training phase
        if phase == 'train':
            loss_triplet.backward()
            optimizer.step()
        # statistics
        if int(version[2]) > 3: # for the new version like 0.4.0 and 0.5.0
            running_loss += loss_triplet.item() #* opt.batchsize
        else :  # for the old version like 0.3.0 and 0.3.1
            running_loss += loss_triplet.item() #*opt.batchsize
        #print( loss_triplet.item())
        running_corrects += float(torch.sum(pscore>nscore+opt.margin))
        running_margin +=float(torch.sum(pscore-nscore))
        running_reg += reg

    datasize = dataset_sizes['train']//opt.batchsize * opt.batchsize
    epoch_loss = running_loss / datasize
    epoch_reg = opt.alpha*running_reg/ datasize
    epoch_acc = running_corrects / datasize
    epoch_margin = running_margin / datasize

    #if epoch_acc>0.75:
    #    opt.margin = min(opt.margin+0.02, 1.0)
    print('now_margin: %.4f'%opt.margin)           
    print('{} Loss: {:.4f} Reg: {:.4f} Acc: {:.4f} MeanMargin: {:.4f}'.format(
        phase, epoch_loss, epoch_reg, epoch_acc, epoch_margin))

    y_loss[phase].append(epoch_loss)
    y_err[phase].append(1.0-epoch_acc)
    # deep copy the model
    if epoch_margin>last_margin:
        last_margin = epoch_margin            
        last_model_wts = model.state_dict()

    #if epoch%10 == 9:
    #    save_network(model, epoch)
    draw_curve(epoch)
    
    return model,last_model_wts,last_margin


class Mask:
    def __init__(self, model):
        self.model_size = {}
        self.model_length = {}
        self.compress_rate = {}
        self.distance_rate = {}
        self.mat = {}
        self.model = model
        self.mask_index = []
        self.filter_small_index = {}
        self.filter_large_index = {}
        self.similar_matrix = {}
        self.decay_rate = 0
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
    
    def get_filter_similar(self, weight_torch, compress_rate, distance_rate, length):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (compress_rate))#pruned rate
            similar_pruned_num = int(weight_torch.size()[0] * distance_rate)
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            # norm1 = torch.norm(weight_vec, 1, 1)
            # norm1_np = norm1.cpu().numpy()
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_small_index = []
            filter_large_index = []
            filter_large_index = norm2_np.argsort()[filter_pruned_num:]
            filter_small_index = norm2_np.argsort()[:filter_pruned_num]

            # # distance using pytorch function
            # similar_matrix = torch.zeros((len(filter_large_index), len(filter_large_index)))
            # for x1, x2 in enumerate(filter_large_index):
            #     for y1, y2 in enumerate(filter_large_index):
            #         # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            #         # similar_matrix[x1, y1] = cos(weight_vec[x2].view(1, -1), weight_vec[y2].view(1, -1))[0]
            #         pdist = torch.nn.PairwiseDistance(p=2)
            #         similar_matrix[x1, y1] = pdist(weight_vec[x2].view(1, -1), weight_vec[y2].view(1, -1))[0][0]
            # # more similar with other filter indicates large in the sum of row
            # similar_sum = torch.sum(torch.abs(similar_matrix), 0).numpy()

            # distance using numpy function
            indices = torch.LongTensor(filter_large_index).cuda()
            weight_vec_after_norm = torch.index_select(weight_vec, 0, indices).cpu().numpy()
            # for euclidean distance
            similar_matrix = distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'euclidean')
            # for cos similarity
            # similar_matrix = 1 - distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'cosine')
            similar_sum = np.sum(np.abs(similar_matrix), axis=0)

            # for distance similar: get the filter index with largest similarity == small distance
            similar_large_index = similar_sum.argsort()[similar_pruned_num:]
            similar_small_index = similar_sum.argsort()[:  similar_pruned_num]
            similar_index_for_filter = [filter_large_index[i] for i in similar_small_index]

            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(similar_index_for_filter)):
                codebook[
                similar_index_for_filter[x] * kernel_length: (similar_index_for_filter[x] + 1) * kernel_length] = 0
            print("similar index done")
        else:
            pass
        return codebook

    def get_filter_codebook(self, weight_torch, compress_rate, length):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
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

    def init_rate(self, rate_norm_per_layer, rate_dist_per_layer):
        if 'vgg' in prune_arch:
            cfg_official = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
            cfg_CP_5x = [24, 22, 41, 51, 108, 89, 111, 184, 276, 228, 512, 512, 512]
            # cfg = [32, 64, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256]
            cfg_Thinet_conv = [32, 32, 64, 64, 128, 128, 128, 256, 256, 256, 512, 512, 512]
            '''
            if args.VGG_pruned_style == "CP_5x":
                cfg_now = cfg_CP_5x
            elif args.VGG_pruned_style == "Thinet_conv":
                cfg_now = cfg_Thinet_conv
            '''
            cfg_index = 0
            previous_cfg = True
            for index, item in enumerate(self.model.named_parameters()):
                self.compress_rate[index] = 1
                if len(item[1].size()) == 4:
                    if not previous_cfg:
                        self.compress_rate[index] = rate_norm_per_layer
                        self.distance_rate[index] = rate_dist_per_layer
                        self.mask_index.append(index)
                        print(item[0], "self.mask_index", self.mask_index)
                    else:
                        self.compress_rate[index] = 1
                        self.distance_rate[index] = 1 - cfg_now[cfg_index] / item[1].size()[0]
                        self.mask_index.append(index)
                        print(item[0], "self.mask_index", self.mask_index, cfg_index, cfg_now[cfg_index])
                        cfg_index += 1
                        
        elif "resnet" in prune_arch:
            for index, item in enumerate(self.model.parameters()):
                self.compress_rate[index] = 1
                self.distance_rate[index] = 1
            for key in range(layer_begin, layer_end + 1, layer_inter):
                self.compress_rate[key] = rate_norm_per_layer
                self.distance_rate[key] = rate_dist_per_layer
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
                    
                    self.distance_rate[x] = 1
                    
                    #remove corresponding bn from list
                    self.compress_rate[x+1] = 1
                    self.compress_rate[x+2] = 1

                    self.mask_index.remove(x+1)
                    self.mask_index.remove(x+2)
                    
                    self.distance_rate[x+1] = 1
                    self.distance_rate[x+2] = 1
                    #print(self.mask_index)
            else:
                pass

    def init_mask(self, rate_norm_per_layer, rate_dist_per_layer,decay_rate = 0.2):
        self.init_rate(rate_norm_per_layer, rate_dist_per_layer)
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
                
                # mask for distance criterion
                self.similar_matrix[index] = self.get_filter_similar(item.data, self.compress_rate[index],
                                                                     self.distance_rate[index],
                                                                     self.model_length[index])
                self.similar_matrix[index] = self.convert2tensor(self.similar_matrix[index])
                self.similar_matrix[index] = self.similar_matrix[index].cuda()
                
                
                #set bn.weight and bias 
                codebook = np.zeros(item.data.size()[0])
                tmp = self.similar_matrix[index].view(item.data.size()[0],-1).sum(axis=1)
                nonzero_idx = np.nonzero(tmp.cpu().numpy())[0].tolist()
                #print(nonzero_idx)
                codebook[nonzero_idx] = 1
                
                self.similar_matrix[index+1] = codebook
                self.similar_matrix[index+2] = codebook
                
                self.similar_matrix[index+1] = self.convert2tensor(self.similar_matrix[index+1])
                self.similar_matrix[index+2] = self.convert2tensor(self.similar_matrix[index+2])
                
                self.similar_matrix[index+1] = self.similar_matrix[index+1].cuda()
                self.similar_matrix[index+2] = self.similar_matrix[index+2].cuda()
                
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
        
    def do_similar_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.data.view(self.model_length[index])
                b = a * self.similar_matrix[index]
                item.data = b.view(self.model_size[index])
        print("mask similar Done")

    def if_zero(self):
        for index, item in enumerate(self.model.parameters()):
            #            if(index in self.mask_index):
            if index in [x for x in range(layer_begin, layer_end + 1, layer_inter)]:
                a = item.data.view(self.model_length[index])
                b = a.cpu().numpy()

                print("layer: %d, number of nonzero weight is %d, zero is %d" % (
                    index, np.count_nonzero(b), len(b) - np.count_nonzero(b)))



######################################################################
# Draw Curve
# ---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="triplet_loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
#    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
#    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig( os.path.join(saving_dir,name,'train.jpg'))

######################################################################
# Save model
# ---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join(saving_dir,name,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda(gpu_ids[0])


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#

print('+'*30,len(class_names))
 

if opt.use_dense:
    model = ft_net_dense(len(class_names))
else:
    model = ft_net(len(class_names),opt.network_arch)

if opt.PCB:
    model = PCB(len(class_names))

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
dir_name = os.path.join(saving_dir,name)
if not os.path.isdir(dir_name):
    os.makedirs(dir_name)
copyfile('./FPGM_train.py', dir_name+'/FPGM_train.py')
copyfile('./model.py', dir_name+'/model.py')
copyfile('./'+triplet_folder_name, dir_name+'/'+triplet_folder_name)

# save opts
with open('%s/opts.json'%dir_name,'w') as fp:
    json.dump(vars(opt), fp, indent=1)

model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=Epochs)
