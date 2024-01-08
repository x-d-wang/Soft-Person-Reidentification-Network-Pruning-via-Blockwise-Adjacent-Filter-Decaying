# -*- coding: utf-8 -*-

from __future__ import print_function, division
import datetime
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
from model import ft_net, ft_net_dense, PCB, PCB_test
from collections import OrderedDict
import models

from ptflops import get_model_complexity_info

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='../Market/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--PCB', action='store_true', help='use PCB' )
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--NonL1', action='store_false', help='test small model for L1_prune' )
parser.add_argument('--prune_rate',default=0.1,type=float, help='prune rate for PFP')
parser.add_argument('--pretrained_dir',default='',type=str, help='pretrained dir path')
parser.add_argument('--network_arch',default='resnet50',type=str, help='network for the training and pruning')

opt = parser.parse_args()

resume_dir = os.path.join(opt.name,'net_'+opt.which_epoch +'.pth')

network_arch = opt.network_arch

if 'resnet18' in network_arch or 'resnet34' in network_arch:
    dim_feature = 512
else:
    dim_feature = 2048

dim_out = 702
if 'Market' in opt.test_dir:
    dim_out = 751

#For L1norm prune
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

prune_rate = opt.prune_rate
load_L1 = opt.NonL1  
print('load_L1:',load_L1)

print('load_L1=',load_L1,'prune_rate=',prune_rate)

str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
data_transforms = transforms.Compose([
        transforms.Resize((256,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
############### Ten Crop        
        #transforms.TenCrop(224),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.ToTensor()(crop) 
          #      for crop in crops]
           # )),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
          #       for crop in crops]
          # ))
])

if opt.PCB:
    data_transforms = transforms.Compose([
        transforms.Resize((384,192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])


data_dir = test_dir
image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery','query']}

class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available()

######################################################################
# Load model
# ---------------------------
def load_network(network):
    save_path = os.path.join(name,'net_%s.pth'%opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network
######################################################################
# Get Small model
# ---------------------------
def check_channel(tensor):
    size_0 = tensor.size()[0]
    size_1 = tensor.size()[1] * tensor.size()[2] * tensor.size()[3]
    tensor_resize = tensor.view(size_0, -1)
    # indicator: if the channel contain all zeros
    channel_if_zero = np.zeros(size_0)
    for x in range(0, size_0, 1):
        channel_if_zero[x] = np.count_nonzero(tensor_resize[x].cpu().numpy()) != 0
    # indices = (torch.LongTensor(channel_if_zero) != 0 ).nonzero().view(-1)

    indices_nonzero = torch.LongTensor((channel_if_zero != 0).nonzero()[0])
    # indices_nonzero = torch.LongTensor((channel_if_zero != 0).nonzero()[0])

    zeros = (channel_if_zero == 0).nonzero()[0]
    #indices_zero = torch.LongTensor(zeros) if zeros != [] else []
    indices_zero = torch.LongTensor(zeros) if zeros.size >0 else []
    return indices_zero, indices_nonzero


def extract_para(big_model,dim_out):
    '''
    :param model:
    :param batch_size:
    :return: num_for_construc: number of remaining filter,
             [conv1,stage1,stage1_expend,stage2,stage2_expend,stage3,stage3_expend,stage4,stage4_expend]

             kept_filter_per_layer: number of remaining filters for every layer
             kept_index_per_layer: filter index of remaining channel
             model: small model
    '''
    state_dict = big_model.state_dict()
    new_state_dict = OrderedDict()
    #remove model. from the state_dict for ft_net
    for k, v in state_dict.items():
        name = k[6:]  # remove `model.`
        new_state_dict[name] = v
    state_dict = new_state_dict

    item = list(state_dict.items())

    kept_index_per_layer = {}
    kept_filter_per_layer = {}
    pruned_index_per_layer = {}
    constrct_flag = []
    conv_idx = 0
    
    for x in range(0, len(item) - 9, 6):#.......wxd
        
        indices_zero, indices_nonzero = check_channel(item[x][1])
        #print('{}:{}:{} '.format(conv_idx,item[x][0],indices_nonzero.size()))
        # indices_list.append(indices_nonzero)
        pruned_index_per_layer[item[x][0]] = indices_zero
        kept_index_per_layer[item[x][0]] = indices_nonzero
        kept_filter_per_layer[item[x][0]] = indices_nonzero.shape[0]
        conv_idx+=1


    if 'resnet50' in network_arch:
        blocks = [3, 4, 6, 3]
        block_flag = "conv3"
    elif 'resnet18' in network_arch:
        blocks = [2, 2, 2, 2]
        block_flag = "conv2"
    elif 'resnet34' in network_arch:
        blocks = [3, 4, 6, 3]
        block_flag = "conv2"
    else:
        pass
    
    # number of nonzero channel in conv1, and four stages
    num_for_construct = {}
    num_for_construct['00']= kept_filter_per_layer['conv1.weight']
    block_idx = 0
    for layer in range(1,5):
        for block_idx in range(blocks[layer-1]):
            conv_key = 'layer'+str(layer)+'.'+str(block_idx)+'.conv'
            for x in kept_filter_per_layer.keys():
                if conv_key in x:
                    num_for_construct[str(layer)+str(block_idx)+x[13]] = kept_filter_per_layer[x]#e.g.'101 for layer1.1.conv1.weight'
    #print('num_for_construct:',num_for_construct)
        
    index_for_construct = dict(
        (key, value) for (key, value) in kept_index_per_layer.items() if block_flag in key)
    bn_value = get_bn_value(big_model, block_flag, pruned_index_per_layer)
   
    if 'resnet50' in network_arch:
        small_model = models.resnet50_small(index=kept_index_per_layer, bn_value=bn_value,
                                            num_for_construct=num_for_construct,num_classes=1000,dim_out=dim_out,network_arch=opt.network_arch)
    elif 'resnet18' in network_arch:
        small_model = models.resnet18_small(index=kept_index_per_layer, bn_value=bn_value,
                                            num_for_construct=num_for_construct,num_classes=1000,dim_out=dim_out,network_arch=opt.network_arch)
    elif 'resnet34' in network_arch:
        small_model = models.resnet34_small(index=kept_index_per_layer, bn_value=bn_value,
                                            num_for_construct=num_for_construct,num_classes=1000,dim_out = dim_out, network_arch=opt.network_arch)
    else:
        pass
        
    return kept_index_per_layer, pruned_index_per_layer, block_flag, small_model


def get_bn_value(big_model, block_flag, pruned_index_per_layer):
    big_model.eval()
    
    #remove model. from the state_dict for ft_net
    state_dict = big_model.state_dict()
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'model.' in k:
            name = k[6:]  # remove `model.`
        else:
            name = k
        new_state_dict[name] = v
        
    state_dict = new_state_dict
    
    bn_flag = "bn3" if block_flag == "conv3" else "bn2"
    key_bn = [x for x in state_dict.keys() if bn_flag in x]
    layer_flag_list = [[x[0:6], x[7], x[9:12], x] for x in key_bn if "weight" in x]
    # layer_flag_list = [['layer1', "0", "bn3",'layer1.0.bn3.weight']]
    bn_value = {}
    for layer_flag in layer_flag_list:
        module_bn = big_model._modules.get('model')._modules.get(
            layer_flag[0])._modules.get(layer_flag[1])._modules.get(layer_flag[2])
        num_feature = module_bn.num_features
        act_bn = module_bn(Variable(torch.zeros(1, num_feature, 1, 1)))

        index_name = layer_flag[3].replace("bn", "conv")
        index = Variable(torch.LongTensor(pruned_index_per_layer[index_name]))
        act_bn = torch.index_select(act_bn, 1, index)

        select = Variable(torch.zeros(1, num_feature, 1, 1))
        select.index_add_(1, index, act_bn)

        bn_value[layer_flag[3]] = select
    return bn_value


def get_small_model(big_model,dim_out):
    indice_dict, pruned_index_per_layer, block_flag, small_model = extract_para(big_model,dim_out)
   
    #remove model. from the state_dict for ft_net
    state_dict = big_model.state_dict()
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'model.' in k:
            name = k[6:]  # remove `model.`
        else:
            name = k
        new_state_dict[name] = v
     
    big_state_dict = new_state_dict
    
    
    small_state_dict = {}
    keys_list = list(big_state_dict.keys())
    for index, [key, value] in enumerate(big_state_dict.items()):
        # all the conv layer excluding downsample layer
        flag_conv_ex_down = not 'bn' in key and not 'downsample' in key and not 'fc' in key and not 'classifier' in key 
        # downsample conv layer
        flag_down = 'downsample.0' in key
        # value for 'output' dimension: all the conv layer including downsample layer
        if flag_conv_ex_down or flag_down:
            small_state_dict[key] = torch.index_select(value, 0, indice_dict[key])
            #if 'layer1.1.conv2.weight' in key:
            conv_index = keys_list.index(key)
            # 4 following bn layer, bn_weight, bn_bias, bn_runningmean, bn_runningvar
            for offset in range(1, 5, 1):
                bn_key = keys_list[conv_index + offset]
                small_state_dict[bn_key] = torch.index_select(big_state_dict[bn_key], 0, indice_dict[key])
                
            #for num_batches_tracked
            bn_key = keys_list[conv_index + 5]
            small_state_dict[bn_key] = big_state_dict[bn_key]
            
            # value for 'input' dimension
            if flag_conv_ex_down:
                # first layer of first block
                if 'layer1.0.conv1.weight' in key:
                    small_state_dict[key] = torch.index_select(small_state_dict[key], 1, indice_dict['conv1.weight'])
                # just conv1 of block, the input dimension should not change for shortcut
                
                elif not "conv1" in key:
                    conv_index = keys_list.index(key)
                    # get the last con layer
                    key_for_input = keys_list[conv_index - 6] # 5-->6
                    small_state_dict[key] = torch.index_select(small_state_dict[key], 1, indice_dict[key_for_input])
                
            # only the first downsample layer should change as conv1 reduced
            elif 'layer1.0.downsample.0.weight' in key:
                small_state_dict[key] = torch.index_select(small_state_dict[key], 1, indice_dict['conv1.weight'])
            
        elif 'fc' in key or 'classifier' in key:
            small_state_dict[key] = value
    

    small_model.load_state_dict(small_state_dict)
                
    
    return small_model
######################################################################
# prune model

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

        return weight_np

    def get_filter_codebook(self, weight_torch, compress_rate, length):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            norm1_1 = torch.norm(weight_vec,1,1) # L1-norm
            norm1 = np.sum(weight_vec.abs().cpu().numpy(), axis=1)
            #norm1_np = norm1.cpu().numpy()
            norm1_np = norm1
            filter_index = norm1_np.argsort()[:filter_pruned_num]
            
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

    def init_rate(self, layer_rate):
        if 'vgg' in prune_arch:
            cfg_5x = [24, 22, 41, 51, 108, 89, 111, 184, 276, 228, 512, 512, 512]
            cfg_official = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
            # cfg = [32, 64, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256]
            cfg_index = 0
            pre_cfg = True
            for index, item in enumerate(self.model.named_parameters()):
                self.compress_rate[index] = 1
                if len(item[1].size()) == 4:
                    if not pre_cfg:
                        self.compress_rate[index] = layer_rate
                        self.mask_index.append(index)
                    else:
                        self.compress_rate[index] =  1 - cfg_5x[cfg_index] / item[1].size()[0]
                        self.mask_index.append(index)

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
            else:
                pass

    def init_mask(self, layer_rate,decay_rate = 0.2):
        self.init_rate(layer_rate)
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index) and len(item.data.size()) == 4: #conv::
                self.decay_rate = decay_rate
                self.mat[index],pruned_list = self.get_filter_codebook(item.data, self.compress_rate[index],
                                                           self.model_length[index])
                self.mat[index] = self.convert2tensor(self.mat[index])
                
                self.mat[index] = self.mat[index].cuda()
                
                #set bn.weight and bias 
                codebook = np.ones(item.data.size()[0])
               
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



######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        ff = torch.FloatTensor(n, dim_feature).zero_()
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            outputs,f = model(input_img) 
            #if i == 1:
            #    sys.exit(0)
            f = f.data.cpu()
            ff = ff+f
        # norm feature
        if opt.PCB:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6) 
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff), 0)
    return features

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_cam,gallery_label = get_id(gallery_path)
query_cam,query_label = get_id(query_path)

######################################################################
# Load Collected data Trained model
if opt.use_dense:
    model_structure = ft_net_dense(dim_out)
else:
    model_structure = ft_net(dim_out,network_arch)

if opt.PCB:
    model_structure = PCB(dim_out)

#model = load_network(model_structure)
model = model_structure



pretrained_dir = opt.pretrained_dir
if pretrained_dir != "":
    state = torch.load(pretrained_dir)
    model.load_state_dict(state)

if load_L1:
    
    m = Mask(model)
    m.init_length()
    print("-" * 10 + "one epoch begin" + "-" * 10)
    print("the compression rate now is {:}".format(prune_rate))
    model = model.cuda()
    #L1-norm pruning
    m.model = model
    m.if_zero()
    m.init_mask(prune_rate,decay_rate = 0)
    m.do_mask()
    m.if_zero()
    model = m.model
    #get small model
    model = model.cpu()
    model = get_small_model(model,dim_out)


state_dict = torch.load(resume_dir)
#if not pruned, remove model. to get a compatible model
if prune_rate == 0 and load_L1:
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'model.' in k:
            name = k[6:]  # remove `model.`
        else:
            name = k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
else:
    model.load_state_dict(state_dict)


'''
print('-'*20,'before prune','-'*20)
for index, [key, value] in enumerate(model.state_dict().items()):
    if(value.numel() > 0):
        print(key,np.sum(value.cpu().numpy()))
'''
if not load_L1:
    model = model.cpu()
    model = get_small_model(model,dim_out)
'''
print('-'*20,'after prune','-'*20)
for index, [key, value] in enumerate(model.state_dict().items()):
    if(value.numel() > 0):
        print(key,np.sum(value.cpu().numpy()))


'''
#add 'model.' for L1-norm methods



#model = model_structure

# Remove the final fc layer and classifier layer
#if not opt.PCB:
#    model.model.fc = nn.Sequential()
#    model.classifier = nn.Sequential()
#else:
#    model = PCB_test(model)

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()
time_start = datetime.datetime.now()
# Extract feature
gallery_feature = extract_feature(model,dataloaders['gallery'])

time_end = datetime.datetime.now()

query_feature = extract_feature(model,dataloaders['query'])
    
# Save to Matlab for check
result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
scipy.io.savemat('pytorch_result.mat',result)
