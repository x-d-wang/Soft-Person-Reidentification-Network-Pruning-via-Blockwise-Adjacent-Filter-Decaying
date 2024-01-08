from __future__ import print_function, division

import cv2
import time
import os
import matplotlib.pyplot as plt
import torch
from PIL import Image
import datetime

import matplotlib.patches as patches

import torch.nn as nn
from torch.autograd import Variable
import numpy as np


# -*- coding: utf-8 -*-

import datetime
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
from torchvision import transforms
import torchvision.models as tms
import time
import os
import scipy.io

from collections import OrderedDict
import models


from ptflops import get_model_complexity_info

dim_feature = 2048

dim_out = 751

prune_arch = ['resnet']
prune_skip_downsample = 1
resnet_arch = 'resnet50'
layer_begin =0
layer_end = 156
layer_inter = 1


prune_rate = 0.9
load_L1 = False




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



    blocks = [3, 4, 6, 3]
    block_flag = "conv3"
  
    
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
    print('num_for_construct:',num_for_construct)
        
    index_for_construct = dict(
        (key, value) for (key, value) in kept_index_per_layer.items() if block_flag in key)
    bn_value = get_bn_value(big_model, block_flag, pruned_index_per_layer)
   
  
    small_model = models.resnet50_small(index=kept_index_per_layer, bn_value=bn_value,
                                            num_for_construct=num_for_construct,num_classes=1000,dim_out=dim_out,network_arch='resnet50')
  
        
    #print("after dele 'model.':",small_model.state_dict().keys())
    #print("kept_index_per_layer:",kept_index_per_layer)
    #print('-'*20,small_model._modules.get('fc'))
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
    #print('layer_flag_list====',layer_flag_list)
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
    # print("keys_list", keys_list)
    for index, [key, value] in enumerate(big_state_dict.items()):
        # all the conv layer excluding downsample layer
        flag_conv_ex_down = not 'bn' in key and not 'downsample' in key and not 'fc' in key and not 'classifier' in key 
        # downsample conv layer
        flag_down = 'downsample.0' in key
        # value for 'output' dimension: all the conv layer including downsample layer
        if flag_conv_ex_down or flag_down:
            small_state_dict[key] = torch.index_select(value, 0, indice_dict[key])
            #if 'layer1.1.conv2.weight' in key:
            #    print('+'*20 +'value:',value.size(), key,small_state_dict[key].sum())
            #print('selecting:',key,indice_dict[key].size())
            conv_index = keys_list.index(key)
            #print('con_index=',conv_index)
            # 4 following bn layer, bn_weight, bn_bias, bn_runningmean, bn_runningvar
            for offset in range(1, 5, 1):
                bn_key = keys_list[conv_index + offset]
                #print('&'*20,big_state_dict[bn_key].sum(),'&'*20)
                
                #print(big_state_dict[bn_key])
                small_state_dict[bn_key] = torch.index_select(big_state_dict[bn_key], 0, indice_dict[key])
                #print('+'*20 + bn_key,small_state_dict[bn_key].sum())
                
                #print(bn_key,small_state_dict[bn_key])
                #print('big:',big_state_dict[bn_key])
                
            #for num_batches_tracked
            bn_key = keys_list[conv_index + 5]
            small_state_dict[bn_key] = big_state_dict[bn_key]
            #print('+'*20 + bn_key,small_state_dict[bn_key].sum())
            
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
                    # print("key_for_input", key, key_for_input)
                    small_state_dict[key] = torch.index_select(small_state_dict[key], 1, indice_dict[key_for_input])
                
            # only the first downsample layer should change as conv1 reduced
            elif 'layer1.0.downsample.0.weight' in key:
                small_state_dict[key] = torch.index_select(small_state_dict[key], 1, indice_dict['conv1.weight'])
            
        elif 'fc' in key or 'classifier' in key:
            small_state_dict[key] = value
    
    print('small_model:',small_model)
    for x in small_state_dict.keys():
        print(x,small_state_dict[x].size())        
    if len(set(big_state_dict.keys()) - set(small_state_dict.keys())) != 0:
        print("different keys of big and small model",
              sorted(set(big_state_dict.keys()) - set(small_state_dict.keys())))
        for x, y in zip(small_state_dict.keys(), small_model.state_dict().keys()):
            if small_state_dict[x].size() != small_model.state_dict()[y].size():
                print("difference with model and dict", x, small_state_dict[x].size(),
                      small_model.state_dict()[y].size())


    #print('small_state_dict keys:',small_state_dict.keys())
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

        print("codebook done")
        return weight_np

    def get_filter_codebook(self, weight_torch, compress_rate, length):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            norm1_1 = torch.norm(weight_vec,1,1) # L1-norm
            norm1 = np.sum(weight_vec.abs().cpu().numpy(), axis=1)
            #print(norm1_1.cpu().numpy()-norm1)
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
                    print(item[1].size())
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
                    #print(self.mask_index)
                    
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
            if (index in self.mask_index) and len(item.data.size()) == 4: #conv::
                self.decay_rate = decay_rate
                self.mat[index],pruned_list = self.get_filter_codebook(item.data, self.compress_rate[index],
                                                           self.model_length[index])
                self.mat[index] = self.convert2tensor(self.mat[index])
                
                self.mat[index] = self.mat[index]
                
                #set bn.weight and bias 
                codebook = np.ones(item.data.size()[0])
               
                #print(nonzero_idx)
                codebook[pruned_list] = decay_rate
                
                self.mat[index+1] = codebook
                self.mat[index+2] = codebook
                
                self.mat[index+1] = self.convert2tensor(self.mat[index+1])
                self.mat[index+2] = self.convert2tensor(self.mat[index+2])
                
                self.mat[index+1] = self.mat[index+1]
                self.mat[index+2] = self.mat[index+2]
                
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




def load_network(network,save_path):
    network.load_state_dict(torch.load(save_path))
    return network

pruned_filter_list = [2, 11, 20, 29, 39, 41, 44, 46, 57, 58, 59, 61, 62]

def draw_features(width,height,x,savename):
    tic=time.time()
    fig = plt.figure(figsize=(12, 4))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width*height):
        ax=plt.subplot(height,width, i + 1)
        #plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        img = x[0, i, :, :]
        
        ax.imshow(img)
    
        #plt.imsave(str(i)+'.png',img)
        print("{}/{}".format(i,width*height))
        ax.set_ylabel('{}'.format(i),rotation=0)
        ax.yaxis.set_label_coords(-0.2, 0.5) 
        if i in pruned_filter_list:
            ax.add_patch(patches.Rectangle((2,0),60, 128,fill=False, edgecolor='red', linewidth = 4))
        
        
        
    fig.savefig(savename, dpi=100)
    fig.clf()
     
    plt.close()
    print("time:{}".format(time.time()-tic))
    
        
 
########################################ft_model###################################

import torch
import torch.nn as nn
from torch.nn import init
import torchvision.models as tms
from torch.autograd import Variable

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=False, relu=False, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        #add_block += [nn.Linear(input_dim, num_bottleneck)] 
        num_bottleneck=input_dim
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        f = self.add_block(x)
        f_norm = f.norm(p=2, dim=1, keepdim=True) + 1e-8
        f = f.div(f_norm)
        x = self.classifier(f)
        return x,f

# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num ):
        super(ft_net, self).__init__()
        model_ft = tms.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num, dropout=False, relu=False)

    def forward(self, x,savepath):
        if True:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
                
            x = torch.squeeze(x)
        else:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)                
            x = torch.squeeze(x)
   
        return x


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

######################################################################
# Load Collected data Trained model
print('-------test-----------')

#APFP_path = '../results/ft_resnet50/prune_APFP/knn2/0.90/no_1x1_3e-1_70_0.01_m95_fast/net_last.pth'
APFP_path = '../results/ft_resnet50/prune_decay/general_k/recheck/knn2/0.9_Rank@1_0.767221_Rank@5_0.892221_Rank@10_0.927553_mAP_0.560703/net_last.pth'

baseline_path = '../ReID_Prune_Triplet/pretrained/Rank@1_0.866390_Rank@5_0.937352_Rank@10_0.960511_mAP_0.708192.pth'

model_structure_APFP = ft_net(751)
model_structure_baseline = ft_net(751)


model_APFP = load_network(model_structure_APFP,APFP_path)
model_baseline = load_network(model_structure_baseline,baseline_path)

model_APFP = model_APFP.eval()
model_baseline = model_baseline.eval()

data_transforms = transforms.Compose([
        transforms.Resize((256,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


img = Image.open('1501_c2s3_069052_00.jpg')
img=data_transforms(img)
img=img.unsqueeze(0)



model_structure_APFP = model_structure_APFP.cpu()
model_APFP_small = get_small_model(model_structure_APFP,dim_out)
model_APFP_small = model_APFP_small.cpu() 
#savepath=r'features_whitegirl'
#if not os.path.exists(savepath):
#    os.mkdir(savepath)
 

net = model_APFP_small
flops, params = get_model_complexity_info(net, (3, 256, 128), as_strings=True, print_per_layer_stat=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))

 
with torch.no_grad():
    time_start = datetime.datetime.now()
    #out=model(img)
    
    time_start = time.time()
    #
    for i in range(10):
        outputs_baseline = model_baseline(img,'features_maps')
        #outputs_APFP = model_APFP_small(img) 
    time_end = time.time()
    print('method4 complexity is:',(time_end-time_start))
    #print(torch.sum(outputs_baseline))
    #print('|PFP - baseline|=',np.linalg.norm(outputs_PFP - outputs_baseline))
    #print('|FPGM - baseline|=',np.linalg.norm(outputs_FPGM - outputs_baseline))
    print("done")



