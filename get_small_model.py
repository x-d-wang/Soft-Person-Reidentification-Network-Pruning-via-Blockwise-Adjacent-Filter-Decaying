import argparse
import os,sys
import shutil
import pdb, time
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import convert_secs2time, time_string, time_file_str
# from models import print_log
#import models
import random
import numpy as np
import copy
import models

from model import ft_net

resume = 'net_last.pth'
save_dir = './small_model'

def main():

    model = ft_net(702)
    model = model.cuda()
    #print("=> Model : {}".format(model))
    #model = torch.nn.DataParallel(model) 
    cudnn.benchmark = True

    # optionally resume from a checkpoint
    if resume:
        if os.path.isfile(resume):
            #checkpoint = torch.load(args.resume)
            state_dict = torch.load(resume)
            #print('state_dict is:{}'.format(state_dict))
            #best_prec1 = checkpoint['best_prec1']
            #state_dict = checkpoint['state_dict']
            #state_dict = remove_module_dict(state_dict)
            #print(' state_dict =',state_dict.keys())
            #return
            model.load_state_dict(state_dict)
            #print(model._modules.get('model')._modules.get('fc'))
            print('-'*20+'beging getting small'+'-'*20)
            
            model = model.cpu()
            small_model = get_small_model(model)
            small_path = os.path.join(save_dir, "small_model.pth")
            torch.save(small_model, small_path)

       
def validate(val_loader, model, criterion, log):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # target = target.cuda(async=True)
        if args.use_cuda:
            input, target = input.cuda(), target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5), log)

    print_log(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                           error1=100 - top1.avg), log)

    return top1.avg


def save_checkpoint(state, is_best, filename, bestname):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestname)


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def remove_module_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def import_sparse(model):
    checkpoint = torch.load('/home/xdwang/projects/deep_learning/pytorch/jupyter/purning/Person-reID-triplet-loss/model/ft_ResNet50/net_59.pth')
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    print("sparse_model_loaded")
    return model


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
    indices_zero = torch.LongTensor(zeros) if zeros != [] else []

    return indices_zero, indices_nonzero


def extract_para(big_model):
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
    try:
        assert len(item) in [102, 182, 327, 522]
        print("state dict length is one of 102, 182, 267, 522")
    except AssertionError as e:
        print("False state dict")

    # indices_list = []
    kept_index_per_layer = {}
    kept_filter_per_layer = {}
    pruned_index_per_layer = {}
    constrct_flag = []
    conv_idx = 0
    for x in range(0, len(item) - 9, 6):#.......wxd
        
        indices_zero, indices_nonzero = check_channel(item[x][1])
        print('{}:{}:{} '.format(conv_idx,item[x][0],indices_nonzero.size()))
        # indices_list.append(indices_nonzero)
        pruned_index_per_layer[item[x][0]] = indices_zero
        kept_index_per_layer[item[x][0]] = indices_nonzero
        kept_filter_per_layer[item[x][0]] = indices_nonzero.shape[0]
        conv_idx+=1
    #for key in kept_index_per_layer:
    #    print('kept_index_per_layer:'+key,kept_index_per_layer[key].size())

    # add 'module.' if state_dict are store in parallel format
    #state_dict = ['module.' + x for x in state_dict]
    #print('-'*20,kept_index_per_layer,'-'*20)
    if len(item) == 102 or len(item) == 182:
        basic_block_flag = ['conv1.weight',
                            'layer1.0.conv1.weight', 'layer1.0.conv2.weight',
                            'layer2.0.conv1.weight', 'layer2.0.conv2.weight',
                            'layer3.0.conv1.weight', 'layer3.0.conv2.weight',
                            'layer4.0.conv1.weight', 'layer4.0.conv2.weight']
        constrct_flag = basic_block_flag
        block_flag = "conv2"
    elif len(item) == 327 or len(item) == 522:
        bottle_block_flag = ['conv1.weight',
                             'layer1.0.conv1.weight', 'layer1.0.conv3.weight',
                             'layer2.0.conv1.weight', 'layer2.0.conv3.weight',
                             'layer3.0.conv1.weight', 'layer3.0.conv3.weight',
                             'layer4.0.conv1.weight', 'layer4.0.conv3.weight']
        constrct_flag = bottle_block_flag
        block_flag = "conv3"

    # number of nonzero channel in conv1, and four stages
    num_for_construct = []
    for key in constrct_flag:
        num_for_construct.append(kept_filter_per_layer[key])
    #print('num_for_construct',num_for_construct)
    index_for_construct = dict(
        (key, value) for (key, value) in kept_index_per_layer.items() if block_flag in key)
    bn_value = get_bn_value(big_model, block_flag, pruned_index_per_layer)
    if len(item) == 102:
        small_model = models.resnet18_small(index=kept_index_per_layer, bn_value=bn_value,
                                            num_for_construct=num_for_construct)
    if len(item) == 182:
        small_model = models.resnet34_small(index=kept_index_per_layer, bn_value=bn_value,
                                            num_for_construct=num_for_construct)
    if len(item) == 327:
        small_model = models.resnet50_small(index=kept_index_per_layer, bn_value=bn_value,
                                            num_for_construct=num_for_construct,num_classes=1000)
    if len(item) == 522:
        small_model = models.resnet101_small(index=kept_index_per_layer, bn_value=bn_value,
                                             num_for_construct=num_for_construct)
        
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
    key_bn = [x for x in state_dict.keys() if "bn3" in x]
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


def get_small_model(big_model):
    indice_dict, pruned_index_per_layer, block_flag, small_model = extract_para(big_model)
    conv_idx = 0
    print('*'*20+'small_model:'+'*'*20)
    for m in small_model.modules():
        if(isinstance(m,nn.Conv2d)):
            print(conv_idx,':',m)
            conv_idx += 1
    
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
            #if 'layer1.0.conv2.weight' in key:
                #print('+'*20 + 'layer1.0.conv2.weight',indice_dict[key].size())
            small_state_dict[key] = torch.index_select(value, 0, indice_dict[key])
            #print('selecting:',key,indice_dict[key].size())
            conv_index = keys_list.index(key)
            # 4 following bn layer, bn_weight, bn_bias, bn_runningmean, bn_runningvar
            for offset in range(1, 5, 1):#5-->6
                bn_key = keys_list[conv_index + offset]
                #print('&'*20,bn_key,'&'*20)
                #print(big_state_dict[bn_key])
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
                    # print("key_for_input", key, key_for_input)
                    small_state_dict[key] = torch.index_select(small_state_dict[key], 1, indice_dict[key_for_input])
            # only the first downsample layer should change as conv1 reduced
            elif 'layer1.0.downsample.0.weight' in key:
                small_state_dict[key] = torch.index_select(small_state_dict[key], 1, indice_dict['conv1.weight'])
        elif 'fc' in key or 'classifier' in key:
            small_state_dict[key] = value
    
    #print('small_model:',small_model)        
    if len(set(big_state_dict.keys()) - set(small_state_dict.keys())) != 0:
        print("different keys of big and small model",
              sorted(set(big_state_dict.keys()) - set(small_state_dict.keys())))
        for x, y in zip(small_state_dict.keys(), small_model.state_dict().keys()):
            if small_state_dict[x].size() != small_model.state_dict()[y].size():
                print("difference with model and dict", x, small_state_dict[x].size(),
                      small_model.state_dict()[y].size())
     
    for key in  small_state_dict.keys():
        if 'bn1.weight' in key or 'bn1.bias' in key or 'bn1.running_mean' in key or 'bn1.running_var' in key:
            print('small_state_dict:'+key,small_state_dict[key])
            print('big_state_dict:'+key,big_state_dict[key])
    
    '''
    new_state_dict = OrderedDict()
    for k, v in small_state_dict.items():
        name = 'model.'+name  # remove `model.`
        new_state_dict[name] = v
    
    small_state_dict = new_state_dict
    '''
    #print('small_state_dict keys:',small_state_dict.keys())
    small_model.load_state_dict(small_state_dict)
    return small_model


if __name__ == '__main__':
    main()
