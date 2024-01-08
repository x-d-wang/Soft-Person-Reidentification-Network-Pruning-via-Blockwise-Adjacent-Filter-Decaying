import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import torch
import time

from torch.nn import init

__all__ = ['ResNet_small', 'resnet18_small', 'resnet34_small', 'resnet50_small', 'resnet101_small', 'resnet152_small']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, num_for_construct,block_key, planes_before_prune, index, bn_value, stride=1,
                 downsample=None):
#    def __init__(self, inplanes, planes_after_prune, planes_expand, planes_before_prune, index, bn_value, stride=1,
#                 downsample=None):
        super(BasicBlock, self).__init__()
        #self.conv1 = conv3x3(inplanes, planes_after_prune, stride)
        #self.bn1 = nn.BatchNorm2d(planes_after_prune)
        self.conv1 = conv3x3(inplanes, num_for_construct[block_key+'1'], stride)
        self.bn1 = nn.BatchNorm2d(num_for_construct[block_key+'1'])
        
        self.relu = nn.ReLU(inplace=True)
        
        #self.conv2 = conv3x3(planes_after_prune, planes_after_prune)
        #self.bn2 = nn.BatchNorm2d(planes_after_prune)
        
        self.conv2 = conv3x3(num_for_construct[block_key+'1'], num_for_construct[block_key+'2'])
        self.bn2 = nn.BatchNorm2d(num_for_construct[block_key+'2'])
        
        self.downsample = downsample
        self.stride = stride

        # for residual index match
        self.index = Variable(index)
        # for bn add
        self.bn_value = bn_value

        # self.out = torch.autograd.Variable(
        #     torch.rand(batch, self.planes_before_prune, 64 * 56 // self.planes_before_prune,
        #                64 * 56 // self.planes_before_prune), volatile=True).cuda()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # setting: without index match
        # out += residual
        # out = self.relu(out)

        # setting: with index match
        residual =  residual + self.bn_value.cuda()
        #residual = residual + torch.zeros(self.bn_value.size())
        residual.index_add_(1, self.index.cuda(), out)

        residual = self.relu(residual)

        return residual


class Bottleneck(nn.Module):
    # expansion is not accurately equals to 4
    expansion = 4

    #def __init__(self, inplanes, planes_after_prune, planes_conv2, planes_expand, planes_before_prune, index, bn_value, stride=1,
    def __init__(self, inplanes, num_for_construct,block_key, planes_before_prune, index, bn_value, stride=1,
                 downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, num_for_construct[block_key+'1'], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_for_construct[block_key+'1'])
        self.conv2 = nn.Conv2d(num_for_construct[block_key+'1'], num_for_construct[block_key+'2'], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_for_construct[block_key+'2'])

        # setting: for accuracy expansion
        self.conv3 = nn.Conv2d(num_for_construct[block_key+'2'], num_for_construct[block_key+'3'], kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_for_construct[block_key+'3'])

        # setting: original resnet, expansion = 4
        # self.conv3 = nn.Conv2d(planes, planes_before_prune * 4, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(planes_before_prune * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        # for residual index match
        self.index = Variable(index)
        # for bn add
        self.bn_value = bn_value

        # self.extend = torch.autograd.Variable(
        #     torch.rand(self.planes_before_prune * 4, 64 * 56 // self.planes_before_prune,
        #                64 * 56 // self.planes_before_prune), volatile=True).cuda()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # setting: without index match
        # print("residual size{},out size{} ".format(residual.size(),   out.size()))
        # out += residual
        # out = self.relu(out)

        # setting: with index match
        #residual += self.bn_value.cuda()
        residual = residual + torch.zeros(self.bn_value.size()).cuda()
        residual.index_add_(1, self.index.cuda(), out)

        residual = self.relu(residual)

        return residual


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
        self.sum = self.sum + val * n
        self.count = self.count + n
        self.avg = self.sum / self.count

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
        

class ResNet_small(nn.Module):

    def __init__(self, block, layers, index, bn_value,
                 num_for_construct=[64, 64, 64 * 4, 128, 128 * 4, 256, 256 * 4, 512, 512 * 4],
                 num_classes=1000,dim_out=751,network_arch='resnet50'):
        super(ResNet_small, self).__init__()
        self.inplanes = num_for_construct['00']

        self.conv1 = nn.Conv2d(3, num_for_construct['00'], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_for_construct['00'])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # setting: expansion = 4
        # self.layer1 = self._make_layer(block, num_for_construct[1], num_for_construct[1] * 4, 64, index,  layers[0])
        # self.layer2 = self._make_layer(block, num_for_construct[2], num_for_construct[2] * 4, 128, index,  layers[1], stride=2)
        # self.layer3 = self._make_layer(block, num_for_construct[3], num_for_construct[3] * 4, 256, index,  layers[2], stride=2)
        # self.layer4 = self._make_layer(block, num_for_construct[4], num_for_construct[4] * 4, 512, index,  layers[3], stride=2)

        # setting: expansion may not accuracy equal to 4
        self.index_layer1 = {key: index[key] for key in index.keys() if 'layer1' in key}
        #for key in self.index_layer1.keys():
        #    print("&"*10+"self.index_layer1 ",key,':',self.index_layer1[key].size())
            
        self.index_layer2 = {key: index[key] for key in index.keys() if 'layer2' in key}
        self.index_layer3 = {key: index[key] for key in index.keys() if 'layer3' in key}
        self.index_layer4 = {key: index[key] for key in index.keys() if 'layer4' in key}
        self.bn_layer1 = {key: bn_value[key] for key in bn_value.keys() if 'layer1' in key}
        self.bn_layer2 = {key: bn_value[key] for key in bn_value.keys() if 'layer2' in key}
        self.bn_layer3 = {key: bn_value[key] for key in bn_value.keys() if 'layer3' in key}
        self.bn_layer4 = {key: bn_value[key] for key in bn_value.keys() if 'layer4' in key}
        # print("bn_layer1", bn_layer1.keys(), bn_layer2.keys(), bn_layer3.keys(), bn_layer4.keys())
        #need revise
        
        self.layer1 = self._make_layer(block, num_for_construct,1, 64, self.index_layer1, self.bn_layer1,
                                       layers[0],stride=1,network_arch=network_arch)
                                       
        #print('-+-'*20+'num_for_construct[0],[1],[2]',num_for_construct[0],num_for_construct[1],num_for_construct[2])
        self.layer2 = self._make_layer(block, num_for_construct,2, 128, self.index_layer2, self.bn_layer2,
                                       layers[1], stride=2,network_arch=network_arch)
        self.layer3 = self._make_layer(block, num_for_construct,3, 256, self.index_layer3, self.bn_layer3,
                                       layers[2], stride=2,network_arch=network_arch)
        self.layer4 = self._make_layer(block, num_for_construct,4, 512, self.index_layer4, self.bn_layer4,
                                       layers[3], stride=2,network_arch=network_arch)
        #self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        #need revise
        if 'resnet18' in network_arch or 'resnet34' in network_arch:
            self.classifier = ClassBlock(512, dim_out, dropout=False, relu=False)
        else: 
            self.classifier = ClassBlock(2048,dim_out, dropout=False, relu=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    #def _make_layer(self, block, planes_after_prune, planes_conv2, planes_expand, planes_before_prune, index, bn_layer, blocks,
    def _make_layer(self, block, num_for_construct, layer_idx, planes_before_prune, index, bn_layer, blocks,
                    stride=1,network_arch='resnet50'):
        downsample = None
        if stride != 1 or self.inplanes != planes_before_prune * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes_before_prune * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes_before_prune * block.expansion),
            )
        #print("before pruning is {}, after pruning is {}:".format(planes_before_prune,planes_after_prune))

        #print(':'*20,self.inplanes,planes_after_prune,planes_before_prune,planes_expand)
        # setting: accu number for_construct expansion
        #need revise
        if 'resnet18' in network_arch or 'resnet34' in network_arch:
            first_conv_flag = '0.conv2'
            block_conv_flag = '.conv2'
            block_bn_flag = '.bn2'
        else:
            first_conv_flag = '0.conv3'
            block_conv_flag = '.conv3'
            block_bn_flag = '.bn3'
        
        index_block_0_dict = {key: index[key] for key in index.keys() if first_conv_flag in key}
        index_block_0_value = list(index_block_0_dict.values())[0]
        
        bn_layer_0_value = list(bn_layer.values())[0]
            

        layers = []
        layers.append(
            block(self.inplanes, num_for_construct,str(layer_idx)+'0', planes_before_prune, index_block_0_value,
                  bn_layer_0_value,
                  stride, downsample))
        #        self.inplanes = planes * block.expansion
        self.inplanes = planes_before_prune * block.expansion
        #print('layers:',layers)
        for i in range(1, blocks):
            #need revise
            index_block_i_dict = {key: index[key] for key in index.keys() if (str(i) + block_conv_flag) in key}
            index_block_i_value = list(index_block_i_dict.values())[0]
            
            #need revise
            bn_layer_i = {key: bn_layer[key] for key in bn_layer.keys() if (str(i) + block_bn_flag) in key}
            bn_layer_i_value = list(bn_layer_i.values())[0]
            layers.append(
                block(self.inplanes, num_for_construct, str(layer_idx)+str(i), planes_before_prune, index_block_i_value,
                      bn_layer_i_value,
                      ))
        #print('nn.Sequential(*layers):',nn.Sequential(*layers))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #x = torch.squeeze(x)
        #x = self.fc(x)
        
        x,f = self.classifier(x)
        return x,f


def resnet18_small(pretrained=False, **kwargs):
    """Constructs a ResNet_small-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_small(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34_small(pretrained=False, **kwargs):
    """Constructs a ResNet_small-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_small(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50_small(pretrained=False, **kwargs):
    """Constructs a ResNet_small-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_small(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101_small(pretrained=False, **kwargs):
    """Constructs a ResNet_small-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_small(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152_small(pretrained=False, **kwargs):
    """Constructs a ResNet_small-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_small(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
