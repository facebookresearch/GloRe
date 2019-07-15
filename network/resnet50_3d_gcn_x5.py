# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn

try:
    from . import initializer
    from .global_reasoning_unit import GloRe_Unit
except:
    import initializer
    from global_reasoning_unit import GloRe_Unit

class BN_AC_CONV2D(nn.Module):

    def __init__(self, num_in, num_filter,
                 kernel=(1,1), pad=(0,0), stride=(1,1), g=1, bias=False):
        super(BN_AC_CONV2D, self).__init__()
        self.bn = nn.BatchNorm2d(num_in, eps=1e-04)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_in, num_filter, kernel_size=kernel, padding=pad,
                               stride=stride, groups=g, bias=bias)

    def forward(self, x):
        h = self.relu(self.bn(x))
        h = self.conv(h)
        return h

class BN_AC_CONV3D(nn.Module):

    def __init__(self, num_in, num_filter,
                 kernel=(1,1,1), pad=(0,0,0), stride=(1,1,1), g=1, bias=False):
        super(BN_AC_CONV3D, self).__init__()
        self.bn = nn.BatchNorm3d(num_in, eps=1e-04)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(num_in, num_filter, kernel_size=kernel, padding=pad,
                               stride=stride, groups=g, bias=bias)

    def forward(self, x):
        h = self.relu(self.bn(x))
        h = self.conv(h)
        return h

class RESIDUAL_BLOCK(nn.Module):

    def __init__(self, num_in, num_mid, num_out, g=1, stride=(1,1,1), first_block=False, use_3d=True):
        super(RESIDUAL_BLOCK, self).__init__()
        kt,pt = (3,1) if use_3d else (1,0)

        self.conv_m1 = BN_AC_CONV3D(num_in=num_in, num_filter=num_mid, kernel=(kt,1,1), pad=(pt,0,0))
        self.conv_m2 = BN_AC_CONV3D(num_in=num_mid, num_filter=num_mid, kernel=(1,3,3), pad=(0,1,1), stride=stride, g=g)
        self.conv_m3 = BN_AC_CONV3D(num_in=num_mid, num_filter=num_out, kernel=(1,1,1), pad=(0,0,0))
        # adapter
        if first_block:
            self.conv_w1 = BN_AC_CONV3D(num_in=num_in,  num_filter=num_out, kernel=(1,1,1), pad=(0,0,0), stride=stride)

    def forward(self, x):

        h = self.conv_m1(x)
        h = self.conv_m2(h)
        h = self.conv_m3(h)

        if hasattr(self, 'conv_w1'):
            x = self.conv_w1(x)

        return h + x


class RESNET50_3D_GCN_X5(nn.Module):

    def __init__(self, num_classes, pretrained=False, **kwargs):
        super(RESNET50_3D_GCN_X5, self).__init__()

        groups = 1
        k_sec  = {  2: 3, \
                    3: 4, \
                    4: 6, \
                    5: 3  }

        # conv1 - x112 (x16)
        conv1_num_out = 32
        self.conv1 = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv3d( 3, conv1_num_out, kernel_size=(3,5,5), padding=(1,2,2), stride=(1,2,2), bias=False)),
                    ('bn', nn.BatchNorm3d(conv1_num_out, eps=1e-04)),
                    ('relu', nn.ReLU(inplace=True)),
                    ('max_pool', nn.MaxPool3d(kernel_size=(1,3,3), padding=(0,1,1), stride=(1,2,2))),
                    ]))

        # conv2 - x56 (x16)
        num_mid = 64
        conv2_num_out = 256
        self.conv2 = nn.Sequential(OrderedDict([
                    ("B%02d"%i, RESIDUAL_BLOCK(num_in=conv1_num_out if i==1 else conv2_num_out,
                                               num_mid=num_mid,
                                               num_out=conv2_num_out,
                                               stride=(1,1,1) if i==1 else (1,1,1),
                                               g=groups,
                                               first_block=(i==1))) for i in range(1,k_sec[2]+1)
                    ]))

        # conv3 - x28 (x8)
        num_mid *= 2
        conv3_num_out = 2 * conv2_num_out
        blocks = []
        for i in range(1,k_sec[3]+1):
            use_3d = bool(i % 2)
            blocks.append(("B%02d"%i, RESIDUAL_BLOCK(num_in=conv2_num_out if i==1 else conv3_num_out,
                                                     num_mid=num_mid,
                                                     num_out=conv3_num_out,
                                                     stride=(2,2,2) if i==1 else (1,1,1),
                                                     use_3d=use_3d,
                                                     g=groups,
                                                     first_block=(i==1))))
            if i in [1,3]:
                blocks.append(("B%02d_extra"%i, GloRe_Unit(num_in=conv3_num_out, num_mid=num_mid)))
        self.conv3 = nn.Sequential(OrderedDict(blocks))
        
        # conv4 - x14 (x8)
        num_mid *= 2
        conv4_num_out = 2 * conv3_num_out
        blocks = []
        for i in range(1,k_sec[4]+1):
            use_3d = bool(i % 2)
            blocks.append(("B%02d"%i, RESIDUAL_BLOCK(num_in=conv3_num_out if i==1 else conv4_num_out,
                                                     num_mid=num_mid,
                                                     num_out=conv4_num_out,
                                                     stride=(1,2,2) if i==1 else (1,1,1),
                                                     use_3d=use_3d,
                                                     g=groups,
                                                     first_block=(i==1))))
            if i in [1,3,5]:
                blocks.append(("B%02d_extra"%i, GloRe_Unit(num_in=conv4_num_out, num_mid=num_mid)))
        self.conv4 = nn.Sequential(OrderedDict(blocks))
        
        # conv5 - x7 (x4)
        num_mid *= 2
        conv5_num_out = 2 * conv4_num_out
        self.conv5 = nn.Sequential(OrderedDict([
                    ("B%02d"%i, RESIDUAL_BLOCK(num_in=conv4_num_out if i==1 else conv5_num_out,
                                               num_mid=num_mid,
                                               num_out=conv5_num_out,
                                               stride=(1,2,2) if i==1 else (1,1,1),
                                               g=groups,
                                               use_3d=(i==2),
                                               first_block=(i==1))) for i in range(1,k_sec[5]+1)
                    ]))

        # final
        self.tail = nn.Sequential(OrderedDict([
                    ('bn', nn.BatchNorm3d(conv5_num_out, eps=1e-04)),
                    ('relu', nn.ReLU(inplace=True))
                    ]))

        self.globalpool = nn.Sequential(OrderedDict([
                        ('avg', nn.AvgPool3d(kernel_size=(4,7,7),  stride=(1,1,1))),
                        ('dropout', nn.Dropout(p=0.5)),
                        ]))
        self.classifier = nn.Linear(conv5_num_out, num_classes)

        #############
        # Initialization
        initializer.xavier(net=self)

        if pretrained:
            import torch
            load_method='inflation' # 'random', 'inflation'
            pretrained_model=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pretrained/resnet50-lite.pth')
            logging.info("Network:: symbol initialized, use pretrained model: `{}'".format(pretrained_model))
            assert os.path.exists(pretrained_model), "cannot locate: `{}'".format(pretrained_model)
            state_dict_2d = torch.load(pretrained_model)
            initializer.init_3d_from_2d_dict(net=self, state_dict=state_dict_2d, method=load_method)
        else:
            logging.info("Network:: symbol initialized, use random inilization!")

        blocker_name_list = []
        for name, param in self.state_dict().items():
            if name.endswith('blocker.weight'):
                blocker_name_list.append(name)
                param[:] = 0. 
        if len(blocker_name_list) > 0:
            logging.info("Network:: change params of the following layer be zeros: {}".format(blocker_name_list))


    def forward(self, x):
        assert x.shape[2] == 8

        h = self.conv1(x)   # x112 ->  x56
        h = self.conv2(h)   #  x56 ->  x56
        h = self.conv3(h)   #  x56 ->  x28
        h = self.conv4(h)   #  x28 ->  x14
        h = self.conv5(h)   #  x14 ->   x7

        # logging.info("{}".format(h.shape))

        h = self.tail(h)
        h = self.globalpool(h)

        h = h.view(h.shape[0], -1)
        h = self.classifier(h)

        return h

if __name__ == "__main__":
    import torch
    logging.getLogger().setLevel(logging.DEBUG)
    # ---------
    net = RESNET50_3D_GCN_X5(num_classes=400, pretrained=False)
    print("--------------")
    data = torch.autograd.Variable(torch.randn(1,3,8,224,224))
    output = net(data)
    torch.save({'state_dict': net.state_dict()}, './tmp.pth')
    print (output.shape)
