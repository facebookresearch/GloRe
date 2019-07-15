# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# from base_cnn import
import logging

from .resnet50_3d_gcn_x5 import RESNET50_3D_GCN_X5
from .resnet101_3d_gcn_x5 import RESNET101_3D_GCN_X5
from .config import get_config

def get_symbol(name, print_net=False, **kwargs):

    if name.upper() == "RESNET50_3D_GCN_X5":
        net = RESNET50_3D_GCN_X5(**kwargs)
    elif name.upper() == "RESNET101_3D_GCN_X5":
        net = RESNET101_3D_GCN_X5(**kwargs)
    else:
        logging.error("network '{}'' not implemented".format(name))
        raise NotImplementedError()

    if print_net:
        logging.debug("Symbol:: Network Architecture:")
        logging.debug(net)

    input_conf = get_config(name, **kwargs)
    return net, input_conf

if __name__ == "__main__":
    import torch
    from torch.autograd import Variable

    name = "RESNET50_3D"
    net, _ = get_symbol(name, num_classes=101)

    x = torch.randn(2, 3, 8, 224, 224)
    out = net(Variable(x))
    print (out)
