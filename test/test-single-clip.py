# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
sys.path.append("..")

import os
import time
import json
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn

import dataset
from train.model import static_model
from train import metric
from data import video_sampler as sampler
from data import video_transforms as transforms
from data.video_iterator import VideoIter
from network.symbol_builder import get_symbol

import pdb

parser = argparse.ArgumentParser(description="PyTorch Video Classification Parser (Evaluation)")
# debug
parser.add_argument('--debug-mode', type=bool, default=True,
                    help="print all setting for debugging.")
# io
parser.add_argument('--dataset', default='Kinetics', choices=['Kinetics'],
                    help="path to dataset")
parser.add_argument('--clip-length', default=8,
                    help="define the length of each input sample.")
parser.add_argument('--frame-interval', type=int, default=8,
                    help="define the sampling interval between frames.")
parser.add_argument('--model-subpath', type=str,
                    default='./exps/models/resnet50-lite_3d_8x8_w-glore_2-3_ep-0000.pth')
parser.add_argument('--model-root', type=str, default="../",
                    help="set logging file.")
parser.add_argument('--log-file', type=str, default="./eval-clips-x1.log",
                    help="set logging file.")
# device
parser.add_argument('--gpus', type=str, default="0,1,2,3,4,5,6,7",
                    help="define gpu id")
# algorithm
parser.add_argument('--network', type=str, default='RESNET50_3D_GCN_X5',
                    help="chose the base network")
# evaluation
parser.add_argument('--batch-size', type=int, default=32,
                    help="batch size")


def autofill(args):
    # customized
    args.model_prefix = os.path.join(args.model_root, args.model_subpath.split('_ep-')[0])
    args.load_epoch = int(args.model_subpath.split('_ep-')[1].split('.pth')[0])
    return args

def set_logger(log_file='', debug_mode=False):
    if log_file:
        if not os.path.exists("./"+os.path.dirname(log_file)):
            os.makedirs("./"+os.path.dirname(log_file))
        handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
    else:
        handlers = [logging.StreamHandler()]

    """ add '%(filename)s' to format show source file """
    logging.basicConfig(level=logging.DEBUG if debug_mode else logging.INFO,
                format='%(asctime)s %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                handlers = handlers)


if __name__ == '__main__':

    # set args
    args = parser.parse_args()
    set_logger(log_file=args.log_file, debug_mode=True)

    args = autofill(args)
    logging.info("Start evaluation with args:\n" +
                 json.dumps(vars(args), indent=4, sort_keys=True))

    # set device states
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus) # before using torch
    assert torch.cuda.is_available(), "CUDA is not available"

    # load dataset related configuration
    dataset_cfg = dataset.get_config(name=args.dataset)

    # creat model
    sym_net, input_config = get_symbol(name=args.network, **dataset_cfg)

    # network
    if torch.cuda.is_available():
        cudnn.benchmark = False
        sym_net = torch.nn.DataParallel(sym_net).cuda()
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        sym_net = torch.nn.DataParallel(sym_net)
        criterion = torch.nn.CrossEntropyLoss()
    net = static_model(net=sym_net,
                       criterion=criterion,
                       model_prefix=args.model_prefix)
    net.load_checkpoint(epoch=args.load_epoch)

    # data iterator: TODO
    num_clips = 1
    num_crops = 1
    data_root = '../dataset/Kinetics'
    normalize = transforms.Normalize(mean=input_config['mean'], std=input_config['std'])
    val_sampler = sampler.EvenlySampling(num=args.clip_length,
                                         interval=args.frame_interval,
                                         num_times=num_clips)
    logging.info("- slen: 256, crop size: 224 x 224. ")
    val_loader = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data', 'val_avi-288p'),
                      txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'kinetics_val_avi.txt'),
                      sampler=val_sampler,
                      force_color=True,
                      video_transform=transforms.Compose([
                                         transforms.Resize(256),
                                         transforms.CenterCrop((224,224)),
                                         transforms.ToTensor(),
                                         normalize,
                                      ]),
                      name='test',
                      return_item_subpath=True,
                      list_repeat_times=(num_clips * num_crops),
                      )

    eval_iter = torch.utils.data.DataLoader(val_loader,
                      batch_size=args.batch_size,
                      shuffle=True,
                      num_workers=50,
                      pin_memory=True)

    # eval metrics
    metrics = metric.MetricList(metric.Loss(name="loss-ce"),
                                metric.Accuracy(topk=1, name="top1"),
                                metric.Accuracy(topk=5, name="top5"))
    metrics.reset()

    # main loop
    net.net.eval()
    avg_score = {}
    best_value = {"Top-1": -1, "Top-5": -1}
    sum_batch_elapse = 0.
    sum_batch_inst = 0
    counter = 1
    softmax = torch.nn.Softmax(dim=1)
    i_batch = 0
    with torch.no_grad():

        for data, target, video_subpath in eval_iter:
            batch_start_time = time.time()
            outputs, losses = net.forward(data, target)

            sum_batch_elapse += time.time() - batch_start_time
            sum_batch_inst += 1

            # recording
            output = softmax(outputs[0]).data.cpu()
            target = target.cpu()
            losses = losses[0].data.cpu()
            for i_item in range(0, output.shape[0]):
                output_i = output[i_item,:].view(1, -1)
                target_i = torch.LongTensor([target[i_item]])
                loss_i = losses
                video_subpath_i = video_subpath[i_item]
                if video_subpath_i in avg_score:
                    avg_score[video_subpath_i][2] += output_i
                    avg_score[video_subpath_i][3] += 1
                    counter = 0.92 * counter + 0.08 * avg_score[video_subpath_i][3]
                else:
                    avg_score[video_subpath_i] = [torch.LongTensor(target_i.numpy().copy()),
                                                  torch.FloatTensor(loss_i.numpy().copy()),
                                                  torch.FloatTensor(output_i.numpy().copy()),
                                                  1] # the last one is counter

            # show progress
            if (i_batch % 100) == 0:
                metrics.reset()
                for _, video_info in avg_score.items():
                    target, loss, pred, _ = video_info
                    metrics.update([pred], target, [loss])
                name_value = metrics.get_name_value()
                if name_value[1][0][1] > best_value["Top-1"] \
                    or (name_value[1][0][1] == best_value["Top-1"] and name_value[2][0][1] > best_value["Top-5"]):
                    best_value = {"Top-1": name_value[1][0][1], "Top-5": name_value[2][0][1]}

                logging.info("{:.1f}: {:.1f}% \t| Batch [0,{}]    \tAvg: {} = {:.5f}, {} = {:.5f}, {} = {:.5f}".format(
                             counter, \
                             float(100*i_batch) / eval_iter.__len__(), \
                             i_batch, \
                             name_value[0][0][0], name_value[0][0][1], \
                             name_value[1][0][0], name_value[1][0][1], \
                             name_value[2][0][0], name_value[2][0][1]))
            i_batch += 1


    # finished
    logging.info("Evaluation Finished!")

    metrics.reset()
    for _, video_info in avg_score.items():
        target, loss, pred, _ = video_info
        metrics.update([pred], target, [loss])

    sk = list(avg_score.keys())
    sk.sort()

    logging.info("Total number of videos: {}".format(len(avg_score)))
    logging.info("Total time cost: {:.1f} sec".format(sum_batch_elapse))
    logging.info("Speed: {:.4f} samples/sec".format(
            args.batch_size * sum_batch_inst / sum_batch_elapse ))
    logging.info("Accuracy:")
    logging.info(json.dumps(metrics.get_name_value(), indent=4, sort_keys=True))

