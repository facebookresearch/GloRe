# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging

import subprocess

from joblib import delayed
from joblib import Parallel

def exe_cmd(cmd):
    try:
        dst_file = cmd.split()[-1]
        if os.path.exists(dst_file) and os.path.getsize(dst_file)/2**10 > 10:
            return "exist"
        cmd = cmd.replace('(', '\(').replace(')', '\)').replace('\'', '\\\'')
        output = subprocess.check_output(cmd, shell=True,
                                        stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        logging.warning("failed: {}".format(cmd))
        # logging.warning("failed: {}: {}".format(cmd, err.output.decode("utf-8"))) # detailed error
        return False
    return output

def convert_video_wapper(video_list,
                         src_root,
                         dst_root,
                         cmd_format,
                         in_parallel):
    commands = []
    for vid_src_path in video_list:
        vid_dst_path = vid_src_path.replace(src_root, dst_root)
        vid_dst_base = vid_dst_path[:vid_dst_path.rfind('.mp4')]
        vid_dst_folder = os.path.dirname(vid_dst_path)
        if not os.path.exists(vid_dst_folder):
            os.makedirs(vid_dst_folder)
        cmd = cmd_format.format(vid_src_path, vid_dst_base)
        commands.append(cmd)

    logging.info("- {} commonds to excute".format(len(commands)))

    if not in_parallel:
        for i, cmd in enumerate(commands):
            # if i % 100 == 0:
            logging.info("{} / {}: '{}'".format(i, len(commands), cmd))
            exe_cmd(cmd=cmd)
    else:
        num_jobs = 20
        logging.info("- processing videos in parallel, num_jobs={}".format(num_jobs))
        Parallel(n_jobs=num_jobs)(delayed(exe_cmd)(cmd) for cmd in commands)

def read_video_list(root_folder):
    video_list = []
    subfolders = [ os.path.join(root_folder, name) for name in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, name)) ]

    for folder_i in subfolders:
        videos = [ os.path.join(folder_i, name) for name in os.listdir(folder_i) if name.endswith('.mp4') ]
        video_list = video_list + videos

    return video_list

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)

    """
    other options
    cmd_format = 'ffmpeg -y -i {} -vcodec flv {}.flv'
    """

    # options
    in_parallel = True
    in_random_order = True # this is useful when you want to run this script on multiple nodes

    # resize to slen = x288
    cmd_format = 'ffmpeg -y -i {} -c:v mpeg4 -filter:v "scale=min(iw\,(288*iw)/min(iw\,ih)):-1" -b:v 512k {}.avi'

    # src_roots = ['../raw/data/val',          '../raw/data/train', ]
    # dst_roots = ['../raw/data/val_avi-288p', '../raw/data/train_avi-288p',]

    src_roots = ['../raw/data/val']
    dst_roots = ['../raw/data/val_x288p-GOPxN']

    for (src_root, dst_root) in zip(src_roots, dst_roots):
        assert os.path.exists(src_root)
        if not os.path.exists(dst_root):
            os.makedirs(dst_root)
        logging.info("- load videos from: {}".format(src_root))
        video_list = read_video_list(root_folder=src_root)
        logging.info("- found {} videos.".format(len(video_list)))

        if in_random_order:
            from random import shuffle
            shuffle(video_list)
            logging.info("- use random order")

        logging.info("- start convert video:")
        logging.info("- option: in_parallel = {}".format(in_parallel))

        convert_video_wapper(video_list=video_list,
                             src_root=src_root,
                             dst_root=dst_root,
                             cmd_format=cmd_format,
                             in_parallel=in_parallel)

        logging.info("- finished!")
