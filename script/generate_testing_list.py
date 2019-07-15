# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging

def duplicate_cache_by_video_length(src_list, 
                                    dst_list, 
                                    clip_length):
    f = open(src_list, 'r')
    head_cached_video_prefix = f.readline()
    head_cached_txt_list = f.readline()
    head_cached_check_video = f.readline()
    lines = f.readlines() # list context
    f.close()

    # decode list
    video_list = []
    for line in lines:
        v_id, label, video_subpath, frame_count, check_video = line.split()
        video_list.append([int(v_id), int(label), video_subpath, \
                                int(frame_count), bool(int(check_video))])

    # write new list
    f = open(dst_list, 'w')
    f.write(head_cached_video_prefix)
    f.write(head_cached_txt_list)
    f.write(head_cached_check_video.replace("\n",""))

    # list context
    for i, info in enumerate(video_list):
        frame_count_i = info[3]
        for i in range(0, frame_count_i, clip_length):
            line = "\n{:d}\t{:d}\t{:s}\t{:d}\t{:d}".format(*info)
            f.write(line)

    f.close()


def duplicate_cache_by_fixed_times(src_list, 
                                   dst_list, 
                                   duplicate_times):
    f = open(src_list, 'r')
    head_cached_video_prefix = f.readline()
    head_cached_txt_list = f.readline()
    head_cached_check_video = f.readline()
    lines = f.readlines() # list context
    f.close()

    # decode list
    video_list = []
    for line in lines:
        v_id, label, video_subpath, frame_count, check_video = line.split()
        video_list.append([int(v_id), int(label), video_subpath, \
                                int(frame_count), bool(int(check_video))])

    # write new list
    f = open(dst_list, 'w')
    f.write(head_cached_video_prefix)
    f.write(head_cached_txt_list)
    f.write(head_cached_check_video.replace("\n",""))

    # list context
    for i, info in enumerate(video_list):
        frame_count_i = info[3]
        for i in range(0, duplicate_times):
            line = "\n{:d}\t{:d}\t{:s}\t{:d}\t{:d}".format(*info)
            f.write(line)

    f.close()

