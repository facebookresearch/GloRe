# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import time
import numpy as np

import torch
import torch.utils.data as data
import logging

class Video(object):
    """basic Video class"""

    def __init__(self):
        self.reset()

    def __del__(self):
        self.close()

    def reset(self):
        self.cap = None
        self.vid_path = None
        self.frame_count = -1
        return self

    def open(self, vid_path):
        if not os.path.exists(vid_path):
            raise IOError("VideoIter:: cannot locate: `{}'".format(vid_path))

        # close previous video
        if self.cap is not None:
            self.close()

        # try to open video
        cap = cv2.VideoCapture(vid_path)
        if cap.isOpened():
            self.cap = cap
            self.vid_path = vid_path
        else:
            cap.release()
            raise IOError("VideoIter:: failed to open video: `{}'".format(vid_path))

        return self

    def count_frames(self):
        offset = 0
        if self.vid_path.endswith('.flv'):
            offset = -1
        unverified_frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) + offset
        self.frame_count = unverified_frame_count
        if self.frame_count <= 0:
            raise IOError("VideoIter:: Video: `{}' has no frames".format(self.vid_path))
        return self.frame_count

    def extract_frames(self, idxs, force_color=True):
        assert self.cap is not None, "No opened video."
        if len(idxs) < 1:
            return []

        frames = []
        pre_idx = max(idxs)
        for idx in idxs:
            assert (self.frame_count < 0) or (idx < self.frame_count), \
                "idxs: {} > total valid frames({})".format(idxs, self.frame_count)
            if pre_idx != (idx - 1):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            res, frame = self.cap.read() # in BGR/GRAY format
            pre_idx = idx
            if not res:
                logging.warning("VideoIter:: >> failed to grab frame {} from `{}'".format(idx, self.vid_path))
                return None
            # Convert to RGB
            if len(frame.shape) < 3:
                if force_color:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        return frames

    def close(self):
        if self.cap is not None:
            self.cap.release()
        self.reset()
        return self



class VideoIter(data.Dataset):

    def __init__(self,
                 video_prefix,
                 txt_list,
                 sampler,
                 video_transform=None,
                 force_color=True,
                 name="<NO_NAME>",
                 cache_root="./.cache",
                 shuffle_list_seed=None,
                 list_repeat_times=1,
                 return_item_subpath=False,
                 tolerant_corrupted_video=True):
        super(VideoIter, self).__init__()
        # get video interpreter
        self.video = Video()
        # load params
        self.sampler = sampler
        self.force_color = force_color
        self.video_prefix = video_prefix
        self.video_transform = video_transform
        self.return_item_subpath = return_item_subpath
        self.backup_item = None
        self.tolerant_corrupted_video = tolerant_corrupted_video
        self.rng = np.random.RandomState(shuffle_list_seed if shuffle_list_seed else 0)
        # load video list
        video_raw_list = self._get_video_list(video_prefix=video_prefix,
                                              txt_list=txt_list,
                                              cache_root=cache_root)
        video_list = []
        for v_id, label, vid_subpath, frame_count in video_raw_list:
            for i_copy in range(list_repeat_times):
                video_list.append([v_id, i_copy, label, vid_subpath, frame_count])
        self.video_list = video_list
        if list_repeat_times > 1:
            logging.warning("VideoIter:: >> repeat sample list by {} times, {} samples to run".format(list_repeat_times, len(self.video_list)))
        if shuffle_list_seed is not None:
            self.rng.shuffle(self.video_list)
        logging.info("VideoIter:: iterator initialized (phase: '{:s}', num: {:d})".format(name, len(self.video_list)))

    def getitem_from_raw_video(self, index):
        # get current video info
        v_id, copy_id, label, vid_subpath, frame_count = self.video_list[index]
        video_path = os.path.join(self.video_prefix, vid_subpath)

        video = None
        successful = False
        try:
            video = self.video.open(vid_path=video_path)
            # cache video info for faster speed
            if frame_count < 0:
                frame_count = video.count_frames()
                if frame_count < 1:
                    raise IOError("video ({}) does not have any frames".format(video_path))
                self.video_list[index][-1] = frame_count
            # extract frames, try 5 times
            num_attempt = 5
            for i_trial in range(0, num_attempt):
                try:
                    # dynamic sampling
                    sampled_idxs = self.sampler.sampling(range_max=frame_count, v_id=v_id, copy_id=copy_id)
                    # extracting frames
                    sampled_frames = video.extract_frames(idxs=sampled_idxs, force_color=self.force_color)
                    break
                except IOError as e:
                    # raise error at the last time
                    if i_trial == (num_attempt-1):
                        raise e
            successful = sampled_frames is not None
        except IOError as e:
            if video:
                video.close()
            logging.warning("VideoIter:: >> I/O error({0}): {1}".format(e.errno, e.strerror))

        if not successful:
            assert (self.backup_item is not None), "Sampling failed, backup inavailable. Terminated!"
            logging.warning("VideoIter:: >> sampling failed, use backup item!")
            video = self.video.open(vid_path=self.backup_item['video_path'])
            sampled_frames = video.extract_frames(idxs=self.backup_item['sampled_idxs'])
        elif self.tolerant_corrupted_video:
            if (self.backup_item is None) or (self.rng.rand() < 0.1):
                self.backup_item = {'video_path': video_path, 'sampled_idxs': sampled_idxs}

        video.close()

        # [(H, W, C), ...] -> (H, W, N*C)
        video_clip = np.concatenate(sampled_frames, axis=2)

        # apply video augmentation
        if self.video_transform is not None:
            if v_id % 100 == 0:
                self.video_transform.set_random_state(seed=v_id+int(time.time()))
            video_clip = self.video_transform(video_clip, idx=v_id, copy_id=copy_id)
        return video_clip, label, vid_subpath


    def __getitem__(self, index):
        succ = False
        while not succ:
            try:
                video_clip, label, vid_subpath = self.getitem_from_raw_video(index)
                succ = True
            except Exception as e:
                logging.error("VideoIter:: UNEXCEPTED ERROR!! <- {}\n{}".format(self.video_list[index], e))
                index = int(self.rng.uniform(0, self.__len__()))
                time.sleep(0.2)

        if self.return_item_subpath:
            return video_clip, label, vid_subpath
        else:
            return video_clip, label


    def __len__(self):
        return len(self.video_list)


    def _get_video_list(self,
                        video_prefix,
                        txt_list,
                        cache_root=None):
        # formate:
        # [vid, label, video_subpath, num_frames]
        assert os.path.exists(video_prefix), "VideoIter:: failed to locate: `{}'".format(video_prefix)
        assert os.path.exists(txt_list), "VideoIter:: failed to locate: `{}'".format(txt_list)
        if cache_root:
            checked_list = os.path.join(cache_root, os.path.basename(txt_list))
            if os.path.exists(checked_list):
                # load checked_list
                video_list = []
                with open(checked_list) as f:
                    lines = f.read().splitlines()
                    logging.info("VideoIter:: found {} videos in `{}'".format(len(lines), checked_list))
                    for i, line in enumerate(lines):
                        v_id, label, video_subpath = line.split()
                        info = [int(v_id), int(label), video_subpath, -1]
                        video_list.append(info)
                return video_list

        # load list and check if file exists
        video_list = []
        with open(txt_list) as f:
            lines = f.read().splitlines()
            logging.info("VideoIter:: found {} videos in `{}'".format(len(lines), txt_list))
            for i, line in enumerate(lines):
                v_id, label, video_subpath = line.split()
                video_path = os.path.join(video_prefix, video_subpath)
                if not os.path.exists(video_path):
                    # logging.warning("VideoIter:: >> cannot locate `{}'".format(video_path))
                    continue
                info = [int(v_id), int(label), video_subpath, -1]
                video_list.append(info)

        # caching video list
        if cache_root and len(video_list) > 0:
            if not os.path.exists(cache_root):
                os.makedirs(cache_root)
            logging.info("VideoIter:: {} videos are found, caching the valid list to: {}".format(len(video_list), checked_list))
            with open(checked_list, 'w') as f:
                for i, (v_id, label, video_subpath, _) in enumerate(video_list):
                    if i > 0:
                        f.write("\n")
                    f.write("{:d}\t{:d}\t{:s}".format(v_id, label, video_subpath))

        return video_list


if __name__ == "__main__":

    import pdb
    import time

    import video_sampler as sampler
    import video_transforms as transforms

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    logging.getLogger().setLevel(logging.DEBUG)

    is_color = True
    normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])

    data_root = '../dataset/Kinetics'

    logging.info("Testing VideoIter without transformer [not torch wapper]")


    # has memory (no metter random or not, it will trival all none overlapped clips)
    clip_length = 8
    interval = 8
    train_sampler = sampler.RandomSampling(num=clip_length,
                                           interval=interval,
                                           speed=[1.0, 1.0],
                                           seed=0)

    # train_dataset = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data', 'val_x288p-GOPxN'),
    train_dataset = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data', 'val_x288p-GOPxN'),
                              txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'kinetics_val_avi.txt'),
                              sampler=train_sampler,
                              force_color=True,
                              video_transform=transforms.Compose([
                                         # transforms.CropScale(crop_size=(224, 224),
                                         #                      crop_type='random_crop',
                                         #                      make_square=False,
                                         #                      aspect_ratio=(1., 1.),
                                         #                      slen=(224, 320)),
                                         transforms.RandomScale(make_square=False,
                                                                aspect_ratio=[0.7, 1./.7],
                                                                slen=[224, 320]),
                                         transforms.RandomCrop((224, 224)), # insert a resize if needed
                                         transforms.RandomHorizontalFlip(),
                                         # transforms.PixelJitter(vars=[-20, 20]), # too slow
                                         # transforms.RandomHLS(vars=[10, 25, 15]), # too slow
                                         transforms.ToTensor(),
                                         # normalize
                                     ],
                                     aug_seed=1),
                              name='debug',
                              shuffle_list_seed=2,
                              )

    '''
    for i in range(1, 2000):
        # logging.info("video id: {}".format(i))
        img, lab = train_dataset.__getitem__(i)
        logging.info("{}: {}".format(i, img.shape))
    '''

    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=1, shuffle=True,
                                               num_workers=12, pin_memory=True)

    logging.info("Start iter")
    tic = time.time()
    for i, (img, lab) in enumerate(train_loader):
        t = time.time() - tic
        logging.info("{} samples/sec. \t img.shape = {}, label = {}.".format(float(i+1)/t, img.shape, lab))
    

    '''
    import matplotlib.pyplot as plt
     
    for vid in range(24):

        img, lab = train_dataset.__getitem__(1)
        img = np.clip(img, 0., 1.)
        logging.info(img.shape)
        for i in range(0, clip_length):
            plt.imshow(img.numpy()[:,i,:,:].transpose(1,2,0))
            plt.draw()
            plt.pause(0.2)
    
    plt.pause(1)
    '''
