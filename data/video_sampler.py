# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np

import logging

class RandomSampling(object):
    def __init__(self, num, interval=1, speed=[1.0, 1.0], seed=0):
        assert num > 0, "at least sampling 1 frame"
        self.num = num
        self.interval = interval if type(interval) == list else [interval]
        self.speed = speed
        self.rng = np.random.RandomState(seed)

    def sampling(self, range_max, v_id=None, copy_id=None):
        assert range_max > 0, \
            ValueError("range_max = {}".format(range_max))
        interval = self.rng.choice(self.interval)
        if self.num == 1:
            return [self.rng.choice(range(0, range_max))]
        # sampling
        speed_min = self.speed[0]
        speed_max = min(self.speed[1], (range_max-1)/((self.num-1)*interval))
        if speed_max < speed_min:
            return (list(range(0, range_max))*self.num)[0:self.num]
            # return [self.rng.choice(range(0, range_max))] * self.num
        random_interval = self.rng.uniform(speed_min, speed_max) * interval
        frame_range = (self.num-1) * random_interval
        clip_start = self.rng.uniform(0, (range_max-1) - frame_range)
        clip_end = clip_start + frame_range
        return np.linspace(clip_start, clip_end, self.num).astype(dtype=np.int).tolist()

class EvenlySampling(object):
    def __init__(self, num, interval=1, num_times=1, seed=0):
        self.num = num
        self.interval = interval
        self.num_times = num_times
        self.rng = np.random.RandomState(seed)

    def sampling(self, range_max, v_id, copy_id):
        assert range_max > 0, \
            ValueError("range_max = {}".format(range_max))
        num = self.num
        num_times = self.num_times
        interval = self.interval
        frame_range = (num - 1) * interval + 1
        # sampling clips
        if frame_range > range_max:
            return (list(range(0, range_max))*self.num)[0:self.num]
            # return [self.rng.choice(range(0, range_max))] * self.num
        if range_max-frame_range*num_times == 0:
            clips = [x*frame_range for x in range(0,num_times)]
        elif range_max-frame_range*num_times > 0:
            step_size = (range_max-frame_range*num_times)/float(num_times+1)+frame_range
            clips = [math.ceil(x*step_size-frame_range) for x in range(1, num_times+1)]
        else:
            step_size = (range_max-frame_range*num_times)/float(num_times-1)+frame_range
            clips = [int(x*step_size) for x in range(0, num_times)] 
        # pickup a clip
        cursor = copy_id % len(clips)
        # sampling within clip
        # logging.info("v_id: {}, cursor: {}, start: {}".format(v_id, cursor, clips[cursor]))
        idxs = range(clips[cursor], clips[cursor]+frame_range, interval)
        return idxs

if __name__ == "__main__":

    import logging
    logging.getLogger().setLevel(logging.DEBUG)

    """ test RandomSampling() """
    num = 8
    interval = 8
    random_sampler = RandomSampling(num=num, interval=interval, speed=[1, 1])

    logging.info("RandomSampling(): range_max < num")
    for i in range(10):
        logging.info("{:d}: {}".format(i, random_sampler.sampling(range_max=2, v_id=1)))

    logging.info("RandomSampling(): range_max == num")
    for i in range(10):
        logging.info("{:d}: {}".format(i, random_sampler.sampling(range_max=num*interval, v_id=1)))

    logging.info("RandomSampling(): range_max > num")
    for i in range(90):
        logging.info("{:d}: {}".format(i, random_sampler.sampling(range_max=num*interval+4, v_id=1)))


    """ test SequentialSampling() """
    # evenly_sampler = EvenlySampling(num=2, interval=1, num_times=2, fix_cursor=False)

    # logging.info("SequentialSampling():")
    # for i in range(14):
    #    logging.info("{:d}: v_id = {}: {}".format(i, 0, list(evenly_sampler.sampling(range_max=4, v_id=0))))
    #    # logging.info("{:d}: v_id = {}: {}".format(i, 1, evenly_sampler.sampling(range_max=9, v_id=1)))
    #    # logging.info("{:d}: v_id = {}: {}".format(i, 2, evenly_sampler.sampling(range_max=2, v_id=2)))
    #    # logging.info("{:d}: v_id = {}: {}".format(i, 3, evenly_sampler.sampling(range_max=3, v_id=3)))

