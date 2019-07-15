# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import numpy as np

import torch
import logging
class Compose(object):
    """Composes several video_transforms together.

    Args:
        transforms (List[Transform]): list of transforms to compose.

    Example:
        >>> video_transforms.Compose([
        >>>     video_transforms.CenterCrop(10),
        >>>     video_transforms.ToTensor(),
        >>> ])
    """
    def __init__(self, transforms, aug_seed=0):
        self.transforms = transforms
        self.set_random_state(aug_seed)

    def __call__(self, data, idx=None, copy_id=0):
        for t in self.transforms:
            data = t(data, idx, copy_id)
        return data

    def set_random_state(self, seed=None):
        for i, t in enumerate(self.transforms):
            t.set_random_state(seed=(seed+i))

class Transform(object):
    """basse class for all transformation"""
    def set_random_state(self, seed=None):
        self.rng = np.random.RandomState(seed)


####################################
# Customized Transformations
####################################

class Normalize(Transform):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor, idx=None, copy_id=0):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

class CropScale(Transform):
    """Combine Crop and scale operation for 
    faster speed & less memory cost
    """
    def __init__(self, crop_size,
                       crop_type='random_crop',
                       make_square=False,
                       aspect_ratio=[1.0, 1.0],
                       slen=[224, 288],
                       interpolation=cv2.INTER_LINEAR):
        # random scale
        assert slen[1] >= slen[0], \
                "slen ({}) should be in increase order".format(scale)
        assert aspect_ratio[1] >= aspect_ratio[0], \
                "aspect_ratio ({}) should be in increase order".format(aspect_ratio)
        self.slen = slen # [min factor, max factor]
        self.aspect_ratio = None if aspect_ratio[0] * aspect_ratio[1] == 1. else aspect_ratio
        self.make_square = make_square

        # random crop
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = crop_size
        if crop_type == 'random_crop':
            self.func_crop = self.random_crop
        elif crop_type == 'center_crop':
            self.func_crop = self.center_crop
        elif crop_type == 'mixed_crop':
            self.func_crop = self.mixed_crop
        else:
            raise NotImplementedError

        # others
        self.interpolation = interpolation
        self.rng = np.random.RandomState(0)

    def random_scale(self, h, w):
        # rescale image
        new_w = w
        new_h = h if not self.make_square else w
        if self.aspect_ratio:
            random_aspect_ratio = 0.5 * (self.rng.uniform(self.aspect_ratio[0], self.aspect_ratio[1])
                                       + self.rng.uniform(self.aspect_ratio[0], self.aspect_ratio[1]))
            if self.rng.rand() > 0.5:
                random_aspect_ratio = 1.0 / random_aspect_ratio
            new_w *= random_aspect_ratio
            new_h /= random_aspect_ratio
        resize_factor = self.rng.uniform(self.slen[0], self.slen[1]) / min(new_w, new_h)
        new_w *= resize_factor
        new_h *= resize_factor
        return (new_h, new_w)

    def random_crop(self, h, w, idx=None, copy_id=0):
        th, tw = self.crop_size
        x1 = self.rng.uniform(0, w - tw)
        y1 = self.rng.uniform(0, h - th)
        return (y1, y1+th, x1, x1+tw)

    def center_crop(self, h, w, idx=None, copy_id=0):
        th, tw = self.crop_size
        x1 = 0.5 * (w - tw)
        y1 = 0.5 * (h - th)
        return (y1, y1+th, x1, x1+tw)

    def mixed_crop(self, h, w, idx=None, copy_id=0):
        if copy_id == 0:
            x1 = 0.5 * (w - tw)
            y1 = 0.5 * (h - th)
        else:
            x1 = self.rng.uniform(0, w - tw)
            y1 = self.rng.uniform(0, h - th)
        return (y1, y1+th, x1, x1+tw)

    def __call__(self, data, idx=None, copy_id=0):
        h, w, c = data.shape
        new_h, new_w = self.random_scale(h, w)
        # compute roi
        new_y1, new_y2, new_x1, new_x2 = self.func_crop(new_h, new_w, idx=idx, copy_id=copy_id)
        # map to orginal image
        y1 = int((new_y1 * h) / new_h)
        y2 = int((new_y2 * h) / new_h)
        x1 = int((new_x1 * w) / new_w)
        x2 = int((new_x2 * w) / new_w)
        # excute
        cropped_data = data[y1:y2, x1:x2, :]
        output = cv2.resize(cropped_data, self.crop_size, self.interpolation)
        return output

class Resize(Transform):
    """ Rescales the input numpy array to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: cv2.INTER_LINEAR
    """
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        self.size = size # [w, h]
        self.interpolation = interpolation

    def __call__(self, data, idx=None, copy_id=0):
        h, w, c = data.shape

        if isinstance(self.size, int):
            slen = self.size
            if min(w, h) == slen:
                return data
            if w < h:
                new_w = self.size
                new_h = int(self.size * h / w)
            else:
                new_w = int(self.size * w / h)
                new_h = self.size
        else:
            new_w = self.size[0]
            new_h = self.size[1]

        if (h != new_h) or (w != new_w):
            scaled_data = cv2.resize(data, (new_w, new_h), self.interpolation)
        else:
            scaled_data = data

        return scaled_data


class RandomScale(Transform):
    """ Rescales the input numpy array to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: cv2.INTER_LINEAR
    """
    def __init__(self, make_square=False,
                       aspect_ratio=[1.0, 1.0],
                       slen=[224, 288],
                       interpolation=cv2.INTER_LINEAR):
        assert slen[1] >= slen[0], \
                "slen ({}) should be in increase order".format(scale)
        assert aspect_ratio[1] >= aspect_ratio[0], \
                "aspect_ratio ({}) should be in increase order".format(aspect_ratio)
        self.slen = slen # [min factor, max factor]
        self.aspect_ratio = aspect_ratio
        self.make_square = make_square
        self.interpolation = interpolation
        self.rng = np.random.RandomState(0)

    def __call__(self, data, idx=None, copy_id=0):
        h, w, c = data.shape
        new_w = w
        new_h = h if not self.make_square else w
        if self.aspect_ratio:
            random_aspect_ratio = 0.5 * (self.rng.uniform(self.aspect_ratio[0], self.aspect_ratio[1])
                                       + self.rng.uniform(self.aspect_ratio[0], self.aspect_ratio[1]))
            if self.rng.rand() > 0.5:
                random_aspect_ratio = 1.0 / random_aspect_ratio
            new_w *= random_aspect_ratio
            new_h /= random_aspect_ratio
        resize_factor = self.rng.uniform(self.slen[0], self.slen[1]) / min(new_w, new_h)
        new_w *= resize_factor
        new_h *= resize_factor
        scaled_data = cv2.resize(data, (int(new_w+1), int(new_h+1)), self.interpolation)
        return scaled_data

class CenterCrop(Transform):
    """Crops the given numpy array at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, data, idx=None, copy_id=0):
        h, w, c = data.shape
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        cropped_data = data[y1:(y1+th), x1:(x1+tw), :]
        return cropped_data

class MixedCrop(Transform):
    """Crops the given numpy array at the random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.rng = np.random.RandomState(0)

    def __call__(self, data, idx=None, copy_id=0):
        h, w, c = data.shape
        th, tw = self.size
        if copy_id == 0:
            x1 = int(round(0.5 * float(w - tw)))
            y1 = int(round(0.5 * float(h - th)))
        else:
            x1 = self.rng.choice(range(w - tw + 1))
            y1 = self.rng.choice(range(h - th + 1))
        cropped_data = data[y1:(y1+th), x1:(x1+tw), :]
        return cropped_data

class RandomCrop(Transform):
    """Crops the given numpy array at the random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.rng = np.random.RandomState(0)

    def __call__(self, data, idx=None, copy_id=0):
        h, w, c = data.shape
        th, tw = self.size
        x1 = self.rng.choice(range(w - tw + 1))
        y1 = self.rng.choice(range(h - th + 1))
        cropped_data = data[y1:(y1+th), x1:(x1+tw), :]
        return cropped_data

class EvenlyHorizontalFlip(Transform):
    """Randomly horizontally flips the given numpy array with a probability of 0.5
    """
    def __init__(self, num_repeat):
        self.num_repeat = num_repeat

    def __call__(self, data, idx=None, copy_id=0):
        if not copy_id < self.num_repeat:
            data = np.fliplr(data)
            data = np.ascontiguousarray(data)
        return data

class RandomHorizontalFlip(Transform):
    """Randomly horizontally flips the given numpy array with a probability of 0.5
    """
    def __init__(self):
        self.rng = np.random.RandomState(0)

    def __call__(self, data, idx=None, copy_id=0):
        if self.rng.rand() < 0.5:
            data = np.fliplr(data)
            data = np.ascontiguousarray(data)
        return data

class RandomVerticalFlip(Transform):
    """Randomly vertically flips the given numpy array with a probability of 0.5
    """
    def __init__(self):
        self.rng = np.random.RandomState(0)

    def __call__(self, data, idx=None, copy_id=0):
        if self.rng.rand() < 0.5:
            data = np.flipud(data)
            data = np.ascontiguousarray(data)
        return data

class PixelJitter(Transform):
    def __init__(self, vars=[-20, 20]):
        self.vars = vars
        self.rng = np.random.RandomState(0)

    def __call__(self, data, idx=None, copy_id=0):
        h, w, c = data.shape
        low = self.rng.uniform(self.vars[0], 0)
        high = self.rng.uniform(0, self.vars[1])
        random_pixel = self.rng.uniform(low=low, high=high, size=(h,w,c)).astype(np.float32)
        augmented_data = data + random_pixel
        return augmented_data

class RandomRGB(Transform):
    def __init__(self, vars=[10, 10, 10]):
        self.vars = vars
        self.rng = np.random.RandomState(0)

    def __call__(self, data, idx=None, copy_id=0):
        h, w, c = data.shape

        random_vars = [int(round(self.rng.uniform(-x, x))) for x in self.vars]

        base = len(random_vars)
        augmented_data = np.zeros(data.shape)
        for ic in range(0, c):
            var = random_vars[ic%base]
            augmented_data[:,:,ic] = np.minimum(np.maximum(data[:,:,ic] + var, 0), 255)
        return augmented_data

class RandomHLS(Transform):
    def __init__(self, vars=[15, 35, 25]):
        self.vars = vars
        self.rng = np.random.RandomState(0)

    def __call__(self, data, idx=None, copy_id=0):
        h, w, c = data.shape
        assert c%3 == 0, "input channel = %d, illegal"%c

        random_vars = [int(self.rng.uniform(-x, x)) for x in self.vars]

        base = len(random_vars)
        augmented_data = np.zeros(data.shape, )

        for i_im in range(0, int(c/3)):
            augmented_data[:,:,3*i_im:(3*i_im+3)] = \
                    cv2.cvtColor(data[:,:,3*i_im:(3*i_im+3)], cv2.COLOR_RGB2HLS)

        hls_limits = [180, 255, 255]
        for ic in range(0, c):
            var = random_vars[ic%base]
            limit = hls_limits[ic%base]
            augmented_data[:,:,ic] = np.minimum(np.maximum(augmented_data[:,:,ic] + var, 0), limit)

        for i_im in range(0, int(c/3)):
            augmented_data[:,:,3*i_im:(3*i_im+3)] = \
                    cv2.cvtColor(augmented_data[:,:,3*i_im:(3*i_im+3)].astype(np.uint8), \
                        cv2.COLOR_HLS2RGB)

        return augmented_data


class ToTensor(Transform):
    """Converts a numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __init__(self, dim=3):
        self.dim = dim

    def __call__(self, image, idx=None, copy_id=0):
        if isinstance(image, np.ndarray):
            # H, W, C = image.shape
            # handle numpy array
            image = torch.from_numpy(image.transpose((2, 0, 1)))
            # backward compatibility
            return image.float() / 255.
