import random
from functools import partial
from typing import Tuple

import numpy as np
import torch
from torchvision.transforms import ColorJitter, RandomAffine, RandomCrop
from torchvision.transforms import functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)

        if len(target) == 1:
            return image[0], target[0]

        return image, target


class Transform:
    def __init__(self):
        pass

    def listfy(self, image, target):
        if not isinstance(image, list):
            image = [image]

        if not isinstance(target, list):
            target = [target]

        assert len(image) == len(target), "Image and target must have the same length."

        return image, target

    def apply_transform(self, image, target, transfrom_function):
        images_t = []
        targets_t = []

        for i, t in zip(image, target):
            i, t = transfrom_function(i, t)
            images_t.append(i)
            targets_t.append(t)

        return images_t, targets_t


class Resize(Transform):
    def __init__(self, size: tuple[int, int], interpolation=F.InterpolationMode.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image, target):
        image, target = self.listfy(image, target)
        return self.apply_transform(image, target, self.transform_function)

    def transform_function(self, image, target):
        image = F.resize(image, self.size, self.interpolation, antialias=True)

        # if target is not none
        if target is not None:
            unsqueeze_needed = True if target.ndim == 2 else False

            if unsqueeze_needed:
                target = torch.unsqueeze(target, 0)  # needed to make resize

            target = F.resize(target, self.size, interpolation=F.InterpolationMode.NEAREST)

            if unsqueeze_needed:
                target = torch.squeeze(target)

        return image, target


class ToTensor(Transform):
    def __call__(self, image, target):
        image, target = self.listfy(image, target)
        return self.apply_transform(image, target, self.transform_function)

    def transform_function(self, image, target):
        image = F.to_tensor(image)

        if target is not None:
            target = torch.as_tensor(np.array(target), dtype=torch.int64)

        return image, target


class MutualColorJitter(Transform):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.jitter = ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, image, target):
        image, target = self.listfy(image, target)
        return self.apply_transform(image, target, self.transform_function)

    def transform_function(self, image, target):
        image = self.jitter(image)

        return image, target


class Normalize(Transform):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image, target = self.listfy(image, target)
        return self.apply_transform(image, target, self.transform_function)

    def transform_function(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class MutualRandomCrop(Transform):
    def __init__(self, size, cat_max_ratio=1.0, ignore_index=255):
        """Randomly crop image and target to a specific size.

        Args:
            size (tuple): Desired output size of the crop.
            cat_max_ratio (float): Maximum ratio of pixels with the same category index.
            ignore_index (int): Category index to be ignored.
        """
        self.size = size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = 255

    def __call__(self, image, target):
        image, target = self.listfy(image, target)
        params_crop = RandomCrop.get_params(image[0], self.size)

        if self.cat_max_ratio < 1.0:
            # find target segmentation with the most unique labels
            num_unique_labels_in_targets = [len(torch.unique(t)) for t in target]
            id_target = num_unique_labels_in_targets.index(max(num_unique_labels_in_targets))
            target_with_most_labels = target[
                id_target
            ]  # needed as we work with gt masks with one default value sometimes

            for _ in range(10):
                seg_tmp = F.crop(target_with_most_labels, *params_crop)
                labels, counts = torch.unique(seg_tmp, return_counts=True)
                counts = counts[labels != self.ignore_index]
                if len(counts) > 0 and counts.max() / counts.sum().float() < self.cat_max_ratio:
                    break
                params_crop = RandomCrop.get_params(image[0], self.size)

        return self.apply_transform(image, target, partial(self.transform_function, params_crop=params_crop))

    def transform_function(self, image, target, params_crop):
        image = F.crop(image, *params_crop)

        if target is not None:
            target = F.crop(target, *params_crop)

        return image, target


class RandomScale(Transform):
    def __init__(self, scale_range: Tuple):
        self.scale_range = scale_range

    def __call__(self, image, target):
        image, target = self.listfy(image, target)
        params_affine = RandomAffine.get_params((0, 0), None, self.scale_range, None, image[0].size())
        return self.apply_transform(image, target, partial(self.transform_function, params_affine=params_affine))

    def transform_function(self, image, target, params_affine):
        image = F.affine(image, *params_affine, interpolation=F.InterpolationMode.BILINEAR, fill=None)

        if target is not None:
            target = target.unsqueeze(0)  # needed to make affine work
            target = F.affine(target, *params_affine, interpolation=F.InterpolationMode.NEAREST, fill=255)
            target = target.squeeze(0)

        return image, target


class RandomHorizontalFlip(Transform):
    def __init__(self, prob: float):
        self.prob = prob

        assert self.prob >= 0 and self.prob <= 1.0, "Probability must be in the range of (0,1)!"

    def __call__(self, image, target):
        image, target = self.listfy(image, target)

        if self.prob > random.random():
            return self.apply_transform(image, target, self.transform_function)

        return image, target

    def transform_function(self, image, target):
        image = F.hflip(image)

        if target is not None:
            target = F.hflip(target)

        return image, target


class CenterCrop(Transform):
    def __init__(self, size):
        """
        Args:
            size (sequence or int): Desired output size of the crop. If size is an
                int instead of sequence like (h, w), a square crop (size, size) is
                made.
        """
        self.size = size

    def __call__(self, image, target):
        image, target = self.listfy(image, target)
        return self.apply_transform(image, target, partial(self.transform_function))

    def transform_function(self, image, target):
        image = F.center_crop(image, self.size)

        if target is not None:
            target = F.center_crop(target, self.size)

        return image, target


class Misalignment(Transform):
    def __init__(self, degrees_rotation: float, translate_range: Tuple, scale_range: Tuple):
        """Applies random affine transformation to image and target.

        Same transformation is applied to both. BUT DIFFERENT if there is more than one pair ->
        simulated misalignment
        """
        self.degrees_rotation = degrees_rotation
        self.translate_range = translate_range
        self.scale_range = scale_range

    def __call__(self, image, target):
        image, target = self.listfy(image, target)

        return self.apply_transform(image, target, partial(self.transform_function))

    def transform_function(self, image, target):
        params_affine = RandomAffine.get_params(
            self.degrees_rotation, self.translate_range, self.scale_range, None, image[0].size()
        )

        image = F.affine(image, *params_affine, interpolation=F.InterpolationMode.BILINEAR, fill=0)

        if target is not None:
            target = target.unsqueeze(0)  # needed to make affine work
            target = F.affine(target, *params_affine, interpolation=F.InterpolationMode.NEAREST, fill=255)
            target = target.squeeze(0)

        return image, target
