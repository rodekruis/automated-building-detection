"""PyTorch-compatible Data Augmentation."""

import sys
import cv2
import torch
import numpy as np
from importlib import import_module


def to_tensor(config, ts, image, mask=None, da=False, resize=False):

    assert len(ts) == 2  # W,H
    assert image is not None

    # Resize, ToTensor and Data Augmentation
    if da:
        assert mask is not None

        try:
            module = import_module("abd_model.da.{}".format(config["train"]["da"]["name"].lower()))
        except:
            sys.exit("Unable to load data augmentation module")

        transform = module.transform(config, image, mask)
        image = cv2.resize(image, ts, interpolation=cv2.INTER_LINEAR) if resize else image
        image = torch.from_numpy(np.moveaxis(transform["image"], 2, 0)).float()
        mask = cv2.resize(mask, ts, interpolation=cv2.INTER_NEAREST) if resize else image
        mask = torch.from_numpy(transform["mask"]).long()
        assert image is not None and mask is not None
        return image, mask

    else:
        image = cv2.resize(image, ts, interpolation=cv2.INTER_LINEAR) if resize else image
        image = torch.from_numpy(np.moveaxis(image, 2, 0)).float()

        if mask is None:
            assert image is not None
            return image

        mask = cv2.resize(mask, ts, interpolation=cv2.INTER_NEAREST) if resize else mask
        mask = torch.from_numpy(mask).long()
        assert image is not None and mask is not None
        return image, mask


def to_tensor_instseg(config, ts, image, mask=None, border_mask=None, da=False, resize=False):

    assert len(ts) == 2  # W,H
    assert image is not None

    # Resize, ToTensor and Data Augmentation
    if da:
        assert mask is not None

        try:
            module = import_module("abd_model.da.{}".format(config["train"]["da"]["name"].lower()))
        except:
            sys.exit("Unable to load data augmentation module")

        transform = module.transform_instseg(config, image, mask, border_mask)
        image = cv2.resize(image, ts, interpolation=cv2.INTER_LINEAR) if resize else image
        image = torch.from_numpy(np.moveaxis(transform["image"], 2, 0)).float()
        mask = cv2.resize(mask, ts, interpolation=cv2.INTER_NEAREST) if resize else image
        mask = torch.from_numpy(transform["mask"]).long()
        border_mask = cv2.resize(border_mask, ts, interpolation=cv2.INTER_NEAREST) if resize else image
        border_mask = torch.from_numpy(transform["border_mask"]).long()
        assert image is not None and mask is not None and border_mask is not None
        return image, mask, border_mask

    else:
        image = cv2.resize(image, ts, interpolation=cv2.INTER_LINEAR) if resize else image
        image = torch.from_numpy(np.moveaxis(image, 2, 0)).float()

        if mask is None:
            assert image is not None
            return image

        mask = cv2.resize(mask, ts, interpolation=cv2.INTER_NEAREST) if resize else mask
        mask = torch.from_numpy(mask).long()

        border_mask = cv2.resize(border_mask, ts, interpolation=cv2.INTER_NEAREST) if resize else border_mask
        border_mask = torch.from_numpy(border_mask).long()

        assert image is not None and mask is not None and border_mask is not None
        return image, mask, border_mask
