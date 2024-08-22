# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Sequence, Union

import numpy as np
from mmengine.dataset import Compose
from mmengine.model import BaseModel

ImageType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]


def _prepare_data(imgs: ImageType, model: BaseModel):

    cfg = model.cfg
    for t in cfg.test_pipeline:
        if t.get('type') == 'LoadAnnotations':
            cfg.test_pipeline.remove(t)

    is_batch = True
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
        is_batch = False

    if isinstance(imgs[0], np.ndarray):
        cfg.test_pipeline[0]['type'] = 'LoadImageFromNDArray'

    # TODO: Consider using the singleton pattern to avoid building
    # a pipeline for each inference
    pipeline = Compose(cfg.test_pipeline)

    ## test time augmentation
    if cfg.get('inference_tta_pipeline') is not None:
        pipeline = Compose(cfg.inference_tta_pipeline)

    data = defaultdict(list)
    for img in imgs:
        if isinstance(img, np.ndarray):
            data_ = dict(img=img)
        else:
            data_ = dict(img_path=img)
        data_ = pipeline(data_)

        if cfg.get('inference_tta_pipeline') is not None:
            data = data_
        else:
            data['inputs'].append(data_['inputs'])
            data['data_samples'].append(data_['data_samples'])

    return data, is_batch


def _stereo_pointmap_prepare_data(imgs1: ImageType, imgs2: ImageType, model: BaseModel):

    cfg = model.cfg
    for t in cfg.test_pipeline:
        if t.get('type') == 'LoadAnnotations':
            cfg.test_pipeline.remove(t)

    is_batch = True
    if not isinstance(imgs1, (list, tuple)):
        imgs1 = [imgs1]
        imgs2 = [imgs2]
        is_batch = False

    if isinstance(imgs1[0], np.ndarray):
        cfg.test_pipeline[0]['type'] = 'LoadImageFromNDArray'

    # a pipeline for each inference
    pipeline = Compose(cfg.test_pipeline)

    data = defaultdict(list)
    for img1, img2 in zip(imgs1, imgs2):

        if isinstance(img1, np.ndarray):
            data1_ = dict(img=img1)
        else:
            data1_ = dict(img_path=img1)

        if isinstance(img2, np.ndarray):
            data2_ = dict(img=img2)
        else:
            data2_ = dict(img_path=img2)

        data1_['is_anchor'] = True
        data2_['is_anchor'] = False

        data1_ = pipeline(data1_)
        data2_ = pipeline(data2_)

        data_ = {}
        for key, val in data1_.items():
            data_[key] = val

        for key, val in data2_.items():
            data_[key] = val

        data['inputs1'].append(data_['inputs1'])
        data['inputs2'].append(data_['inputs2'])

        data['data_samples1'].append(data_['data_samples1'])
        data['data_samples2'].append(data_['data_samples2'])

    return data, is_batch
