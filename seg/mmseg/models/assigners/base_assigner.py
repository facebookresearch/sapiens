# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABCMeta, abstractmethod
from typing import Optional

from mmengine.structures import InstanceData


class BaseAssigner(metaclass=ABCMeta):
    """Base assigner that assigns masks to ground truth class labels."""

    @abstractmethod
    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               **kwargs):
        """Assign masks to either a ground truth class label or a negative
        label."""
