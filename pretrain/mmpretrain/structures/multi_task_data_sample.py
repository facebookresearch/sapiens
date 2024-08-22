# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from mmengine.structures import BaseDataElement


class MultiTaskDataSample(BaseDataElement):

    @property
    def tasks(self):
        return self._data_fields
