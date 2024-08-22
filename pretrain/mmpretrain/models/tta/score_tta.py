# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

from mmengine.model import BaseTTAModel

from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample


@MODELS.register_module()
class AverageClsScoreTTA(BaseTTAModel):

    def merge_preds(
        self,
        data_samples_list: List[List[DataSample]],
    ) -> List[DataSample]:
        """Merge predictions of enhanced data to one prediction.

        Args:
            data_samples_list (List[List[DataSample]]): List of predictions
                of all enhanced data.

        Returns:
            List[DataSample]: Merged prediction.
        """
        merged_data_samples = []
        for data_samples in data_samples_list:
            merged_data_samples.append(self._merge_single_sample(data_samples))
        return merged_data_samples

    def _merge_single_sample(self, data_samples):
        merged_data_sample: DataSample = data_samples[0].new()
        merged_score = sum(data_sample.pred_score
                           for data_sample in data_samples) / len(data_samples)
        merged_data_sample.set_pred_score(merged_score)
        return merged_data_sample
