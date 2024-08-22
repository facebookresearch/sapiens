# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
import os
import os.path as osp
from typing import List, Sequence

import numpy as np
import torch
from mmengine.dist.utils import get_rank
from mmengine.evaluator import BaseMetric

from mmpretrain.registry import METRICS


@METRICS.register_module()
class ShapeBiasMetric(BaseMetric):
    """Evaluate the model on ``cue_conflict`` dataset.

    This module will evaluate the model on an OOD dataset, cue_conflict, in
    order to measure the shape bias of the model. In addition to compuate the
    Top-1 accuracy, this module also generate a csv file to record the
    detailed prediction results, such that this csv file can be used to
    generate the shape bias curve.

    Args:
        csv_dir (str): The directory to save the csv file.
        model_name (str): The name of the csv file. Please note that the
            model name should be an unique identifier.
        dataset_name (str): The name of the dataset. Default: 'cue_conflict'.
    """

    # mapping several classes from ImageNet-1K to the same category
    airplane_indices = [404]
    bear_indices = [294, 295, 296, 297]
    bicycle_indices = [444, 671]
    bird_indices = [
        8, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 22, 23, 24, 80, 81, 82, 83,
        87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100, 127, 128, 129,
        130, 131, 132, 133, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
        145
    ]
    boat_indices = [472, 554, 625, 814, 914]
    bottle_indices = [440, 720, 737, 898, 899, 901, 907]
    car_indices = [436, 511, 817]
    cat_indices = [281, 282, 283, 284, 285, 286]
    chair_indices = [423, 559, 765, 857]
    clock_indices = [409, 530, 892]
    dog_indices = [
        152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165,
        166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
        180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 193, 194,
        195, 196, 197, 198, 199, 200, 201, 202, 203, 205, 206, 207, 208, 209,
        210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
        224, 225, 226, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238,
        239, 240, 241, 243, 244, 245, 246, 247, 248, 249, 250, 252, 253, 254,
        255, 256, 257, 259, 261, 262, 263, 265, 266, 267, 268
    ]
    elephant_indices = [385, 386]
    keyboard_indices = [508, 878]
    knife_indices = [499]
    oven_indices = [766]
    truck_indices = [555, 569, 656, 675, 717, 734, 864, 867]

    def __init__(self,
                 csv_dir: str,
                 model_name: str,
                 dataset_name: str = 'cue_conflict',
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.categories = sorted([
            'knife', 'keyboard', 'elephant', 'bicycle', 'airplane', 'clock',
            'oven', 'chair', 'bear', 'boat', 'cat', 'bottle', 'truck', 'car',
            'bird', 'dog'
        ])
        self.csv_dir = csv_dir
        self.model_name = model_name
        self.dataset_name = dataset_name
        if get_rank() == 0:
            self.csv_path = self.create_csv()

    def process(self, data_batch, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            result = dict()
            if 'pred_score' in data_sample:
                result['pred_score'] = data_sample['pred_score'].cpu()
            else:
                result['pred_label'] = data_sample['pred_label'].cpu()
            result['gt_label'] = data_sample['gt_label'].cpu()
            result['gt_category'] = data_sample['img_path'].split('/')[-2]
            result['img_name'] = data_sample['img_path'].split('/')[-1]

            aggregated_category_probabilities = []
            # get the prediction for each category of current instance
            for category in self.categories:
                category_indices = getattr(self, f'{category}_indices')
                category_probabilities = torch.gather(
                    result['pred_score'], 0,
                    torch.tensor(category_indices)).mean()
                aggregated_category_probabilities.append(
                    category_probabilities)
            # sort the probabilities in descending order
            pred_indices = torch.stack(aggregated_category_probabilities
                                       ).argsort(descending=True).numpy()
            result['pred_category'] = np.take(self.categories, pred_indices)

            # Save the result to `self.results`.
            self.results.append(result)

    def create_csv(self) -> str:
        """Create a csv file to store the results."""
        session_name = 'session-1'
        csv_path = osp.join(
            self.csv_dir, self.dataset_name + '_' + self.model_name + '_' +
            session_name + '.csv')
        if osp.exists(csv_path):
            os.remove(csv_path)
        directory = osp.dirname(csv_path)
        if not osp.exists(directory):
            os.makedirs(directory, exist_ok=True)
        with open(csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow([
                'subj', 'session', 'trial', 'rt', 'object_response',
                'category', 'condition', 'imagename'
            ])
        return csv_path

    def dump_results_to_csv(self, results: List[dict]) -> None:
        """Dump the results to a csv file.

        Args:
            results (List[dict]): A list of results.
        """
        for i, result in enumerate(results):
            img_name = result['img_name']
            category = result['gt_category']
            condition = 'NaN'
            with open(self.csv_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.model_name, 1, i + 1, 'NaN',
                    result['pred_category'][0], category, condition, img_name
                ])

    def compute_metrics(self, results: List[dict]) -> dict:
        """Compute the metrics from the results.

        Args:
            results (List[dict]): A list of results.

        Returns:
            dict: A dict of metrics.
        """
        if get_rank() == 0:
            self.dump_results_to_csv(results)
        metrics = dict()
        metrics['accuracy/top1'] = np.mean([
            result['pred_category'][0] == result['gt_category']
            for result in results
        ])

        return metrics
