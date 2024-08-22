# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Callable, List, Union

from mmcv.transforms import BaseTransform, Compose

from mmpretrain.registry import TRANSFORMS

# Define type of transform or transform config
Transform = Union[dict, Callable[[dict], dict]]


@TRANSFORMS.register_module()
class MultiView(BaseTransform):
    """A transform wrapper for multiple views of an image.

    Args:
        transforms (list[dict | callable], optional): Sequence of transform
            object or config dict to be wrapped.
        mapping (dict): A dict that defines the input key mapping.
            The keys corresponds to the inner key (i.e., kwargs of the
            ``transform`` method), and should be string type. The values
            corresponds to the outer keys (i.e., the keys of the
            data/results), and should have a type of string, list or dict.
            None means not applying input mapping. Default: None.
        allow_nonexist_keys (bool): If False, the outer keys in the mapping
            must exist in the input data, or an exception will be raised.
            Default: False.

    Examples:
        >>> # Example 1: MultiViews 1 pipeline with 2 views
        >>> pipeline = [
        >>>     dict(type='MultiView',
        >>>         num_views=2,
        >>>         transforms=[
        >>>             [
        >>>                dict(type='Resize', scale=224))],
        >>>         ])
        >>> ]
        >>> # Example 2: MultiViews 2 pipelines, the first with 2 views,
        >>> # the second with 6 views
        >>> pipeline = [
        >>>     dict(type='MultiView',
        >>>         num_views=[2, 6],
        >>>         transforms=[
        >>>             [
        >>>                dict(type='Resize', scale=224)],
        >>>             [
        >>>                dict(type='Resize', scale=224),
        >>>                dict(type='RandomSolarize')],
        >>>         ])
        >>> ]
    """

    def __init__(self, transforms: List[List[Transform]],
                 num_views: Union[int, List[int]]) -> None:

        if isinstance(num_views, int):
            num_views = [num_views]
        assert isinstance(num_views, List)
        assert len(num_views) == len(transforms)
        self.num_views = num_views

        self.pipelines = []
        for trans in transforms:
            pipeline = Compose(trans)
            self.pipelines.append(pipeline)

        self.transforms = []
        for i in range(len(num_views)):
            self.transforms.extend([self.pipelines[i]] * num_views[i])

    def transform(self, results: dict) -> dict:
        """Apply transformation to inputs.

        Args:
            results (dict): Result dict from previous pipelines.

        Returns:
            dict: Transformed results.
        """
        multi_views_outputs = dict(img=[])
        for trans in self.transforms:
            inputs = copy.deepcopy(results)
            outputs = trans(inputs)

            multi_views_outputs['img'].append(outputs['img'])
        results.update(multi_views_outputs)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__ + '('
        for i, p in enumerate(self.pipelines):
            repr_str += f'\nPipeline {i + 1} with {self.num_views[i]} views:\n'
            repr_str += str(p)
        repr_str += ')'
        return repr_str


@TRANSFORMS.register_module()
class ApplyToList(BaseTransform):
    """A transform wrapper to apply the wrapped transforms to a list of items.
    For example, to load and resize a list of images.

    Args:
        transforms (list[dict | callable]): Sequence of transform config dict
            to be wrapped.
        scatter_key (str): The key to scatter data dict. If the field is a
            list, scatter the list to multiple data dicts to do transformation.
        collate_keys (List[str]): The keys to collate from multiple data dicts.
            The fields in ``collate_keys`` will be composed into a list after
            transformation, and the other fields will be adopted from the
            first data dict.
    """

    def __init__(self, transforms, scatter_key, collate_keys):
        super().__init__()

        self.transforms = Compose([TRANSFORMS.build(t) for t in transforms])
        self.scatter_key = scatter_key
        self.collate_keys = set(collate_keys)
        self.collate_keys.add(self.scatter_key)

    def transform(self, results: dict):
        scatter_field = results.get(self.scatter_key)

        if isinstance(scatter_field, list):
            scattered_results = []
            for item in scatter_field:
                single_results = copy.deepcopy(results)
                single_results[self.scatter_key] = item
                scattered_results.append(self.transforms(single_results))

            final_output = scattered_results[0]

            # merge output list to single output
            for key in scattered_results[0].keys():
                if key in self.collate_keys:
                    final_output[key] = [
                        single[key] for single in scattered_results
                    ]
            return final_output
        else:
            return self.transforms(results)
