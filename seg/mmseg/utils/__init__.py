# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# yapf: disable
from .class_names import (ade_classes, ade_palette, bdd100k_classes,
                          bdd100k_palette, cityscapes_classes,
                          cityscapes_palette, cocostuff_classes,
                          cocostuff_palette, dataset_aliases, get_classes,
                          get_palette, isaid_classes, isaid_palette,
                          loveda_classes, loveda_palette, potsdam_classes,
                          potsdam_palette, stare_classes, stare_palette,
                          synapse_classes, synapse_palette, vaihingen_classes,
                          vaihingen_palette, voc_classes, voc_palette)
# yapf: enable
from .collect_env import collect_env
from .get_templates import get_predefined_templates
from .io import datafrombytes
from .misc import add_prefix, stack_batch
from .misc import stack_batch_stereo_pointmap, stack_batch_stereo_correspondences
from .set_env import register_all_modules
from .tokenizer import tokenize
from .typing_utils import (ConfigType, ForwardResults, MultiConfig,
                           OptConfigType, OptMultiConfig, OptSampleList,
                           SampleList, TensorDict, TensorList)

# isort: off
from .mask_classification import MatchMasks, seg_data_to_instance_data
from .nearest_neighbours import bruteforce_reciprocal_nns, bruteforce_nn_ratio

__all__ = [
    'collect_env',
    'register_all_modules',
    'stack_batch',
    'add_prefix',
    'ConfigType',
    'OptConfigType',
    'MultiConfig',
    'OptMultiConfig',
    'SampleList',
    'OptSampleList',
    'TensorDict',
    'TensorList',
    'ForwardResults',
    'cityscapes_classes',
    'ade_classes',
    'voc_classes',
    'cocostuff_classes',
    'loveda_classes',
    'potsdam_classes',
    'vaihingen_classes',
    'isaid_classes',
    'stare_classes',
    'cityscapes_palette',
    'ade_palette',
    'voc_palette',
    'cocostuff_palette',
    'loveda_palette',
    'potsdam_palette',
    'vaihingen_palette',
    'isaid_palette',
    'stare_palette',
    'dataset_aliases',
    'get_classes',
    'get_palette',
    'datafrombytes',
    'synapse_palette',
    'synapse_classes',
    'get_predefined_templates',
    'tokenize',
    'seg_data_to_instance_data',
    'MatchMasks',
    'bdd100k_classes',
    'bdd100k_palette',
]
