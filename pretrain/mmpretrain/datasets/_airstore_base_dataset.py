# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import functools
import gc
import logging
import pickle
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from torch.utils.data import Dataset, IterableDataset

from mmengine.fileio import join_path, list_from_file, load
from mmengine.logging import print_log
from mmengine.registry import TRANSFORMS
from mmengine.utils import is_abs


class Compose:
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict, callable], optional): Sequence of transform
            object or config dict to be composed.
    """

    def __init__(self, transforms: Optional[Sequence[Union[dict, Callable]]]):
        self.transforms: List[Callable] = []

        if transforms is None:
            transforms = []

        for transform in transforms:
            # `Compose` can be built with config dict with type and
            # corresponding arguments.
            if isinstance(transform, dict):
                transform = TRANSFORMS.build(transform)
                if not callable(transform):
                    raise TypeError(f'transform should be a callable object, '
                                    f'but got {type(transform)}')
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(
                    f'transform must be a callable object or dict, '
                    f'but got {type(transform)}')

    def __call__(self, data: dict) -> Optional[dict]:
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """
        for t in self.transforms:
            data = t(data)
            # The transform will return None when it failed to load images or
            # cannot find suitable augmentation parameters to augment the data.
            # Here we simply return None if the transform returns None and the
            # dataset will handle it by randomly selecting another data sample.
            if data is None:
                return None
        return data

    def __repr__(self):
        """Print ``self.transforms`` in sequence.

        Returns:
            str: Formatted string.
        """
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


def force_full_init(old_func: Callable) -> Any:
    """Those methods decorated by ``force_full_init`` will be forced to call
    ``full_init`` if the instance has not been fully initiated.

    Args:
        old_func (Callable): Decorated function, make sure the first arg is an
            instance with ``full_init`` method.

    Returns:
        Any: Depends on old_func.
    """

    @functools.wraps(old_func)
    def wrapper(obj: object, *args, **kwargs):
        # The instance must have `full_init` method.
        if not hasattr(obj, 'full_init'):
            raise AttributeError(f'{type(obj)} does not have full_init '
                                 'method.')
        # If instance does not have `_fully_initialized` attribute or
        # `_fully_initialized` is False, call `full_init` and set
        # `_fully_initialized` to True
        if not getattr(obj, '_fully_initialized', False):
            print_log(
                f'Attribute `_fully_initialized` is not defined in '
                f'{type(obj)} or `type(obj)._fully_initialized is '
                'False, `full_init` will be called and '
                f'{type(obj)}._fully_initialized will be set to True',
                logger='current',
                level=logging.WARNING)
            obj.full_init()  # type: ignore
            obj._fully_initialized = True  # type: ignore

        return old_func(obj, *args, **kwargs)

    return wrapper

## iterable dataset
class _AirstoreBaseDataset(IterableDataset):
    r"""BaseDataset for open source projects in OpenMMLab.

    The annotation format is shown as follows.

    .. code-block:: none

        {
            "metainfo":
            {
              "dataset_type": "test_dataset",
              "task_name": "test_task"
            },
            "data_list":
            [
              {
                "img_path": "test_img.jpg",
                "height": 604,
                "width": 640,
                "instances":
                [
                  {
                    "bbox": [0, 0, 10, 20],
                    "bbox_label": 1,
                    "mask": [[0,0],[0,10],[10,20],[20,0]],
                    "extra_anns": [1,2,3]
                  },
                  {
                    "bbox": [10, 10, 110, 120],
                    "bbox_label": 2,
                    "mask": [[10,10],[10,110],[110,120],[120,10]],
                    "extra_anns": [4,5,6]
                  }
                ]
              },
            ]
        }

    Args:
        ann_file (str, optional): Annotation file path. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        data_prefix (dict): Prefix for training data. Defaults to
            dict(img_path='').
        filter_cfg (dict, optional): Config for filter data. Defaults to None.
        indices (int or Sequence[int], optional): Support using first few
            data in annotation file to facilitate training/testing on a smaller
        serialize_data (bool, optional): Whether to hold memory using
            serialized objects, when enabled, data loader workers can use
            shared RAM from master process instead of making a copy. Defaults
            to True.
        pipeline (list, optional): Processing pipeline. Defaults to [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Defaults to False.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. In some cases, such as visualization, only the meta
            information of the dataset is needed, which is not necessary to
            load annotation file. ``Basedataset`` can skip load annotations to
            save time by set ``lazy_init=True``. Defaults to False.
        max_refetch (int, optional): If ``Basedataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Defaults to 1000.

    Note:
        BaseDataset collects meta information from ``annotation file`` (the
        lowest priority), ``BaseDataset.METAINFO``(medium) and ``metainfo
        parameter`` (highest) passed to constructors. The lower priority meta
        information will be overwritten by higher one.

    Note:
        Dataset wrapper such as ``ConcatDataset``, ``RepeatDataset`` .etc.
        should not inherit from ``BaseDataset`` since ``get_subset`` and
        ``get_subset_`` could produce ambiguous meaning sub-dataset which
        conflicts with original dataset.

    Examples:
        >>> # Assume the annotation file is given above.
        >>> class CustomDataset(BaseDataset):
        >>>     METAINFO: dict = dict(task_name='custom_task',
        >>>                           dataset_type='custom_type')
        >>> metainfo=dict(task_name='custom_task_name')
        >>> custom_dataset = CustomDataset(
        >>>                      'path/to/ann_file',
        >>>                      metainfo=metainfo)
        >>> # meta information of annotation file will be overwritten by
        >>> # `CustomDataset.METAINFO`. The merged meta information will
        >>> # further be overwritten by argument `metainfo`.
        >>> custom_dataset.metainfo
        {'task_name': custom_task_name, dataset_type: custom_type}
    """

    METAINFO: dict = dict()
    _fully_initialized: bool = False

    def __init__(self,
                 ann_file: Optional[str] = '',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = '',
                 data_prefix: dict = dict(img_path=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000):
        self.ann_file = ann_file
        self._metainfo = self._load_metainfo(copy.deepcopy(metainfo))
        self.data_root = data_root
        self.data_prefix = copy.copy(data_prefix)
        self.filter_cfg = copy.deepcopy(filter_cfg)
        self._indices = indices
        self.serialize_data = serialize_data
        self.test_mode = test_mode
        self.max_refetch = max_refetch
        self.data_list: List[dict] = []
        self.data_bytes: np.ndarray

        # Join paths.
        self._join_prefix()

        # Build pipeline.
        self.pipeline = Compose(pipeline)
        # Full initialize the dataset.
        if not lazy_init:
            self.full_init()

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index and automatically call ``full_init`` if the
        dataset has not been fully initialized.

        Args:
            idx (int): The index of data.

        Returns:
            dict: The idx-th annotation of the dataset.
        """
        if self.serialize_data:
            start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
            end_addr = self.data_address[idx].item()
            bytes = memoryview(
                self.data_bytes[start_addr:end_addr])  # type: ignore
            data_info = pickle.loads(bytes)  # type: ignore
        else:
            data_info = copy.deepcopy(self.data_list[idx])
        # Some codebase needs `sample_idx` of data information. Here we convert
        # the idx to a positive number and save it in data information.
        if idx >= 0:
            data_info['sample_idx'] = idx
        else:
            data_info['sample_idx'] = len(self) + idx

        return data_info

    def full_init(self):
        """Load annotation file and set ``BaseDataset._fully_initialized`` to
        True.

        If ``lazy_init=False``, ``full_init`` will be called during the
        instantiation and ``self._fully_initialized`` will be set to True. If
        ``obj._fully_initialized=False``, the class method decorated by
        ``force_full_init`` will call ``full_init`` automatically.

        Several steps to initialize annotation:

            - load_data_list: Load annotations from annotation file.
            - filter data information: Filter annotations according to
              filter_cfg.
            - slice_data: Slice dataset according to ``self._indices``
            - serialize_data: Serialize ``self.data_list`` if
              ``self.serialize_data`` is True.
        """
        if self._fully_initialized:
            return
        # load data information
        self.data_list = self.load_data_list()
        # filter illegal data, such as data that has no annotations.
        self.data_list = self.filter_data()
        # Get subset data according to indices.
        if self._indices is not None:
            self.data_list = self._get_unserialized_subset(self._indices)

        # serialize data_list
        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()

        self._fully_initialized = True

    @property
    def metainfo(self) -> dict:
        """Get meta information of dataset.

        Returns:
            dict: meta information collected from ``BaseDataset.METAINFO``,
            annotation file and metainfo argument during instantiation.
        """
        return copy.deepcopy(self._metainfo)

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        This method should return dict or list of dict. Each dict or list
        contains the data information of a training sample. If the protocol of
        the sample annotations is changed, this function can be overridden to
        update the parsing logic while keeping compatibility.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            list or list[dict]: Parsed annotation.
        """
        for prefix_key, prefix in self.data_prefix.items():
            assert prefix_key in raw_data_info, (
                f'raw_data_info: {raw_data_info} dose not contain prefix key'
                f'{prefix_key}, please check your data_prefix.')
            raw_data_info[prefix_key] = join_path(prefix,
                                                  raw_data_info[prefix_key])
        return raw_data_info

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg. Defaults return all
        ``data_list``.

        If some ``data_list`` could be filtered according to specific logic,
        the subclass should override this method.

        Returns:
            list[int]: Filtered results.
        """
        return self.data_list

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category ids by index. Dataset wrapped by ClassBalancedDataset
        must implement this method.

        The ``ClassBalancedDataset`` requires a subclass which implements this
        method.

        Args:
            idx (int): The index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """
        raise NotImplementedError(f'{type(self)} must implement `get_cat_ids` '
                                  'method')

    # def __getitem__(self, idx: int) -> dict:
    #     """Get the idx-th image and data information of dataset after
    #     ``self.pipeline``, and ``full_init`` will be called if the dataset has
    #     not been fully initialized.

    #     During training phase, if ``self.pipeline`` get ``None``,
    #     ``self._rand_another`` will be called until a valid image is fetched or
    #      the maximum limit of refetech is reached.

    #     Args:
    #         idx (int): The index of self.data_list.

    #     Returns:
    #         dict: The idx-th image and data information of dataset after
    #         ``self.pipeline``.
    #     """
    #     # Performing full initialization by calling `__getitem__` will consume
    #     # extra memory. If a dataset is not fully initialized by setting
    #     # `lazy_init=True` and then fed into the dataloader. Different workers
    #     # will simultaneously read and parse the annotation. It will cost more
    #     # time and memory, although this may work. Therefore, it is recommended
    #     # to manually call `full_init` before dataset fed into dataloader to
    #     # ensure all workers use shared RAM from master process.
    #     if not self._fully_initialized:
    #         print_log(
    #             'Please call `full_init()` method manually to accelerate '
    #             'the speed.',
    #             logger='current',
    #             level=logging.WARNING)
    #         self.full_init()

    #     if self.test_mode:
    #         data = self.prepare_data(idx)
    #         if data is None:
    #             raise Exception('Test time pipline should not get `None` '
    #                             'data_sample')
    #         return data

    #     for _ in range(self.max_refetch + 1):
    #         data = self.prepare_data(idx)
    #         # Broken images or random augmentations may cause the returned data
    #         # to be None
    #         if data is None:
    #             idx = self._rand_another()
    #             continue
    #         return data

    #     raise Exception(f'Cannot find valid image after {self.max_refetch}! '
    #                     'Please check your image path and pipeline')

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        If the annotation file does not follow `OpenMMLab 2.0 format dataset
        <https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html>`_ .
        The subclass must override this method for load annotations. The meta
        information of annotation file will be overwritten :attr:`METAINFO`
        and ``metainfo`` argument of constructor.

        Returns:
            list[dict]: A list of annotation.
        """  # noqa: E501
        # `self.ann_file` denotes the absolute annotation file path if
        # `self.root=None` or relative path if `self.root=/path/to/data/`.
        annotations = load(self.ann_file)
        if not isinstance(annotations, dict):
            raise TypeError(f'The annotations loaded from annotation file '
                            f'should be a dict, but got {type(annotations)}!')
        if 'data_list' not in annotations or 'metainfo' not in annotations:
            raise ValueError('Annotation must have data_list and metainfo '
                             'keys')
        metainfo = annotations['metainfo']
        raw_data_list = annotations['data_list']

        # Meta information load from annotation file will not influence the
        # existed meta information load from `BaseDataset.METAINFO` and
        # `metainfo` arguments defined in constructor.
        for k, v in metainfo.items():
            self._metainfo.setdefault(k, v)

        # load and parse data_infos.
        data_list = []
        for raw_data_info in raw_data_list:
            # parse raw data information to target format
            data_info = self.parse_data_info(raw_data_info)
            if isinstance(data_info, dict):
                # For image tasks, `data_info` should information if single
                # image, such as dict(img_path='xxx', width=360, ...)
                data_list.append(data_info)
            elif isinstance(data_info, list):
                # For video tasks, `data_info` could contain image
                # information of multiple frames, such as
                # [dict(video_path='xxx', timestamps=...),
                #  dict(video_path='xxx', timestamps=...)]
                for item in data_info:
                    if not isinstance(item, dict):
                        raise TypeError('data_info must be list of dict, but '
                                        f'got {type(item)}')
                data_list.extend(data_info)
            else:
                raise TypeError('data_info should be a dict or list of dict, '
                                f'but got {type(data_info)}')

        return data_list

    @classmethod
    def _load_metainfo(cls, metainfo: dict = None) -> dict:
        """Collect meta information from the dictionary of meta.

        Args:
            metainfo (dict): Meta information dict. If ``metainfo``
                contains existed filename, it will be parsed by
                ``list_from_file``.

        Returns:
            dict: Parsed meta information.
        """
        # avoid `cls.METAINFO` being overwritten by `metainfo`
        cls_metainfo = copy.deepcopy(cls.METAINFO)
        if metainfo is None:
            return cls_metainfo
        if not isinstance(metainfo, dict):
            raise TypeError(
                f'metainfo should be a dict, but got {type(metainfo)}')

        for k, v in metainfo.items():
            if isinstance(v, str):
                # If type of value is string, and can be loaded from
                # corresponding backend. it means the file name of meta file.
                try:
                    cls_metainfo[k] = list_from_file(v)
                except (TypeError, FileNotFoundError):
                    print_log(
                        f'{v} is not a meta file, simply parsed as meta '
                        'information',
                        logger='current',
                        level=logging.WARNING)
                    cls_metainfo[k] = v
            else:
                cls_metainfo[k] = v
        return cls_metainfo

    def _join_prefix(self):
        """Join ``self.data_root`` with ``self.data_prefix`` and
        ``self.ann_file``.

        Examples:
            >>> # self.data_prefix contains relative paths
            >>> self.data_root = 'a/b/c'
            >>> self.data_prefix = dict(img='d/e/')
            >>> self.ann_file = 'f'
            >>> self._join_prefix()
            >>> self.data_prefix
            dict(img='a/b/c/d/e')
            >>> self.ann_file
            'a/b/c/f'
            >>> # self.data_prefix contains absolute paths
            >>> self.data_root = 'a/b/c'
            >>> self.data_prefix = dict(img='/d/e/')
            >>> self.ann_file = 'f'
            >>> self._join_prefix()
            >>> self.data_prefix
            dict(img='/d/e')
            >>> self.ann_file
            'a/b/c/f'
        """
        # Automatically join annotation file path with `self.root` if
        # `self.ann_file` is not an absolute path.
        if self.ann_file and not is_abs(self.ann_file) and self.data_root:
            self.ann_file = join_path(self.data_root, self.ann_file)
        # Automatically join data directory with `self.root` if path value in
        # `self.data_prefix` is not an absolute path.
        for data_key, prefix in self.data_prefix.items():
            if not isinstance(prefix, str):
                raise TypeError('prefix should be a string, but got '
                                f'{type(prefix)}')
            if not is_abs(prefix) and self.data_root:
                self.data_prefix[data_key] = join_path(self.data_root, prefix)
            else:
                self.data_prefix[data_key] = prefix

    @force_full_init
    def get_subset_(self, indices: Union[Sequence[int], int]) -> None:
        """The in-place version of ``get_subset`` to convert dataset to a
        subset of original dataset.

        This method will convert the original dataset to a subset of dataset.
        If type of indices is int, ``get_subset_`` will return a subdataset
        which contains the first or last few data information according to
        indices is positive or negative. If type of indices is a sequence of
        int, the subdataset will extract the data information according to
        the index given in indices.

        Examples:
              >>> dataset = BaseDataset('path/to/ann_file')
              >>> len(dataset)
              100
              >>> dataset.get_subset_(90)
              >>> len(dataset)
              90
              >>> # if type of indices is sequence, extract the corresponding
              >>> # index data information
              >>> dataset.get_subset_([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
              >>> len(dataset)
              10
              >>> dataset.get_subset_(-3)
              >>> len(dataset) # Get the latest few data information.
              3

        Args:
            indices (int or Sequence[int]): If type of indices is int, indices
                represents the first or last few data of dataset according to
                indices is positive or negative. If type of indices is
                Sequence, indices represents the target data information
                index of dataset.
        """
        # Get subset of data from serialized data or data information sequence
        # according to `self.serialize_data`.
        if self.serialize_data:
            self.data_bytes, self.data_address = \
                self._get_serialized_subset(indices)
        else:
            self.data_list = self._get_unserialized_subset(indices)

    @force_full_init
    def get_subset(self, indices: Union[Sequence[int], int]) -> 'BaseDataset':
        """Return a subset of dataset.

        This method will return a subset of original dataset. If type of
        indices is int, ``get_subset_`` will return a subdataset which
        contains the first or last few data information according to
        indices is positive or negative. If type of indices is a sequence of
        int, the subdataset will extract the information according to the index
        given in indices.

        Examples:
              >>> dataset = BaseDataset('path/to/ann_file')
              >>> len(dataset)
              100
              >>> subdataset = dataset.get_subset(90)
              >>> len(sub_dataset)
              90
              >>> # if type of indices is list, extract the corresponding
              >>> # index data information
              >>> subdataset = dataset.get_subset([0, 1, 2, 3, 4, 5, 6, 7,
              >>>                                  8, 9])
              >>> len(sub_dataset)
              10
              >>> subdataset = dataset.get_subset(-3)
              >>> len(subdataset) # Get the latest few data information.
              3

        Args:
            indices (int or Sequence[int]): If type of indices is int, indices
                represents the first or last few data of dataset according to
                indices is positive or negative. If type of indices is
                Sequence, indices represents the target data information
                index of dataset.

        Returns:
            BaseDataset: A subset of dataset.
        """
        # Get subset of data from serialized data or data information list
        # according to `self.serialize_data`. Since `_get_serialized_subset`
        # will recalculate the subset data information,
        # `_copy_without_annotation` will copy all attributes except data
        # information.
        sub_dataset = self._copy_without_annotation()
        # Get subset of dataset with serialize and unserialized data.
        if self.serialize_data:
            data_bytes, data_address = \
                self._get_serialized_subset(indices)
            sub_dataset.data_bytes = data_bytes.copy()
            sub_dataset.data_address = data_address.copy()
        else:
            data_list = self._get_unserialized_subset(indices)
            sub_dataset.data_list = copy.deepcopy(data_list)
        return sub_dataset

    def _get_serialized_subset(self, indices: Union[Sequence[int], int]) \
            -> Tuple[np.ndarray, np.ndarray]:
        """Get subset of serialized data information list.

        Args:
            indices (int or Sequence[int]): If type of indices is int,
                indices represents the first or last few data of serialized
                data information list. If type of indices is Sequence, indices
                represents the target data information index which consist of
                subset data information.

        Returns:
            Tuple[np.ndarray, np.ndarray]: subset of serialized data
            information.
        """
        sub_data_bytes: Union[List, np.ndarray]
        sub_data_address: Union[List, np.ndarray]
        if isinstance(indices, int):
            if indices >= 0:
                assert indices < len(self.data_address), \
                    f'{indices} is out of dataset length({len(self)}'
                # Return the first few data information.
                end_addr = self.data_address[indices - 1].item() \
                    if indices > 0 else 0
                # Slicing operation of `np.ndarray` does not trigger a memory
                # copy.
                sub_data_bytes = self.data_bytes[:end_addr]
                # Since the buffer size of first few data information is not
                # changed,
                sub_data_address = self.data_address[:indices]
            else:
                assert -indices <= len(self.data_address), \
                    f'{indices} is out of dataset length({len(self)}'
                # Return the last few data information.
                ignored_bytes_size = self.data_address[indices - 1]
                start_addr = self.data_address[indices - 1].item()
                sub_data_bytes = self.data_bytes[start_addr:]
                sub_data_address = self.data_address[indices:]
                sub_data_address = sub_data_address - ignored_bytes_size
        elif isinstance(indices, Sequence):
            sub_data_bytes = []
            sub_data_address = []
            for idx in indices:
                assert len(self) > idx >= -len(self)
                start_addr = 0 if idx == 0 else \
                    self.data_address[idx - 1].item()
                end_addr = self.data_address[idx].item()
                # Get data information by address.
                sub_data_bytes.append(self.data_bytes[start_addr:end_addr])
                # Get data information size.
                sub_data_address.append(end_addr - start_addr)
            # Handle indices is an empty list.
            if sub_data_bytes:
                sub_data_bytes = np.concatenate(sub_data_bytes)
                sub_data_address = np.cumsum(sub_data_address)
            else:
                sub_data_bytes = np.array([])
                sub_data_address = np.array([])
        else:
            raise TypeError('indices should be a int or sequence of int, '
                            f'but got {type(indices)}')
        return sub_data_bytes, sub_data_address  # type: ignore

    def _get_unserialized_subset(self, indices: Union[Sequence[int],
                                                      int]) -> list:
        """Get subset of data information list.

        Args:
            indices (int or Sequence[int]): If type of indices is int,
                indices represents the first or last few data of data
                information. If type of indices is Sequence, indices represents
                the target data information index which consist of subset data
                information.

        Returns:
            Tuple[np.ndarray, np.ndarray]: subset of data information.
        """
        if isinstance(indices, int):
            if indices >= 0:
                # Return the first few data information.
                sub_data_list = self.data_list[:indices]
            else:
                # Return the last few data information.
                sub_data_list = self.data_list[indices:]
        elif isinstance(indices, Sequence):
            # Return the data information according to given indices.
            sub_data_list = []
            for idx in indices:
                sub_data_list.append(self.data_list[idx])
        else:
            raise TypeError('indices should be a int or sequence of int, '
                            f'but got {type(indices)}')
        return sub_data_list

    def _serialize_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Serialize ``self.data_list`` to save memory when launching multiple
        workers in data loading. This function will be called in ``full_init``.

        Hold memory using serialized objects, and data loader workers can use
        shared RAM from master process instead of making a copy.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Serialized result and corresponding
            address.
        """

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=4)
            return np.frombuffer(buffer, dtype=np.uint8)

        # Serialize data information list avoid making multiple copies of
        # `self.data_list` when iterate `import torch.utils.data.dataloader`
        # with multiple workers.
        data_list = [_serialize(x) for x in self.data_list]
        address_list = np.asarray([len(x) for x in data_list], dtype=np.int64)
        data_address: np.ndarray = np.cumsum(address_list)
        # TODO Check if np.concatenate is necessary
        data_bytes = np.concatenate(data_list)
        # Empty cache for preventing making multiple copies of
        # `self.data_info` when loading data multi-processes.
        self.data_list.clear()
        gc.collect()
        return data_bytes, data_address

    def _rand_another(self) -> int:
        """Get random index.

        Returns:
            int: Random index from 0 to ``len(self)-1``
        """
        return np.random.randint(0, len(self))

    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = self.get_data_info(idx)
        return self.pipeline(data_info)

    @force_full_init
    def __len__(self) -> int:
        """Get the length of filtered dataset and automatically call
        ``full_init`` if the  dataset has not been fully init.

        Returns:
            int: The length of filtered dataset.
        """
        if self.serialize_data:
            return len(self.data_address)
        else:
            return len(self.data_list)

    def _copy_without_annotation(self, memo=dict()) -> 'BaseDataset':
        """Deepcopy for all attributes other than ``data_list``,
        ``data_address`` and ``data_bytes``.

        Args:
            memo: Memory dict which used to reconstruct complex object
                correctly.
        """
        cls = self.__class__
        other = cls.__new__(cls)
        memo[id(self)] = other

        for key, value in self.__dict__.items():
            if key in ['data_list', 'data_address', 'data_bytes']:
                continue
            super(BaseDataset, other).__setattr__(key,
                                                  copy.deepcopy(value, memo))

        return other
