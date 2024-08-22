from typing import List, Any, Optional, Union, Iterator, Tuple
import torch
import functools
from mmengine import fileio
from mmengine.logging import MMLogger
import numpy as np
import cv2
import os
import io
import random
import pickle
import torchvision.transforms.v2 as T
from mmpretrain.registry import DATASETS

# from .airstore.base_dataset import AirstoreBaseDataset
from .airstore_base_dataset import AirstoreBaseDataset

## internal tools
from airstore.client.airstore_tabular import AIRStorePathHandler
from iopath.common.file_io import PathManager
from airstore.client.fetch_failure import AirstoreFetchFailureBehavior

from PIL import Image
from pillow_heif import register_heif_opener ## to load heic images

@DATASETS.register_module()
class Instagram(AirstoreBaseDataset):
    METAINFO = {'classes': 'person'}

    def __init__(self,
                 data_root: str = '',
                 airstore_id: str = '',
                 split: str = '',
                 data_prefix: Union[str, dict] = '',
                 ann_file: str = '',
                 serialize_data: bool = False, ## no need to serialize data since we are using airstore
                 metainfo: Optional[dict] = None,
                 **kwargs):

        self.global_rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.batch_size = int(os.environ.get("TRAIN_BATCH_SIZE_PER_GPU", 512))

        self.path_manager = PathManager()
        self.path_manager.register_handler(AIRStorePathHandler())
        self.airstore_id = airstore_id

        self.limit = None
        self._cached_iterator = None

        if split:
            splits = ['train', 'val', 'test']
            assert split in splits, \
                f"The split must be one of {splits}, but get '{split}'"

            if split == 'test':
                logger = MMLogger.get_current_instance()
                logger.info(
                    'Since the Shutterstock test set does not provide label'
                    'annotations, `with_label` is set to False')
                kwargs['with_label'] = False

            data_prefix = split if data_prefix == '' else data_prefix

        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            ann_file=ann_file,
            metainfo=metainfo,
            serialize_data=serialize_data,
            **kwargs)
        return

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [f'Root of dataset: \t{self.data_root}']
        return body

    def _map_item(self, row, mode='BGR') -> Tuple[Any, torch.Tensor]:
        img = np.array(Image.open(io.BytesIO(row["image"])).convert("RGB"))
        if mode == 'BGR':
            img = img[:, :, ::-1]

        data_info = {'img': img, 'img_shape': img.shape[:2], 'ori_shape': img.shape[:2]}

        return self.pipeline(data_info)

    ## this is a dummy list to achieve similar behavior as a index based dataset. This is fundamentally an iter based dataset.
    def __len__(self) -> int:
        total_samples = self.num_global_samples # Total samples in the dataset
        samples_per_gpu = total_samples // self.world_size # Number of samples each GPU will process
        return samples_per_gpu

    def __iter__(self, max_tries=10000):
        if self._cached_iterator is None:
            self._cached_iterator = self._open_iterator()

        try_count = 0
        while True:
            try_count += 1

            try:
                row = next(self._cached_iterator)
                row = self._map_item(row)
                yield row
            except Exception as e:  # General catch if you're unsure about the types of exceptions
                print(f"Unexpected error during data fetch: {e}. Retrying.")
                self._cached_iterator = self._open_iterator()  # Reset the cached iterator if there's an error
                continue
            if try_count >= max_tries:
                print(f'\033[91mMax try count {try_count} is reached during fetch!\033[0m')
                break

    def _open_iterator(self) -> Iterator[Any]:
        # extract numbers of dataloading workers and current worker id (range from
        # 0 to num_workers-1) from torch.utils. If we can't get worker_info we
        # assume the current process is the only dataloading worker.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        # split the dataset for each worker
        airstore_world_size = self.world_size * num_workers
        # each worker take it's split by it's parent process rank and worker id
        airstore_rank = self.global_rank * num_workers + worker_id

        # Randomly sample a seed
        random_seed = random.randint(0, 1000000)

        register_heif_opener()
        return self.path_manager.opent(
            f"airstore://{self.airstore_id}",
            seed=random_seed,
            world_size=airstore_world_size,
            rank=airstore_rank,
            enable_shuffle=True,
            limit=self.limit,
            fetch_failure_behavior=AirstoreFetchFailureBehavior.RETURN_NONE
        )

    @functools.cached_property
    def num_global_samples(self):
        """Returns the total number of samples in the dataset without sharding."""
        return self.path_manager.opent(
            f"airstore://{self.airstore_id}",
            seed=0,
            world_size=1,  # Will retrieve the entire dataset, not for distributed training
            rank=0,
            enable_shuffle=False,
        ).total_size

        import ipdb; ipdb.set_trace()
        return 10

    def load_data_list(self):
        """Load image paths and gt_labels."""
        data_list = []
        total_samples = self.num_global_samples

        print('\033[92m' + 'Rank:{}, World Size:{}, Batch Size:{}, Loading {} instagram dummy samples from airstore'.format(\
            self.global_rank, self.world_size, self.batch_size, total_samples) + '\033[0m')

        # data_list = [{'img_path': '', 'gt_label': int(0)} for _ in range(total_samples)]
        data_list = [None]*total_samples

        assert len(data_list) == total_samples

        print('\033[92m' + 'Done!' + '\033[0m')

        return data_list

    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index and automatically call ``full_init`` if the
        dataset has not been fully initialized.

        Args:
            idx (int): The index of data.

        Returns:
            dict: The idx-th annotation of the dataset.
        """
        data_info = {'img_path': '', 'gt_label': int(0)}

        # Some codebase needs `sample_idx` of data information. Here we convert
        # the idx to a positive number and save it in data information.
        if idx >= 0:
            data_info['sample_idx'] = idx
        else:
            data_info['sample_idx'] = len(self) + idx

        return data_info
