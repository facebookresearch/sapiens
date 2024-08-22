# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os.path as osp
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import rich.progress as progress
import torch
import torch.nn.functional as F
from mmengine.config import Config, DictAction
from mmengine.device import get_device
from mmengine.logging import MMLogger
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist

from mmpretrain.apis import get_model
from mmpretrain.registry import DATASETS

try:
    from sklearn.manifold import TSNE
except ImportError as e:
    raise ImportError('Please install `sklearn` to calculate '
                      'TSNE by `pip install scikit-learn`') from e


def parse_args():
    parser = argparse.ArgumentParser(description='t-SNE visualization')
    parser.add_argument('config', help='tsne config file path')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--test-cfg',
        help='tsne config file path to load config of test dataloader.')
    parser.add_argument(
        '--vis-stage',
        choices=['backbone', 'neck', 'pre_logits'],
        help='The visualization stage of the model')
    parser.add_argument(
        '--class-idx',
        nargs='+',
        type=int,
        help='The categories used to calculate t-SNE.')
    parser.add_argument(
        '--max-num-class',
        type=int,
        default=20,
        help='The first N categories to apply t-SNE algorithms. '
        'Defaults to 20.')
    parser.add_argument(
        '--max-num-samples',
        type=int,
        default=100,
        help='The maximum number of samples per category. '
        'Higher number need longer time to calculate. Defaults to 100.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument('--device', help='Device used for inference')
    parser.add_argument(
        '--legend',
        action='store_true',
        help='Show the legend of all categories.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the result in a graphical window.')

    # t-SNE settings
    parser.add_argument(
        '--n-components', type=int, default=2, help='the dimension of results')
    parser.add_argument(
        '--perplexity',
        type=float,
        default=30.0,
        help='The perplexity is related to the number of nearest neighbors'
        'that is used in other manifold learning algorithms.')
    parser.add_argument(
        '--early-exaggeration',
        type=float,
        default=12.0,
        help='Controls how tight natural clusters in the original space are in'
        'the embedded space and how much space will be between them.')
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=200.0,
        help='The learning rate for t-SNE is usually in the range'
        '[10.0, 1000.0]. If the learning rate is too high, the data may look'
        'like a ball with any point approximately equidistant from its nearest'
        'neighbours. If the learning rate is too low, most points may look'
        'compressed in a dense cloud with few outliers.')
    parser.add_argument(
        '--n-iter',
        type=int,
        default=1000,
        help='Maximum number of iterations for the optimization. Should be at'
        'least 250.')
    parser.add_argument(
        '--n-iter-without-progress',
        type=int,
        default=300,
        help='Maximum number of iterations without progress before we abort'
        'the optimization.')
    parser.add_argument(
        '--init', type=str, default='random', help='The init method')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        work_type = args.config.split('/')[1]
        cfg.work_dir = osp.join('./work_dirs', work_type,
                                osp.splitext(osp.basename(args.config))[0])

    # create work_dir
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    tsne_work_dir = osp.join(cfg.work_dir, f'tsne_{timestamp}/')
    mkdir_or_exist(osp.abspath(tsne_work_dir))

    # init the logger before other steps
    log_file = osp.join(tsne_work_dir, 'tsne.log')
    logger = MMLogger.get_instance(
        'mmpretrain',
        logger_name='mmpretrain',
        log_file=log_file,
        log_level=cfg.log_level)

    # build the model from a config file and a checkpoint file
    device = args.device or get_device()
    model = get_model(cfg, args.checkpoint, device=device)
    logger.info('Model loaded.')

    # build the dataset
    if args.test_cfg is not None:
        dataloader_cfg = Config.fromfile(args.test_cfg).get('test_dataloader')
    elif 'test_dataloader' not in cfg:
        raise ValueError('No `test_dataloader` in the config, you can '
                         'specify another config file that includes test '
                         'dataloader settings by the `--test-cfg` option.')
    else:
        dataloader_cfg = cfg.get('test_dataloader')

    dataset = DATASETS.build(dataloader_cfg.pop('dataset'))
    classes = dataset.metainfo.get('classes')

    if args.class_idx is None:
        num_classes = args.max_num_class if classes is None else len(classes)
        args.class_idx = list(range(num_classes))[:args.max_num_class]

    if classes is not None:
        classes = [classes[idx] for idx in args.class_idx]
    else:
        classes = args.class_idx

    # compress dataset, select that the label is less then max_num_class
    subset_idx_list = []
    counter = defaultdict(int)
    for i in range(len(dataset)):
        gt_label = dataset.get_data_info(i)['gt_label']
        if (gt_label in args.class_idx
                and counter[gt_label] < args.max_num_samples):
            subset_idx_list.append(i)
            counter[gt_label] += 1
    dataset.get_subset_(subset_idx_list)
    logger.info(f'Apply t-SNE to visualize {len(subset_idx_list)} samples.')

    dataloader_cfg.dataset = dataset
    dataloader_cfg.setdefault('collate_fn', dict(type='default_collate'))
    dataloader = Runner.build_dataloader(dataloader_cfg)

    results = dict()
    features = []
    labels = []
    for data in progress.track(dataloader, description='Calculating...'):
        with torch.no_grad():
            # preprocess data
            data = model.data_preprocessor(data)
            batch_inputs, batch_data_samples = \
                data['inputs'], data['data_samples']
            batch_labels = torch.cat([i.gt_label for i in batch_data_samples])

            # extract backbone features
            extract_args = {}
            if args.vis_stage:
                extract_args['stage'] = args.vis_stage
            batch_features = model.extract_feat(batch_inputs, **extract_args)

            # post process
            if batch_features[0].ndim == 4:
                # For (N, C, H, W) feature
                batch_features = [
                    F.adaptive_avg_pool2d(inputs, 1).squeeze()
                    for inputs in batch_features
                ]
            elif batch_features[0].ndim == 3:
                # For (N, L, C) feature
                batch_features = [inputs.mean(1) for inputs in batch_features]

        # save batch features
        features.append(batch_features)
        labels.extend(batch_labels.cpu().numpy())

    for i in range(len(features[0])):
        key = 'feat_' + str(model.backbone.out_indices[i])
        results[key] = np.concatenate(
            [batch[i].cpu().numpy() for batch in features], axis=0)

    # save features
    for key, val in results.items():
        output_file = f'{tsne_work_dir}{key}.npy'
        np.save(output_file, val)

    # build t-SNE model
    tsne_model = TSNE(
        n_components=args.n_components,
        perplexity=args.perplexity,
        early_exaggeration=args.early_exaggeration,
        learning_rate=args.learning_rate,
        n_iter=args.n_iter,
        n_iter_without_progress=args.n_iter_without_progress,
        init=args.init)

    # run and get results
    logger.info('Running t-SNE.')
    for key, val in results.items():
        result = tsne_model.fit_transform(val)
        res_min, res_max = result.min(0), result.max(0)
        res_norm = (result - res_min) / (res_max - res_min)
        _, ax = plt.subplots(figsize=(10, 10))
        scatter = ax.scatter(
            res_norm[:, 0],
            res_norm[:, 1],
            alpha=1.0,
            s=15,
            c=labels,
            cmap='tab20')
        if args.legend:
            legend = ax.legend(scatter.legend_elements()[0], classes)
            ax.add_artist(legend)
        plt.savefig(f'{tsne_work_dir}{key}.png')
        if args.show:
            plt.show()
    logger.info(f'Save features and results to {tsne_work_dir}')


if __name__ == '__main__':
    main()
