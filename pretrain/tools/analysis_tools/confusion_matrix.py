# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import tempfile

import mmengine
from mmengine.config import Config, DictAction
from mmengine.evaluator import Evaluator
from mmengine.runner import Runner

from mmpretrain.evaluation import ConfusionMatrix
from mmpretrain.registry import DATASETS
from mmpretrain.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(
        description='Eval a checkpoint and draw the confusion matrix.')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'ckpt_or_result',
        type=str,
        help='The checkpoint file (.pth) or '
        'dumpped predictions pickle file (.pkl).')
    parser.add_argument('--out', help='the file to save the confusion matrix.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to display the metric result by matplotlib if supports.')
    parser.add_argument(
        '--show-path', type=str, help='Path to save the visualization image.')
    parser.add_argument(
        '--include-values',
        action='store_true',
        help='To draw the values in the figure.')
    parser.add_argument(
        '--cmap',
        type=str,
        default='viridis',
        help='The color map to use. Defaults to "viridis".')
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
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # register all modules in mmpretrain into the registries
    # do not init the default scope here because it will be init in the runner
    register_all_modules(init_default_scope=False)

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.ckpt_or_result.endswith('.pth'):
        # Set confusion matrix as the metric.
        cfg.test_evaluator = dict(type='ConfusionMatrix')

        cfg.load_from = str(args.ckpt_or_result)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.work_dir = tmpdir
            runner = Runner.from_cfg(cfg)
            classes = runner.test_loop.dataloader.dataset.metainfo.get(
                'classes')
            cm = runner.test()['confusion_matrix/result']
    else:
        predictions = mmengine.load(args.ckpt_or_result)
        evaluator = Evaluator(ConfusionMatrix())
        metrics = evaluator.offline_evaluate(predictions, None)
        cm = metrics['confusion_matrix/result']
        try:
            # Try to build the dataset.
            dataset = DATASETS.build({
                **cfg.test_dataloader.dataset, 'pipeline': []
            })
            classes = dataset.metainfo.get('classes')
        except Exception:
            classes = None

    if args.out is not None:
        mmengine.dump(cm, args.out)

    if args.show or args.show_path is not None:
        fig = ConfusionMatrix.plot(
            cm,
            show=args.show,
            classes=classes,
            include_values=args.include_values,
            cmap=args.cmap)
        if args.show_path is not None:
            fig.savefig(args.show_path)
            print(f'The confusion matrix is saved at {args.show_path}.')


if __name__ == '__main__':
    main()
