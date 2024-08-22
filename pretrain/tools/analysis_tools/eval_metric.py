# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import mmengine
import rich
from mmengine import DictAction
from mmengine.evaluator import Evaluator

from mmpretrain.registry import METRICS

HELP_URL = (
    'https://mmpretrain.readthedocs.io/en/latest/useful_tools/'
    'log_result_analysis.html#how-to-conduct-offline-metric-evaluation')

prog_description = f"""\
Evaluate metric of the results saved in pkl format.

The detailed usage can be found in {HELP_URL}
"""


def parse_args():
    parser = argparse.ArgumentParser(description=prog_description)
    parser.add_argument('pkl_results', help='Results in pickle format')
    parser.add_argument(
        '--metric',
        nargs='+',
        action='append',
        dest='metric_options',
        help='The metric config, the key-value pair in xxx=yyy format will be '
        'parsed as the metric config items. You can specify multiple metrics '
        'by use multiple `--metric`. For list type value, you can use '
        '"key=[a,b]" or "key=a,b", and it also allows nested list/tuple '
        'values, e.g. "key=[(a,b),(c,d)]".')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.metric_options is None:
        raise ValueError('Please speicfy at least one `--metric`. '
                         f'The detailed usage can be found in {HELP_URL}')

    test_metrics = []
    for metric_option in args.metric_options:
        metric_cfg = {}
        for kv in metric_option:
            k, v = kv.split('=', maxsplit=1)
            metric_cfg[k] = DictAction._parse_iterable(v)
        test_metrics.append(METRICS.build(metric_cfg))

    predictions = mmengine.load(args.pkl_results)

    evaluator = Evaluator(test_metrics)
    eval_results = evaluator.offline_evaluate(predictions, None)
    rich.print(eval_results)


if __name__ == '__main__':
    main()
