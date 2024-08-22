# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import re
from itertools import groupby

import matplotlib.pyplot as plt
import numpy as np

from mmpretrain.utils import load_json_log


def cal_train_time(log_dicts, args):
    """Compute the average time per training iteration."""
    for i, log_dict in enumerate(log_dicts):
        print(f'{"-" * 5}Analyze train time of {args.json_logs[i]}{"-" * 5}')
        train_logs = log_dict['train']

        if 'epoch' in train_logs[0]:
            epoch_ave_times = []
            for _, logs in groupby(train_logs, lambda log: log['epoch']):
                if args.include_outliers:
                    all_time = np.array([log['time'] for log in logs])
                else:
                    all_time = np.array([log['time'] for log in logs])[1:]
                epoch_ave_times.append(all_time.mean())
            epoch_ave_times = np.array(epoch_ave_times)
            slowest_epoch = epoch_ave_times.argmax()
            fastest_epoch = epoch_ave_times.argmin()
            std_over_epoch = epoch_ave_times.std()
            print(f'slowest epoch {slowest_epoch + 1}, '
                  f'average time is {epoch_ave_times[slowest_epoch]:.4f}')
            print(f'fastest epoch {fastest_epoch + 1}, '
                  f'average time is {epoch_ave_times[fastest_epoch]:.4f}')
            print(f'time std over epochs is {std_over_epoch:.4f}')

        avg_iter_time = np.array([log['time'] for log in train_logs]).mean()
        print(f'average iter time: {avg_iter_time:.4f} s/iter')
        print()


def get_legends(args):
    """if legend is None, use {filename}_{key} as legend."""
    legend = args.legend
    if legend is None:
        legend = []
        for json_log in args.json_logs:
            for metric in args.keys:
                # remove '.json' in the end of log names
                basename = os.path.basename(json_log)[:-5]
                if basename.endswith('.log'):
                    basename = basename[:-4]
                legend.append(f'{basename}_{metric}')
    assert len(legend) == (len(args.json_logs) * len(args.keys))
    return legend


def plot_phase_train(metric, train_logs, curve_label):
    """plot phase of train curve."""
    xs = np.array([log['step'] for log in train_logs])
    ys = np.array([log[metric] for log in train_logs])

    if 'epoch' in train_logs[0]:
        scale_factor = train_logs[-1]['step'] / train_logs[-1]['epoch']
        xs = xs / scale_factor
        plt.xlabel('Epochs')
    else:
        plt.xlabel('Iters')

    plt.plot(xs, ys, label=curve_label, linewidth=0.75)


def plot_phase_val(metric, val_logs, curve_label):
    """plot phase of val curve."""
    xs = np.array([log['step'] for log in val_logs])
    ys = np.array([log[metric] for log in val_logs])

    plt.xlabel('Steps')
    plt.plot(xs, ys, label=curve_label, linewidth=0.75)


def plot_curve_helper(log_dicts, metrics, args, legend):
    """plot curves from log_dicts by metrics."""
    num_metrics = len(metrics)
    for i, log_dict in enumerate(log_dicts):
        for j, key in enumerate(metrics):
            json_log = args.json_logs[i]
            print(f'plot curve of {json_log}, metric is {key}')
            curve_label = legend[i * num_metrics + j]

            train_keys = {} if len(log_dict['train']) == 0 else set(
                log_dict['train'][0].keys()) - {'step', 'epoch'}
            val_keys = {} if len(log_dict['val']) == 0 else set(
                log_dict['val'][0].keys()) - {'step'}

            if key in val_keys:
                plot_phase_val(key, log_dict['val'], curve_label)
            elif key in train_keys:
                plot_phase_train(key, log_dict['train'], curve_label)
            else:
                raise ValueError(
                    f'Invalid key "{key}", please choose from '
                    f'{set.union(set(train_keys), set(val_keys))}.')
            plt.legend()


def plot_curve(log_dicts, args):
    """Plot train metric-iter graph."""
    # set style
    try:
        import seaborn as sns
        sns.set_style(args.style)
    except ImportError:
        pass

    # set plot window size
    wind_w, wind_h = args.window_size.split('*')
    wind_w, wind_h = int(wind_w), int(wind_h)
    plt.figure(figsize=(wind_w, wind_h))

    # get legends and metrics
    legends = get_legends(args)
    metrics = args.keys

    # plot curves from log_dicts by metrics
    plot_curve_helper(log_dicts, metrics, args, legends)

    # set title and show or save
    if args.title is not None:
        plt.title(args.title)
    if args.out is None:
        plt.show()
    else:
        print(f'save curve to: {args.out}')
        plt.savefig(args.out)
        plt.cla()


def add_plot_parser(subparsers):
    parser_plt = subparsers.add_parser(
        'plot_curve', help='parser for plotting curves')
    parser_plt.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser_plt.add_argument(
        '--keys',
        type=str,
        nargs='+',
        default=['loss'],
        help='the metric that you want to plot')
    parser_plt.add_argument('--title', type=str, help='title of figure')
    parser_plt.add_argument(
        '--legend',
        type=str,
        nargs='+',
        default=None,
        help='legend of each plot')
    parser_plt.add_argument(
        '--style',
        type=str,
        default='whitegrid',
        help='style of the figure, need `seaborn` package.')
    parser_plt.add_argument('--out', type=str, default=None)
    parser_plt.add_argument(
        '--window-size',
        default='12*7',
        help='size of the window to display images, in format of "$W*$H".')


def add_time_parser(subparsers):
    parser_time = subparsers.add_parser(
        'cal_train_time',
        help='parser for computing the average time per training iteration')
    parser_time.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser_time.add_argument(
        '--include-outliers',
        action='store_true',
        help='include the first value of every epoch when computing '
        'the average time')


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Json Log')
    # currently only support plot curve and calculate average train time
    subparsers = parser.add_subparsers(dest='task', help='task parser')
    add_plot_parser(subparsers)
    add_time_parser(subparsers)
    args = parser.parse_args()

    if hasattr(args, 'window_size') and args.window_size != '':
        assert re.match(r'\d+\*\d+', args.window_size), \
            "'window-size' must be in format 'W*H'."
    return args


def main():
    args = parse_args()

    json_logs = args.json_logs
    for json_log in json_logs:
        assert json_log.endswith('.json')

    log_dicts = [load_json_log(json_log) for json_log in json_logs]

    if args.task == 'cal_train_time':
        cal_train_time(log_dicts, args)
    elif args.task == 'plot_curve':
        plot_curve(log_dicts, args)


if __name__ == '__main__':
    main()
