# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modified from https://github.com/bethgelab/model-vs-human
import argparse
import os
import os.path as osp

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mmengine.logging import MMLogger
from utils import FormatStrFormatter, ShapeBias

# global default boundary settings for thin gray transparent
# boundaries to avoid not being able to see the difference
# between two partially overlapping datapoints of the same color:
PLOTTING_EDGE_COLOR = (0.3, 0.3, 0.3, 0.3)
PLOTTING_EDGE_WIDTH = 0.02
ICONS_DIR = osp.join(
    osp.dirname(__file__), '..', '..', 'resources', 'shape_bias_icons')

parser = argparse.ArgumentParser()
parser.add_argument('--csv-dir', type=str, help='directory of csv files')
parser.add_argument(
    '--result-dir', type=str, help='directory to save plotting results')
parser.add_argument('--model-names', nargs='+', default=[], help='model name')
parser.add_argument(
    '--colors',
    nargs='+',
    type=float,
    default=[],
    help=  # noqa
    'the colors for the plots of each model, and they should be in the same order as model_names'  # noqa: E501
)
parser.add_argument(
    '--markers',
    nargs='+',
    type=str,
    default=[],
    help=  # noqa
    'the markers for the plots of each model, and they should be in the same order as model_names'  # noqa: E501
)
parser.add_argument(
    '--plotting-names',
    nargs='+',
    default=[],
    help=  # noqa
    'the plotting names for the plots of each model, and they should be in the same order as model_names'  # noqa: E501
)
parser.add_argument(
    '--delete-icons',
    action='store_true',
    help='whether to delete the icons after plotting')

humans = [
    'subject-01', 'subject-02', 'subject-03', 'subject-04', 'subject-05',
    'subject-06', 'subject-07', 'subject-08', 'subject-09', 'subject-10'
]

icon_names = [
    'airplane.png', 'response_icons_vertical_reverse.png', 'bottle.png',
    'car.png', 'oven.png', 'elephant.png', 'dog.png', 'boat.png', 'clock.png',
    'chair.png', 'keyboard.png', 'bird.png', 'bicycle.png',
    'response_icons_horizontal.png', 'cat.png', 'bear.png', 'colorbar.pdf',
    'knife.png', 'response_icons_vertical.png', 'truck.png'
]


def read_csvs(csv_dir: str) -> pd.DataFrame:
    """Reads all csv files in a directory and returns a single dataframe.

    Args:
        csv_dir (str): directory of csv files.

    Returns:
        pd.DataFrame: dataframe containing all csv files
    """
    df = pd.DataFrame()
    for csv in os.listdir(csv_dir):
        if csv.endswith('.csv'):
            cur_df = pd.read_csv(osp.join(csv_dir, csv))
            cur_df.columns = [c.lower() for c in cur_df.columns]
            df = df.append(cur_df)
    df.condition = df.condition.astype(str)
    return df


def plot_shape_bias_matrixplot(args, analysis=ShapeBias()) -> None:
    """Plots a matrixplot of shape bias.

    Args:
        args (argparse.Namespace): arguments.
        analysis (ShapeBias): shape bias analysis. Defaults to ShapeBias().
    """
    mpl.rcParams['font.family'] = ['serif']
    mpl.rcParams['font.serif'] = ['Times New Roman']

    plt.figure(figsize=(9, 7))
    df = read_csvs(args.csv_dir)

    fontsize = 15
    ticklength = 10
    markersize = 250
    label_size = 20

    classes = df['category'].unique()
    num_classes = len(classes)

    # plot setup
    fig = plt.figure(1, figsize=(12, 12), dpi=300.)
    ax = plt.gca()

    ax.set_xlim([0, 1])
    ax.set_ylim([-.5, num_classes - 0.5])

    # secondary reversed x axis
    ax_top = ax.secondary_xaxis(
        'top', functions=(lambda x: 1 - x, lambda x: 1 - x))

    # labels, ticks
    plt.tick_params(
        axis='y', which='both', left=False, right=False, labelleft=False)
    ax.set_ylabel('Shape categories', labelpad=60, fontsize=label_size)
    ax.set_xlabel(
        "Fraction of 'texture' decisions", fontsize=label_size, labelpad=25)
    ax_top.set_xlabel(
        "Fraction of 'shape' decisions", fontsize=label_size, labelpad=25)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax_top.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.get_xaxis().set_ticks(np.arange(0, 1.1, 0.1))
    ax_top.set_ticks(np.arange(0, 1.1, 0.1))
    ax.tick_params(
        axis='both', which='major', labelsize=fontsize, length=ticklength)
    ax_top.tick_params(
        axis='both', which='major', labelsize=fontsize, length=ticklength)

    # arrows on x axes
    plt.arrow(
        x=0,
        y=-1.75,
        dx=1,
        dy=0,
        fc='black',
        head_width=0.4,
        head_length=0.03,
        clip_on=False,
        length_includes_head=True,
        overhang=0.5)
    plt.arrow(
        x=1,
        y=num_classes + 0.75,
        dx=-1,
        dy=0,
        fc='black',
        head_width=0.4,
        head_length=0.03,
        clip_on=False,
        length_includes_head=True,
        overhang=0.5)

    # icons besides y axis
    # determine order of icons
    df_selection = df.loc[(df['subj'].isin(humans))]
    class_avgs = []
    for cl in classes:
        df_class_selection = df_selection.query("category == '{}'".format(cl))
        class_avgs.append(1 - analysis.analysis(
            df=df_class_selection)['shape-bias'])
    sorted_indices = np.argsort(class_avgs)
    classes = classes[sorted_indices]

    # icon placement is calculated in axis coordinates
    WIDTH = 1 / num_classes
    # placement left of yaxis (-WIDTH) plus some spacing (-.25*WIDTH)
    XPOS = -1.25 * WIDTH
    YPOS = -0.5
    HEIGHT = 1
    MARGINX = 1 / 10 * WIDTH  # vertical whitespace between icons
    MARGINY = 1 / 10 * HEIGHT  # horizontal whitespace between icons

    left = XPOS + MARGINX
    right = XPOS + WIDTH - MARGINX

    for i in range(num_classes):
        bottom = i + MARGINY + YPOS
        top = (i + 1) - MARGINY + YPOS
        iconpath = osp.join(ICONS_DIR, '{}.png'.format(classes[i]))
        plt.imshow(
            plt.imread(iconpath),
            extent=[left, right, bottom, top],
            aspect='auto',
            clip_on=False)

    # plot horizontal intersection lines
    for i in range(num_classes - 1):
        plt.plot([0, 1], [i + .5, i + .5],
                 c='gray',
                 linestyle='dotted',
                 alpha=0.4)

    # plot average shapebias + scatter points
    for i in range(len(args.model_names)):
        df_selection = df.loc[(df['subj'].isin(args.model_names[i]))]
        result_df = analysis.analysis(df=df_selection)
        avg = 1 - result_df['shape-bias']
        ax.plot([avg, avg], [-1, num_classes], color=args.colors[i])
        class_avgs = []
        for cl in classes:
            df_class_selection = df_selection.query(
                "category == '{}'".format(cl))
            class_avgs.append(1 - analysis.analysis(
                df=df_class_selection)['shape-bias'])

        ax.scatter(
            class_avgs,
            classes,
            color=args.colors[i],
            marker=args.markers[i],
            label=args.plotting_names[i],
            s=markersize,
            clip_on=False,
            edgecolors=PLOTTING_EDGE_COLOR,
            linewidths=PLOTTING_EDGE_WIDTH,
            zorder=3)
    plt.legend(frameon=True, labelspacing=1, loc=9)

    figure_path = osp.join(args.result_dir,
                           'cue-conflict_shape-bias_matrixplot.pdf')
    fig.savefig(figure_path, bbox_inches='tight')
    plt.close()


def check_icons() -> bool:
    """Check if icons are present, if not download them."""
    if not osp.exists(ICONS_DIR):
        return False
    for icon_name in icon_names:
        if not osp.exists(osp.join(ICONS_DIR, icon_name)):
            return False
    return True


if __name__ == '__main__':

    if not check_icons():
        root_url = 'https://github.com/bethgelab/model-vs-human/raw/master/assets/icons'  # noqa: E501
        os.makedirs(ICONS_DIR, exist_ok=True)
        MMLogger.get_current_instance().info(
            f'Downloading icons to {ICONS_DIR}')
        for icon_name in icon_names:
            url = osp.join(root_url, icon_name)
            os.system('wget -O {} {}'.format(
                osp.join(ICONS_DIR, icon_name), url))

    args = parser.parse_args()
    assert len(args.model_names) * 3 == len(args.colors), 'Number of colors \
        must be 3 times the number of models. Every three colors are the RGB \
            values for one model.'

    # preprocess colors
    args.colors = [c / 255. for c in args.colors]
    colors = []
    for i in range(len(args.model_names)):
        colors.append(args.colors[3 * i:3 * i + 3])
    args.colors = colors
    args.colors.append([165 / 255., 30 / 255., 55 / 255.])  # human color

    # if plotting names are not specified, use model names
    if len(args.plotting_names) == 0:
        args.plotting_names = args.model_names

    # preprocess markers
    args.markers.append('D')  # human marker

    # preprocess model names
    args.model_names = [[m] for m in args.model_names]
    args.model_names.append(humans)

    # preprocess plotting names
    args.plotting_names.append('Humans')

    plot_shape_bias_matrixplot(args)
    if args.delete_icons:
        os.system('rm -rf {}'.format(ICONS_DIR))
