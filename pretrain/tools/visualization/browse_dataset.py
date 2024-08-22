# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os.path as osp
import sys
import textwrap

from matplotlib import transforms
from mmengine.config import Config, DictAction
from mmengine.dataset import Compose
from mmengine.registry import init_default_scope
from mmengine.utils import ProgressBar
from mmengine.visualization.utils import img_from_canvas

from mmpretrain.datasets.builder import build_dataset
from mmpretrain.structures import DataSample
from mmpretrain.visualization import UniversalVisualizer, create_figure

try:
    from matplotlib._tight_bbox import adjust_bbox
except ImportError:
    # To be compatible with matplotlib 3.5
    from matplotlib.tight_bbox import adjust_bbox


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--output-dir',
        '-o',
        default=None,
        type=str,
        help='If there is no display interface, you can save it.')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--phase',
        '-p',
        default='train',
        type=str,
        choices=['train', 'test', 'val'],
        help='phase of dataset to visualize, accept "train" "test" and "val".'
        ' Defaults to "train".')
    parser.add_argument(
        '--show-number',
        '-n',
        type=int,
        default=sys.maxsize,
        help='number of images selected to visualize, must bigger than 0. if '
        'the number is bigger than length of dataset, show all the images in '
        'dataset; default "sys.maxsize", show all images in dataset')
    parser.add_argument(
        '--show-interval',
        '-i',
        type=float,
        default=2,
        help='the interval of show (s)')
    parser.add_argument(
        '--mode',
        '-m',
        default='transformed',
        type=str,
        choices=['original', 'transformed', 'concat', 'pipeline'],
        help='display mode; display original pictures or transformed pictures'
        ' or comparison pictures. "original" means show images load from disk'
        '; "transformed" means to show images after transformed; "concat" '
        'means show images stitched by "original" and "output" images. '
        '"pipeline" means show all the intermediate images. '
        'Defaults to "transformed".')
    parser.add_argument(
        '--rescale-factor',
        '-r',
        type=float,
        help='(For `mode=original`) Image rescale factor, which is useful if'
        'the output is too large or too small.')
    parser.add_argument(
        '--channel-order',
        '-c',
        default='BGR',
        choices=['BGR', 'RGB'],
        help='The channel order of the showing images, could be "BGR" '
        'or "RGB", Defaults to "BGR".')
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


def make_grid(imgs, names):
    """Concat list of pictures into a single big picture, align height here."""
    # A large canvas to ensure all text clear.
    figure = create_figure(dpi=150, figsize=(16, 9))

    # deal with imgs
    max_nrows = 1
    img_shapes = []
    for img in imgs:
        if isinstance(img, list):
            max_nrows = max(len(img), max_nrows)
            img_shapes.append([i.shape[:2] for i in img])
        else:
            img_shapes.append(img.shape[:2])
    gs = figure.add_gridspec(max_nrows, len(imgs))

    for i, img in enumerate(imgs):
        if isinstance(img, list):
            for j in range(len(img)):
                subplot = figure.add_subplot(gs[j, i])
                subplot.axis(False)
                subplot.imshow(img[j])
                name = '\n'.join(textwrap.wrap(names[i] + str(j), width=20))
                subplot.set_title(
                    f'{name}\n{img_shapes[i][j]}',
                    fontsize=15,
                    family='monospace')
        else:
            subplot = figure.add_subplot(gs[:, i])
            subplot.axis(False)
            subplot.imshow(img)
            name = '\n'.join(textwrap.wrap(names[i], width=20))
            subplot.set_title(
                f'{name}\n{img_shapes[i]}', fontsize=15, family='monospace')

    # Manage the gap of subplots
    figure.tight_layout()

    # Remove the white boundary (reserve 0.5 inches at the top to show label)
    points = figure.get_tightbbox(
        figure.canvas.get_renderer()).get_points() + [[0, 0], [0, 0.5]]
    adjust_bbox(figure, transforms.Bbox(points))

    return img_from_canvas(figure.canvas)


class InspectCompose(Compose):
    """Compose multiple transforms sequentially.

    And record "img" field of all results in one list.
    """

    def __init__(self, transforms, intermediate_imgs, visualizer):
        super().__init__(transforms=transforms)
        self.intermediate_imgs = intermediate_imgs
        self.visualizer = visualizer

    def __call__(self, data):
        if 'img' in data:
            self.intermediate_imgs.append({
                'name': 'Original',
                'img': data['img'].copy()
            })

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
            if 'img' in data:
                img = data['img'].copy()
                if 'mask' in data:
                    tmp_img = img[0] if isinstance(img, list) else img
                    tmp_img = self.visualizer.add_mask_to_image(
                        tmp_img,
                        DataSample().set_mask(data['mask']),
                        resize=tmp_img.shape[:2])
                    img = [tmp_img] + img[1:] if isinstance(img,
                                                            list) else tmp_img
                self.intermediate_imgs.append({
                    'name': t.__class__.__name__,
                    'img': img
                })
        return data


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope('mmpretrain')  # Use mmpretrain as default scope.

    dataset_cfg = cfg.get(args.phase + '_dataloader').get('dataset')
    dataset = build_dataset(dataset_cfg)

    # init visualizer
    cfg.visualizer.pop('type')
    fig_cfg = dict(figsize=(16, 10))
    visualizer = UniversalVisualizer(
        **cfg.visualizer, fig_show_cfg=fig_cfg, fig_save_cfg=fig_cfg)
    visualizer.dataset_meta = dataset.metainfo

    # init inspection
    intermediate_imgs = []
    dataset.pipeline = InspectCompose(dataset.pipeline.transforms,
                                      intermediate_imgs, visualizer)

    # init visualization image number
    display_number = min(args.show_number, len(dataset))
    progress_bar = ProgressBar(display_number)

    for i, item in zip(range(display_number), dataset):

        rescale_factor = None
        if args.mode == 'original':
            image = intermediate_imgs[0]['img']
            # Only original mode need rescale factor, `make_grid` will use
            # matplotlib to manage the size of subplots.
            rescale_factor = args.rescale_factor
        elif args.mode == 'transformed':
            image = make_grid([intermediate_imgs[-1]['img']], ['transformed'])
        elif args.mode == 'concat':
            ori_image = intermediate_imgs[0]['img']
            trans_image = intermediate_imgs[-1]['img']
            image = make_grid([ori_image, trans_image],
                              ['original', 'transformed'])
        else:
            image = make_grid([result['img'] for result in intermediate_imgs],
                              [result['name'] for result in intermediate_imgs])

        intermediate_imgs.clear()

        data_sample = item['data_samples'].numpy()

        # get filename from dataset or just use index as filename
        if hasattr(item['data_samples'], 'img_path'):
            filename = osp.basename(item['data_samples'].img_path)
        else:
            # some dataset have not image path
            filename = f'{i}.jpg'

        out_file = osp.join(args.output_dir,
                            filename) if args.output_dir is not None else None

        visualizer.visualize_cls(
            image if args.channel_order == 'RGB' else image[..., ::-1],
            data_sample,
            rescale_factor=rescale_factor,
            show=not args.not_show,
            wait_time=args.show_interval,
            name=filename,
            out_file=out_file)
        progress_bar.update()


if __name__ == '__main__':
    main()
