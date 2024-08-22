# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
from typing import Dict, Optional, Sequence
import numpy as np
import os
from mmpose.registry import METRICS
from .coco_metric import CocoMetric
import os.path as osp
import tempfile
from collections import OrderedDict, defaultdict
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump, get_local_path, load
from mmengine.logging import MMLogger
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval
from ..functional import oks_nms, soft_oks_nms

from .coco_wholebody_metric import CocoWholeBodyMetric
from mmpose.datasets.datasets.utils import parse_pose_metainfo

try:
    from configs._base_.datasets.coco_wholebody import dataset_info as coco_wholebody_dataset_meta
    coco_wholebody_dataset_meta = parse_pose_metainfo(coco_wholebody_dataset_meta)
except Exception as e:
    pass

@METRICS.register_module()
class GoliathCocoWholeBodyMetric(CocoMetric):
    """
    """
    default_prefix: Optional[str] = 'goliath'
    body_num = 17
    foot_num = 6
    face_num = 238
    left_hand_num = 20
    right_hand_num = 20
    remaining_extra_num = 7 ## total to 308

    def __init__(self,
                 ann_file: Optional[str] = None,
                 coco_wholebody_ann_file: Optional[str] = None,
                 use_area: bool = True,
                 iou_type: str = 'keypoints',
                 score_mode: str = 'bbox_keypoint',
                 keypoint_score_thr: float = 0.2,
                 nms_mode: str = 'oks_nms',
                 nms_thr: float = 0.9,
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:

        super().__init__(ann_file, use_area, iou_type, score_mode, keypoint_score_thr, nms_mode,\
                        nms_thr, format_only, outfile_prefix, collect_device, prefix)

        self.coco_wholebody_metric = CocoWholeBodyMetric(coco_wholebody_ann_file, use_area, iou_type, score_mode, keypoint_score_thr, \
                                nms_mode, nms_thr, format_only, outfile_prefix, collect_device, 'coco-wholebody')

        self.coco_wholebody_metric._dataset_meta = coco_wholebody_dataset_meta
        self.coco_wholebody_num_keypoints = self.coco_wholebody_metric._dataset_meta['num_keypoints']

        self.goliath_num_images = len(self.coco.getImgIds())
        self.coco_wholebody_num_images = len(self.coco_wholebody_metric.coco.getImgIds())

        ## uncomment to debug
        # self.goliath_num_images = 16
        # self.coco_wholebody_num_images = 16

        return


    def gt_to_coco_json(self, gt_dicts: Sequence[dict],
                        outfile_prefix: str) -> str:
        """
        """
        image_infos = []
        annotations = []
        img_ids = []
        ann_ids = []

        for gt_dict in gt_dicts:
            # filter duplicate image_info
            if gt_dict['img_id'] not in img_ids:
                image_info = dict(
                    id=gt_dict['img_id'],
                    width=gt_dict['width'],
                    height=gt_dict['height'],
                )
                if self.iou_type == 'keypoints_crowd':
                    image_info['crowdIndex'] = gt_dict['crowd_index']

                image_infos.append(image_info)
                img_ids.append(gt_dict['img_id'])

            # filter duplicate annotations
            for ann in gt_dict['raw_ann_info']:
                annotation = dict(
                    id=ann['id'],
                    image_id=ann['image_id'],
                    category_id=ann['category_id'],
                    bbox=ann['bbox'],
                    keypoints=ann['keypoints'],
                    foot_kpts=ann['foot_kpts'],
                    face_kpts=ann['face_kpts'],
                    lefthand_kpts=ann['lefthand_kpts'],
                    righthand_kpts=ann['righthand_kpts'],
                    iscrowd=ann['iscrowd'],
                )
                if self.use_area:
                    assert 'area' in ann, \
                        '`area` is required when `self.use_area` is `True`'
                    annotation['area'] = ann['area']

                annotations.append(annotation)
                ann_ids.append(ann['id'])

        info = dict(
            date_created=str(datetime.datetime.now()),
            description='Coco json file converted by mmpose CocoMetric.')
        coco_json: dict = dict(
            info=info,
            images=image_infos,
            categories=self.dataset_meta['CLASSES'],
            licenses=None,
            annotations=annotations,
        )
        converted_json_path = f'{outfile_prefix}.gt.json'
        dump(coco_json, converted_json_path, sort_keys=True, indent=4)
        return converted_json_path

    def convert_results_goliath_to_coco_wholebody(self, results: list) -> list:
        """Convert the results of Goliath to COCO-WholeBody format.

        Args:
            results (list): The processed results of each batch.

        Returns:
            list: The converted results.
        """
        # split prediction and gt list
        preds, gts = zip(*results) ## gts are don't care here
        coco_wholebody_to_goliath_mapping = self.dataset_meta['coco_wholebody_to_goliath_mapping'] ## coco_wholebody_index to goliath_index mapping
        coco_wholebody_indexes, goliath_indexes = zip(*[(k, v) for k, v in coco_wholebody_to_goliath_mapping.items()])

        for pred in preds:
            goliath_keypoints = pred['keypoints'] ## 1 x 308 x 2
            goliath_keypoint_scores = pred['keypoint_scores'] ## 1 x 308

            num_detections = goliath_keypoints.shape[0]

            coco_wholebody_keypoints = np.zeros((num_detections, self.coco_wholebody_num_keypoints, 2))
            coco_wholebody_keypoint_scores = np.zeros((num_detections, self.coco_wholebody_num_keypoints))

            coco_wholebody_keypoints[:, coco_wholebody_indexes, :] = goliath_keypoints[:, goliath_indexes, :]
            coco_wholebody_keypoint_scores[:, coco_wholebody_indexes] = goliath_keypoint_scores[:, goliath_indexes]

            pred['keypoints'] = coco_wholebody_keypoints
            pred['keypoint_scores'] = coco_wholebody_keypoint_scores

        # combine the predictions and ground truths
        results = list(zip(preds, gts))

        return results

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        print(f'goliath num images:{self.goliath_num_images}, coco_wholebody num images:{self.coco_wholebody_num_images}, results length:{len(results)}')

        assert len(results) > self.goliath_num_images

        results_goliath = results[:self.goliath_num_images]
        results_coco_wholebody = self.convert_results_goliath_to_coco_wholebody(results[self.goliath_num_images:])

        # Print goliath metrics in orange
        print('\033[38;5;208m' + '-----------------------------------start goliath eval------------------------------------------'+ '\033[0m')
        goliath_metrics = self.compute_goliath_metrics(results_goliath)
        print('\033[38;5;208m' + '-----------------------------------end goliath eval------------------------------------------' + '\033[0m')

        # Print coco_wholebody metrics in green
        print('\033[32m' + '-----------------------------------start coco_wholebody eval------------------------------------------'+ '\033[0m')
        coco_wholebody_metrics = self.coco_wholebody_metric.compute_metrics(results_coco_wholebody)
        print('\033[32m' + '-----------------------------------end coco_wholebody eval------------------------------------------' + '\033[0m')

        metrics = goliath_metrics.copy()

        for metric_name, metric_value in coco_wholebody_metrics.items():
            metrics[f'coco_wholebody_{metric_name}'] = metric_value

        return metrics

    def compute_goliath_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # split prediction and gt list
        preds, gts = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        if self.coco is None:
            # use converted gt json file to initialize coco helper
            logger.info('Converting ground truth to coco format...')
            coco_json_path = self.gt_to_coco_json(
                gt_dicts=gts, outfile_prefix=outfile_prefix)
            self.coco = COCO(coco_json_path)

        kpts = defaultdict(list)

        # group the preds by img_id
        for pred in preds:
            img_id = pred['img_id']
            assert pred['img_id'] == pred['id']
            for idx in range(len(pred['keypoints'])):
                instance = {
                    'id': pred['id'],
                    'img_id': pred['img_id'],
                    'category_id': pred['category_id'],
                    'keypoints': pred['keypoints'][idx],
                    'keypoint_scores': pred['keypoint_scores'][idx],
                    'bbox_score': pred['bbox_scores'][idx],
                }

                if 'areas' in pred:
                    instance['area'] = pred['areas'][idx]
                else:
                    # use keypoint to calculate bbox and get area
                    keypoints = pred['keypoints'][idx]
                    area = (
                        np.max(keypoints[:, 0]) - np.min(keypoints[:, 0])) * (
                            np.max(keypoints[:, 1]) - np.min(keypoints[:, 1]))
                    instance['area'] = area

                kpts[img_id].append(instance)

        # sort keypoint results according to id and remove duplicate ones
        kpts = self._sort_and_unique_bboxes(kpts, key='id')

        # score the prediction results according to `score_mode`
        # and perform NMS according to `nms_mode`
        valid_kpts = defaultdict(list)
        num_keypoints = self.dataset_meta['num_keypoints']
        for img_id, instances in kpts.items():
            for instance in instances:
                # concatenate the keypoint coordinates and scores
                instance['keypoints'] = np.concatenate([
                        instance['keypoints'], instance['keypoint_scores'][:, None]
                    ], axis=-1)
                if self.score_mode == 'bbox':
                    instance['score'] = instance['bbox_score']
                elif self.score_mode == 'keypoint':
                    instance['score'] = np.mean(instance['keypoint_scores'])
                else:
                    bbox_score = instance['bbox_score']
                    if self.score_mode == 'bbox_rle':
                        keypoint_scores = instance['keypoint_scores']
                        instance['score'] = float(bbox_score +
                                                  np.mean(keypoint_scores) +
                                                  np.max(keypoint_scores))

                    else:  # self.score_mode == 'bbox_keypoint':
                        mean_kpt_score = 0
                        valid_num = 0
                        for kpt_idx in range(num_keypoints):
                            kpt_score = instance['keypoint_scores'][kpt_idx]
                            if kpt_score > self.keypoint_score_thr:
                                mean_kpt_score += kpt_score
                                valid_num += 1
                        if valid_num != 0:
                            mean_kpt_score /= valid_num
                        instance['score'] = bbox_score * mean_kpt_score
            # perform nms
            if self.nms_mode == 'none':
                valid_kpts[img_id] = instances
            else:
                nms = oks_nms if self.nms_mode == 'oks_nms' else soft_oks_nms
                keep = nms(
                    instances,
                    self.nms_thr,
                    sigmas=self.dataset_meta['sigmas'])
                valid_kpts[img_id] = [instances[_keep] for _keep in keep]

        # convert results to coco style and dump into a json file
        self.results2json(valid_kpts, outfile_prefix=outfile_prefix)

        # only format the results without doing quantitative evaluation
        if self.format_only:
            logger.info('results are saved in '
                        f'{osp.dirname(outfile_prefix)}')
            return {}

        # evaluation results
        eval_results = OrderedDict()
        logger.info(f'Evaluating {self.__class__.__name__}...')
        info_str = self._do_python_keypoint_eval(outfile_prefix)
        name_value = OrderedDict(info_str)
        eval_results.update(name_value)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results

    def results2json(self, keypoints: Dict[int, list],
                     outfile_prefix: str) -> str:
        """Dump the keypoint detection results to a COCO style json file.

        Args:
            keypoints (Dict[int, list]): Keypoint detection results
                of the dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.keypoints.json",

        Returns:
            str: The json file name of keypoint results.
        """
        # the results with category_id
        cat_id = 1
        cat_results = []

        self.goliath_info = self.coco.__dict__['dataset']['info']['goliath_info']
        self.body_keypoint_ids = self.goliath_info['body_keypoint_ids']
        self.foot_keypoint_ids = self.goliath_info['foot_keypoint_ids']
        self.face_keypoint_ids = self.goliath_info['face_keypoint_ids']
        self.left_hand_keypoint_ids = self.goliath_info['left_hand_keypoint_ids']
        self.right_hand_keypoint_ids = self.goliath_info['right_hand_keypoint_ids']

        assert len(self.body_keypoint_ids) == self.body_num
        assert len(self.foot_keypoint_ids) == self.foot_num
        assert len(self.face_keypoint_ids) == self.face_num
        assert len(self.left_hand_keypoint_ids) == self.left_hand_num
        assert len(self.right_hand_keypoint_ids) == self.right_hand_num

        for _, img_kpts in keypoints.items():
            _keypoints = np.array(
                [img_kpt['keypoints'] for img_kpt in img_kpts])
            num_keypoints = self.dataset_meta['num_keypoints']
            # collect all the person keypoints in current image
            _body_keypoints = _keypoints[:, self.body_keypoint_ids].copy() ## get only body keypoints
            _foot_keypoints = _keypoints[:, self.foot_keypoint_ids].copy() ## get only foot keypoints
            _face_keypoints = _keypoints[:, self.face_keypoint_ids].copy() ## get only face keypoints
            _left_hand_keypoints = _keypoints[:, self.left_hand_keypoint_ids].copy() ## get only left hand keypoints
            _right_hand_keypoints = _keypoints[:, self.right_hand_keypoint_ids].copy() ## get only right hand keypoints

            _keypoints = _keypoints.reshape(-1, num_keypoints * 3) ## flatten
            _body_keypoints = _body_keypoints.reshape(-1, self.body_num * 3) ## flatten
            _foot_keypoints = _foot_keypoints.reshape(-1, self.foot_num * 3) ## flatten
            _face_keypoints = _face_keypoints.reshape(-1, self.face_num * 3) ## flatten
            _left_hand_keypoints = _left_hand_keypoints.reshape(-1, self.left_hand_num * 3) ## flatten
            _right_hand_keypoints = _right_hand_keypoints.reshape(-1, self.right_hand_num * 3) ## flatten

            result = [{
                'image_id': img_kpt['img_id'],
                'category_id': cat_id,
                'goliath_wholebody_kpts': _keypoint.tolist(), ## all keypoints. Modified in xtcocotools
                'keypoints': _body_keypoint.tolist(), ## xtcocotools treats this as body keypoints, 17 default
                'foot_kpts': _foot_keypoint.tolist(),
                'face_kpts': _face_keypoint.tolist(),
                'lefthand_kpts': _left_hand_keypoint.tolist(),
                'righthand_kpts': _right_hand_keypoint.tolist(),
                'score': float(img_kpt['score']),
            } for img_kpt, _keypoint, _body_keypoint, _foot_keypoint, _face_keypoint, \
                _left_hand_keypoint, _right_hand_keypoint in zip(img_kpts, _keypoints, \
                _body_keypoints, _foot_keypoints, _face_keypoints, \
                _left_hand_keypoints, _right_hand_keypoints)]

            cat_results.extend(result)

        res_file = f'{outfile_prefix}.keypoints.json'
        dump(cat_results, res_file, sort_keys=True, indent=4)

    def _do_python_keypoint_eval(self, outfile_prefix: str) -> list:
        """Do keypoint evaluation using COCOAPI.

        Args:
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.keypoints.json",

        Returns:
            list: a list of tuples. Each tuple contains the evaluation stats
            name and corresponding stats value.
        """
        res_file = f'{outfile_prefix}.keypoints.json'
        coco_det = self.coco.loadRes(res_file)
        sigmas = self.dataset_meta['sigmas']

        self.goliath_info = self.coco.__dict__['dataset']['info']['goliath_info']
        self.body_keypoint_ids = self.goliath_info['body_keypoint_ids']
        self.foot_keypoint_ids = self.goliath_info['foot_keypoint_ids']
        self.face_keypoint_ids = self.goliath_info['face_keypoint_ids']
        self.left_hand_keypoint_ids = self.goliath_info['left_hand_keypoint_ids']
        self.right_hand_keypoint_ids = self.goliath_info['right_hand_keypoint_ids']

        assert len(self.body_keypoint_ids) == self.body_num
        assert len(self.foot_keypoint_ids) == self.foot_num
        assert len(self.face_keypoint_ids) == self.face_num
        assert len(self.left_hand_keypoint_ids) == self.left_hand_num
        assert len(self.right_hand_keypoint_ids) == self.right_hand_num

        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_body',
            sigmas[self.body_keypoint_ids],
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_foot',
            sigmas[self.foot_keypoint_ids],
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_face',
            sigmas[self.face_keypoint_ids],
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_lefthand',
            sigmas[self.left_hand_keypoint_ids],
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_righthand',
            sigmas[self.right_hand_keypoint_ids],
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(
            self.coco, coco_det, 'keypoints_wholebody_goliath', sigmas, use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = [
            'AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
            'AR .75', 'AR (M)', 'AR (L)'
        ]

        info_str = list(zip(stats_names, coco_eval.stats))

        return info_str
