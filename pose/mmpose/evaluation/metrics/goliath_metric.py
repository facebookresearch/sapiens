# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
from typing import Dict, Optional, Sequence

import numpy as np
from mmengine.fileio import dump
from xtcocotools.cocoeval import COCOeval

from mmpose.registry import METRICS
from .coco_metric import CocoMetric


@METRICS.register_module()
class GoliathMetric(CocoMetric):
    """
    """
    default_prefix: Optional[str] = 'goliath'
    body_num = 17
    foot_num = 6
    face_num = 238
    left_hand_num = 20
    right_hand_num = 20
    remaining_extra_num = 7 ## total to 308

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
