from argparse import ArgumentParser
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import cv2
import numpy as np

def print_metrics(metrics):
    for metric_name, metric_val in metrics.items():
        print('{:<10}: {:.3f}'.format(metric_name, metric_val))
    return

##--------------------------------------------------
class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg

class RunningAverageDict:
    """A dictionary of running averages."""
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if new_dict is None:
            return

        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        if self._dict is None:
            return None
        return {key: value.get_value() for key, value in self._dict.items()}

# https://github.com/isl-org/ZoeDepth/blob/edb6daf45458569e24f50250ef1ed08c015f17a7/zoedepth/utils/misc.py#L159
def compute_errors(gt, pred):
    """Compute metrics for 'pred' compared to 'gt'

    Args:
        gt (numpy.ndarray): Ground truth values
        pred (numpy.ndarray): Predicted values

        gt.shape should be equal to pred.shape

    Returns:
        dict: Dictionary containing the following metrics:
            'a1': Delta1 accuracy: Fraction of pixels that are within a scale factor of 1.25
            'a2': Delta2 accuracy: Fraction of pixels that are within a scale factor of 1.25^2
            'a3': Delta3 accuracy: Fraction of pixels that are within a scale factor of 1.25^3
            'abs_rel': Absolute relative error
            'rmse': Root mean squared error
            'log_10': Absolute log10 error
            'sq_rel': Squared relative error
            'rmse_log': Root mean squared error on the log scale
            'silog': Scale invariant log error
    """

    if len(gt) == 0:
        print('invalid!')

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    a1_human = (thresh < 1.05).mean()
    a2_human = (thresh < 1.05 ** 2).mean()
    a3_human = (thresh < 1.05 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    l1_err = np.mean(np.abs(gt - pred))

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, 
                a1_human=a1_human, a2_human=a2_human, a3_human=a3_human,
                l1_err=l1_err,
                abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)

def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = np.sum(mask * prediction * prediction)
    a_01 = np.sum(mask * prediction)
    a_11 = np.sum(mask)

    # right hand side: b = [b_0, b_1]
    b_0 = np.sum(mask * prediction * target)
    b_1 = np.sum(mask * target)

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = 0
    x_1 = 0

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    if valid:
        x_0 = (a_11 * b_0 - a_01 * b_1) / det
        x_1 = (-a_01 * b_0 + a_00 * b_1) / det

    return x_0, x_1


def process_image(name, depth_dir, mask_dir, pred_root, min_depth_eval=0.001, max_depth_eval=100):
    gt_depth_path = os.path.join(depth_dir, name.replace('.png', '.npy'))
    gt_mask_path = os.path.join(mask_dir, name)

    gt_depth = np.load(gt_depth_path)  # H x W
    gt_mask = cv2.imread(gt_mask_path)  # H x W
    gt_mask = gt_mask[:, :, 0]

    pred_depth_path = os.path.join(pred_root, name.replace('.png', '.npy'))  # H x W
    pred_depth = np.load(pred_depth_path)
    
    pred_depth[pred_depth < min_depth_eval] = min_depth_eval
    pred_depth[pred_depth > max_depth_eval] = max_depth_eval
    pred_depth[np.isinf(pred_depth)] = max_depth_eval
    pred_depth[np.isnan(pred_depth)] = min_depth_eval

    scale, shift = compute_scale_and_shift(pred_depth, gt_depth, gt_mask > 0)
    aligned_pred_depth = scale * pred_depth + shift
    aligned_pred_depth[aligned_pred_depth < min_depth_eval] = min_depth_eval
    aligned_pred_depth[aligned_pred_depth > max_depth_eval] = max_depth_eval

    return compute_errors(gt_depth[gt_mask > 0], aligned_pred_depth[gt_mask > 0])

def depth_evaluate(data_root, pred_root, round_vals=True, round_precision=3):
    rgb_dir = os.path.join(data_root, 'rgb')
    depth_dir = os.path.join(data_root, 'depth')
    mask_dir = os.path.join(data_root, 'mask')

    rgb_files = {x for x in os.listdir(rgb_dir) if x.endswith('.png')}
    mask_files = {x for x in os.listdir(mask_dir) if x.endswith('.png')}
    depth_files = {x.replace('.npy', '.png') for x in os.listdir(depth_dir) if x.endswith('.npy')}

    # Find the intersection of file names between images, masks, and depths
    common_names = rgb_files & mask_files & depth_files
    common_names = sorted(common_names)

    metrics = RunningAverageDict()

    # Load predictions in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_name = {executor.submit(process_image, name, depth_dir, mask_dir, pred_root): name for name in common_names}
        for future in tqdm(as_completed(future_to_name), total=len(common_names)):
            name = future_to_name[future]
            image_metrics = future.result()
            metrics.update(image_metrics)

    if round_vals:
        def r(m): return round(m, round_precision)
    else:
        def r(m): return m
    metrics = {k: r(v) for k, v in metrics.get_value().items()}

    return metrics