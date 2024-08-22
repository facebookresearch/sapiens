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

def compute_errors(gt, pred):
    if len(gt) == 0:
        print('Invalid input: ground truth is empty.')
        return

    # Normalize ground truth and prediction to unit norm
    gt_norm = np.linalg.norm(gt, axis=1, keepdims=True)
    pred_norm = np.linalg.norm(pred, axis=1, keepdims=True)
    gt_unit = gt / gt_norm
    pred_unit = pred / pred_norm

    # Compute angular error in degrees
    dot_product = np.sum(gt_unit * pred_unit, axis=1)
    dot_product = np.clip(dot_product, -1.0, 1.0)  # Ensure numerical stability
    angular_error = np.arccos(dot_product) * (180.0 / np.pi)

    # Compute error metrics
    angular_error_mean = np.mean(angular_error)
    angular_error_median = np.median(angular_error)
    within_11_point_5_deg = np.mean(angular_error < 11.5) * 100
    within_22_point_5_deg = np.mean(angular_error < 22.5) * 100
    within_30_deg = np.mean(angular_error < 30.0) * 100

    return dict(
            angular_error_mean=angular_error_mean,
            angular_error_median=angular_error_median,
            within_11_point_5_deg=within_11_point_5_deg,
            within_22_point_5_deg=within_22_point_5_deg,
            within_30_deg=within_30_deg,
        )


def process_image(name, normal_dir, mask_dir, pred_root):
    gt_normal_path = os.path.join(normal_dir, name.replace('.png', '.npy'))
    gt_mask_path = os.path.join(mask_dir, name)

    gt_normal = np.load(gt_normal_path)  # H x W x 3. RGB format
    gt_mask = cv2.imread(gt_mask_path)  # H x W
    gt_mask = gt_mask[:, :, 0]

    pred_normal_path = os.path.join(pred_root, name.replace('.png', '.npy'))  # H x W
    pred_normal = np.load(pred_normal_path) ## H x W x 3. RGB format

    return compute_errors(gt_normal[gt_mask > 0], pred_normal[gt_mask > 0])

def normal_evaluate(data_root, pred_root, prev_metrics=None, round_vals=True, round_precision=2):
    rgb_dir = os.path.join(data_root, 'rgb')
    normal_dir = os.path.join(data_root, 'normal')
    mask_dir = os.path.join(data_root, 'mask')

    rgb_files = {x for x in os.listdir(rgb_dir) if x.endswith('.png')}
    mask_files = {x for x in os.listdir(mask_dir) if x.endswith('.png')}
    normal_files = {x.replace('.npy', '.png') for x in os.listdir(normal_dir) if x.endswith('.npy')}

    # Find the intersection of file names between images, masks, and normal
    common_names = rgb_files & mask_files & normal_files
    common_names = sorted(common_names)

    if prev_metrics is None:
        metrics = RunningAverageDict()
    else:
        metrics = prev_metrics

    ##------------parallel-----------------------
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_name = {executor.submit(process_image, name, normal_dir, mask_dir, pred_root): name for name in common_names}
        for future in tqdm(as_completed(future_to_name), total=len(common_names)):
            name = future_to_name[future]
            image_metrics = future.result()
            metrics.update(image_metrics)

    # ##------------sequential-----------------------
    # for name in tqdm(common_names):
    #     image_metrics = process_image(name, normal_dir, mask_dir, pred_root)
    #     metrics.update(image_metrics)

    ##-----------------------------------------
    if round_vals:
        def r(m): return round(m, round_precision)
    else:
        def r(m): return m
    metrics_dict = {k: r(v) for k, v in metrics.get_value().items()}

    return metrics_dict, metrics
