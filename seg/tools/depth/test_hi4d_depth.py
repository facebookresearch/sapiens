from argparse import ArgumentParser
import os
from tqdm import tqdm
import cv2
import numpy as np
from utils import *

##--------------------------------------------------
def main():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', help='Test data dir')
    parser.add_argument('--pred_dir', '--pred-dir', default=None, help='Path to output dir')
    args = parser.parse_args()

    data_dir = args.data_dir
    pred_dir = args.pred_dir

    ##-----------------for hi4d--------------------
    metrics = depth_evaluate(data_dir, os.path.join(pred_dir, 'rgb'))
    print('pred_dir:', pred_dir)

    print_metrics(metrics)

if __name__ == '__main__':
    main()
