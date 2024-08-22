import os, sys, shutil
import os.path as osp
import pdb
import numpy as np
# import cv2
import torch as th
import torch.nn.functional as thf
from collections import defaultdict, OrderedDict
import subprocess as sp
import multiprocessing as mp
import time
import copy
from tqdm import tqdm
from care.data.io import typed
from typing import cast, Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING, Any
from care.data.io.typed.container import ContainerType
os.environ["RSC_JOB_UUID"] = "Dummy" # On RSC, we need to set RSC_JOB_UUID before import typed from care, but RSC_JOB_UUID is not actually used in this example code


def _load_uv(uv_img_path):
    uv = typed.load(uv_img_path)

    uv = uv.astype(np.int32)
    uv = np.concatenate(
        [
            (uv[..., 0] | (uv[..., 1] << 8))[..., None],
            (uv[..., 2] | (uv[..., 3] << 8))[..., None]
        ],
        axis=-1
    )
    uv = uv.astype(np.float32)
    uv = uv / ((1 << 16) - 1)
    uv = uv * 2.0 - 1.0
    return uv


def load_data_raw(
    data_root:str, 
    vn_img_cont:ContainerType,
    v_img_cont:ContainerType,
    ident:str, 
    segment:str, 
    frame:int
) -> Dict[str, np.ndarray]:

    # image
    # img_path = osp.join(data_root, ident, segment, "image", f"{frame:06d}.png")
    # img = typed.load(img_path) # (h, w, 3) & uint8

    # mean tex
    tex_path = osp.join(data_root, ident, "w_mean_tex.png")
    tex_mean = typed.load(tex_path)[..., :3] # (1024, 1024, 3)

    # uv
    uv_img_path = osp.join(data_root, ident, segment, "vt_img", f"{frame:06d}.png")
    uv = _load_uv(uv_img_path) # (h, w, 2) & float32

    # normal
    nrml = vn_img_cont.load(f"{segment}/{frame:06d}.npy") # (h, w, 3) & float32

    img = nrml.copy()

    # vertex
    vertex = v_img_cont.load(f"{segment}/{frame:06d}.npy") # (h, w, 3) & float32

    # mean tex image
    tex_mean_path = osp.join(data_root, ident, segment, "tex_mean", f"{frame:06d}.png")
    tex_mean_img = typed.load(tex_mean_path) # (h, w, 3) & uint8

    # exp tex image
    tex_exp_path = osp.join(data_root, ident, segment, "tex_exp", f"{frame:06d}.png")
    tex_exp_img = typed.load(tex_exp_path) # (h, w, 3) & uint8

    # mask
    mask_path = osp.join(data_root, ident, segment, "mask", f"{frame:06d}.png")
    mask = typed.load(mask_path) # (h, w, 1) & uint8
    mask = np.repeat(mask, 3, axis=2)

    # neck weights
    neck_weights_path = osp.join(data_root, ident, segment, "neck_weights", f"{frame:06d}.png")
    neck_weights = typed.load(neck_weights_path) # (h, w, 1) & uint8
    neck_weights = np.repeat(neck_weights, 3, axis=2)

    data_raw = dict(
        img=img,
        uv=uv,
        nrml=nrml,
        vertex=vertex,
        tex_mean=tex_mean,
        tex_mean_img=tex_mean_img,
        tex_exp_img=tex_exp_img,
        mask=mask,
        neck_weights=neck_weights
    )
    return data_raw


def visualize(res_path: str, data_raw: Dict[str, np.ndarray]) -> None:
    img = data_raw["img"]
    uv = data_raw["uv"]
    nrml = data_raw["nrml"]
    vertex = data_raw["vertex"]
    tex_mean = data_raw["tex_mean"]
    tex_mean_img = data_raw["tex_mean_img"]
    tex_exp_img = data_raw["tex_exp_img"]
    mask = data_raw["mask"]
    neck_weights = data_raw["neck_weights"]

    # visualize uv
    mask_t = th.tensor(mask)[None].permute(0, 3, 1, 2).float() / 255.0
    uv_t = th.tensor(uv)[None].float() # (1, h, w, 2)
    tex_mean_t = th.tensor(tex_mean)[None].permute(0, 3, 1, 2).float() # (1, 3, h, w)
    uv_vis = thf.grid_sample(tex_mean_t, uv_t, align_corners=False) * mask_t
    uv_vis = uv_vis[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    # visualize nrml
    nrml_vis = ((nrml + 1) / 2).clip(0, 1)
    nrml_vis[..., 2] = (1.0 - (nrml_vis[..., 2]))
    nrml_vis = (nrml_vis * 255.0).astype(np.uint8)

    # visualize vertex
    v_min = vertex.min()
    v_max = vertex.max()
    vertex_vis = (vertex - v_min) / (v_max - v_min) * 255.0
    vertex_vis = vertex_vis.clip(0, 255).astype(np.uint8)

    # compose all visualization
    img_list = [img, tex_mean_img, tex_exp_img, uv_vis, nrml_vis, vertex_vis, mask, neck_weights]
    for img in img_list:
        print(img.shape, img.dtype)
    img_vis = np.concatenate(img_list, axis=1)
    typed.save(res_path, img_vis)


def main():
    # identity, segment, frame id
    ident = "1189279795225648"
    segment = "dynamic_range-of-motion-1"
    frame = 0

    # path
    # data_root = "/mnt/home/rongyu/data/ica/mgr_new"
    data_root = '/uca/rongyu/data/mgr'

    # load raw data
    vn_img_cont = typed.open_container(
        osp.join(data_root, ident, "vn_img.zip"),
        mode="r"
    )
    v_img_cont = typed.open_container(
        osp.join(data_root, ident, "v_img.zip"),
        mode="r"
    )
    data_raw = load_data_raw(data_root, vn_img_cont, v_img_cont, ident, segment, frame)

    # visualize
    res_vis_path = "runs/tmp/0.jpg"
    visualize(res_vis_path, data_raw)


if __name__ == '__main__':
    main()
