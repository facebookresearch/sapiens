# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from mmengine.model import is_model_wrapper
from mmengine.runner import TestLoop, ValLoop, autocast

from mmpretrain.registry import LOOPS


@LOOPS.register_module()
class RetrievalValLoop(ValLoop):
    """Loop for multimodal retrieval val.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 valing. Defaults to
            False.
    """

    def run(self) -> dict:
        """Launch val."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()

        feats_local = []
        data_samples_local = []

        for idx, data_batch in enumerate(self.dataloader):
            with torch.no_grad():
                self.runner.call_hook(
                    'before_val_iter', batch_idx=idx, data_batch=data_batch)
                # predictions should be sequence of BaseDataElement
                with autocast(enabled=self.fp16):
                    if is_model_wrapper(self.runner.model):
                        data_preprocessor = self.runner.model.module.data_preprocessor  # noqa: E501
                    else:
                        data_preprocessor = self.runner.model.data_preprocessor

                    # get features for retrieval instead of data samples
                    data_batch = data_preprocessor(data_batch, False)
                    feats = self.runner.model._run_forward(
                        data_batch, mode='tensor')
                    feats_local.append(feats)
                    data_samples_local.extend(data_batch['data_samples'])
                self.runner.call_hook(
                    'after_val_iter',
                    batch_idx=idx,
                    data_batch=data_batch,
                    outputs=feats)

        # concatenate different features
        feats_local = {
            k: torch.cat([dic[k] for dic in feats_local])
            for k in feats_local[0]
        }

        # get predictions
        if is_model_wrapper(self.runner.model):
            predict_all_fn = self.runner.model.module.predict_all
        else:
            predict_all_fn = self.runner.model.predict_all

        img_size = self.dataloader.dataset.img_size
        text_size = self.dataloader.dataset.text_size
        with torch.no_grad():
            i2t_data_samples, t2i_data_samples = predict_all_fn(
                feats_local,
                data_samples_local,
                num_images=img_size,
                num_texts=text_size,
            )

        # process in evaluator and compute metrics
        self.evaluator.process(i2t_data_samples, None)
        i2t_metrics = self.evaluator.evaluate(img_size)
        i2t_metrics = {f'i2t/{k}': v for k, v in i2t_metrics.items()}
        self.evaluator.process(t2i_data_samples, None)
        t2i_metrics = self.evaluator.evaluate(text_size)
        t2i_metrics = {f't2i/{k}': v for k, v in t2i_metrics.items()}
        metrics = {**i2t_metrics, **t2i_metrics}

        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')
        return metrics


@LOOPS.register_module()
class RetrievalTestLoop(TestLoop):
    """Loop for multimodal retrieval test.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 testing. Defaults to
            False.
    """

    def run(self) -> dict:
        """Launch test."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()

        feats_local = []
        data_samples_local = []

        for idx, data_batch in enumerate(self.dataloader):
            with torch.no_grad():
                self.runner.call_hook(
                    'before_test_iter', batch_idx=idx, data_batch=data_batch)
                # predictions should be sequence of BaseDataElement
                with autocast(enabled=self.fp16):
                    if is_model_wrapper(self.runner.model):
                        data_preprocessor = self.runner.model.module.data_preprocessor  # noqa: E501
                    else:
                        data_preprocessor = self.runner.model.data_preprocessor
                    # get features for retrieval instead of data samples
                    data_batch = data_preprocessor(data_batch, False)
                    feats = self.runner.model._run_forward(
                        data_batch, mode='tensor')
                    feats_local.append(feats)
                    data_samples_local.extend(data_batch['data_samples'])
                self.runner.call_hook(
                    'after_test_iter',
                    batch_idx=idx,
                    data_batch=data_batch,
                    outputs=feats)

        # concatenate different features
        feats_local = {
            k: torch.cat([dic[k] for dic in feats_local])
            for k in feats_local[0]
        }

        # get predictions
        if is_model_wrapper(self.runner.model):
            predict_all_fn = self.runner.model.module.predict_all
        else:
            predict_all_fn = self.runner.model.predict_all

        img_size = self.dataloader.dataset.img_size
        text_size = self.dataloader.dataset.text_size
        with torch.no_grad():
            i2t_data_samples, t2i_data_samples = predict_all_fn(
                feats_local,
                data_samples_local,
                num_images=img_size,
                num_texts=text_size,
            )

        # process in evaluator and compute metrics
        self.evaluator.process(i2t_data_samples, None)
        i2t_metrics = self.evaluator.evaluate(img_size)
        i2t_metrics = {f'i2t/{k}': v for k, v in i2t_metrics.items()}
        self.evaluator.process(t2i_data_samples, None)
        t2i_metrics = self.evaluator.evaluate(text_size)
        t2i_metrics = {f't2i/{k}': v for k, v in t2i_metrics.items()}
        metrics = {**i2t_metrics, **t2i_metrics}

        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')
        return metrics
