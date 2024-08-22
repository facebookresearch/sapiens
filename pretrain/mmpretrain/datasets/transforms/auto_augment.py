# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from copy import deepcopy
from math import ceil
from numbers import Number
from typing import List, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
from mmcv.transforms import BaseTransform, Compose, RandomChoice
from mmcv.transforms.utils import cache_randomness
from mmengine.utils import is_list_of, is_seq_of
from PIL import Image, ImageFilter

from mmpretrain.registry import TRANSFORMS


def merge_hparams(policy: dict, hparams: dict) -> dict:
    """Merge hyperparameters into policy config.

    Only merge partial hyperparameters required of the policy.

    Args:
        policy (dict): Original policy config dict.
        hparams (dict): Hyperparameters need to be merged.

    Returns:
        dict: Policy config dict after adding ``hparams``.
    """
    policy = deepcopy(policy)
    op = TRANSFORMS.get(policy['type'])
    assert op is not None, f'Invalid policy type "{policy["type"]}".'

    op_args = inspect.getfullargspec(op.__init__).args
    for key, value in hparams.items():
        if key in op_args and key not in policy:
            policy[key] = value
    return policy


@TRANSFORMS.register_module()
class AutoAugment(RandomChoice):
    """Auto augmentation.

    This data augmentation is proposed in `AutoAugment: Learning Augmentation
    Policies from Data <https://arxiv.org/abs/1805.09501>`_.

    Args:
        policies (str | list[list[dict]]): The policies of auto augmentation.
            If string, use preset policies collection like "imagenet". If list,
            Each item is a sub policies, composed by several augmentation
            policy dicts. When AutoAugment is called, a random sub policies in
            ``policies`` will be selected to augment images.
        hparams (dict): Configs of hyperparameters. Hyperparameters will be
            used in policies that require these arguments if these arguments
            are not set in policy dicts. Defaults to ``dict(pad_val=128)``.

    .. admonition:: Available preset policies

        - ``"imagenet"``: Policy for ImageNet, come from
          `DeepVoltaire/AutoAugment`_

    .. _DeepVoltaire/AutoAugment: https://github.com/DeepVoltaire/AutoAugment
    """

    def __init__(self,
                 policies: Union[str, List[List[dict]]],
                 hparams: dict = dict(pad_val=128)):
        if isinstance(policies, str):
            assert policies in AUTOAUG_POLICIES, 'Invalid policies, ' \
                f'please choose from {list(AUTOAUG_POLICIES.keys())}.'
            policies = AUTOAUG_POLICIES[policies]
        self.hparams = hparams
        self.policies = [[merge_hparams(t, hparams) for t in sub]
                         for sub in policies]
        transforms = [[TRANSFORMS.build(t) for t in sub] for sub in policies]

        super().__init__(transforms=transforms)

    def __repr__(self) -> str:
        policies_str = ''
        for sub in self.policies:
            policies_str += '\n    ' + ', \t'.join([t['type'] for t in sub])

        repr_str = self.__class__.__name__
        repr_str += f'(policies:{policies_str}\n)'
        return repr_str


@TRANSFORMS.register_module()
class RandAugment(BaseTransform):
    r"""Random augmentation.

    This data augmentation is proposed in `RandAugment: Practical automated
    data augmentation with a reduced search space
    <https://arxiv.org/abs/1909.13719>`_.

    Args:
        policies (str | list[dict]): The policies of random augmentation.
            If string, use preset policies collection like "timm_increasing".
            If list, each item is one specific augmentation policy dict.
            The policy dict shall should have these keys:

            - ``type`` (str), The type of augmentation.
            - ``magnitude_range`` (Sequence[number], optional): For those
              augmentation have magnitude, you need to specify the magnitude
              level mapping range. For example, assume ``total_level`` is 10,
              ``magnitude_level=3`` specify magnitude is 3 if
              ``magnitude_range=(0, 10)`` while specify magnitude is 7 if
              ``magnitude_range=(10, 0)``.
            - other keyword arguments of the augmentation.

        num_policies (int): Number of policies to select from policies each
            time.
        magnitude_level (int | float): Magnitude level for all the augmentation
            selected.
        magnitude_std (Number | str): Deviation of magnitude noise applied.

            - If positive number, the magnitude obeys normal distribution
              :math:`\mathcal{N}(magnitude_level, magnitude_std)`.
            - If 0 or negative number, magnitude remains unchanged.
            - If str "inf", the magnitude obeys uniform distribution
              :math:`Uniform(min, magnitude)`.
        total_level (int | float): Total level for the magnitude. Defaults to
            10.
        hparams (dict): Configs of hyperparameters. Hyperparameters will be
            used in policies that require these arguments if these arguments
            are not set in policy dicts. Defaults to ``dict(pad_val=128)``.

    .. admonition:: Available preset policies

        - ``"timm_increasing"``: The ``_RAND_INCREASING_TRANSFORMS`` policy
          from `timm`_

    .. _timm: https://github.com/rwightman/pytorch-image-models

    Examples:

        To use "timm-increasing" policies collection, select two policies every
        time, and magnitude_level of every policy is 6 (total is 10 by default)

        >>> import numpy as np
        >>> from mmpretrain.datasets import RandAugment
        >>> transform = RandAugment(
        ...     policies='timm_increasing',
        ...     num_policies=2,
        ...     magnitude_level=6,
        ... )
        >>> data = {'img': np.random.randint(0, 256, (224, 224, 3))}
        >>> results = transform(data)
        >>> print(results['img'].shape)
        (224, 224, 3)

        If you want the ``magnitude_level`` randomly changes every time, you
        can use ``magnitude_std`` to specify the random distribution. For
        example, a normal distribution :math:`\mathcal{N}(6, 0.5)`.

        >>> transform = RandAugment(
        ...     policies='timm_increasing',
        ...     num_policies=2,
        ...     magnitude_level=6,
        ...     magnitude_std=0.5,
        ... )

        You can also use your own policies:

        >>> policies = [
        ...     dict(type='AutoContrast'),
        ...     dict(type='Rotate', magnitude_range=(0, 30)),
        ...     dict(type='ColorTransform', magnitude_range=(0, 0.9)),
        ... ]
        >>> transform = RandAugment(
        ...     policies=policies,
        ...     num_policies=2,
        ...     magnitude_level=6
        ... )

    Note:
        ``magnitude_std`` will introduce some randomness to policy, modified by
        https://github.com/rwightman/pytorch-image-models.

        When magnitude_std=0, we calculate the magnitude as follows:

        .. math::
            \text{magnitude} = \frac{\text{magnitude_level}}
            {\text{totallevel}} \times (\text{val2} - \text{val1})
            + \text{val1}
    """

    def __init__(self,
                 policies: Union[str, List[dict]],
                 num_policies: int,
                 magnitude_level: int,
                 magnitude_std: Union[Number, str] = 0.,
                 total_level: int = 10,
                 hparams: dict = dict(pad_val=128)):
        if isinstance(policies, str):
            assert policies in RANDAUG_POLICIES, 'Invalid policies, ' \
                f'please choose from {list(RANDAUG_POLICIES.keys())}.'
            policies = RANDAUG_POLICIES[policies]

        assert is_list_of(policies, dict), 'policies must be a list of dict.'

        assert isinstance(magnitude_std, (Number, str)), \
            '`magnitude_std` must be of number or str type, ' \
            f'got {type(magnitude_std)} instead.'
        if isinstance(magnitude_std, str):
            assert magnitude_std == 'inf', \
                '`magnitude_std` must be of number or "inf", ' \
                f'got "{magnitude_std}" instead.'

        assert num_policies > 0, 'num_policies must be greater than 0.'
        assert magnitude_level >= 0, 'magnitude_level must be no less than 0.'
        assert total_level > 0, 'total_level must be greater than 0.'

        self.num_policies = num_policies
        self.magnitude_level = magnitude_level
        self.magnitude_std = magnitude_std
        self.total_level = total_level
        self.hparams = hparams
        self.policies = []
        self.transforms = []

        randaug_cfg = dict(
            magnitude_level=magnitude_level,
            total_level=total_level,
            magnitude_std=magnitude_std)

        for policy in policies:
            self._check_policy(policy)
            policy = merge_hparams(policy, hparams)
            policy.pop('magnitude_key', None)  # For backward compatibility
            if 'magnitude_range' in policy:
                policy.update(randaug_cfg)
            self.policies.append(policy)
            self.transforms.append(TRANSFORMS.build(policy))

    def __iter__(self):
        """Iterate all transforms."""
        return iter(self.transforms)

    def _check_policy(self, policy):
        """Check whether the sub-policy dict is available."""
        assert isinstance(policy, dict) and 'type' in policy, \
            'Each policy must be a dict with key "type".'
        type_name = policy['type']

        if 'magnitude_range' in policy:
            magnitude_range = policy['magnitude_range']
            assert is_seq_of(magnitude_range, Number), \
                f'`magnitude_range` of RandAugment policy {type_name} ' \
                'should be a sequence with two numbers.'

    @cache_randomness
    def random_policy_indices(self) -> np.ndarray:
        """Return the random chosen transform indices."""
        indices = np.arange(len(self.policies))
        return np.random.choice(indices, size=self.num_policies).tolist()

    def transform(self, results: dict) -> Optional[dict]:
        """Randomly choose a sub-policy to apply."""

        chosen_policies = [
            self.transforms[i] for i in self.random_policy_indices()
        ]

        sub_pipeline = Compose(chosen_policies)
        return sub_pipeline(results)

    def __repr__(self) -> str:
        policies_str = ''
        for policy in self.policies:
            policies_str += '\n    ' + f'{policy["type"]}'
            if 'magnitude_range' in policy:
                val1, val2 = policy['magnitude_range']
                policies_str += f' ({val1}, {val2})'

        repr_str = self.__class__.__name__
        repr_str += f'(num_policies={self.num_policies}, '
        repr_str += f'magnitude_level={self.magnitude_level}, '
        repr_str += f'total_level={self.total_level}, '
        repr_str += f'policies:{policies_str}\n)'
        return repr_str


class BaseAugTransform(BaseTransform):
    r"""The base class of augmentation transform for RandAugment.

    This class provides several common attributions and methods to support the
    magnitude level mapping and magnitude level randomness in
    :class:`RandAugment`.

    Args:
        magnitude_level (int | float): Magnitude level.
        magnitude_range (Sequence[number], optional): For augmentation have
            magnitude argument, maybe "magnitude", "angle" or other, you can
            specify the magnitude level mapping range to generate the magnitude
            argument. For example, assume ``total_level`` is 10,
            ``magnitude_level=3`` specify magnitude is 3 if
            ``magnitude_range=(0, 10)`` while specify magnitude is 7 if
            ``magnitude_range=(10, 0)``. Defaults to None.
        magnitude_std (Number | str): Deviation of magnitude noise applied.

            - If positive number, the magnitude obeys normal distribution
              :math:`\mathcal{N}(magnitude, magnitude_std)`.
            - If 0 or negative number, magnitude remains unchanged.
            - If str "inf", the magnitude obeys uniform distribution
              :math:`Uniform(min, magnitude)`.

            Defaults to 0.
        total_level (int | float): Total level for the magnitude. Defaults to
            10.
        prob (float): The probability for performing transformation therefore
            should be in range [0, 1]. Defaults to 0.5.
        random_negative_prob (float): The probability that turns the magnitude
            negative, which should be in range [0,1]. Defaults to 0.
    """

    def __init__(self,
                 magnitude_level: int = 10,
                 magnitude_range: Tuple[float, float] = None,
                 magnitude_std: Union[str, float] = 0.,
                 total_level: int = 10,
                 prob: float = 0.5,
                 random_negative_prob: float = 0.5):
        self.magnitude_level = magnitude_level
        self.magnitude_range = magnitude_range
        self.magnitude_std = magnitude_std
        self.total_level = total_level
        self.prob = prob
        self.random_negative_prob = random_negative_prob

    @cache_randomness
    def random_disable(self):
        """Randomly disable the transform."""
        return np.random.rand() > self.prob

    @cache_randomness
    def random_magnitude(self):
        """Randomly generate magnitude."""
        magnitude = self.magnitude_level
        # if magnitude_std is positive number or 'inf', move
        # magnitude_value randomly.
        if self.magnitude_std == 'inf':
            magnitude = np.random.uniform(0, magnitude)
        elif self.magnitude_std > 0:
            magnitude = np.random.normal(magnitude, self.magnitude_std)
            magnitude = np.clip(magnitude, 0, self.total_level)

        val1, val2 = self.magnitude_range
        magnitude = (magnitude / self.total_level) * (val2 - val1) + val1
        return magnitude

    @cache_randomness
    def random_negative(self, value):
        """Randomly negative the value."""
        if np.random.rand() < self.random_negative_prob:
            return -value
        else:
            return value

    def extra_repr(self):
        """Extra repr string when auto-generating magnitude is enabled."""
        if self.magnitude_range is not None:
            repr_str = f', magnitude_level={self.magnitude_level}, '
            repr_str += f'magnitude_range={self.magnitude_range}, '
            repr_str += f'magnitude_std={self.magnitude_std}, '
            repr_str += f'total_level={self.total_level}, '
            return repr_str
        else:
            return ''


@TRANSFORMS.register_module()
class Shear(BaseAugTransform):
    """Shear images.

    Args:
        magnitude (int | float | None): The magnitude used for shear. If None,
            generate from ``magnitude_range``, see :class:`BaseAugTransform`.
            Defaults to None.
        pad_val (int, Sequence[int]): Pixel pad_val value for constant fill.
            If a sequence of length 3, it is used to pad_val R, G, B channels
            respectively. Defaults to 128.
        prob (float): The probability for performing shear therefore should be
            in range [0, 1]. Defaults to 0.5.
        direction (str): The shearing direction. Options are 'horizontal' and
            'vertical'. Defaults to 'horizontal'.
        random_negative_prob (float): The probability that turns the magnitude
            negative, which should be in range [0,1]. Defaults to 0.5.
        interpolation (str): Interpolation method. Options are 'nearest',
            'bilinear', 'bicubic', 'area', 'lanczos'. Defaults to 'bicubic'.
        **kwargs: Other keyword arguments of :class:`BaseAugTransform`.
    """

    def __init__(self,
                 magnitude: Union[int, float, None] = None,
                 pad_val: Union[int, Sequence[int]] = 128,
                 prob: float = 0.5,
                 direction: str = 'horizontal',
                 random_negative_prob: float = 0.5,
                 interpolation: str = 'bicubic',
                 **kwargs):
        super().__init__(
            prob=prob, random_negative_prob=random_negative_prob, **kwargs)
        assert (magnitude is None) ^ (self.magnitude_range is None), \
            'Please specify only one of `magnitude` and `magnitude_range`.'

        self.magnitude = magnitude
        if isinstance(pad_val, Sequence):
            self.pad_val = tuple(pad_val)
        else:
            self.pad_val = pad_val

        assert direction in ('horizontal', 'vertical'), 'direction must be ' \
            f'either "horizontal" or "vertical", got "{direction}" instead.'
        self.direction = direction

        self.interpolation = interpolation

    def transform(self, results):
        """Apply transform to results."""
        if self.random_disable():
            return results

        if self.magnitude is not None:
            magnitude = self.random_negative(self.magnitude)
        else:
            magnitude = self.random_negative(self.random_magnitude())

        img = results['img']
        img_sheared = mmcv.imshear(
            img,
            magnitude,
            direction=self.direction,
            border_value=self.pad_val,
            interpolation=self.interpolation)
        results['img'] = img_sheared.astype(img.dtype)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(magnitude={self.magnitude}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'direction={self.direction}, '
        repr_str += f'random_negative_prob={self.random_negative_prob}, '
        repr_str += f'interpolation={self.interpolation}{self.extra_repr()})'
        return repr_str


@TRANSFORMS.register_module()
class Translate(BaseAugTransform):
    """Translate images.

    Args:
        magnitude (int | float | None): The magnitude used for translate. Note
            that the offset is calculated by magnitude * size in the
            corresponding direction. With a magnitude of 1, the whole image
            will be moved out of the range. If None, generate from
            ``magnitude_range``, see :class:`BaseAugTransform`.
        pad_val (int, Sequence[int]): Pixel pad_val value for constant fill.
            If a sequence of length 3, it is used to pad_val R, G, B channels
            respectively. Defaults to 128.
        prob (float): The probability for performing translate therefore should
             be in range [0, 1]. Defaults to 0.5.
        direction (str): The translating direction. Options are 'horizontal'
            and 'vertical'. Defaults to 'horizontal'.
        random_negative_prob (float): The probability that turns the magnitude
            negative, which should be in range [0,1]. Defaults to 0.5.
        interpolation (str): Interpolation method. Options are 'nearest',
            'bilinear', 'bicubic', 'area', 'lanczos'. Defaults to 'nearest'.
        **kwargs: Other keyword arguments of :class:`BaseAugTransform`.
    """

    def __init__(self,
                 magnitude: Union[int, float, None] = None,
                 pad_val: Union[int, Sequence[int]] = 128,
                 prob: float = 0.5,
                 direction: str = 'horizontal',
                 random_negative_prob: float = 0.5,
                 interpolation: str = 'nearest',
                 **kwargs):
        super().__init__(
            prob=prob, random_negative_prob=random_negative_prob, **kwargs)
        assert (magnitude is None) ^ (self.magnitude_range is None), \
            'Please specify only one of `magnitude` and `magnitude_range`.'

        self.magnitude = magnitude
        if isinstance(pad_val, Sequence):
            self.pad_val = tuple(pad_val)
        else:
            self.pad_val = pad_val

        assert direction in ('horizontal', 'vertical'), 'direction must be ' \
            f'either "horizontal" or "vertical", got "{direction}" instead.'
        self.direction = direction

        self.interpolation = interpolation

    def transform(self, results):
        """Apply transform to results."""
        if self.random_disable():
            return results

        if self.magnitude is not None:
            magnitude = self.random_negative(self.magnitude)
        else:
            magnitude = self.random_negative(self.random_magnitude())

        img = results['img']
        height, width = img.shape[:2]
        if self.direction == 'horizontal':
            offset = magnitude * width
        else:
            offset = magnitude * height
        img_translated = mmcv.imtranslate(
            img,
            offset,
            direction=self.direction,
            border_value=self.pad_val,
            interpolation=self.interpolation)
        results['img'] = img_translated.astype(img.dtype)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(magnitude={self.magnitude}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'direction={self.direction}, '
        repr_str += f'random_negative_prob={self.random_negative_prob}, '
        repr_str += f'interpolation={self.interpolation}{self.extra_repr()})'
        return repr_str


@TRANSFORMS.register_module()
class Rotate(BaseAugTransform):
    """Rotate images.

    Args:
        angle (float, optional): The angle used for rotate. Positive values
            stand for clockwise rotation. If None, generate from
            ``magnitude_range``, see :class:`BaseAugTransform`.
            Defaults to None.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If None, the center of the image will be used.
            Defaults to None.
        scale (float): Isotropic scale factor. Defaults to 1.0.
        pad_val (int, Sequence[int]): Pixel pad_val value for constant fill.
            If a sequence of length 3, it is used to pad_val R, G, B channels
            respectively. Defaults to 128.
        prob (float): The probability for performing rotate therefore should be
            in range [0, 1]. Defaults to 0.5.
        random_negative_prob (float): The probability that turns the angle
            negative, which should be in range [0,1]. Defaults to 0.5.
        interpolation (str): Interpolation method. Options are 'nearest',
            'bilinear', 'bicubic', 'area', 'lanczos'. Defaults to 'nearest'.
        **kwargs: Other keyword arguments of :class:`BaseAugTransform`.
    """

    def __init__(self,
                 angle: Optional[float] = None,
                 center: Optional[Tuple[float]] = None,
                 scale: float = 1.0,
                 pad_val: Union[int, Sequence[int]] = 128,
                 prob: float = 0.5,
                 random_negative_prob: float = 0.5,
                 interpolation: str = 'nearest',
                 **kwargs):
        super().__init__(
            prob=prob, random_negative_prob=random_negative_prob, **kwargs)
        assert (angle is None) ^ (self.magnitude_range is None), \
            'Please specify only one of `angle` and `magnitude_range`.'

        self.angle = angle
        self.center = center
        self.scale = scale
        if isinstance(pad_val, Sequence):
            self.pad_val = tuple(pad_val)
        else:
            self.pad_val = pad_val

        self.interpolation = interpolation

    def transform(self, results):
        """Apply transform to results."""
        if self.random_disable():
            return results

        if self.angle is not None:
            angle = self.random_negative(self.angle)
        else:
            angle = self.random_negative(self.random_magnitude())

        img = results['img']
        img_rotated = mmcv.imrotate(
            img,
            angle,
            center=self.center,
            scale=self.scale,
            border_value=self.pad_val,
            interpolation=self.interpolation)
        results['img'] = img_rotated.astype(img.dtype)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(angle={self.angle}, '
        repr_str += f'center={self.center}, '
        repr_str += f'scale={self.scale}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'random_negative_prob={self.random_negative_prob}, '
        repr_str += f'interpolation={self.interpolation}{self.extra_repr()})'
        return repr_str


@TRANSFORMS.register_module()
class AutoContrast(BaseAugTransform):
    """Auto adjust image contrast.

    Args:
        prob (float): The probability for performing auto contrast
            therefore should be in range [0, 1]. Defaults to 0.5.
        **kwargs: Other keyword arguments of :class:`BaseAugTransform`.
    """

    def __init__(self, prob: float = 0.5, **kwargs):
        super().__init__(prob=prob, **kwargs)

    def transform(self, results):
        """Apply transform to results."""
        if self.random_disable():
            return results

        img = results['img']
        img_contrasted = mmcv.auto_contrast(img)
        results['img'] = img_contrasted.astype(img.dtype)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob})'
        return repr_str


@TRANSFORMS.register_module()
class Invert(BaseAugTransform):
    """Invert images.

    Args:
        prob (float): The probability for performing invert therefore should
             be in range [0, 1]. Defaults to 0.5.
        **kwargs: Other keyword arguments of :class:`BaseAugTransform`.
    """

    def __init__(self, prob: float = 0.5, **kwargs):
        super().__init__(prob=prob, **kwargs)

    def transform(self, results):
        """Apply transform to results."""
        if self.random_disable():
            return results

        img = results['img']
        img_inverted = mmcv.iminvert(img)
        results['img'] = img_inverted.astype(img.dtype)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob})'
        return repr_str


@TRANSFORMS.register_module()
class Equalize(BaseAugTransform):
    """Equalize the image histogram.

    Args:
        prob (float): The probability for performing equalize therefore should
             be in range [0, 1]. Defaults to 0.5.
        **kwargs: Other keyword arguments of :class:`BaseAugTransform`.
    """

    def __init__(self, prob: float = 0.5, **kwargs):
        super().__init__(prob=prob, **kwargs)

    def transform(self, results):
        """Apply transform to results."""
        if self.random_disable():
            return results

        img = results['img']
        img_equalized = mmcv.imequalize(img)
        results['img'] = img_equalized.astype(img.dtype)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob})'
        return repr_str


@TRANSFORMS.register_module()
class Solarize(BaseAugTransform):
    """Solarize images (invert all pixel values above a threshold).

    Args:
        thr (int | float | None): The threshold above which the pixels value
            will be inverted. If None, generate from ``magnitude_range``,
            see :class:`BaseAugTransform`. Defaults to None.
        prob (float): The probability for solarizing therefore should be in
            range [0, 1]. Defaults to 0.5.
        **kwargs: Other keyword arguments of :class:`BaseAugTransform`.
    """

    def __init__(self,
                 thr: Union[int, float, None] = None,
                 prob: float = 0.5,
                 **kwargs):
        super().__init__(prob=prob, random_negative_prob=0., **kwargs)
        assert (thr is None) ^ (self.magnitude_range is None), \
            'Please specify only one of `thr` and `magnitude_range`.'

        self.thr = thr

    def transform(self, results):
        """Apply transform to results."""
        if self.random_disable():
            return results

        if self.thr is not None:
            thr = self.thr
        else:
            thr = self.random_magnitude()

        img = results['img']
        img_solarized = mmcv.solarize(img, thr=thr)
        results['img'] = img_solarized.astype(img.dtype)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(thr={self.thr}, '
        repr_str += f'prob={self.prob}{self.extra_repr()}))'
        return repr_str


@TRANSFORMS.register_module()
class SolarizeAdd(BaseAugTransform):
    """SolarizeAdd images (add a certain value to pixels below a threshold).

    Args:
        magnitude (int | float | None): The value to be added to pixels below
            the thr. If None, generate from ``magnitude_range``, see
            :class:`BaseAugTransform`. Defaults to None.
        thr (int | float): The threshold below which the pixels value will be
            adjusted.
        prob (float): The probability for solarizing therefore should be in
            range [0, 1]. Defaults to 0.5.
        **kwargs: Other keyword arguments of :class:`BaseAugTransform`.
    """

    def __init__(self,
                 magnitude: Union[int, float, None] = None,
                 thr: Union[int, float] = 128,
                 prob: float = 0.5,
                 **kwargs):
        super().__init__(prob=prob, random_negative_prob=0., **kwargs)
        assert (magnitude is None) ^ (self.magnitude_range is None), \
            'Please specify only one of `magnitude` and `magnitude_range`.'

        self.magnitude = magnitude

        assert isinstance(thr, (int, float)), 'The thr type must '\
            f'be int or float, but got {type(thr)} instead.'
        self.thr = thr

    def transform(self, results):
        """Apply transform to results."""
        if self.random_disable():
            return results

        if self.magnitude is not None:
            magnitude = self.magnitude
        else:
            magnitude = self.random_magnitude()

        img = results['img']
        img_solarized = np.where(img < self.thr,
                                 np.minimum(img + magnitude, 255), img)
        results['img'] = img_solarized.astype(img.dtype)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(magnitude={self.magnitude}, '
        repr_str += f'thr={self.thr}, '
        repr_str += f'prob={self.prob}{self.extra_repr()})'
        return repr_str


@TRANSFORMS.register_module()
class Posterize(BaseAugTransform):
    """Posterize images (reduce the number of bits for each color channel).

    Args:
        bits (int, optional): Number of bits for each pixel in the output img,
            which should be less or equal to 8. If None, generate from
            ``magnitude_range``, see :class:`BaseAugTransform`.
            Defaults to None.
        prob (float): The probability for posterizing therefore should be in
            range [0, 1]. Defaults to 0.5.
        **kwargs: Other keyword arguments of :class:`BaseAugTransform`.
    """

    def __init__(self,
                 bits: Optional[int] = None,
                 prob: float = 0.5,
                 **kwargs):
        super().__init__(prob=prob, random_negative_prob=0., **kwargs)
        assert (bits is None) ^ (self.magnitude_range is None), \
            'Please specify only one of `bits` and `magnitude_range`.'

        if bits is not None:
            assert bits <= 8, \
                f'The bits must be less than 8, got {bits} instead.'
        self.bits = bits

    def transform(self, results):
        """Apply transform to results."""
        if self.random_disable():
            return results

        if self.bits is not None:
            bits = self.bits
        else:
            bits = self.random_magnitude()

        # To align timm version, we need to round up to integer here.
        bits = ceil(bits)

        img = results['img']
        img_posterized = mmcv.posterize(img, bits=bits)
        results['img'] = img_posterized.astype(img.dtype)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(bits={self.bits}, '
        repr_str += f'prob={self.prob}{self.extra_repr()})'
        return repr_str


@TRANSFORMS.register_module()
class Contrast(BaseAugTransform):
    """Adjust images contrast.

    Args:
        magnitude (int | float | None): The magnitude used for adjusting
            contrast. A positive magnitude would enhance the contrast and
            a negative magnitude would make the image grayer. A magnitude=0
            gives the origin img. If None, generate from ``magnitude_range``,
            see :class:`BaseAugTransform`. Defaults to None.
        prob (float): The probability for performing contrast adjusting
            therefore should be in range [0, 1]. Defaults to 0.5.
        random_negative_prob (float): The probability that turns the magnitude
            negative, which should be in range [0,1]. Defaults to 0.5.
    """

    def __init__(self,
                 magnitude: Union[int, float, None] = None,
                 prob: float = 0.5,
                 random_negative_prob: float = 0.5,
                 **kwargs):
        super().__init__(
            prob=prob, random_negative_prob=random_negative_prob, **kwargs)
        assert (magnitude is None) ^ (self.magnitude_range is None), \
            'Please specify only one of `magnitude` and `magnitude_range`.'

        self.magnitude = magnitude

    def transform(self, results):
        """Apply transform to results."""
        if self.random_disable():
            return results

        if self.magnitude is not None:
            magnitude = self.random_negative(self.magnitude)
        else:
            magnitude = self.random_negative(self.random_magnitude())

        img = results['img']
        img_contrasted = mmcv.adjust_contrast(img, factor=1 + magnitude)
        results['img'] = img_contrasted.astype(img.dtype)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(magnitude={self.magnitude}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'random_negative_prob={self.random_negative_prob}'
        repr_str += f'{self.extra_repr()})'
        return repr_str


@TRANSFORMS.register_module()
class ColorTransform(BaseAugTransform):
    """Adjust images color balance.

    Args:
        magnitude (int | float | None): The magnitude used for color transform.
            A positive magnitude would enhance the color and a negative
            magnitude would make the image grayer. A magnitude=0 gives the
            origin img. If None, generate from ``magnitude_range``, see
            :class:`BaseAugTransform`. Defaults to None.
        prob (float): The probability for performing ColorTransform therefore
            should be in range [0, 1]. Defaults to 0.5.
        random_negative_prob (float): The probability that turns the magnitude
            negative, which should be in range [0,1]. Defaults to 0.5.
        **kwargs: Other keyword arguments of :class:`BaseAugTransform`.
    """

    def __init__(self,
                 magnitude: Union[int, float, None] = None,
                 prob: float = 0.5,
                 random_negative_prob: float = 0.5,
                 **kwargs):
        super().__init__(
            prob=prob, random_negative_prob=random_negative_prob, **kwargs)
        assert (magnitude is None) ^ (self.magnitude_range is None), \
            'Please specify only one of `magnitude` and `magnitude_range`.'

        self.magnitude = magnitude

    def transform(self, results):
        """Apply transform to results."""
        if self.random_disable():
            return results

        if self.magnitude is not None:
            magnitude = self.random_negative(self.magnitude)
        else:
            magnitude = self.random_negative(self.random_magnitude())

        img = results['img']
        img_color_adjusted = mmcv.adjust_color(img, alpha=1 + magnitude)
        results['img'] = img_color_adjusted.astype(img.dtype)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(magnitude={self.magnitude}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'random_negative_prob={self.random_negative_prob}'
        repr_str += f'{self.extra_repr()})'
        return repr_str


@TRANSFORMS.register_module()
class Brightness(BaseAugTransform):
    """Adjust images brightness.

    Args:
        magnitude (int | float | None): The magnitude used for adjusting
            brightness. A positive magnitude would enhance the brightness and a
            negative magnitude would make the image darker. A magnitude=0 gives
            the origin img. If None, generate from ``magnitude_range``, see
            :class:`BaseAugTransform`. Defaults to None.
        prob (float): The probability for performing brightness adjusting
            therefore should be in range [0, 1]. Defaults to 0.5.
        random_negative_prob (float): The probability that turns the magnitude
            negative, which should be in range [0,1]. Defaults to 0.5.
        **kwargs: Other keyword arguments of :class:`BaseAugTransform`.
    """

    def __init__(self,
                 magnitude: Union[int, float, None] = None,
                 prob: float = 0.5,
                 random_negative_prob: float = 0.5,
                 **kwargs):
        super().__init__(
            prob=prob, random_negative_prob=random_negative_prob, **kwargs)
        assert (magnitude is None) ^ (self.magnitude_range is None), \
            'Please specify only one of `magnitude` and `magnitude_range`.'

        self.magnitude = magnitude

    def transform(self, results):
        """Apply transform to results."""
        if self.random_disable():
            return results

        if self.magnitude is not None:
            magnitude = self.random_negative(self.magnitude)
        else:
            magnitude = self.random_negative(self.random_magnitude())

        img = results['img']
        img_brightened = mmcv.adjust_brightness(img, factor=1 + magnitude)
        results['img'] = img_brightened.astype(img.dtype)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(magnitude={self.magnitude}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'random_negative_prob={self.random_negative_prob}'
        repr_str += f'{self.extra_repr()})'
        return repr_str


@TRANSFORMS.register_module()
class Sharpness(BaseAugTransform):
    """Adjust images sharpness.

    Args:
        magnitude (int | float | None): The magnitude used for adjusting
            sharpness. A positive magnitude would enhance the sharpness and a
            negative magnitude would make the image bulr. A magnitude=0 gives
            the origin img. If None, generate from ``magnitude_range``, see
            :class:`BaseAugTransform`. Defaults to None.
        prob (float): The probability for performing sharpness adjusting
            therefore should be in range [0, 1]. Defaults to 0.5.
        random_negative_prob (float): The probability that turns the magnitude
            negative, which should be in range [0,1]. Defaults to 0.5.
        **kwargs: Other keyword arguments of :class:`BaseAugTransform`.
    """

    def __init__(self,
                 magnitude: Union[int, float, None] = None,
                 prob: float = 0.5,
                 random_negative_prob: float = 0.5,
                 **kwargs):
        super().__init__(
            prob=prob, random_negative_prob=random_negative_prob, **kwargs)
        assert (magnitude is None) ^ (self.magnitude_range is None), \
            'Please specify only one of `magnitude` and `magnitude_range`.'

        self.magnitude = magnitude

    def transform(self, results):
        """Apply transform to results."""
        if self.random_disable():
            return results

        if self.magnitude is not None:
            magnitude = self.random_negative(self.magnitude)
        else:
            magnitude = self.random_negative(self.random_magnitude())

        img = results['img']
        img_sharpened = mmcv.adjust_sharpness(img, factor=1 + magnitude)
        results['img'] = img_sharpened.astype(img.dtype)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(magnitude={self.magnitude}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'random_negative_prob={self.random_negative_prob}'
        repr_str += f'{self.extra_repr()})'
        return repr_str


@TRANSFORMS.register_module()
class Cutout(BaseAugTransform):
    """Cutout images.

    Args:
        shape (int | tuple(int) | None): Expected cutout shape (h, w).
            If given as a single value, the value will be used for both h and
            w. If None, generate from ``magnitude_range``, see
            :class:`BaseAugTransform`. Defaults to None.
        pad_val (int, Sequence[int]): Pixel pad_val value for constant fill.
            If it is a sequence, it must have the same length with the image
            channels. Defaults to 128.
        prob (float): The probability for performing cutout therefore should
            be in range [0, 1]. Defaults to 0.5.
        **kwargs: Other keyword arguments of :class:`BaseAugTransform`.
    """

    def __init__(self,
                 shape: Union[int, Tuple[int], None] = None,
                 pad_val: Union[int, Sequence[int]] = 128,
                 prob: float = 0.5,
                 **kwargs):
        super().__init__(prob=prob, random_negative_prob=0., **kwargs)
        assert (shape is None) ^ (self.magnitude_range is None), \
            'Please specify only one of `shape` and `magnitude_range`.'

        self.shape = shape
        if isinstance(pad_val, Sequence):
            self.pad_val = tuple(pad_val)
        else:
            self.pad_val = pad_val

    def transform(self, results):
        """Apply transform to results."""
        if self.random_disable():
            return results

        if self.shape is not None:
            shape = self.shape
        else:
            shape = int(self.random_magnitude())

        img = results['img']
        img_cutout = mmcv.cutout(img, shape, pad_val=self.pad_val)
        results['img'] = img_cutout.astype(img.dtype)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(shape={self.shape}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'prob={self.prob}{self.extra_repr()})'
        return repr_str


@TRANSFORMS.register_module()
class GaussianBlur(BaseAugTransform):
    """Gaussian blur images.

    Args:
        radius (int, float, optional): The blur radius. If None, generate from
            ``magnitude_range``, see :class:`BaseAugTransform`.
            Defaults to None.
        prob (float): The probability for posterizing therefore should be in
            range [0, 1]. Defaults to 0.5.
        **kwargs: Other keyword arguments of :class:`BaseAugTransform`.
    """

    def __init__(self,
                 radius: Union[int, float, None] = None,
                 prob: float = 0.5,
                 **kwargs):
        super().__init__(prob=prob, random_negative_prob=0., **kwargs)
        assert (radius is None) ^ (self.magnitude_range is None), \
            'Please specify only one of `radius` and `magnitude_range`.'

        self.radius = radius

    def transform(self, results):
        """Apply transform to results."""
        if self.random_disable():
            return results

        if self.radius is not None:
            radius = self.radius
        else:
            radius = self.random_magnitude()

        img = results['img']
        pil_img = Image.fromarray(img)
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
        results['img'] = np.array(pil_img, dtype=img.dtype)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(radius={self.radius}, '
        repr_str += f'prob={self.prob}{self.extra_repr()})'
        return repr_str


# yapf: disable
# flake8: noqa
AUTOAUG_POLICIES = {
    # Policy for ImageNet, refers to
    # https://github.com/DeepVoltaire/AutoAugment/blame/master/autoaugment.py
    'imagenet': [
        [dict(type='Posterize', bits=4, prob=0.4),             dict(type='Rotate', angle=30., prob=0.6)],
        [dict(type='Solarize', thr=256 / 9 * 4, prob=0.6),     dict(type='AutoContrast', prob=0.6)],
        [dict(type='Equalize', prob=0.8),                      dict(type='Equalize', prob=0.6)],
        [dict(type='Posterize', bits=5, prob=0.6),             dict(type='Posterize', bits=5, prob=0.6)],
        [dict(type='Equalize', prob=0.4),                      dict(type='Solarize', thr=256 / 9 * 5, prob=0.2)],
        [dict(type='Equalize', prob=0.4),                      dict(type='Rotate', angle=30 / 9 * 8, prob=0.8)],
        [dict(type='Solarize', thr=256 / 9 * 6, prob=0.6),     dict(type='Equalize', prob=0.6)],
        [dict(type='Posterize', bits=6, prob=0.8),             dict(type='Equalize', prob=1.)],
        [dict(type='Rotate', angle=10., prob=0.2),             dict(type='Solarize', thr=256 / 9, prob=0.6)],
        [dict(type='Equalize', prob=0.6),                      dict(type='Posterize', bits=5, prob=0.4)],
        [dict(type='Rotate', angle=30 / 9 * 8, prob=0.8),      dict(type='ColorTransform', magnitude=0., prob=0.4)],
        [dict(type='Rotate', angle=30., prob=0.4),             dict(type='Equalize', prob=0.6)],
        [dict(type='Equalize', prob=0.0),                      dict(type='Equalize', prob=0.8)],
        [dict(type='Invert', prob=0.6),                        dict(type='Equalize', prob=1.)],
        [dict(type='ColorTransform', magnitude=0.4, prob=0.6), dict(type='Contrast', magnitude=0.8, prob=1.)],
        [dict(type='Rotate', angle=30 / 9 * 8, prob=0.8),      dict(type='ColorTransform', magnitude=0.2, prob=1.)],
        [dict(type='ColorTransform', magnitude=0.8, prob=0.8), dict(type='Solarize', thr=256 / 9 * 2, prob=0.8)],
        [dict(type='Sharpness', magnitude=0.7, prob=0.4),      dict(type='Invert', prob=0.6)],
        [dict(type='Shear', magnitude=0.3 / 9 * 5, prob=0.6, direction='horizontal'), dict(type='Equalize', prob=1.)],
        [dict(type='ColorTransform', magnitude=0., prob=0.4),  dict(type='Equalize', prob=0.6)],
        [dict(type='Equalize', prob=0.4),                      dict(type='Solarize', thr=256 / 9 * 5, prob=0.2)],
        [dict(type='Solarize', thr=256 / 9 * 4, prob=0.6),     dict(type='AutoContrast', prob=0.6)],
        [dict(type='Invert', prob=0.6),                        dict(type='Equalize', prob=1.)],
        [dict(type='ColorTransform', magnitude=0.4, prob=0.6), dict(type='Contrast', magnitude=0.8, prob=1.)],
        [dict(type='Equalize', prob=0.8),                      dict(type='Equalize', prob=0.6)],
    ],
}

RANDAUG_POLICIES = {
    # Refers to `_RAND_INCREASING_TRANSFORMS` in pytorch-image-models
    'timm_increasing': [
        dict(type='AutoContrast'),
        dict(type='Equalize'),
        dict(type='Invert'),
        dict(type='Rotate', magnitude_range=(0, 30)),
        dict(type='Posterize', magnitude_range=(4, 0)),
        dict(type='Solarize', magnitude_range=(256, 0)),
        dict(type='SolarizeAdd', magnitude_range=(0, 110)),
        dict(type='ColorTransform', magnitude_range=(0, 0.9)),
        dict(type='Contrast', magnitude_range=(0, 0.9)),
        dict(type='Brightness', magnitude_range=(0, 0.9)),
        dict(type='Sharpness', magnitude_range=(0, 0.9)),
        dict(type='Shear', magnitude_range=(0, 0.3), direction='horizontal'),
        dict(type='Shear', magnitude_range=(0, 0.3), direction='vertical'),
        dict(type='Translate', magnitude_range=(0, 0.45), direction='horizontal'),
        dict(type='Translate', magnitude_range=(0, 0.45), direction='vertical'),
    ],
    'simple_increasing': [
        dict(type='AutoContrast'),
        dict(type='Equalize'),
        dict(type='Rotate', magnitude_range=(0, 30)),
        dict(type='Shear', magnitude_range=(0, 0.3), direction='horizontal'),
        dict(type='Shear', magnitude_range=(0, 0.3), direction='vertical'),
    ],
}
