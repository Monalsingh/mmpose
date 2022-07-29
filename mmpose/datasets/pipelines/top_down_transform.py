# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import cv2
import numpy as np

from mmpose.core.bbox import bbox_xywh2cs
from mmpose.core.post_processing import (affine_transform, fliplr_joints,
                                         get_affine_transform, get_warp_matrix,
                                         warp_affine_joints)
from mmpose.datasets.builder import PIPELINES


@PIPELINES.register_module()
class TopDownGetBboxCenterScale:
    """Convert bbox from [x, y, w, h] to center and scale.

    The center is the coordinates of the bbox center, and the scale is the
    bbox width and height normalized by a scale factor.

    Required key: 'bbox', 'ann_info'

    Modifies key: 'center', 'scale'

    Args:
        padding (float): bbox padding scale that will be multilied to scale.
            Default: 1.25
    """
    # Pixel std is 200.0, which serves as the normalization factor to
    # to calculate bbox scales.
    pixel_std: float = 200.0

    def __init__(self, padding: float = 1.25):
        self.padding = padding

    def __call__(self, results):

        if 'center' in results and 'scale' in results:
            warnings.warn(
                'Use the "center" and "scale" that already exist in the data '
                'sample. The padding will still be applied.')
            results['scale'] *= self.padding
        else:
            bbox = results['bbox']
            image_size = results['ann_info']['image_size']
            aspect_ratio = image_size[0] / image_size[1]

            center, scale = bbox_xywh2cs(
                bbox,
                aspect_ratio=aspect_ratio,
                padding=self.padding,
                pixel_std=self.pixel_std)

            results['center'] = center
            results['scale'] = scale
        return results


@PIPELINES.register_module()
class TopDownAffine:
    """Affine transform the image to make input.

    Required key:'img', 'joints_3d', 'joints_3d_visible', 'ann_info','scale',
    'rotation' and 'center'.

    Modified key:'img', 'joints_3d', and 'joints_3d_visible'.

    Args:
        use_udp (bool): To use unbiased data processing.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
    """

    def __init__(self, use_udp=False):
        self.use_udp = use_udp

    def __call__(self, results):
        image_size = results['ann_info']['image_size']

        img = results['img']
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']
        c = results['center']
        s = results['scale']
        r = results['rotation']

        if self.use_udp:
            trans = get_warp_matrix(r, c * 2.0, image_size - 1.0, s * 200.0)
            if not isinstance(img, list):
                img = cv2.warpAffine(
                    img,
                    trans, (int(image_size[0]), int(image_size[1])),
                    flags=cv2.INTER_LINEAR)
            else:
                img = [
                    cv2.warpAffine(
                        i,
                        trans, (int(image_size[0]), int(image_size[1])),
                        flags=cv2.INTER_LINEAR) for i in img
                ]

            joints_3d[:, 0:2] = \
                warp_affine_joints(joints_3d[:, 0:2].copy(), trans)

        else:
            trans = get_affine_transform(c, s, r, image_size)
            if not isinstance(img, list):
                img = cv2.warpAffine(
                    img,
                    trans, (int(image_size[0]), int(image_size[1])),
                    flags=cv2.INTER_LINEAR)
            else:
                img = [
                    cv2.warpAffine(
                        i,
                        trans, (int(image_size[0]), int(image_size[1])),
                        flags=cv2.INTER_LINEAR) for i in img
                ]
            for i in range(results['ann_info']['num_joints']):
                if joints_3d_visible[i, 0] > 0.0:
                    joints_3d[i,
                              0:2] = affine_transform(joints_3d[i, 0:2], trans)

        results['img'] = img
        results['joints_3d'] = joints_3d
        results['joints_3d_visible'] = joints_3d_visible

        return results


