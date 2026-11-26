"""Shared building blocks of DBSTrack: BatchNorm, ECA attention, RichProjection, image / weight utilities."""
import cv2
import numpy as np
import torch.nn as nn


class _BatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, *args, **kwargs):
        super(_BatchNorm2d, self).__init__(
            num_features, *args, eps=1e-6, momentum=0.05, **kwargs)


class ECALayer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class ECALayer1(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(ECALayer1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        w = self.max_pool(x)
        w = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        w = self.sigmoid(w)


        y = y + w
        return x * y.expand_as(x)


class RichProjection(nn.Module):
    """Dual-branch bottleneck projection that preserves information better than
    a simple 1×1 convolution when reducing channels before cross-correlation.

    Architecture:
        x (in_channels) ─┬─ DepthWise 3×3 → BN → ReLU → PointWise 1×1 (out_ch) → BN ─┐
                          │                                                           ├→ + → ReLU → out
                          └─ PointWise 1×1 (out_ch) → BN ────────────────────────────┘

    - Depthwise branch: captures spatial context before projection (local patterns)
    - Pointwise branch: direct channel mixing (global relationships)
    - Sum fusion: both streams contribute without information bottleneck

    The output channel count matches the old simple projection, so cross-correlation
    cost is unchanged (128ch → 4× cheaper than 256ch).
    """

    def __init__(self, in_channels=256, out_channels=128):
        super(RichProjection, self).__init__()
        # Depthwise branch: spatial context then project
        self.dw_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1,
            padding=1, groups=in_channels, bias=False)  # depthwise
        self.dw_bn = _BatchNorm2d(in_channels)
        self.dw_relu = nn.ReLU(inplace=True)
        self.dw_pw = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False)  # pointwise proj
        self.dw_pw_bn = _BatchNorm2d(out_channels)

        # Pointwise branch: direct channel projection (no spatial mixing)
        self.pw_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False)
        self.pw_bn = _BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Branch 1: depthwise 3×3 → pointwise 1×1
        dw = self.dw_conv(x)
        dw = self.dw_bn(dw)
        dw = self.dw_relu(dw)
        dw = self.dw_pw(dw)
        dw = self.dw_pw_bn(dw)

        # Branch 2: direct pointwise
        pw = self.pw_conv(x)
        pw = self.pw_bn(pw)

        # Fuse: both branches contribute
        return self.relu(dw + pw)


def init_weights(model, gain=1):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def read_image(img_file, cvt_code=cv2.COLOR_BGR2RGB):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)
    return img


def show_image(img, boxes=None, box_fmt='ltwh', colors=None,
               thickness=3, fig_n=1, delay=1, visualize=True,
               cvt_code=cv2.COLOR_RGB2BGR):
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)

    # resize img if necessary
    max_size = 960
    if max(img.shape[:2]) > max_size:
        scale = max_size / max(img.shape[:2])
        out_size = (
            int(img.shape[1] * scale),
            int(img.shape[0] * scale))
        img = cv2.resize(img, out_size)
        if boxes is not None:
            boxes = np.array(boxes, dtype=np.float32) * scale

    if boxes is not None:
        assert box_fmt in ['ltwh', 'ltrb']
        boxes = np.array(boxes, dtype=np.int32)
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
        if box_fmt == 'ltrb':
            boxes[:, 2:] -= boxes[:, :2]

        # clip bounding boxes
        bound = np.array(img.shape[1::-1])[None, :]
        boxes[:, :2] = np.clip(boxes[:, :2], 0, bound)
        boxes[:, 2:] = np.clip(boxes[:, 2:], 0, bound - boxes[:, :2])

        if colors is None:
            colors = [
                (0, 0, 255),
                (0, 255, 0),
                (255, 0, 0),
                (0, 255, 255),
                (255, 0, 255),
                (255, 255, 0),
                (0, 0, 128),
                (0, 128, 0),
                (128, 0, 0),
                (0, 128, 128),
                (128, 0, 128),
                (128, 128, 0)]
        colors = np.array(colors, dtype=np.int32)
        if colors.ndim == 1:
            colors = np.expand_dims(colors, axis=0)

        for i, box in enumerate(boxes):
            color = colors[i % len(colors)]
            pt1 = (box[0], box[1])
            pt2 = (box[0] + box[2], box[1] + box[3])
            img = cv2.rectangle(img, pt1, pt2, color.tolist(), thickness)

    if visualize:
        winname = 'window_{}'.format(fig_n)
        cv2.imshow(winname, img)
        cv2.waitKey(delay)

    return img


def filter_frames(img0, anno, vis_ratios=None):
    """Indices of the frames usable for training (target size / aspect / visibility rules of SiamFC)."""
    size = np.array(img0.shape[1::-1])[np.newaxis, :]
    areas = anno[:, 2] * anno[:, 3]
    c1 = areas >= 20
    c2 = np.all(anno[:, 2:] >= 20, axis=1)
    c3 = np.all(anno[:, 2:] <= 500, axis=1)
    c4 = np.all((anno[:, 2:] / size) >= 0.01, axis=1)
    c5 = np.all((anno[:, 2:] / size) <= 0.5, axis=1)
    c6 = (anno[:, 2] / np.maximum(1, anno[:, 3])) >= 0.25
    c7 = (anno[:, 2] / np.maximum(1, anno[:, 3])) <= 4
    if vis_ratios is not None:
        c8 = (vis_ratios > max(1, vis_ratios.max() * 0.3))
    else:
        c8 = np.ones_like(c1)
    return np.where(np.logical_and.reduce((c1, c2, c3, c4, c5, c6, c7, c8)))[0]
