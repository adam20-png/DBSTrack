"""
DBSTrack network.

    shared MobileNetV3-Large (first 13 stages), read at stride 8 (40 ch) and stride 16 (112 ch)
    template modulation M of the search stream (prototypes + template-generated 1x3 / 3x1 depthwise kernels)
    one low-resolution attention block A on the 19 x 19 stride-16 search map
    two branches (crops 3 / 5, ECA, projection to 128 channels)
    pixel-wise correlation of both branches (100 + 36 channels) + SE, concatenated with the search features
    anchor-free head (classification, centerness, ltrb box regression)

The template is encoded once per sequence (encode_template) and its outputs are reused for every frame.
Geometry: total stride 8, template 127, search 303 -> 23 x 23 score map. Location (i, j) <-> search-crop pixel
    (x, y) = (OFFSET + 8 j, OFFSET + 8 i),  OFFSET = 63.5 (the map is centred on the crop)
"""
from __future__ import absolute_import

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tracker.backbone import LightBackbone
from tracker.common import ECALayer, ECALayer1, _BatchNorm2d
from tracker.common import RichProjection

TOTAL_STRIDE = 8
EXEMPLAR_SZ = 127
INSTANCE_SZ = 303          # search region. Any 255 + 8k works (fully convolutional).
SCORE_SIZE = 23            # 17 + (303 - 255) / 8


def score_size_for(instance_sz):
    assert (instance_sz - 255) % TOTAL_STRIDE == 0, 'instance size must be 255 + 8k'
    return 17 + (instance_sz - 255) // TOTAL_STRIDE


def instance_for(score_size):
    return 255 + (score_size - 17) * TOTAL_STRIDE


def score_points(device=None, score_size=SCORE_SIZE):
    """(S*S, 2) x, y (continuous search-crop coordinates) of every score-map location, row-major.
    The map is centred on the crop: offset = instance/2 - (S-1)/2 * 8 = 63.5 for every size."""
    offset = instance_for(score_size) / 2.0 - (score_size - 1) / 2.0 * TOTAL_STRIDE
    r = torch.arange(score_size, dtype=torch.float32, device=device) * TOTAL_STRIDE + offset
    yy, xx = torch.meshgrid(r, r, indexing='ij')
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], 1)


def xcorr_depthwise(x, kernel):
    """x (B,C,H,W), kernel (B,C,h,w) -> (B,C,H-h+1,W-w+1)"""
    b, c = kernel.shape[:2]
    out = F.conv2d(x.reshape(1, b * c, x.size(2), x.size(3)),
                   kernel.reshape(b * c, 1, kernel.size(2), kernel.size(3)), groups=b * c)
    return out.reshape(b, c, out.size(2), out.size(3))


def _conv_bn_relu(cin, cout, k):
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), _BatchNorm2d(cout),
                         nn.ReLU(inplace=True))


class AnchorFreeHead(nn.Module):
    """~0.78 M parameters (128 channels, 2 cls convs + 3 reg convs)."""

    def __init__(self, in_channels=256, mid=128, n_cls=2, n_reg=3, prior=0.01):
        super(AnchorFreeHead, self).__init__()
        self.fuse = _conv_bn_relu(in_channels, mid, 1)
        self.cls_tower = nn.Sequential(*[_conv_bn_relu(mid, mid, 3) for _ in range(n_cls)])
        self.reg_tower = nn.Sequential(*[_conv_bn_relu(mid, mid, 3) for _ in range(n_reg)])
        self.cls = nn.Conv2d(mid, 1, 3, padding=1)
        self.ctr = nn.Conv2d(mid, 1, 3, padding=1)
        self.reg = nn.Conv2d(mid, 4, 3, padding=1)
        # learnable scale / bias of the exponential box parametrisation (SiamFC++)
        self.si = nn.Parameter(torch.tensor(1.0))
        self.bi = nn.Parameter(torch.tensor(0.0))
        self.prior = prior

    def reset_parameters(self):
        """Called AFTER common.init_weights: focal-loss prior on the cls bias, small init of the outputs."""
        for m in (self.cls, self.ctr, self.reg):
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0.0)
        nn.init.constant_(self.cls.bias, -math.log((1 - self.prior) / self.prior))
        # exp(bi) * stride * 2 ~ 16 px at start (half-size of a small target in the 255 crop) -> bi = log(2)
        with torch.no_grad():
            self.si.fill_(1.0)
            self.bi.fill_(math.log(2.0))

    def forward(self, feat):
        f = self.fuse(feat)
        c = self.cls_tower(f)
        r = self.reg_tower(f)
        cls = self.cls(c)                                              # logits (B,1,17,17)
        ctr = self.ctr(c)                                              # logits (B,1,17,17)
        ltrb = torch.exp(self.si * self.reg(r) + self.bi) * TOTAL_STRIDE   # px, > 0 (B,4,17,17)
        return cls, ctr, ltrb


def xcorr_pixelwise(x, kernel):
    """pixel-wise correlation (LightTrack / FEAR): every template position is a 1x1 filter.
    x (B,C,H,W), kernel (B,C,h,w) -> (B, h*w, H', W') with H' = H-h+1, ALIGNED with xcorr_depthwise:
    output (i,j) = mean of the 2x2 search positions around the centre of the depthwise window (h, w even)."""
    b, c, h, w = kernel.shape
    corr = torch.einsum('bck,bchw->bkhw', kernel.reshape(b, c, h * w), x) / math.sqrt(c)
    corr = F.avg_pool2d(corr, 2, stride=1) if h % 2 == 0 else corr
    oh, ow = x.size(2) - h + 1, x.size(3) - w + 1
    sy, sx = (h // 2 - 1, w // 2 - 1) if h % 2 == 0 else (h // 2, w // 2)
    return corr[:, :, sy:sy + oh, sx:sx + ow]


class _SE(nn.Module):
    def __init__(self, c, r=4):
        super(_SE, self).__init__()
        self.fc = nn.Sequential(nn.Linear(c, c // r), nn.ReLU(inplace=True), nn.Linear(c // r, c), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x.mean((2, 3)))[:, :, None, None]


def _crop_like(x, kernel):
    """search features cropped to the correlation output grid (same alignment as xcorr_pixelwise)."""
    h, w = kernel.shape[2:]
    oh, ow = x.size(2) - h + 1, x.size(3) - w + 1
    if h % 2 == 0:
        x = F.avg_pool2d(x, 2, stride=1)
        sy, sx = h // 2 - 1, w // 2 - 1
    else:
        sy, sx = h // 2, w // 2
    return x[:, :, sy:sy + oh, sx:sx + ow]


class SiameseNet(nn.Module):

    def __init__(self, backbone=None, head_mid=128, corr='dw', z_cells=(10, 6)):
        """Dual-branch Siamese network WITHOUT template conditioning (used as the 'None' ablation variant).
           corr='pw' : pixel-wise correlation of both branches + SE, concatenated with the search features
                       of both branches -> 1x1 fuse (DBSTrack). corr='dw' : depthwise xcorr (2 x 128 channels).
           z_cells = template map sizes (127 input)."""
        super(SiameseNet, self).__init__()
        self.corr = corr
        self.backbone = backbone if backbone is not None else LightBackbone()
        self.tematt = ECALayer(256)
        self.detatt = ECALayer(256)
        self.attse = ECALayer1(256)
        self.proj1 = RichProjection(256, 128)
        self.proj2 = RichProjection(256, 128)
        if corr == 'pw':
            n_corr = z_cells[0] ** 2 + z_cells[1] ** 2                   # 100 + 36
            self.corr_se = _SE(n_corr)
            self.head = AnchorFreeHead(n_corr + 256, head_mid)
        else:
            self.head = AnchorFreeHead(256, head_mid)

    def encode_template(self, z):
        z1, z2 = self.backbone(z)
        z1 = self.tematt(z1) + self.attse(z1) + z1
        z2 = self.tematt(z2) + self.attse(z2) + z2
        return self.proj1(z1), self.proj2(z2)

    def encode_search(self, x, kernels=None):          # kernels unused here (used by DBSTrackNet)
        x1, x2 = self.backbone(x)
        x1 = self.detatt(x1) + self.attse(x1) + x1
        x2 = self.detatt(x2) + self.attse(x2) + x2
        return self.proj1(x1), self.proj2(x2)

    def head_forward(self, kernels, xs):
        if self.corr == 'pw':
            c = torch.cat([xcorr_pixelwise(xs[0], kernels[0]), xcorr_pixelwise(xs[1], kernels[1])], 1)
            f = torch.cat([self.corr_se(c), _crop_like(xs[0], kernels[0]), _crop_like(xs[1], kernels[1])], 1)
            return self.head(f)
        f1 = xcorr_depthwise(xs[0], kernels[0])
        f2 = xcorr_depthwise(xs[1], kernels[1])
        return self.head(torch.cat([f1, f2], 1))

    def forward(self, z, x, precomputed_kernels=None):
        kernels = precomputed_kernels if precomputed_kernels is not None else self.encode_template(z)
        return self.head_forward(kernels, self.encode_search(x, kernels))


# ============================================================================== template conditioning
class TemplateModulation(nn.Module):
    """Template modulation M: template -> search modulation INSIDE the backbone (inspired by AsymTrack's ETM).

    Asymmetric: the template goes through the backbone unchanged (once, at init) and gives a context
    (n_proto prototypes + template-conditioned separable 1D kernels); every search frame is modulated by it:
        y = dynamic depthwise (1 x k) then (k x 1) convolution of x with template-generated kernels
        r = attention of every search position over the template prototypes
        x' = x + gamma * conv1x1([y, r])          gamma = 0.1 at init -> small perturbation
    """

    def __init__(self, c, k=3, n_proto=8, d=None):
        super(TemplateModulation, self).__init__()
        self.c, self.k, self.d = c, k, d or max(32, c // 2)
        self.proto_logits = nn.Conv2d(c, n_proto, 1)
        self.gen = nn.Sequential(nn.Linear(c, c // 4), nn.ReLU(inplace=True), nn.Linear(c // 4, 2 * c * k))
        self.q = nn.Conv2d(c, self.d, 1, bias=False)
        self.kp = nn.Linear(c, self.d, bias=False)
        self.vp = nn.Linear(c, c, bias=False)
        self.out = nn.Conv2d(2 * c, c, 1)
        # 0.1 (not 0): with gamma = 0 the out-conv receives no gradient at all at the start (tested:
        # d loss / d gamma ~ 4e-3 -> very slow start); 0.1 = small perturbation of the pretrained features
        self.gamma = nn.Parameter(torch.full((1,), 0.1))

    def context(self, z):
        """z (B,C,h,w) template features -> prototypes (B,K,C), separable kernels (B,2*C*k)."""
        a = torch.softmax(self.proto_logits(z).flatten(2), -1)              # (B,K,hw)
        protos = torch.bmm(a, z.flatten(2).transpose(1, 2))                 # (B,K,C)
        return protos, self.gen(z.mean((2, 3)))

    def forward(self, x, protos, dk):
        b, c, h, w = x.shape
        k = self.k
        kh, kv = dk.reshape(b, 2, c, k).unbind(1)
        y = F.conv2d(x.reshape(1, b * c, h, w), kh.reshape(b * c, 1, 1, k), padding=(0, k // 2), groups=b * c)
        y = F.conv2d(y, kv.reshape(b * c, 1, k, 1), padding=(k // 2, 0), groups=b * c).reshape(b, c, h, w)
        q = self.q(x).flatten(2).transpose(1, 2)                            # (B,HW,d)
        att = torch.softmax(torch.bmm(q, self.kp(protos).transpose(1, 2)) / math.sqrt(self.d), -1)   # (B,HW,K)
        r = torch.bmm(att, self.vp(protos)).transpose(1, 2).reshape(b, c, h, w)
        return x + self.gamma * self.out(torch.cat([y, r], 1))


class LowResAttention(nn.Module):
    """Low-resolution attention A: one global self-attention block on the stride-16 SEARCH map (19 x 19 for 303),
    where it is cheap (AsymTrack / HiT put attention only after the last, smallest stage).
    Conditional position encoding (depthwise 3x3) -> works for any search size. Residuals gated by
    zero-initialised scalars -> starts as identity."""

    def __init__(self, c, heads=4, ffn=2):
        super(LowResAttention, self).__init__()
        self.cpe = nn.Conv2d(c, c, 3, padding=1, groups=c)
        self.norm1 = nn.GroupNorm(1, c)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)
        self.norm2 = nn.GroupNorm(1, c)
        self.ffn = nn.Sequential(nn.Conv2d(c, c * ffn, 1), nn.GELU(), nn.Conv2d(c * ffn, c, 1))
        self.g1 = nn.Parameter(torch.zeros(1))
        self.g2 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.cpe(x)
        t = self.norm1(x).flatten(2).transpose(1, 2)                        # (B,HW,C)
        a = self.attn(t, t, t, need_weights=False)[0].transpose(1, 2).reshape(b, c, h, w)
        x = x + self.g1 * a
        return x + self.g2 * self.ffn(self.norm2(x))


class DBSTrackNet(SiameseNet):
    """DBSTrack: dual-branch network + template modulation of the search stream at stride 8 (after features[6],
    40 ch) and stride 16 (after features[12], 112 ch) + one low-resolution attention block on the search map.
    The template branch is computed once per sequence (asymmetric).
    kernels = (k1, k2, protos8, dyn8, protos16, dyn16) -> the tracker's kernel blending still works."""

    def __init__(self, backbone=None, head_mid=128, z_cells=(10, 6), n_proto=8):
        super(DBSTrackNet, self).__init__(backbone, head_mid, corr='pw', z_cells=z_cells)
        bb = self.backbone
        self.tm8 = TemplateModulation(bb.CH_STRIDE8, n_proto=n_proto)
        self.tm16 = TemplateModulation(bb.CH_STRIDE16, n_proto=n_proto)
        self.lowatt = LowResAttention(bb.CH_STRIDE16)

    def _features(self, x, ctx=None):
        bb = self.backbone
        if bb.normalize_input:
            x = (x - bb.input_mean) / bb.input_std
        f8 = None
        for i, layer in enumerate(bb.body):
            x = layer(x)
            if i == bb.IDX_STRIDE8:
                if ctx is not None:
                    x = self.tm8(x, ctx[0], ctx[1])                          # depth 1: modulated BEFORE stages 7-12
                f8 = x
        f16 = x
        if ctx is not None:
            f16 = self.lowatt(self.tm16(f16, ctx[2], ctx[3]))               # depth 2 + global context
        b1 = bb.neck1(f8)
        b2 = F.interpolate(bb.neck2(f16), size=b1.shape[-2:], mode='bilinear', align_corners=False)
        return bb._crop(b1, bb.crop1), bb._crop(b2, bb.crop2), f8, f16

    def encode_template(self, z):
        z1, z2, f8, f16 = self._features(z)
        z1 = self.tematt(z1) + self.attse(z1) + z1
        z2 = self.tematt(z2) + self.attse(z2) + z2
        return (self.proj1(z1), self.proj2(z2)) + self.tm8.context(f8) + self.tm16.context(f16)

    def encode_search(self, x, kernels=None):
        assert kernels is not None, 'DBSTrackNet.encode_search needs the template kernels'
        x1, x2, _, _ = self._features(x, kernels[2:])
        x1 = self.detatt(x1) + self.attse(x1) + x1
        x2 = self.detatt(x2) + self.attse(x2) + x2
        return self.proj1(x1), self.proj2(x2)


def build_net(arch='dbstrack'):
    """'dbstrack' = full model; 'none' = same network without template modulation / attention."""
    if arch == 'dbstrack':
        return DBSTrackNet()
    return SiameseNet(corr='pw')


def net_from_state_dict(sd):
    """build the network matching a checkpoint (full DBSTrack or the unconditioned variant) and load it."""
    net = DBSTrackNet() if any(k.startswith('tm8.') for k in sd) else \
        SiameseNet(corr='pw' if any(k.startswith('corr_se.') for k in sd) else 'dw')
    net.load_state_dict(sd, strict=True)
    return net


# ----------------------------------------------------------------------------------------- loss
def make_targets(boxes, neg, score_size=SCORE_SIZE):
    """boxes (B,4) x1y1x2y2 in search-crop pixels, neg (B,) bool.
    returns cls (B,N) {0,1}, ctr (B,N) [0,1], ltrb (B,N,4), with N = S*S."""
    pts = score_points(boxes.device, score_size)                       # (N,2)
    x, y = pts[:, 0][None], pts[:, 1][None]
    l = x - boxes[:, 0:1]
    t = y - boxes[:, 1:2]
    r = boxes[:, 2:3] - x
    b = boxes[:, 3:4] - y
    ltrb = torch.stack([l, t, r, b], 2)                                # (B,N,4)
    inside = ltrb.min(2).values > 0
    pos = inside & (~neg[:, None])
    lr_ = torch.stack([l, r], 2).clamp(min=1e-6)
    tb_ = torch.stack([t, b], 2).clamp(min=1e-6)
    ctr = torch.sqrt((lr_.min(2).values / lr_.max(2).values) * (tb_.min(2).values / tb_.max(2).values))
    return pos.float(), torch.where(pos, ctr, torch.zeros_like(ctr)), ltrb


def sigmoid_focal_loss(logits, target, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    p_t = p * target + (1 - p) * (1 - target)
    loss = ce * ((1 - p_t) ** gamma)
    a_t = alpha * target + (1 - alpha) * (1 - target)
    return (a_t * loss).sum()


def iou_loss(pred, target, weight):
    """pred / target ltrb (M,4) > 0 ; linear IoU loss weighted by centerness target."""
    pa = (pred[:, 0] + pred[:, 2]) * (pred[:, 1] + pred[:, 3])
    ta = (target[:, 0] + target[:, 2]) * (target[:, 1] + target[:, 3])
    w = torch.min(pred[:, 0], target[:, 0]) + torch.min(pred[:, 2], target[:, 2])
    h = torch.min(pred[:, 1], target[:, 1]) + torch.min(pred[:, 3], target[:, 3])
    inter = w.clamp(min=0) * h.clamp(min=0)
    iou = inter / (pa + ta - inter).clamp(min=1e-6)
    return ((1 - iou) * weight).sum() / weight.sum().clamp(min=1e-6), iou


def reg_loss(outputs, boxes, neg, w_cls=1.0, w_ctr=1.0, w_reg=3.0):
    cls, ctr, ltrb = outputs
    b = cls.size(0)
    t_cls, t_ctr, t_ltrb = make_targets(boxes, neg, cls.size(-1))      # geometry read from the output size
    cls = cls.reshape(b, -1)
    ctr = ctr.reshape(b, -1)
    ltrb = ltrb.reshape(b, 4, -1).permute(0, 2, 1)
    pos = t_cls > 0
    n_pos = max(float(pos.sum()), 1.0)
    l_cls = sigmoid_focal_loss(cls, t_cls) / n_pos
    if pos.any():
        l_ctr = F.binary_cross_entropy_with_logits(ctr[pos], t_ctr[pos])
        l_reg, iou = iou_loss(ltrb[pos], t_ltrb[pos], t_ctr[pos])
        mean_iou = float(iou.detach().mean())
    else:
        l_ctr = l_reg = cls.sum() * 0
        mean_iou = 0.0
    total = w_cls * l_cls + w_ctr * l_ctr + w_reg * l_reg
    return total, {'cls': float(l_cls.detach()), 'ctr': float(l_ctr.detach()), 'reg': float(l_reg.detach()), 'iou': mean_iou,
                   'npos': n_pos / b}
