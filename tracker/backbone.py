"""
LightBackbone: MobileNetV3-Large read at two depths (DBSTrack).

Versions of the reference run: torch 2.14.1 | torchvision 0.29.1
ImageNet weights: torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 (default)

One forward pass, B images of size S x S (S = 127 template, 303 search):

    x (0..255 RGB) --normalize--> features[0..6]  ---------------> f8  (B, 40, S/8, S/8)
                                        |                              |
                                        v                              v
                              features[7..12] -> f16 (B,112,S/16)    neck1  (pw 40->256, dw3x3, pw 256->256)
                                        |                              |
                                      neck2 (pw 112->256, BN)       crop1 (3 cells per side)
                                        |                              |
                          bilinear x2 -> (B,256,S/8,S/8)             branch1
                                        |
                                      crop2 (5 cells per side)
                                        |
                                    branch2

  * branch1 = stride-8 features (features[6]); branch2 = stride-16 features (features[12]) brought back to
    stride 8 by a bilinear x2 upsample. The stride-32 stage is discarded (it holds most of the parameters).
  * MobileNetV3 uses 'same' padding, which leaks absolute position (SiamDW, SiamRPN++). The border cells of
    each branch are cropped, identically for template and search, so the correlation stays centred.
  * Template maps: 10 x 10 (branch1) and 6 x 6 (branch2).

forward(x) -> (branch1, branch2), both 256 channels.
"""
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import mobilenet_v3_large

__all__ = ['LightBackbone', 'light_backbone']


class _BN(nn.BatchNorm2d):
    """Same hyper-parameters as tracker.common._BatchNorm2d (eps=1e-6, momentum=0.05).
    Duplicated here to keep this module free of circular imports."""

    def __init__(self, num_features):
        super(_BN, self).__init__(num_features, eps=1e-6, momentum=0.05)


# ImageNet statistics expressed in the 0..255 domain: the tracker feeds RAW 0..255 RGB tensors
# whereas torchvision weights expect
# ((x/255) - mean) / std.
_IMAGENET_MEAN_255 = (123.675, 116.28, 103.53)
_IMAGENET_STD_255 = (58.395, 57.12, 57.375)


class LightBackbone(nn.Module):
    """Two-branch MobileNetV3-Large feature extractor.

    Args:
        out_channels:    channels of both branches (256 — imposed by the head, do not change).
        crop1, crop2:    border cells removed on each side of branch1 / branch2 (padding removal).
        b1_decorr:       keep the branch-exclusive depthwise-separable block on branch1
                         Ablation switch.
        freeze_bn_stats: keep BatchNorm of the PRETRAINED body in eval mode while training
                         (batch=8 and two different distributions z/x make BN statistics noisy;
                         standard in SiamRPN++-style trackers). BN affine params stay trainable.
        normalize_input: apply ImageNet mean/std inside the backbone (input stays raw 0..255).

    Attributes used by the training code:
        body   : pretrained MobileNetV3 features[0..12]  (loaded by load_imagenet_to_light_backbone)
        neck1, neck2 : NEW layers (random init through common.init_weights) — must stay trainable
                       even when the pretrained body is frozen.
    """

    N_LAYERS = 13          # features[0..12]
    IDX_STRIDE8 = 6        # features[6]  -> 40 ch,  stride 8
    CH_STRIDE8 = 40
    CH_STRIDE16 = 112      # features[12] -> 112 ch, stride 16

    def __init__(self, out_channels=256, crop1=3, crop2=5, b1_decorr=True,
                 freeze_bn_stats=True, normalize_input=True):
        super(LightBackbone, self).__init__()
        self.out_channels = out_channels
        self.crop1, self.crop2 = int(crop1), int(crop2)
        self.freeze_bn_stats = freeze_bn_stats
        self.normalize_input = normalize_input

        # --- pretrained body (random init here; ImageNet weights are loaded separately so that
        #     constructing the model never triggers a download) ---------------------------------
        feats = mobilenet_v3_large(weights=None).features
        self.body = nn.Sequential(*list(feats.children())[:self.N_LAYERS])  # keys '0'..'12'

        mean = torch.tensor(_IMAGENET_MEAN_255).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD_255).view(1, 3, 1, 1)
        self.register_buffer('input_mean', mean, persistent=False)
        self.register_buffer('input_std', std, persistent=False)

        # --- branch1 neck: stride-8 (40ch) -> 256ch -----------------------------------------------
        layers = [nn.Conv2d(self.CH_STRIDE8, out_channels, 1, bias=False), _BN(out_channels),
                  nn.ReLU(inplace=True)]
        if b1_decorr:
            layers += [
                nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
                _BN(out_channels), nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 1, bias=False), _BN(out_channels),
                nn.ReLU(inplace=True)]
        self.neck1 = nn.Sequential(*layers)

        # --- branch2 neck: stride-16 (112ch) -> 256ch, linear output ----------
        # (done at LOW resolution, before the upsample: 4x cheaper)
        self.neck2 = nn.Sequential(nn.Conv2d(self.CH_STRIDE16, out_channels, 1, bias=False),
                                   _BN(out_channels))

    # ------------------------------------------------------------------------------------------
    def train(self, mode=True):
        """Keep pretrained-body BN statistics frozen during training (see freeze_bn_stats)."""
        super(LightBackbone, self).train(mode)
        if mode and self.freeze_bn_stats:
            for m in self.body.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return self

    def pretrained_parameters(self):
        """Parameters initialised from ImageNet (low LR / frozen in phase 1)."""
        return self.body.parameters()

    def new_parameters(self):
        """Randomly-initialised adaptation layers (always trained at the 'head' LR)."""
        return list(self.neck1.parameters()) + list(self.neck2.parameters())

    @staticmethod
    def _crop(x, c):
        return x if c == 0 else x[:, :, c:-c, c:-c].contiguous()

    def forward(self, x):
        """
        Args:
            x: (B, 3, S, S) RGB float tensor in 0..255.  S = 127 (exemplar) or 255 + 8k (search).
        Returns:
            branch1: (B, 256, S/8 - 2*crop1, ...)   -> (B,256,10,10) @127 | (B,256,26,26) @255
            branch2: (B, 256, S/8 - 2*crop2, ...)   -> (B,256, 6, 6) @127 | (B,256,22,22) @255
            Both are stride-8 maps, so  26-10+1 = 22-6+1 = 17  (SiamFC response 17 x 17).
        """
        if self.normalize_input:
            x = (x - self.input_mean) / self.input_std

        f8 = None
        for i, layer in enumerate(self.body):
            x = layer(x)
            if i == self.IDX_STRIDE8:
                f8 = x                                   # (B, 40, S/8, S/8)
        f16 = x                                          # (B, 112, S/16, S/16)

        b1 = self.neck1(f8)                              # (B, 256, S/8, S/8)
        b2 = self.neck2(f16)                             # (B, 256, S/16, S/16)
        # explicit reconciliation of resolutions: stride 16 -> stride 8 (same op for z and x,
        # so the constant quarter-cell phase shift of bilinear x2 cancels in the correlation)
        b2 = F.interpolate(b2, size=b1.shape[-2:], mode='bilinear', align_corners=False)

        return self._crop(b1, self.crop1), self._crop(b2, self.crop2)


def light_backbone(**kwargs):
    """Factory."""
    return LightBackbone(**kwargs)


def load_imagenet_to_light_backbone(model, weights='IMAGENET1K_V2', verbose=True):
    """Load torchvision MobileNetV3-Large ImageNet weights into `LightBackbone.body` ONLY.

    Loaded  : features[0..12] (conv + BN weight/bias/running stats), strict=True.
    NOT touched (verified at the end by a bit-exact comparison): neck1, neck2, tematt, detatt,
    attse, proj1, proj2 (RichProjection), the fusion conv and the head. They keep common.init_weights.

    Args:
        model:   a tracker (model.net.backbone is a LightBackbone), a network, or a LightBackbone.
        weights: name of a torchvision MobileNet_V3_Large_Weights member
                 ('IMAGENET1K_V2' default, 'IMAGENET1K_V1' for the ablation).
    """
    net = getattr(model, 'net', model)
    backbone = getattr(net, 'backbone', net)
    assert isinstance(backbone, LightBackbone), 'load_imagenet_to_light_backbone needs a LightBackbone'

    w_enum = torchvision.models.MobileNet_V3_Large_Weights[weights]
    full_sd = w_enum.get_state_dict(progress=True, check_hash=True)   # hash-checked download
    n = len(backbone.body)
    body_sd = {k[len('features.'):]: v for k, v in full_sd.items()
               if k.startswith('features.') and int(k.split('.')[1]) < n}

    # snapshot of everything that must NOT change
    others_before = {k: v.detach().clone() for k, v in net.state_dict().items()
                     if not k.startswith('backbone.body.')} if hasattr(net, 'backbone') else {}

    backbone.body.load_state_dict(body_sd, strict=True)               # raises on any mismatch

    if hasattr(net, 'backbone'):
        after = net.state_dict()
        for k, v in others_before.items():
            assert torch.equal(after[k], v), f'non-backbone tensor modified by ImageNet loading: {k}'

    if verbose:
        print(f'  Loaded {len(body_sd)} tensors into LightBackbone.body from MobileNetV3-Large '
              f'{weights} (top-1 {w_enum.meta["_metrics"]["ImageNet-1K"]["acc@1"]}%)')
        print(f'  Source: {w_enum.url}')
        print(f'  torch {torch.__version__} | torchvision {torchvision.__version__}')
        print('  VERIFIED: necks / ECA / RichProjection / fusion conv / head left untouched.')
    return model
