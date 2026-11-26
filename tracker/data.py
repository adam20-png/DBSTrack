"""
Training pairs WITH the target box in search-crop coordinates (needed by the regression head).

Differences with the SiamFC pair sampler (SiamFC labels are a fixed disk at the centre):
  * the search crop is randomly SHIFTED (+-64 px in the 303 crop) and RESCALED (x e^U(-0.25,0.25)),
    so the target is not always centred and the head must localise / regress it;
  * the box of the target in the 255 crop is returned (x1, y1, x2, y2);
  * a fraction of NEGATIVE pairs (template and search from two different sequences) teaches the
    classifier to reject distractors (SiamFC++ / DaSiamRPN practice);
  * the SAME crop function (crop_patch) is used by the tracker at test time.

Frame validity: same filter as the SiamFC pair sampler (size / aspect / GOT-10k cover / LaSOT occlusion).
Images are RGB float 0..255 (the backbone normalises internally).
"""
from __future__ import absolute_import, division

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from tracker.common import filter_frames
from tracker.model import EXEMPLAR_SZ, INSTANCE_SZ


def context_size(w, h, context=0.5):
    p = context * (w + h)
    return np.sqrt((w + p) * (h + p))


def crop_patch(img, center, size, out_size, avg=None):
    """Square crop of side `size` (image pixels) centred on `center`=(cx,cy), resized to out_size.
    Continuous coordinates (pixel i covers [i, i+1)):  u = (X - cx + size/2) * out_size / size,
    so the crop centre out_size/2 corresponds to (cx, cy). Outside the image: mean colour padding."""
    if avg is None:
        avg = img.reshape(-1, img.shape[2]).mean(0)
    s = out_size / float(size)
    m = np.array([[s, 0, s * (0.5 - center[0] + size / 2.0) - 0.5],
                  [0, s, s * (0.5 - center[1] + size / 2.0) - 0.5]], dtype=np.float64)
    return cv2.warpAffine(img, m, (out_size, out_size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=tuple(float(a) for a in avg))


def box_to_crop(box_xyxy, center, size, out_size):
    """image box (continuous x1,y1,x2,y2) -> crop box, consistent with crop_patch."""
    s = out_size / float(size)
    b = np.asarray(box_xyxy, dtype=np.float64)
    return np.array([(b[0] - center[0]) * s, (b[1] - center[1]) * s,
                     (b[2] - center[0]) * s, (b[3] - center[1]) * s]) + out_size / 2.0


def crop_to_image(pt_xy, center, size, out_size):
    s = out_size / float(size)
    return (pt_xy[0] - out_size / 2.0) / s + center[0], (pt_xy[1] - out_size / 2.0) / s + center[1]


def _color_aug(img, rng):
    img = img.astype(np.float32)
    if rng.rand() < 0.05:                                       # rare grayscale
        g = img.mean(2, keepdims=True)
        img = np.repeat(g, 3, 2)
    img = img * rng.uniform(0.8, 1.2, size=(1, 1, 3)) + rng.uniform(-15, 15)
    return np.clip(img, 0, 255)


def _motion_blur(img, rng, max_k=9):
    """random linear motion blur (fast target / camera motion)."""
    k = int(rng.randint(3, max_k + 1)) | 1
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    m = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), float(rng.uniform(0, 180)), 1.0)
    kernel = cv2.warpAffine(kernel, m, (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img.astype(np.float32), -1, kernel)


class RegPairs(Dataset):

    def __init__(self, seqs, neg_ratio=0.2, max_gap=100, shift=64, scale_jitter=0.25,
                 z_shift=4, z_scale=0.05, context=0.5, epoch_len=None, instance_sz=INSTANCE_SZ,
                 max_gaps=None, hard_neg=0.0, blur_prob=0.0):
        """options:
           max_gaps : per-dataset max frame gap, e.g. [100, 300] for [GOT-10k, LaSOT] (MixedSeqs order)
           hard_neg : fraction of negative pairs taken from ANOTHER sequence of the SAME category (LaSOT)
           blur_prob: probability of motion blur on the search crop"""
        self.seqs = seqs
        self.max_gaps, self.hard_neg, self.blur_prob = max_gaps, hard_neg, blur_prob
        # category of every entry (LaSOT: <root>/Training/<category>/<seq>) for same-category negatives
        self.cat_of, self.by_cat = {}, {}
        if hard_neg > 0 and hasattr(seqs, 'index'):
            import os
            for e, (k, i) in enumerate(seqs.index):
                ds = seqs.datasets[k]
                if hasattr(ds, 'seq_dirs') and type(ds).__name__ == 'LaSOTTrain':
                    cat = os.path.basename(os.path.dirname(ds.seq_dirs[i]))
                    self.cat_of[e] = cat
                    self.by_cat.setdefault(cat, set()).add((k, i))
            self.by_cat = {c: sorted(v) for c, v in self.by_cat.items()}
        self.instance_sz = instance_sz
        self.neg_ratio, self.max_gap = neg_ratio, max_gap
        self.shift, self.scale_jitter = shift, scale_jitter
        self.z_shift, self.z_scale, self.context = z_shift, z_scale, context
        self.epoch_len = epoch_len or len(seqs)
        self.return_meta = getattr(seqs, 'return_meta', False)

    def __len__(self):
        return self.epoch_len

    # ------------------------------------------------------------------ sampling
    def _seq(self, idx):
        item = self.seqs[idx]
        if len(item) == 3:
            img_files, anno, meta = item
            vis = meta.get('cover', None) if meta is not None else None
        else:
            img_files, anno = item[:2]
            vis = None
        return img_files, np.asarray(anno, dtype=np.float64), vis

    def _valid(self, idx):
        img_files, anno, vis = self._seq(idx)
        img0 = cv2.imread(img_files[0], cv2.IMREAD_COLOR)
        if img0 is None:
            return img_files, anno, np.array([], dtype=int)
        return img_files, anno, filter_frames(img0, anno, vis)

    def _pick(self, rng, index=None):
        for _ in range(20):
            idx = index if index is not None else rng.randint(len(self.seqs))
            img_files, anno, val = self._valid(idx)
            if len(val) >= 2:
                return idx, img_files, anno, val
            index = None
        raise RuntimeError('could not find a valid sequence')

    # ------------------------------------------------------------------ crops
    def _crop(self, img, box_xywh, out_size, rng, shift, scale_j, search):
        x, y, w, h = box_xywh
        cx, cy = x + w / 2.0, y + h / 2.0
        sz = context_size(w, h, self.context) * (out_size / float(EXEMPLAR_SZ) if search else 1.0)
        sz *= np.exp(rng.uniform(-scale_j, scale_j))
        # a shift of d crop-pixels = d * sz / out_size image pixels; the crop centre moves, the target does not
        dx, dy = rng.uniform(-shift, shift, size=2) * sz / out_size
        center = (cx - dx, cy - dy)
        patch = crop_patch(img, center, sz, out_size)
        box = box_to_crop([x, y, x + w, y + h], center, sz, out_size)
        return patch, box

    def __getitem__(self, index):
        rng = np.random.RandomState((np.random.randint(1 << 30) + index) % (1 << 31))
        idx, img_files, anno, val = self._pick(rng, index % len(self.seqs))
        neg = rng.rand() < self.neg_ratio
        # template frame / search frame (gap <= max_gap, per dataset)
        iz = val[rng.randint(len(val))]
        gap = self.max_gap
        if self.max_gaps is not None and hasattr(self.seqs, 'index'):
            gap = self.max_gaps[self.seqs.index[idx][0]]
        if neg:
            cat = self.cat_of.get(idx)
            hard = cat is not None and len(self.by_cat.get(cat, [])) > 1 and rng.rand() < self.hard_neg
            if hard:                                   # same category, different sequence
                me = self.seqs.index[idx]
                pool = [e for e in self.by_cat[cat] if e != me]
                k2, i2 = pool[rng.randint(len(pool))]
                item = self.seqs.datasets[k2][i2]
                img_files2, anno2 = item[0], np.asarray(item[1], dtype=np.float64)
                vis2 = item[2].get('cover') if len(item) == 3 and item[2] is not None else None
                img0 = cv2.imread(img_files2[0], cv2.IMREAD_COLOR)
                val2 = filter_frames(img0, anno2, vis2) if img0 is not None else np.array([], dtype=int)
                if len(val2) == 0:
                    hard = False
            if not hard:
                idx2, img_files2, anno2, val2 = self._pick(rng)
                while idx2 == idx and len(self.seqs) > 1:
                    idx2, img_files2, anno2, val2 = self._pick(rng)
            ix = val2[rng.randint(len(val2))]
            x_file, x_box = img_files2[ix], anno2[ix]
        else:
            near = val[np.abs(val - iz) <= gap]
            ix = near[rng.randint(len(near))]
            x_file, x_box = img_files[ix], anno[ix]

        z_img = cv2.cvtColor(cv2.imread(img_files[iz], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        x_img = cv2.cvtColor(cv2.imread(x_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        z, _ = self._crop(z_img, anno[iz], EXEMPLAR_SZ, rng, self.z_shift, self.z_scale, search=False)
        x, box = self._crop(x_img, x_box, self.instance_sz, rng, self.shift, self.scale_jitter, search=True)
        if self.blur_prob > 0 and rng.rand() < self.blur_prob:
            x = _motion_blur(x, rng)
        z, x = _color_aug(z, rng), _color_aug(x, rng)
        z = torch.from_numpy(z).permute(2, 0, 1).float()
        x = torch.from_numpy(x).permute(2, 0, 1).float()
        return z, x, torch.from_numpy(box).float(), torch.tensor(bool(neg))
