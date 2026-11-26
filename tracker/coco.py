"""
COCO 2017 (still images) as an extra source of training pairs (SiamRPN++ / AsymTrack / MaST practice).

A COCO "pair" = template and search cropped around the SAME object of the SAME photo; they differ only by
the augmentations (search: shift +-64 px, scale x e^U(-0.25,0.25), motion blur, colour; template: small
shift / scale jitter, colour). Negative pairs: half from another object of the SAME category (another
person, another car... = hard distractor), half from a random object.

Objects kept (same acceptance rules as tracker.common.filter_frames for video frames):
    iscrowd == 0, w, h >= 20 px, w, h <= 500 px, w/W, h/H in [0.01, 0.5], aspect ratio in [0.25, 4].
The index (file names, boxes, categories) is built once from instances_train2017.json and cached in
<coco>/annotations/coco_train_index.npz (json.load of the 470 MB file takes ~1 min and several GB).

Output contract = tracker.data.RegPairs: (z, x, box_xyxy_in_search_crop, neg).
"""
from __future__ import absolute_import

import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from tracker.data import RegPairs, _color_aug, _motion_blur
from tracker.model import EXEMPLAR_SZ, INSTANCE_SZ


def build_coco_index(root, split='train2017'):
    cache = os.path.join(root, 'annotations', 'coco_%s_index.npz' % split)
    if os.path.isfile(cache):
        d = np.load(cache, allow_pickle=False)
        return d['files'], d['img_idx'], d['boxes'], d['cats']
    print('building the COCO index (once) from instances_%s.json ...' % split)
    ann = json.load(open(os.path.join(root, 'annotations', 'instances_%s.json' % split)))
    imgs = {im['id']: (k, im['file_name'], im['width'], im['height']) for k, im in enumerate(ann['images'])}
    files = np.array([im['file_name'] for im in ann['images']])
    img_idx, boxes, cats = [], [], []
    for a in ann['annotations']:
        if a.get('iscrowd', 0):
            continue
        k, _, W, H = imgs[a['image_id']]
        x, y, w, h = a['bbox']
        if w < 20 or h < 20 or w > 500 or h > 500:
            continue
        if not (0.01 <= w / W <= 0.5 and 0.01 <= h / H <= 0.5):
            continue
        if not (0.25 <= w / h <= 4.0):
            continue
        img_idx.append(k)
        boxes.append((x, y, w, h))
        cats.append(a['category_id'])
    img_idx, boxes, cats = np.array(img_idx, np.int64), np.array(boxes, np.float32), np.array(cats, np.int64)
    np.savez(cache, files=files, img_idx=img_idx, boxes=boxes, cats=cats)
    return files, img_idx, boxes, cats


class CocoPairs(RegPairs):

    def __init__(self, root, epoch_len, neg_ratio=0.2, hard_neg=0.5, shift=64, scale_jitter=0.25,
                 z_shift=4, z_scale=0.05, context=0.5, instance_sz=INSTANCE_SZ, blur_prob=0.2, split='train2017'):
        # RegPairs attributes needed by _crop (no video sequences here)
        self.shift, self.scale_jitter, self.z_shift, self.z_scale, self.context = shift, scale_jitter, z_shift, z_scale, context
        self.instance_sz, self.neg_ratio, self.hard_neg, self.blur_prob = instance_sz, neg_ratio, hard_neg, blur_prob
        self.epoch_len = int(epoch_len)
        self.img_dir = os.path.join(root, split)
        self.files, self.img_idx, self.boxes, self.cats = build_coco_index(root, split)
        self.by_cat = {}
        for i, c in enumerate(self.cats):
            self.by_cat.setdefault(int(c), []).append(i)
        self.by_cat = {c: np.array(v) for c, v in self.by_cat.items()}

    def __len__(self):
        return self.epoch_len

    def summary(self):
        return (f'  COCO    : {len(self.boxes):6d} objects in {len(np.unique(self.img_idx))} images '
                f'({len(self.by_cat)} categories) -> {self.epoch_len} pairs/epoch')

    def _img(self, obj):
        img = cv2.imread(os.path.join(self.img_dir, self.files[self.img_idx[obj]]), cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, index):
        rng = np.random.RandomState((np.random.randint(1 << 30) + index) % (1 << 31))
        obj = rng.randint(len(self.boxes))
        neg = rng.rand() < self.neg_ratio
        x_obj = obj
        if neg:
            pool = self.by_cat[int(self.cats[obj])]
            if rng.rand() < self.hard_neg and len(pool) > 1:
                while x_obj == obj:
                    x_obj = int(pool[rng.randint(len(pool))])
            else:
                while x_obj == obj:
                    x_obj = rng.randint(len(self.boxes))
        z_img = self._img(obj)
        x_img = z_img if x_obj == obj or self.img_idx[x_obj] == self.img_idx[obj] else self._img(x_obj)
        z, _ = self._crop(z_img, self.boxes[obj], EXEMPLAR_SZ, rng, self.z_shift, self.z_scale, search=False)
        x, box = self._crop(x_img, self.boxes[x_obj], self.instance_sz, rng, self.shift, self.scale_jitter, search=True)
        if self.blur_prob > 0 and rng.rand() < self.blur_prob:
            x = _motion_blur(x, rng)
        z, x = _color_aug(z, rng), _color_aug(x, rng)
        z = torch.from_numpy(z).permute(2, 0, 1).float()
        x = torch.from_numpy(x).permute(2, 0, 1).float()
        return z, x, torch.from_numpy(box).float(), torch.tensor(bool(neg))
