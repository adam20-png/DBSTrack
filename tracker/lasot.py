"""
LaSOT training split + GOT-10k/LaSOT mixture for the SiamFC pair sampler

Pair() only needs  len(seqs), seqs[i] -> (img_files, anno, meta)  and  seqs.return_meta.
meta['cover'] is used by Pair._filter (keeps frames with cover > max(1, 0.3 * max)):
  GOT-10k  : cover in 0..8 (official absence/cover labels)
  LaSOT    : we build cover = 8 when the target is visible, 0 when full_occlusion or out_of_view
             -> occluded / out-of-view frames are never sampled, exactly like GOT-10k absent frames.

Expected LaSOT layout (official, already split):
    <root>/Training/<category>/<category>-<id>/{img/00000001.jpg, groundtruth.txt,
                                               full_occlusion.txt, out_of_view.txt}
"""
from __future__ import absolute_import

import glob
import os

import numpy as np

__all__ = ['LaSOTTrain', 'MixedSeqs']


def _read_flags(path, n):
    """LaSOT flag files: one line of comma-separated 0/1. Missing file -> all visible."""
    if not os.path.isfile(path):
        return np.zeros(n, dtype=bool)
    with open(path) as f:
        txt = f.read().strip().replace('\n', ',')
    vals = [int(float(v)) for v in txt.split(',') if v.strip() != '']
    flags = np.zeros(n, dtype=bool)
    flags[:min(n, len(vals))] = np.asarray(vals[:n], dtype=bool)
    return flags


class LaSOTTrain(object):
    """Official LaSOT training split (1120 sequences). Lazy: frames are listed on first access."""

    def __init__(self, root_dir, subset='Training', return_meta=True):
        self.root_dir = os.path.join(root_dir, subset)
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f'LaSOT folder not found: {self.root_dir}')
        self.return_meta = return_meta
        self.seq_dirs = sorted(os.path.dirname(p) for p in
                               glob.glob(os.path.join(self.root_dir, '*', '*', 'groundtruth.txt')))
        self.seq_names = [os.path.basename(d) for d in self.seq_dirs]
        if not self.seq_dirs:
            raise RuntimeError(f'no LaSOT sequence (groundtruth.txt) under {self.root_dir}')

    def __len__(self):
        return len(self.seq_dirs)

    def __getitem__(self, index):
        d = self.seq_dirs[index]
        anno = np.loadtxt(os.path.join(d, 'groundtruth.txt'), delimiter=',', dtype=np.float64)
        anno = anno.reshape(-1, 4)
        img_files = sorted(glob.glob(os.path.join(d, 'img', '*.jpg')))
        n = min(len(img_files), len(anno))
        img_files, anno = img_files[:n], anno[:n]
        if not self.return_meta:
            return img_files, anno
        hidden = _read_flags(os.path.join(d, 'full_occlusion.txt'), n) | \
                 _read_flags(os.path.join(d, 'out_of_view.txt'), n)
        cover = np.where(hidden, 0, 8).astype(np.int64)
        return img_files, anno, {'cover': cover}


class MixedSeqs(object):
    """Concatenation of several sequence datasets; each one can be repeated r times per epoch
    (Pair samples one pair per entry and per epoch, so repeat = pairs per sequence and per epoch)."""

    def __init__(self, datasets, repeats=None, names=None):
        repeats = repeats or [1] * len(datasets)
        self.datasets, self.repeats = datasets, repeats
        self.names = names or [type(d).__name__ for d in datasets]
        self.return_meta = True
        self.index = [(k, i) for k, (ds, r) in enumerate(zip(datasets, repeats))
                      for _ in range(r) for i in range(len(ds))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        k, i = self.index[index]
        item = self.datasets[k][i]
        if len(item) == 2:                       # dataset without meta -> everything visible
            img_files, anno = item
            item = (img_files, anno, {'cover': None})
        return item

    def summary(self):
        lines, tot = [], len(self)
        for name, ds, r in zip(self.names, self.datasets, self.repeats):
            n = len(ds) * r
            lines.append(f'  {name:8s}: {len(ds):5d} sequences x {r} = {n:6d} pairs/epoch ({100 * n / tot:4.1f} %)')
        lines.append(f'  total   : {tot:6d} pairs/epoch')
        return '\n'.join(lines)
