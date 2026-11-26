"""
DBSTrack tracker (got10k Tracker API: init(image, box) / update(image) -> box xywh).

Single-scale search (the box size comes from the regression head):
    score  = sigmoid(cls) * sigmoid(ctr)
    pscore = score * scale/ratio-change penalty, blended with a cosine window
    best location -> regressed box -> smoothed size update (lr = penalty * score * test_lr)

Two inference-time mechanisms (no extra weights), both off by default; their values are tuned ONLY on the
held-out validation split (tune.py):
  1) lost-target re-detection: when the best raw score < lost_thr, the next frames are searched on a LARGER
     search region at the SAME scale (lost_instance_sz, e.g. 527), the cosine window is weakened and the box is
     frozen until the score goes back above lost_thr.
  2) confidence-gated template update: every upd_interval frames, if the score > upd_thr and the target is not
     lost, a template is cropped on the current box; kernels used for tracking =
     (1 - upd_w) * first-frame kernels + upd_w * dynamic kernels (the first frame is always kept).
"""
from __future__ import absolute_import

import numpy as np
import torch
from got10k.trackers import Tracker as BaseTracker

from tracker.common import read_image, show_image
from tracker.model import SiameseNet, net_from_state_dict, score_points, score_size_for, EXEMPLAR_SZ, INSTANCE_SZ
from tracker.data import context_size, crop_patch


class Tracker(BaseTracker):

    def __init__(self, net_path=None, name='Tracker', penalty_k=0.04, window_influence=0.21,
                 test_lr=0.52, context=0.5, device=None, instance_sz=INSTANCE_SZ, net=None,
                 lost_thr=0.0, lost_instance_sz=463, lost_window=0.05,
                 upd_w=0.0, upd_thr=0.6, upd_interval=10, fast=True):
        super(Tracker, self).__init__(name, True)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.instance_sz = int(instance_sz)
        if net is not None:
            self.net = net
        elif net_path is not None:
            sd = torch.load(net_path, map_location='cpu')
            self.net = net_from_state_dict(sd.get('net', sd))
        else:
            self.net = SiameseNet()
        self.net.to(self.device).eval()
        self.penalty_k, self.wi, self.test_lr, self.context = penalty_k, window_influence, test_lr, context
        self.lost_thr, self.lost_instance_sz, self.lost_window = float(lost_thr), int(lost_instance_sz), lost_window
        self.upd_w, self.upd_thr, self.upd_interval = float(upd_w), float(upd_thr), int(upd_interval)
        # fast=True: crop uploaded as uint8 and converted on the GPU + ONE GPU->CPU transfer per frame instead of
        # three. Same numbers as fast=False (checked on GOT-10k val: max box difference 0.0 px), ~+30 % fps.
        self.fast = bool(fast)
        self._geo = {}

    def _geometry(self, S):
        if S not in self._geo:
            n = score_size_for(S)
            h = np.hanning(n)
            self._geo[S] = (score_points(score_size=n).numpy(), np.outer(h, h).reshape(-1))
        return self._geo[S]

    def _to_tensor(self, patch):
        if self.fast:        # uint8 upload (4x less data), exact same float values after conversion on the GPU
            return torch.from_numpy(np.ascontiguousarray(patch)).to(self.device).permute(2, 0, 1).float()[None]
        return torch.from_numpy(patch).permute(2, 0, 1).float()[None].to(self.device)

    @torch.no_grad()
    def _template(self, img, center, size):
        sz = context_size(size[0], size[1], self.context)
        z = crop_patch(img, center, sz, EXEMPLAR_SZ, self.avg)
        z = self._to_tensor(z)
        return self.net.encode_template(z)

    @torch.no_grad()
    def init(self, image, box):
        img = np.asarray(image)
        x, y, w, h = [float(v) for v in box]
        self.center = np.array([x + w / 2.0, y + h / 2.0])
        self.size = np.array([w, h])
        self.avg = img.reshape(-1, 3).mean(0)
        self.kernels0 = self._template(img, self.center, self.size)
        self.kernels = self.kernels0
        self.frame = 0
        self.lost = False
        self.scores = []                       # best raw score per frame (used to set thresholds on validation)

    @torch.no_grad()
    def update(self, image):
        img = np.asarray(image)
        self.frame += 1
        w, h = self.size
        S = self.lost_instance_sz if (self.lost and self.lost_thr > 0) else self.instance_sz
        points, window = self._geometry(S)
        wi = self.lost_window if (self.lost and self.lost_thr > 0) else self.wi
        # the crop SCALE does not depend on S: s_x grows with S so that pixels/target stay the same
        s_x = context_size(w, h, self.context) * S / float(EXEMPLAR_SZ)
        scale = S / s_x
        x = crop_patch(img, self.center, s_x, S, self.avg)
        x = self._to_tensor(x)
        cls, ctr, ltrb = self.net.head_forward(self.kernels, self.net.encode_search(x, self.kernels))
        if self.fast:          # one GPU -> CPU transfer (one synchronisation) instead of three
            c = torch.sigmoid(cls).reshape(-1)
            s = c * torch.sigmoid(ctr).reshape(-1)
            out = torch.cat([s, c, ltrb.reshape(-1)]).cpu().numpy()
            N = s.numel()
            score, cls_s, ltrb = out[:N], out[N:2 * N], out[2 * N:].reshape(4, -1).T
        else:
            score = (torch.sigmoid(cls) * torch.sigmoid(ctr)).reshape(-1).cpu().numpy()
            cls_s = torch.sigmoid(cls).reshape(-1).cpu().numpy()
            ltrb = ltrb.reshape(4, -1).t().cpu().numpy()

        x1 = points[:, 0] - ltrb[:, 0]
        y1 = points[:, 1] - ltrb[:, 1]
        x2 = points[:, 0] + ltrb[:, 2]
        y2 = points[:, 1] + ltrb[:, 3]
        bw, bh = x2 - x1, y2 - y1

        def change(r):
            return np.maximum(r, 1.0 / r)

        def sz(a, b):
            p = (a + b) * 0.5
            return np.sqrt((a + p) * (b + p))

        tw, th = w * scale, h * scale
        s_c = change(sz(bw, bh) / sz(tw, th))
        r_c = change((tw / th) / np.maximum(bw / np.maximum(bh, 1e-6), 1e-6))
        penalty = np.exp(-(r_c * s_c - 1) * self.penalty_k)
        pscore = penalty * score
        pscore = pscore * (1 - wi) + window * wi
        best = int(np.argmax(pscore))
        best_score = float(score[best])
        self.scores.append(best_score)

        found = not (self.lost_thr > 0 and best_score < self.lost_thr)
        if found:
            cx_c, cy_c = (x1[best] + x2[best]) / 2.0, (y1[best] + y2[best]) / 2.0
            lr = penalty[best] * cls_s[best] * self.test_lr
            self.center = self.center + (np.array([cx_c, cy_c]) - S / 2.0) / scale
            new_wh = np.array([bw[best], bh[best]]) / scale
            self.size = (1 - lr) * self.size + lr * new_wh
        # else: target considered lost -> keep the last box, search wider next frame
        self.lost = not found

        H, W = img.shape[:2]
        self.center = np.clip(self.center, 0, [W, H])
        self.size = np.clip(self.size, 10, [W, H])

        if self.upd_w > 0 and found and best_score > self.upd_thr and self.frame % self.upd_interval == 0:
            kd = self._template(img, self.center, self.size)
            self.kernels = tuple((1 - self.upd_w) * k0 + self.upd_w * k1 for k0, k1 in zip(self.kernels0, kd))

        return np.array([self.center[0] - self.size[0] / 2.0, self.center[1] - self.size[1] / 2.0,
                         self.size[0], self.size[1]])

    def track(self, img_files, box, visualize=False):
        """same contract as got10k Tracker.track (timing excludes image reading / display)."""
        import time
        n = len(img_files)
        boxes = np.zeros((n, 4))
        boxes[0] = box
        times = np.zeros(n)
        for f, img_file in enumerate(img_files):
            img = read_image(img_file)
            t0 = time.time()
            if f == 0:
                self.init(img, box)
            else:
                boxes[f, :] = self.update(img)
            times[f] = time.time() - t0
            if visualize:
                show_image(img, boxes[f, :])
        return boxes, times
