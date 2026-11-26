# DBSTrack

**DBSTrack: Depth-Specialized Dual-Branch Siamese Tracking in a Single Lightweight Backbone**

Alhou Harouna Adamou, Wenjun Zhou — Southwest Petroleum University

DBSTrack is a lightweight Siamese tracker (2.03 M parameters, 1.13 GMACs) that reads one MobileNetV3-Large
backbone at two depths. A stride-8 branch keeps the spatial detail needed for localization and a stride-16 branch
carries the semantic context; the stride-32 stage is discarded. The template conditions the search stream inside
the backbone (template modulation at stride 8 and stride 16, one low-resolution attention block on the stride-16
map), so the two branches interact before the final fusion. Both branches are correlated pixel-wise and fed to an
anchor-free head.

## Results

| Benchmark | Metric | DBSTrack |
|---|---|---|
| OTB100 | AUC / Precision | 71.1 / 93.1 |
| LaSOT | AUC / P<sub>norm</sub> / P | 57.0 / 65.6 / 57.2 |
| TrackingNet | AUC / P<sub>norm</sub> / P | 74.6 / 80.4 / 69.4 |
| TC128 | AUC | 57.1 |
| NFS30 | AUC | 60.6 |
| UAV123 | AUC | 63.4 |
| DTB70 | AUC | 64.5 |

Speed (batch 1, full tracker with re-detection and template update): about 285 FPS on an RTX 4060 Ti GPU and
80 FPS on a Ryzen 5 7600X CPU.

## Installation

```bash
conda create -n dbstrack python=3.13 -y
conda activate dbstrack
pip install -r requirements.txt
```

## Pretrained model

`models/dbstrack.pth` is the published model, the one evaluated in the paper. It is the last epoch of the
fine-tuning run (`model_e20.pth` of run 2 below), which starts from the 60-epoch checkpoint of run 1: 80 training
epochs in total. It is loaded by default by `test.py`.

## Evaluation

Set `BENCHMARK` and the dataset path in the configuration block of `test.py`, then:

```bash
python test.py
```

Supported: OTB100, LaSOT, TrackingNet (writes a submission archive), UAV123, DTB70, TC128, NFS30, GOT-10k test.
Scores, raw boxes and plots go to `results/<BENCHMARK>/`; every run adds one line to `results/all_results.csv`.
The OTB100 numbers above were computed with the pysot toolkit from the raw boxes written by `test.py`.

CPU speed: `python cpu_fps.py OTB100`.

## Raw results

`raw_results/` holds the boxes of the published model (`models/dbstrack.pth`, tracker parameters of `test.py`) on
OTB100, LaSOT, TC128, NFS30, UAV123 and DTB70: one `<sequence>.txt` per sequence, one box per frame
(x, y, w, h, 1-based pixel coordinates as written by the got10k toolkit). `raw_results/TrackingNet/` holds the
archive submitted to the TrackingNet evaluation server.

## Training

Training data: GOT-10k train, LaSOT training split and COCO 2017 train. Set the paths in `train.py`, then:

```bash
# 1) 60 epochs from ImageNet weights (10 epochs with the backbone frozen, then 50 epochs end to end)
python train.py                     # VARIANT = 'scratch'
# 2) 20 more epochs from the last checkpoint (warm restart)
python train.py                     # VARIANT = 'finetune'
```

Tracker hyper-parameters are chosen on a held-out validation split (GOT-10k val + 20 LaSOT training sequences),
never on the test benchmarks:

```bash
python tune.py
```

## Code layout

```
tracker/backbone.py    MobileNetV3-Large read at stride 8 and stride 16
tracker/model.py       template modulation, low-resolution attention, correlation, anchor-free head, loss
tracker/tracker.py     online tracker (re-detection, template update)
tracker/data.py        training pairs (GOT-10k, LaSOT), tracker/coco.py (COCO pairs)
tracker/benchmarks.py  benchmark wrappers (got10k toolkit), tracker/lasot_official.py (LaSOT metrics)
train.py | test.py | tune.py | cpu_fps.py
```

## Acknowledgements

The evaluation relies on the [got10k toolkit](https://github.com/got-10k/toolkit) and the
[pysot toolkit](https://github.com/StrangerZhang/pysot-toolkit). The template modulation is inspired by the
efficient template modulation of AsymTrack.

## Citation

The paper is under review. A BibTeX entry will be added here after publication.

## License

The code is released under the [MIT License](LICENSE). The pretrained model is provided for research purposes;
its use is also subject to the terms of the training datasets (GOT-10k, LaSOT, COCO).
