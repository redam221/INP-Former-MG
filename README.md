# INP-Former-MG: Multi-Granularity Intrinsic Normal Prototypes for Anomaly Detection

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-orange.svg)](https://pytorch.org/)
[![CI](https://github.com/redam221/INP-Former-MG/actions/workflows/ci.yml/badge.svg)](https://github.com/redam221/INP-Former-MG/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An architecture-level extension of **INP-Former** (CVPR 2025, *"Exploring
Intrinsic Normal Prototypes within a Single Image for Universal Anomaly
Detection"*). INP-Former-MG replaces the single, depth-agnostic prototype set
of the original with **independent prototype sets at multiple feature
granularities**, plus three auxiliary improvements (soft coherence assignment,
prototype diversity regularization, and adaptive prototype gating).

> This repository is a research-focused re-implementation and extension. It is
> intended for reproducible single-/multi-/few-shot anomaly detection on
> MVTec-AD, VisA, and Real-IAD.

---

## What's wrong with a single prototype set?

The original INP-Former fuses 8 intermediate ViT layers into **one** feature
tensor (`mean`), aggregates a **single** set of `INP_num` learnable prototypes
from it, and guides the **entire** decoder with that one set:

```
encoder (layers 2..9) ──mean──> one fused feature
                                      │
                           single INP set (6)  <── one Aggregation_Block
                                      │
                           one decoder (8 blocks) ──> reconstruction error
```

Shallow layers encode local texture / edges; deep layers encode part identity
and semantic layout. Collapsing them into a single prototype space is a
representational bottleneck: one prototype set must simultaneously describe
fine-grained texture *and* coarse structure.

## What INP-Former-MG does

INP-Former-MG partitions the intermediate layers into `G` **granularity
groups** (e.g. shallow / middle / deep) and gives each group its **own**
prototype set, extractor, bottleneck, and decoder:

```
encoder (layers 2..9)
   ├─ shallow [2,3,4] ──> INP set A ──> decoder A ─┐
   ├─ middle [5,6,7] ───> INP set B ──> decoder B ─┼──> fused anomaly map
   └─ deep   [8,9]  ───> INP set C ──> decoder C ─┘
```

With `G = 1` (a single group) the model **reduces exactly to the original
INP-Former**, which makes the single-vs-multi granularity ablation trivial and
the comparison against the paper's reported numbers direct.

### The three auxiliary improvements

| Improvement | Problem it solves | Where |
|---|---|---|
| **Soft-assignment coherence** | hard `torch.min` blocks gradient to non-nearest prototypes | `models/uad.py` → `soft_coherence_loss` |
| **Prototype diversity regularization** | several INPs can collapse to the same normal mode | `models/uad.py` → `prototype_diversity_loss` |
| **Adaptive prototype gating** | `INP_num` is a hand-tuned global constant | `models/uad.py` → `adaptive_prototype_gate` |

---

## Installation

```bash
conda create -n inp-mg python=3.12
conda activate inp-mg

# 1. Install PyTorch matching your CUDA version first (example: CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Then the remaining dependencies
pip install -r requirements.txt
```

For CPU-only, replace step 1 with
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.

The encoder backbones (DINOv2 / DINOv1 / BEiT) are downloaded automatically on
first use.

## Prepare datasets

Follow the same layout as the original INP-Former:

- **MVTec-AD** — download from [mvtec.com](https://www.mvtec.com/company/research/datasets/mvtec-ad), unzip to `../mvtec_anomaly_detection`.
- **VisA** — preprocess into `../VisA_pytorch/1cls` using the official splitter.
- **Real-IAD** — contact the authors at [realiad4ad.github.io](https://realiad4ad.github.io/Real-IAD/), unzip to `../Real-IAD`.

## Quick start

Single-granularity baseline (equivalent to the original INP-Former):

```bash
python train.py --dataset MVTec-AD --data-path ../mvtec_anomaly_detection \
    --layer-groups "0,1,2,3,4,5,6,7" --inp-nums 6 --phase train
```

3-granularity variant (shallow / middle / deep), soft coherence + diversity:

```bash
python train.py --dataset MVTec-AD --data-path ../mvtec_anomaly_detection \
    --layer-groups "0,1,2;3,4,5;6,7" --inp-nums 4,4,4 \
    --coherence-mode soft --diversity-weight 0.05 --phase train
```

Evaluate a trained checkpoint:

```bash
python train.py --dataset MVTec-AD --data-path ../mvtec_anomaly_detection \
    --layer-groups "0,1,2;3,4,5;6,7" --inp-nums 4,4,4 --phase test
```

Verify the core logic without any dataset or backbone download:

```bash
python smoke_test.py
```

---

## Project structure

```
INP-Former-MG/
├── train.py                  # training / evaluation entry point
├── smoke_test.py             # core-logic smoke test (no data / no backbone)
├── dataset.py                # MVTec-AD / Real-IAD dataset classes
├── aug_funcs.py              # geometric augmentation helpers (kornia)
├── utils.py                  # metrics, schedulers, logging, evaluation
├── models/
│   ├── uad.py                # INP_Former + 3 auxiliary losses (the core change)
│   ├── builder.py            # build_model() from a flat config
│   ├── vision_transformer.py # Aggregation_Block / Prototype_Block
│   └── vit_encoder.py        # encoder loading (DINOv2 / DINOv1 / BEiT)
├── optimizers/
│   └── StableAdamW.py        # stable AdamW optimizer
├── dinov1/  dinov2/  beit/   # vendored encoder backbones
├── scripts/
│   └── run_ablation.sh       # one-shot ablation ladder
├── docs/
│   └── technical_details.md  # code-level diff vs the original INP-Former
└── requirements.txt
```

---

## Configuration

| Argument | Meaning | Example |
|---|---|---|
| `--layer-groups` | Partition of the 8 intermediate layers into granularity groups | `"0,1,2;3,4,5;6,7"` |
| `--inp-nums` | Number of INPs per group | `"4,4,4"` |
| `--decoder-layers` | Decoder depth per group (default = group size) | `"4,4,2"` |
| `--coherence-mode` | `soft` (ours) or `hard` (original `gather_loss`) | `soft` |
| `--coherence-temperature` | Temperature for soft assignment | `0.1` |
| `--diversity-weight` | Prototype diversity regularizer weight | `0.05` |
| `--adaptive-prototypes` | Enable learned per-prototype gating | flag |
| `--gate-sparsity-weight` | Sparsity weight for the adaptive gate | `0.01` |

---

## Ablation design

The change is deliberately decomposed into independent, toggleable axes so each
can be ablated cleanly against the original:

| # | Axis | Original | MG | Command switch |
|---|---|---|---|---|
| A | Granularity | 1 group | 2 / 3 groups | `--layer-groups` |
| B | Coherence assignment | `hard` | `soft` | `--coherence-mode` |
| C | Prototype diversity | off | on | `--diversity-weight` |
| D | INP count | fixed `6` | adaptive | `--adaptive-prototypes` |

A recommended ablation ladder (each row keeps everything else at the original
setting):

1. `--layer-groups "0,1,2,3,4,5,6,7" --inp-nums 6 --coherence-mode hard` → **baseline**
2. `--layer-groups "0,1,2,3;4,5,6,7" --inp-nums 6,6` → 2-granularity
3. `--layer-groups "0,1,2;3,4,5;6,7" --inp-nums 4,4,4` → 3-granularity
4. row 3 + `--coherence-mode soft` → soft coherence
5. row 4 + `--diversity-weight 0.05` → diversity regularization
6. row 5 + `--adaptive-prototypes --gate-sparsity-weight 0.01` → adaptive INP count

## Results

> **Status: training in progress.** The benchmark numbers below are not yet
> published; they will be filled in as soon as training on MVTec-AD / VisA /
> Real-IAD completes. Every row is fully reproducible via the commands in the
> [Ablation design](#ablation-design) section and the ladder script
> `scripts/run_ablation.sh`, so the tables can be regenerated from scratch.

### Metric convention

We report the same metrics as the original INP-Former / Dinomaly line of work,
evaluated per category and averaged:

| Level | Metrics |
|---|---|
| Image-level | AUROC · AP · max-F1 |
| Pixel-level | AUROC · AP · max-F1 · AUPRO |

### Reproducing a result row

Each configuration is reproduced in two steps — train, then evaluate (the
checkpoint path is derived deterministically from the same flags):

```bash
# (1) train the 3-granularity + soft-coherence + diversity variant
python train.py --dataset MVTec-AD --data-path ../mvtec_anomaly_detection \
    --layer-groups "0,1,2;3,4,5;6,7" --inp-nums 4,4,4 \
    --coherence-mode soft --diversity-weight 0.05 \
    --total-epochs 200 --batch-size 16 --phase train

# (2) evaluate the saved checkpoint
python train.py --dataset MVTec-AD --data-path ../mvtec_anomaly_detection \
    --layer-groups "0,1,2;3,4,5;6,7" --inp-nums 4,4,4 \
    --coherence-mode soft --diversity-weight 0.05 --phase test
```

Per-category and mean metrics are printed to the console and written to
`saved_results/<save_name>/log.txt`. To reproduce the full ladder in one shot:

```bash
bash scripts/run_ablation.sh MVTec-AD ../mvtec_anomaly_detection
```

### Benchmark tables

> Placeholder — to be filled in once training completes.

**MVTec-AD (mean over 15 categories)**

| Configuration | I-AUROC | I-AP | I-max-F1 | P-AUROC | P-AP | P-max-F1 | AUPRO |
|---|---|---|---|---|---|---|---|
| baseline (single granularity) | — | — | — | — | — | — | — |
| 2-granularity | — | — | — | — | — | — | — |
| 3-granularity | — | — | — | — | — | — | — |
| + soft coherence | — | — | — | — | — | — | — |
| + diversity reg. | — | — | — | — | — | — | — |
| + adaptive gating | — | — | — | — | — | — | — |

**VisA / Real-IAD**

| Dataset | I-AUROC | I-AP | I-max-F1 | P-AUROC | P-AP | P-max-F1 | AUPRO |
|---|---|---|---|---|---|---|---|
| VisA | — | — | — | — | — | — | — |
| Real-IAD | — | — | — | — | — | — | — |

## Citation

If this repository is useful for your research, please cite the original work:

```bibtex
@InProceedings{luo2025INP-Former,
    author    = {Luo, Wei and Cao, Yunkang and Yao, Haiming and Zhang, Xiaotian and Lou, Jianan and Cheng, Yuqi and Shen, Weiming and Yu, Wenyong},
    title     = {Exploring Intrinsic Normal Prototypes within a Single Image for Universal Anomaly Detection},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    year      = {2025}
}
```

## Acknowledgements

This project builds on [INP-Former](https://github.com/WeiLuo-cyber/INP-Former),
which itself builds on [Dinomaly](https://github.com/guojiajeremy/Dinomaly).
We thank the original authors for their foundational work.

## License

[MIT](LICENSE)
