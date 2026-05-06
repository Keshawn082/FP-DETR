# FP-DETR

**FP-DETR** is a novel real-time object detection model designed for surrounding personnel detection around underground Load-Haul-Dump (LHD) vehicles in complex mining environments. Built on the RT-DETR framework, it introduces three key innovations to tackle the challenges of dust, uneven lighting, and occlusions.

## Key Results

| Metric | FP-DETR | Baseline (RT-DETR) | Δ |
|---|---|---|---|
| mAP@0.5 | **87.3%** | 85.7% | +1.6% |
| Parameters | **21% fewer** | — | — |
| False alarm rate | Lower (field tests) | — | — |

---

## Architecture

```
Image  ──►  PResNet Backbone  ──►  HybridEncoder  ──►  Decoder  ──►  Heads
                                   (CSP-FFCM + POTE)   (POTE)
```

### CSP-FFCM — CSP Fast Fourier Convolution Module

Found in `src/nn/model/csp_ffcm.py`.

Fuses **spatial** and **frequency-domain** features:

- **CSP stream**: Cross Stage Partial bottleneck blocks capture local spatial context efficiently.
- **FFC stream**: Fast Fourier Convolution applies 1×1 convolutions in the frequency domain (via rFFT2/irFFT2), capturing global context without the O(N²) cost of standard convolutions.
- The two streams are concatenated and projected back, with a residual shortcut.

This helps the model remain robust to dust and uneven illumination, which primarily manifest as low-frequency perturbations in the spectrum.

### POTE — POlarity-aware Transformer Encoder

Found in `src/nn/model/pote.py`.

Replaces standard softmax attention with **linear** attention using a polarity decomposition:

- Each query/key vector is split into a **positive polarity** (foreground-attending) and a **negative polarity** (background-suppressing) using an ELU+1 kernel.
- Attention complexity is reduced from O(N²) to O(N·d), enabling processing of large feature maps in real time.
- The polarity split doubles effective feature capacity without adding parameters.

### MAL — Multi-scale Adaptive Loss

Found in `src/nn/criterion/mal_criterion.py`.

Combines three terms with a scale-adaptive weight:

1. **Varifocal classification loss** — soft IoU-weighted focal loss that focuses training on uncertain predictions.
2. **Scale-adaptive GIoU regression loss** — small targets (common in overhead mine views) receive stronger gradients (`w = 2 − area`).
3. **L1 coordinate loss** — lightweight auxiliary regression signal.

---

## Repository Structure

```
FP-DETR/
├── configs/
│   ├── fp_detr_r18vd.yaml      # Lightweight R-18 config
│   └── fp_detr_r50vd.yaml      # Full R-50 config
├── src/
│   ├── core/                   # Config loading utilities
│   ├── data/                   # COCO dataset + transforms
│   ├── misc/                   # Logging
│   └── nn/
│       ├── backbone/           # PResNet (R-18/R-50)
│       ├── criterion/          # MAL loss
│       └── model/
│           ├── csp_ffcm.py     # CSP-FFCM module
│           ├── pote.py         # POTE attention
│           ├── hybrid_encoder.py
│           └── fp_detr.py      # Full model
├── tools/
│   ├── train.py                # Training script
│   └── eval.py                 # Evaluation / inference script
└── tests/                      # Unit + integration tests (83 tests)
```

---

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train

```bash
python tools/train.py --config configs/fp_detr_r50vd.yaml
```

### Evaluate

```bash
python tools/eval.py \
    --config configs/fp_detr_r50vd.yaml \
    --checkpoint outputs/fp_detr_r50/checkpoint_epoch0119.pth
```

### Run tests

```bash
python -m pytest tests/ -v
```

---

## Citation

If you use FP-DETR in your research, please cite:

```bibtex
@article{fpdetr2024,
  title   = {FP-DETR: Real-Time Surrounding Personnel Detection for Underground LHD Vehicles},
  year    = {2024},
}
```

