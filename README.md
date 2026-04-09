# NPS-Net: Nested Radially Monotone Polar Occupancy Estimation Clinically-Grounded Optic Disc and Cup Segmentation for Glaucoma Screening

Rimsa Goperma, Rojan Basnet, and Liang Zhao  
Graduate School of Advanced Integrated Studies in Human Survivability (GSAIS), Kyoto University

Valid segmentation of the optic disc (OD) and optic cup (OC) from fundus photographs is essential for glaucoma screening. Unfortunately, existing deep learning methods do not guarantee clinical validness including star-convexity and nested structure of OD and OC, resulting in corruption of diagnostic metrics, especially under cross-dataset domain shift. This paper proposes NPS-Net (Nested Polar Shape Network), the first framework that formulates the OD/OC segmentation as nested radially monotone polar occupancy estimation. This output representation can guarantee the aforementioned clinical validness and achieve high accuracy.

Evaluated across seven public datasets, NPS-Net shows strong zero-shot generalization. On RIM-ONE, it maintains 100% anatomical validity and improves Cup Dice by 12.8% absolute over the best baseline, reducing vCDR MAE by over 56%. On PAPILA, it achieves Disc Dice of 0.9438 and Disc HD95 of 2.78px, an 83% reduction over the best competing method.

![NPS-Net Architecture](Utils/NSPNet.png)

## Quick Start

### Training

NPS-Net uses staged training - B4 variant trains in 3 stages:
- Stage A (epochs 1-20): Train polar encoder + monotone heads only
- Stage B (epochs 21-30): Enable shape prior branch
- Stage C (epochs 31-80): Enable consistency loss + full optimization

```bash
# Train baseline models
python training/train.py --model vanilla
python training/train.py --model attunet

# Train NPS-Net ablation variants (handles staged training internally)
python training/train_ablation.py --variant b2
python training/train_ablation.py --variant b3
python training/train_ablation.py --variant b4
```

### Evaluation

```bash
# Baseline inference (vanilla, attunet, resunet, polar_unet, transunet, beal, dofe)
python evaluation/inference.py --model vanilla --test

# NPS-Net evaluation on external datasets
python evaluation/inference_combined.py --model npsnet --test
python evaluation/inference_papila.py --model npsnet
python evaluation/inference_refuge.py --model npsnet

# External dataset evaluation
python evaluation/inference_papila.py --model all
python evaluation/inference_refuge.py --model all

# Ablation evaluation
python evaluation/inference_ablation.py --variant b4 --test
python evaluation/inference_polar_tta_ablation.py --test
```

## Model Weights

Pre-trained weights are available on HuggingFace: https://huggingface.co/Rimsa66/nps-net

### NPS-Net (Ours)

| Variant | Description |
|---------|-------------|
| B2 | Monotone heads |
| B3 | + Nesting |
| B4 | + Shape prior |

### Baselines

Vanilla UNet, Attention UNet, ResUNet, PolarUNet, TransUNet, BEAL, DoFE

Download using Python:
```python
from huggingface_hub import hf_hub_download

# Download NPS-Net B4
model_path = hf_hub_download(
    repo_id="Rimsa66/nps-net", 
    filename="b4/best_model.pth"
)
```

## Requirements

```
torch>=2.0
torchvision
numpy
pandas
opencv-python
scipy
scikit-learn
wandb (optional)
```

## Key Features

- **Nested Polar Representation**: Guarantees OD⊇OC (anatomical validity)
- **Monotone Polar Occupancy**: Star-convex shape via cumulative-decrement
- **Shape Prior Branch**: Learned boundary distributions with confidence gating
- **Polar-TTA**: Test-time augmentation for improved localization
