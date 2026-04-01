# Scarfold — Protein Fold Visualizer

> A simplified **AlphaFold-inspired** pipeline: amino acid sequence → predicted inter-residue distance map → 3-D coordinate reconstruction → interactive scoring & visualization.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch) ![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

1. [What is this?](#what-is-this)
2. [Pipeline at a glance](#pipeline-at-a-glance)
3. [Model architecture](#model-architecture)
4. [Results & metrics](#results--metrics)
5. [Project structure](#project-structure)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Interactive outputs](#interactive-outputs)
9. [Ideas to extend](#ideas-to-extend)

---

## What is this?

Scarfold is an educational project that shows **how modern protein structure prediction works** — without the complexity of AlphaFold's full codebase.

It takes a raw amino-acid sequence and:
- predicts all pairwise **inter-residue distances** (a distogram),
- reconstructs **3-D coordinates** from those distances (classical MDS),
- aligns the prediction to a native structure (Kabsch alignment),
- scores quality with **pLDDT**, **local lDDT**, **contact-map F1**, and **TM-score proxy**,
- renders everything as **interactive HTML files** (Plotly).

---

## Pipeline at a glance

```
 Amino-acid sequence
        │
        ▼
  One-hot encoding              (20 features per residue)
        │
        ▼
 ┌──────────────────┐
 │  Neural Network  │  ──── MLP  or  Transformer
 └──────────────────┘
        │   predicts
        ▼
  Distance matrix (L × L)      (Å between every residue pair)
        │
        ▼
  Classical MDS                 (distance → 3-D coordinates)
        │
        ▼
  Kabsch alignment              (rigid-body fit to native)
        │
        ▼
  Scoring & Visualization
     • pLDDT per residue
     • local lDDT per residue
     • Contact-map F1 / Precision / Recall
     • TM-score proxy
     • Interactive 3-D HTML outputs
```

---

## Model architecture

Two model variants are available, both predicting the symmetric distance matrix **D[i,j]**:

### Input features (48-dim per residue)

Instead of a simple 20-dim one-hot, every residue is encoded as a **48-dimensional vector**:

```
One-hot      (20) — residue identity
BLOSUM62     (20) — substitution log-odds row (evolutionary context)
Physicochemical (8) — hydrophobicity, charge, polarity, volume,
                       aromaticity, helix/sheet/coil propensity
```

This gives the model evolutionary and biophysical context without requiring a full MSA.

### MLP (baseline)

```
Input  (L × 48)  ──flatten──►  Linear(L·48 → 1024)  ──LayerNorm──ReLU──Dropout
                                Linear(1024 → 512)   ──LayerNorm──ReLU──Dropout
                                Linear(512 → L²)
                                reshape → (L, L)  →  symmetrize  →  ReLU + ε
```

### Transformer (recommended)

```
Input  (B, L, 48)
   │
   ▼  Linear projection  →  (B, L, 256)  + learnable positional embedding
   │
   ▼  TransformerEncoder  (3 layers, 4 heads, GELU, pre-LN, dropout 0.1)
   │
   ▼  Outer-product pair features  →  (B, L, L, 512)
   │
   ▼  Pair MLP (3 layers, GELU)  →  (B, L, L, 1)  →  symmetrize  →  ReLU + ε
```

### Training objective

Combined loss: **MSE + 0.5 × Contact BCE**

$$\mathcal{L} = \mathcal{L}_{\text{MSE}} + 0.5 \cdot \mathcal{L}_{\text{contact BCE}}$$

The contact BCE loss provides explicit signal for predicting which residues are within 8 Å — sparse in MSE but critical for function. Optimizer: **AdamW** + **cosine annealing** LR + gradient clipping.

### Structure reconstruction

Coordinates are recovered from the predicted distance matrix using **gradient-based metric optimisation** (warm-started from classical MDS, refined with Adam + Huber loss for 500 iterations), which is more robust to noisy predicted distances than closed-form MDS.

---

## Results & metrics

Evaluation on PDB 1A3N (chain A, 100 residues), `model_final_real.pt`:

| Metric | Value |
|---|---|
| RMSD (Kabsch aligned) | **7.86 Å** |
| RMSD (unaligned) | 15.24 Å |
| Mean pseudo pLDDT | 16.2 |
| Mean local lDDT | 10.3 |
| Contact-map F1 | **0.533** |
| TM-score proxy | 0.009 |

### Training loss curves

The plots below represent the training and validation loss curves saved during training (`train_history_real.csv`, `train_history.csv`).

```
Loss
 40 │▓
    │▓▓
 30 │  ▓▓
    │    ▓▓▓
 20 │       ▓▓▓▓
    │            ▓▓▓▓▓
 10 │                  ▓▓▓▓▓▓▓▓▓▓▓▓
    └─────────────────────────────────  Epoch →
         Train loss converges smoothly
```

> Full per-epoch CSV logs are in `train_history.csv` and `train_history_real.csv`.

---

## Project structure

```
Scarfold/
├── src/
│   ├── main.py           ← unified entry point (train + evaluate + demo)
│   ├── model.py          ← MLP & Transformer distance predictors (PyTorch)
│   ├── train.py          ← training loop with checkpointing & CSV logging
│   ├── evaluate.py       ← evaluation script, saves JSON results
│   ├── benchmark.py      ← statistical MLP vs Transformer comparison (t-test)
│   ├── download_data.py  ← download real PDB structures for training
│   ├── utils.py          ← MDS, Kabsch, pLDDT, lDDT, TM-score, BLOSUM62 helpers
│   └── visualize.py      ← interactive Plotly HTML visualizations
│
├── data/
│   └── pdbs/             ← place .pdb files here (or use download_data.py)
│
├── checkpoints/          ← saved model checkpoints per epoch
├── results/              ← evaluation JSON outputs
├── report/
│   └── report.md         ← 4-page paper-style write-up
│
├── model_final.pt        ← trained model (synthetic data)
├── model_final_real.pt   ← trained model (real PDB data)
│
├── train_history.csv          ← epoch / train_loss / val_loss (synthetic)
├── train_history_real.csv     ← epoch / train_loss / val_loss (real PDB)
│
├── out_pred_vs_native.html        ← 3-D predicted vs native overlay
├── out_pred_struct_colored.html   ← 3-D structure colored by pLDDT
├── out_contact_map.html           ← contact map heatmap
├── out_plddt_profile.html         ← per-residue pLDDT bar chart
├── out_local_lddt_profile.html    ← per-residue local lDDT bar chart
├── out_tm_score.html              ← TM-score gauge
│
├── requirements.txt
└── README.md
```

---

## Installation

**Requirements:** Python 3.10+, pip

```bash
# 1. Clone the repo
git clone https://github.com/immortal71/Scarfold.git
cd Scarfold

# 2. (Recommended) create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

`requirements.txt` installs: `numpy`, `torch`, `plotly`, `biopython`, `scipy`

---

## Usage

All commands are run from the **project root**.

### Quick demo (synthetic data, no PDB needed)

```bash
python src/main.py --demo
```

### Download real PDB training data

```bash
# Downloads ~80 small, high-quality single-chain proteins from RCSB
python src/download_data.py --n 80 --out data/pdbs

# If the API search fails, use the built-in curated list of known structures
python src/download_data.py --n 50 --out data/pdbs --use-fallback
```

### Train on real PDB files

```bash
python src/train.py \
    --train-from-pdb \
    --pdb-dir data/pdbs \
    --chain A \
    --model transformer \
    --epochs 60 \
    --lr 5e-4 \
    --checkpoint-dir checkpoints \
    --save-path model_final_real.pt \
    --csv train_history_real.csv
```

### Train on synthetic data

```bash
python src/main.py \
    --train \
    --model transformer \
    --samples 400 \
    --length 40 \
    --epochs 120 \
    --lr 5e-4 \
    --checkpoint-dir checkpoints \
    --save-path model_final.pt \
    --csv train_history.csv
```

### Statistical benchmark (MLP vs Transformer)

```bash
# Synthetic benchmark with t-test results
python src/benchmark.py --samples 400 --length 40 --epochs 80 --n-test 30

# Benchmark on real PDB data (requires data/pdbs to be populated)
python src/benchmark.py --train-from-pdb --pdb-dir data/pdbs --epochs 60
```

### Evaluate a trained model

```bash
# On a PDB ID (auto-download)
python src/evaluate.py --model transformer --load-model model_final_real.pt --pdb-id 1crn --chain A

# On a local PDB file
python src/evaluate.py --model transformer --load-model model_final_real.pt --pdb data/my.pdb --chain A

# On a FASTA file
python src/evaluate.py --model mlp --load-model model_final.pt --fasta data/example.fasta
```

---

## Interactive outputs

After running the pipeline, open any `.html` file in your browser:

| File | What you see |
|---|---|
| `out_pred_vs_native.html` | 3-D overlay of predicted vs native backbone |
| `out_pred_struct_colored.html` | Predicted backbone colored by per-residue confidence |
| `out_contact_map.html` | Predicted vs native contact-map heatmap (Å) |
| `out_plddt_profile.html` | Per-residue pseudo pLDDT bar chart |
| `out_local_lddt_profile.html` | Per-residue local lDDT bar chart |
| `out_tm_score.html` | TM-score gauge chart |

---

## Ideas to extend

- **PSSM from PSI-BLAST** — run 3 iterations of PSI-BLAST against UniRef50 to build per-position profiles (the single biggest improvement possible)
- **Binned distogram** — classify distances into 64 bins (like real AlphaFold) instead of direct regression for better-calibrated confidence
- **Evoformer-lite** — add pair-bias attention: update pair representations directly instead of just per-residue representations
- **End-to-end training** — backpropagate through the structure module with a differentiable lDDT loss
- **Larger dataset** — train on CATH / SCOPe domain libraries for realistic benchmarking

See [`report/report.md`](report/report.md) for a full paper-style write-up of the methodology, results, and discussion.
We now persist training and evaluation outputs in a dedicated path to make reproducibility easy.

- Default checkpoints: `checkpoints/best_model_epoch_{n}.pt`
- Final model export: `model_final.pt` (or any path via `--save-path` / `--save-model`)
- CSV training history: `train_history.csv`
- Synthetic demo summaries: output printed in console + HTML files (`out_*.html`)
- Recommended persisted test output file: `results/eval_{timestamp}.json` (create and write in evaluate scripts)

If you want, add a lightweight method to `src/evaluate.py`:

```python
import json, os, datetime

def save_evaluation_results(results, out_dir='results'):
    os.makedirs(out_dir, exist_ok=True)
    filename = f"eval_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
    path = os.path.join(out_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    return path
```

Use-case summary
----------------
This project is designed for educational research and proof-of-concept work in protein structure prediction, such as:
- Demonstrating AlphaFold-like workflows for academic applications
- Benchmarking lightweight sequence->distogram models on small PDBs
- Comparing model variants (MLP, Transformer) in an undergraduate research project
- Visualizing contacts, confidence scores, and RMSD in interactive outputs
- Extending to real CASP-like datasets, MSA-based features, and structural classification

Target users:
- students applying to computational biology programs
- researchers prototyping new scoring or contact prediction methods
- instructors teaching structural bioinformatics techniques
- hackathon or early-stage proof-of-concept teams

Future directions:
- real CASP/PDB dataset loader + MSA profiles
- pairwise distance binning instead of regression
- TM-score curves, residue-level error heatmaps
- Streamlit/Gradio interactive model comparison UI

Important command usage notes
-----------------------------

- If you are in the project root (`C:\Users\HUAWEI\Downloads\bioproject`):
  - run `python src/main.py`
  - Example:
    `python src/main.py --demo`

- If you are in the `src` folder (`bioproject\src`):
  - run `python main.py `
  - Example:
    `python main.py --demo`

- Do not run `python main.py ` from project root (no `main.py` there).
- Do not split options into separate commands:
  - WRONG:
    - `--csv train_history.csv` (by itself)
    - `--checkpoint-dir checkpoints` (by itself)
  - RIGHT:
    - `python src/main.py --train  --checkpoint-dir checkpoints --save-model model_final.pt --csv train_history.csv`

- 