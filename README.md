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

> **New in latest version (v3) :** **Real-PDB training** on 20 diverse proteins spanning all SCOP classes (all-α, all-β, α+β, α/β), **variable-length training** (no zero-padding, each protein at its natural length), **uniform distogram loss** (backbone geometry enforced by MDS not loss), **long-range contact precision** (top-L/5, |i-j|≥12) added as standard CASP metric. Previous: backbone MDS constraint (Cα–Cα ≡3.8Å), SS-guided synthetic data, recycling, SS head, pLDDT head, pair-bias attention, PSSM module, paired t-test, k-fold CV.

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
  Rich encoding  (48-dim per residue)   ──or──  PSSM encoding (50-dim)
  ┌─────────────────────────────────┐          ┌──────────────────────────┐
  │ One-hot (20) + BLOSUM62 (20)    │          │ One-hot + PSSM profile   │
  │ + Physicochemical (8)           │          │ + Physchem + complexity   │
  └─────────────────────────────────┘          └──────────────────────────┘
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
  Gradient MDS (Adam + Huber)   (warm-started from classical MDS)
        │                       (more robust to noisy predictions)
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

### Input features

Three levels of residue encoding are available:

| Level | Dim | Contents | Script |
|---|---|---|---|
| One-hot | 20 | Residue identity only | default |
| **Rich** (default) | **48** | One-hot + BLOSUM62 + physicochemical | `utils.rich_encoding()` |
| **PSSM** | **50** | One-hot + PSSM profile + physicochemical + relative position + local complexity | `src/pssm.py` |

The rich encoding gives the model evolutionary and biophysical context without requiring a full MSA. The PSSM encoding adds per-position evolutionary profiles — either derived from the sequence itself (pseudo-PSSM, no tools) or from a real PSI-BLAST run.

### MLP (baseline)

```
Input  (L × 48)  ──flatten──►  Linear(L·48 → 1024)  ──LayerNorm──GELU──Dropout
                                Linear(1024 → 512)   ──LayerNorm──GELU──Dropout
                                Linear(512 → L×L×65)
                                reshape → (L, L, 65 bins)  →  softmax  →  expected distance
```

### Transformer — Evoformer-lite (recommended)

```
Input  (B, L, 48)
   │
   ▼  Linear projection  →  (B, L, 256)  + learnable positional embedding
   │
   ▼  Initial pair representation  (outer sum + relative-position embedding)
   │
   ▼  Evoformer-lite stack  (4 layers)  ×  N_recycles = 3:
   │    ┌── Pair-bias attention: pair(i,j) biases attention between residues i,j
   │    │     (manual QKV proj — correct per-batch per-head, no batch averaging)
   │    ├── Pair track update: outer-product mean of residue embeddings
   │    └── Pre-LN FFN on residue track
   │
   ▼  After each recycle (except last): feed stop-gradient distogram logits
   │   back into pair representation via learned linear layer
   │
   ▼  Final outputs:
        │
        ├── Distogram head  →  (B, L, L, 65 bins)  →  expected distance
        ├── SS head         →  (B, L, 3)  coil / helix / strand
        └── pLDDT head      →  (B, L, 4)  confidence bins (< 1 / 1–2 / 2–4 / > 4 Å error)
```

### Training objective

Combined loss over **64+1 distance bins** + auxiliary heads (Transformer only):

$$\mathcal{L} = \mathcal{L}_{\text{distogram CE}} + 0.5 \cdot \mathcal{L}_{\text{contact BCE}} + 0.2 \cdot \mathcal{L}_{\text{SS CE}} + 0.1 \cdot \mathcal{L}_{\text{pLDDT CE}}$$

- **Distogram CE** (uniform): 64 uniform bins 2–22 Å + 1 “too-far” bin. Same as AlphaFold’s distogram head — far sharper gradients than MSE regression. Backbone geometry is enforced at reconstruction time (MDS bond constraint) rather than in the loss, giving the model’s full gradient budget to long-range contact prediction.
- **Contact BCE**: binary classification for pairs < 8 Å.
- **Secondary structure CE**: unsupervised 3-class labels derived from Cα geometry (no DSSP needed).
- **pLDDT CE**: 4-bin per-residue confidence derived from current-step mean absolute distance error.
- Optimizer: **AdamW** + **cosine annealing** LR + gradient clipping 1.0.

### Structure reconstruction

Coordinates are recovered from expected distances via **gradient-based metric optimisation** (warm-started from classical MDS, refined with Adam + Huber loss + lDDT-proxy regulariser). The MDS now enforces a **backbone bond constraint** (Cα–Cα = 3.8 Å for consecutive residues, penalty weight 5.0), ensuring a physically connected chain even when long-range predictions are noisy. Synthetic training data uses **SS-guided Cα geometry** (helix: 1.5 Å rise + 100°/res, strand: 3.5 Å rise, coil: virtual-bond model) instead of a random walk, giving realistic distance statistics for training.

---

## Results & metrics

Evaluation on three completely **held-out** benchmark proteins (never seen during training), `model_v3.pt` — 200-epoch real-PDB training on 20 diverse proteins across all major structural classes.

| Protein | Length | Class | RMSD (aligned) | local lDDT | Contact F1 | TM-score proxy |
|---|---|---|---|---|---|---|
| 1CRN (crambin) | 46 aa | α+β | 9.6 Å | **43.2** | **0.683** | **0.105** |
| 2GB1 (protein G B1) | 56 aa | α+β | **8.7 Å** | **41.9** | **0.632** | **0.143** |
| 1VII (villin hp) | 36 aa | all-α | 12.5 Å | **46.8** | **0.787** | 0.020 |
| **Mean** | | | **10.3 Å** | **44.0** | **0.700** | **0.089** |

**Improvement over v1 baseline** (synthetic training, one-hot encoding, unconstrained MDS):

| Metric | v1 (synthetic) | **v3 (real PDB)** | Δ |
|---|---|---|---|
| Mean local lDDT | 10.3 | **44.0** | **+327 %** |
| Contact-map F1 | 0.533 | **0.700** | **+31 %** |
| TM-score proxy | 0.009 | **0.089** | **+889 %** |
| Val distance MSE | ~88 Å² | **21 Å²** | **−4×** |

> **Why contact F1 > 0.70 is meaningful**: F1 includes all pairs |i-j| ≥ 1, which is the standard benchmark for educational distance-prediction models. Long-range precision (|i-j| ≥ 12) remains near zero without MSA — this is expected: AlphaFold2, RoseTTAFold, etc. all require hundreds of homologous sequences (MSA) to learn evolutionary couplings. The metric clearly shows where the model stands and what would be needed to raise it further.

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

> Full per-epoch CSV logs are in `train_history.csv`, `train_history_real.csv`, `train_v2.csv`, and `train_v3.csv` (v3 run, 200 epochs on 20 real PDB structures, seq_len=60).

---

## Project structure

```
Scarfold/
├── src/
│   ├── main.py           ← unified entry point (train + evaluate + demo)
│   ├── model.py          ← MLP & Transformer distance predictors (PyTorch)
│   ├── train.py          ← training loop with checkpointing & CSV logging
│   ├── evaluate.py       ← evaluation script, saves JSON results
│   ├── benchmark.py      ← statistical comparison: MLP vs Transformer vs 3 naive baselines
│   ├── ablation.py       ← systematic ablation study (11 conditions, all component combos)
│   ├── pssm.py           ← PSSM encoding: pseudo / PSI-BLAST / runner (50-dim features)
│   ├── download_data.py  ← download PDB structures (RCSB search or CATH S35 non-redundant)
│   ├── utils.py          ← MDS, Kabsch, pLDDT, lDDT, TM-score, BLOSUM62/rich encoding
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
# Option A — CATH S35 non-redundant representatives (recommended for benchmarking)
# Standard set used in published protein structure prediction papers
python src/download_data.py --cath-s35 --n 80 --out data/pdbs

# Option B — RCSB search (small X-ray structures, resolution ≤ 2.5 Å)
python src/download_data.py --n 80 --out data/pdbs

# Option C — built-in curated fallback list (no internet API needed)
python src/download_data.py --n 50 --out data/pdbs --use-fallback
```

### Train on real PDB files

```bash
# Standard training (proteins padded to --max-residues)
python src/train.py \
    --train-from-pdb \
    --pdb-dir data/pdbs \
    --chain A \
    --model transformer \
    --epochs 60 \
    --lr 5e-4 \
    --batch-size 8 \
    --max-residues 80 \
    --checkpoint-dir checkpoints \
    --save-path model_final_real.pt \
    --csv train_history_real.csv

# Variable-length training: each protein trained at its native length (no padding)
python src/train.py \
    --train-from-pdb \
    --variable-length \
    --pdb-dir data/pdbs \
    --model transformer \
    --epochs 60 \
    --lr 5e-4 \
    --save-path model_final_real.pt
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

### Statistical benchmark (MLP vs Transformer vs naive baselines)

```bash
# Synthetic benchmark — compares 5 methods with paired t-tests:
# Random | Seq-separation | Mean-distance | MLP | Transformer
python src/benchmark.py --samples 400 --length 40 --epochs 80 --n-test 30

# Benchmark on real PDB data (requires data/pdbs to be populated)
python src/benchmark.py --train-from-pdb --pdb-dir data/pdbs --epochs 60
```

### Component ablation study

Sweeps all combinations of architecture × features × loss × reconstruction to isolate each improvement:

```bash
# Full ablation (takes ~10–30 min)
python src/ablation.py --samples 400 --length 40 --epochs 60 --n-test 30

# Quick sanity check (~2 min)
python src/ablation.py --quick
```

Results are saved to `results/ablation_<timestamp>.csv` and `.json`.

### Evaluate a trained model

```bash
# On a PDB ID (auto-download)
python src/evaluate.py --model transformer --load-model model_final_real.pt --pdb-id 1crn --chain A

# On a local PDB file
python src/evaluate.py --model transformer --load-model model_final_real.pt --pdb data/my.pdb --chain A

# On a FASTA file
python src/evaluate.py --model mlp --load-model model_final.pt --fasta data/example.fasta

# With a PSI-BLAST PSSM file (upgrades encoding to 50-dim for better accuracy)
python src/evaluate.py --model transformer --load-model model_final_real.pt \
    --pdb-id 1crn --pssm 1crn.pssm
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

**Already implemented:**
- ✅ PSSM encoding (`src/pssm.py`) — pseudo-PSSM (no tools) and real PSI-BLAST parser
- ✅ CATH S35 downloader (`--cath-s35` flag) — standard non-redundant benchmark set
- ✅ Systematic ablation study (`src/ablation.py`) — 11 conditions, isolates each component
- ✅ Naive baselines in `benchmark.py` — random, seq-separation, mean-distance
- ✅ Gradient MDS with lDDT-proxy regulariser — better than closed-form MDS
- ✅ **Distogram head (64+1 bins)** — like AlphaFold; replaces MSE regression
- ✅ **Evoformer-lite Transformer** — pair-bias attention + pair track update (4 layers)
- ✅ Mini-batch training + variable-length PDB training (`--variable-length`)
- ✅ Combined distogram CE + contact BCE loss with AdamW + cosine annealing

**Remaining gaps (true research frontier without GPU cluster):**
- **Full MSA via PSI-BLAST** — `src/pssm.py` has `run_psiblast()` ready; needs UniRef50 database (~70 GB)
- **End-to-end training** — backpropagate through gradient MDS with differentiable lDDT loss
- **Template features** — use known structure templates as additional input (AlphaFold-style)

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