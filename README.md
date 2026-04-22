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

> **New in latest version (v4):** **Triangle multiplication** (AlphaFold2 Algorithms 11 & 12) added to each Evoformer-lite layer — pair (i,j) now aggregates over ALL intermediate residues k, enabling genuine geometric transitivity (if d(i,k) and d(k,j) are known, d(i,j) is inferred). **Expanded training** to 50 diverse proteins across all SCOP classes with **crop augmentation** (4 random L=60 windows per long chain per epoch, turning 50 proteins into thousands of effective training examples), trained for 300 epochs; **contact BCE** auxiliary loss added. Previous (v3): real-PDB training on 20 proteins, variable-length, uniform distogram loss, long-range contact precision metric. Earlier: backbone MDS bond constraint, SS-guided synthetic data, recycling, SS/pLDDT heads, pair-bias attention, PSSM module.

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

### Transformer — Evoformer-lite + Triangle Multiplication (recommended)

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
   │    ├── Pre-LN FFN on residue track
   │    └── Triangle multiplication (NEW — AF2 Algorithms 11 & 12):
   │          • Outgoing:  z_ij += gate ⊙ LN(Σ_k  a_ik ⊙ b_jk)  → shares k between columns
   │          • Incoming:  z_ij += gate ⊙ LN(Σ_k  a_ki ⊙ b_kj)  → shares k between rows
   │          Enforces geometric transitivity: d(i,k)+d(k,j)→d(i,j) without cubic memory
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

**Why triangle multiplication matters**: Without it, the pair track can only update pair (i,j) from sequence information — it can't reason that "i is close to k, and k is close to j, therefore i should be close to j". This is the core geometric primitive that allows AlphaFold's pair representation to converge on a globally consistent 3-D structure. Adding it is the single most impactful architectural change beyond the basic Evoformer.

### Training objective

Combined loss over **64+1 distance bins** + auxiliary heads (Transformer only):

$$\mathcal{L} = \mathcal{L}_{\text{distogram CE}} + 0.3 \cdot \mathcal{L}_{\text{contact BCE}} + 0.2 \cdot \mathcal{L}_{\text{SS CE}}$$

- **Distogram CE** (uniform): 64 uniform bins 2–22 Å + 1 "too-far" bin. Same as AlphaFold's distogram head — far sharper gradients than MSE regression. Backbone geometry is enforced at reconstruction time (MDS bond constraint) rather than in the loss, giving the model's full gradient budget to long-range contact prediction.
- **Contact BCE**: binary cross-entropy for pairs < 8 Å (weight 0.3). Provides direct supervision on the most biologically relevant distance threshold.
- **Secondary structure CE**: unsupervised 3-class labels derived from Cα geometry (no DSSP needed, weight 0.2).
- Optimizer: **AdamW** + **cosine annealing** LR (warmup None, decay to 1% of peak LR) + gradient clipping 1.0.

### Structure reconstruction

Coordinates are recovered from expected distances via **gradient-based metric optimisation** (warm-started from classical MDS, refined with Adam + Huber loss + lDDT-proxy regulariser). The MDS now enforces a **backbone bond constraint** (Cα–Cα = 3.8 Å for consecutive residues, penalty weight 5.0), ensuring a physically connected chain even when long-range predictions are noisy. Synthetic training data uses **SS-guided Cα geometry** (helix: 1.5 Å rise + 100°/res, strand: 3.5 Å rise, coil: virtual-bond model) instead of a random walk, giving realistic distance statistics for training.

---

## Results & metrics

### v5 — LR-weighted contact loss fine-tuning (current best)

Evaluation on **7 diverse completely held-out** proteins (never seen during training), `model_v5.pt` — v4 fine-tuned for 30 epochs with 8× upweighted long-range contact BCE loss.

| Protein | Length | Class | local lDDT | Contact F1 | Long-range P@L/5 | TM-score proxy |
|---|---|---|---|---|---|---|
| 1CRN (crambin) | 46 aa | α+β | **56.1** | 0.706 | **0.778** | 0.067 |
| 1VII (villin hp) | 36 aa | all-α | 40.7 | **0.833** | 0.000 | 0.068 |
| 1LYZ (lysozyme) | 60 aa | α+β | 42.8 | 0.681 | **0.250** | 0.093 |
| 1TRZ (insulin) | 21 aa | all-α | **59.1** | **0.892** | **0.250** | 0.010 |
| 1AHO (scorpion toxin)† | 60 aa | α+β | 26.3 | 0.604 | 0.083 | 0.076 |
| 2PTL (protein L)† | 60 aa | α+β | 53.0 | 0.749 | **0.417** | 0.036 |
| 1TIG (trigger factor)† | 60 aa | α+β | 36.3 | 0.766 | 0.083 | 0.066 |
| **Mean** | | | **44.9** | **0.747** | **0.266** | **0.059** |

*†L=60 crop (model trained on 60aa crops). 1AHO has 64 aa total, 2PTL 62 aa, 1TIG 88 aa.*

**Progression across versions:**

| Metric | v3 | v4 | **v5** | Δ (v3→v5) |
|---|---|---|---|---|
| Contact F1 | 0.700 | 0.720 | **0.747** | **+6.7%** |
| Long-range precision (P@L/5, \|i-j\|≥12) | 0.000 | 0.111 | **0.266** | **first non-zero → +140%** |
| local lDDT | 44.0 | 39.5 | **44.9** | **+2.1%** |

> **Honest context on contact F1**: Our controlled ablation (see `report/report.md`) reveals that a zero-learning sequence-distance baseline achieves F1 = **0.712**, because most contacts in small proteins are between sequence-adjacent residues. Contact F1 alone is insufficient. The scientifically meaningful metric is **long-range precision P@L/5** (|i-j| ≥ 12): v3 achieves 0.000, v4 achieves 0.111, v5 achieves **0.266** across 7 diverse proteins — the pair track with triangle multiplication + LR-weighted training loss is what drives this.

### v3 — Real-PDB training (20 proteins, 200 epochs)

Evaluation on three completely **held-out** benchmark proteins, `model_v3.pt`.

| Protein | Length | Class | RMSD (aligned) | local lDDT | Contact F1 | TM-score proxy |
|---|---|---|---|---|---|---|
| 1CRN (crambin) | 46 aa | α+β | 9.6 Å | **43.2** | **0.683** | **0.105** |
| 2GB1 (protein G B1) | 56 aa | α+β | **8.7 Å** | **41.9** | **0.632** | **0.143** |
| 1VII (villin hp) | 36 aa | all-α | 12.5 Å | **46.8** | **0.787** | 0.020 |
| **Mean** | | | **10.3 Å** | **44.0** | **0.700** | **0.089** |

**Improvement over v1 baseline** (synthetic training, one-hot encoding, unconstrained MDS):

| Metric | v1 (synthetic) | v3 (real PDB) | Δ v1→v3 |
|---|---|---|---|
| Mean local lDDT | 10.3 | **44.0** | **+327 %** |
| Contact-map F1 | 0.533 | **0.700** | **+31 %** |
| TM-score proxy | 0.009 | **0.089** | **+889 %** |
| Val distance MSE | ~88 Å² | **21 Å²** | **−4×** |

> **Why contact F1 > 0.70 alone is NOT the key metric**: Our ablation shows a zero-learning baseline achieves F1 = 0.737, because most contacts in small proteins (L < 60) are between residues close in sequence. The meaningful metric is **long-range precision P@L/5** (|i-j| ≥ 12) — v3 achieves 0.000, v4 achieves 0.178. See `report/report.md` for the full controlled ablation.

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
- ✅ **Triangle multiplication** (v4) — AF2 Algorithms 11 & 12; outgoing + incoming passes; enforces geometric transitivity in the pair track
- ✅ **Crop augmentation** (v4) — 4 random L=60 windows per long chain per epoch; 50 proteins acts as thousands of training examples
- ✅ **Real-PDB training on 50+ diverse proteins** (v4) — all major SCOP structural classes

**Remaining gaps (true research frontier):**
- **Full MSA via PSI-BLAST** — the single biggest gap from AlphaFold. `src/pssm.py` has `run_psiblast()` ready; needs UniRef50 database (~70 GB). Evolutionary coevolution from deep MSA is how AlphaFold learns long-range contacts. Without it, `long_range_precision_L5` stays near zero for this model.
- **Row/column-wise attention on pair representation** — AF2's full Evoformer has row-wise gated self-attention on the pair matrix (quadratic in L, currently omitted for tractability)
- **End-to-end training** — backpropagate through gradient MDS with differentiable lDDT loss
- **Template features** — use known structure templates as additional input (AlphaFold-style)
- **GPU training** — scale to L=256, 500+ proteins, 1000 epochs

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